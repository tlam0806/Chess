#!/usr/bin/env python3
"""Validate and run the production NNUE stage benchmark inside one dyno."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import subprocess
import zlib
from pathlib import Path
from typing import Any, Sequence


CHUNK_SENTINEL = "CHESS_NNUE_FORWARD_STAGE_CHUNK="
END_SENTINEL = "CHESS_NNUE_FORWARD_STAGE_END="
ERROR_SENTINEL = "CHESS_NNUE_FORWARD_STAGE_ERROR="
EXPECTED_BACKEND = "vnni"
EXPECTED_KERNEL = "x86_avx512vnni_256"
STAGES = (
    "s1_input_to_hidden2",
    "s2_hidden2_to_hidden3",
    "s3_hidden3_to_output",
    "full",
)
CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_CPU_STAT = Path("/sys/fs/cgroup/cpu.stat")
OFFICIAL_CORPUS_CHECKSUM = "0xf5399956b2fa9ea8"
OFFICIAL_STAGE_CHECKSUMS = {
    "s1_input_to_hidden2": "0xd264620b4b94ba37",
    "s2_hidden2_to_hidden3": "0x89dbdb9bf3a1ddde",
    "s3_hidden3_to_output": "0x61bbf9ec6191a24b",
    "full": "0x61bbf9ec6191a24b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def read_cpu_identity() -> dict[str, Any]:
    model: str | None = None
    flags: list[str] = []
    text = read_optional_text(Path("/proc/cpuinfo"))
    if text is not None:
        for line in text.splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            normalized = key.strip().lower()
            if normalized == "model name" and model is None:
                model = value.strip()
            elif normalized in {"flags", "features"} and not flags:
                flags = sorted(set(value.split()))
    return {"model": model, "flags": flags}


def read_affinity() -> dict[str, Any] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except OSError:
        return None
    return {"cpus": cpus, "count": len(cpus)}


def parse_cpu_stat(text: str | None) -> dict[str, int] | None:
    if text is None:
        return None
    result: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            result[fields[0]] = int(fields[1])
    return result


def cpu_stat_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {
        key: after[key] - before.get(key, 0)
        for key in sorted(after)
        if after[key] >= before.get(key, 0)
    }


def host_metadata() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "uname": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "architecture": platform.machine(),
        "cpu": read_cpu_identity(),
        "affinity": read_affinity(),
        # DYNO is the only allowlisted environment value. No environment dump
        # is collected or emitted by this runner.
        "dyno": os.environ.get("DYNO"),
        "cgroup": {"cpu_max": read_optional_text(CGROUP_CPU_MAX)},
    }


def validate_benchmark_result(
    result: Any,
    *,
    expected_model_sha256: str,
    expected_backend: str = EXPECTED_BACKEND,
    expected_kernel: str = EXPECTED_KERNEL,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise RuntimeError("benchmark output is not a JSON object")
    if result.get("schema_version") != 2:
        raise RuntimeError("benchmark schema_version is not 2")
    if result.get("protocol") != "phase_nnue_forward_stages_v2":
        raise RuntimeError("benchmark protocol marker is not v2")
    if result.get("status") != "valid":
        raise RuntimeError(f"benchmark status is not valid: {result.get('status')!r}")
    if result.get("backend_requested") != expected_backend:
        raise RuntimeError(
            "benchmark did not confirm requested backend: "
            f"{result.get('backend_requested')!r}"
        )
    if result.get("kernel") != expected_kernel:
        raise RuntimeError(
            f"benchmark reported kernel {result.get('kernel')!r}, "
            f"expected {expected_kernel!r}"
        )
    if result.get("model_sha256") != expected_model_sha256:
        raise RuntimeError("benchmark model SHA-256 does not match staged model")

    correctness = result.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("status") != "pass":
        raise RuntimeError("benchmark SIMD/scalar parity gate did not pass")
    if correctness.get("scalar_parity") is not True:
        raise RuntimeError("benchmark scalar parity was not confirmed")
    if correctness.get("stage_parity") is not True:
        raise RuntimeError("benchmark stage parity was not confirmed")
    if correctness.get("timed_sink_parity") is not True:
        raise RuntimeError("benchmark timed-sink parity was not confirmed")

    expected_timing = {
        "wall_clock": "std::chrono::steady_clock",
        "process_cpu_clock": "std::clock",
        "subtraction": "none",
        "stage_order": "rotating",
    }
    if result.get("timing") != expected_timing:
        raise RuntimeError("benchmark timing declaration does not match protocol v2")

    stages = result.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        raise RuntimeError(f"benchmark stages must be exactly {','.join(STAGES)}")
    for name in STAGES:
        row = stages[name]
        if not isinstance(row, dict):
            raise RuntimeError(f"stage {name} is not an object")
        for field in (
            "median_ns_per_evaluation",
            "mean_ns_per_evaluation",
            "min_ns_per_evaluation",
            "max_ns_per_evaluation",
            "median_process_cpu_ns_per_evaluation",
            "median_cpu_wall_ratio",
            "median_batch_ns",
            "elapsed_ns",
            "process_cpu_ns",
            "checksum",
            "timed_sink",
        ):
            if field not in row:
                raise RuntimeError(f"stage {name} is missing {field}")
        minimum = float(row["min_ns_per_evaluation"])
        median = float(row["median_ns_per_evaluation"])
        maximum = float(row["max_ns_per_evaluation"])
        process_cpu = float(row["median_process_cpu_ns_per_evaluation"])
        cpu_wall_ratio = float(row["median_cpu_wall_ratio"])
        if not all(
            math.isfinite(value)
            for value in (minimum, median, maximum, process_cpu, cpu_wall_ratio)
        ):
            raise RuntimeError(f"stage {name} has non-finite timing values")
        if minimum <= 0 or process_cpu <= 0 or not minimum <= median <= maximum:
            raise RuntimeError(f"stage {name} has invalid timing values")
        if cpu_wall_ratio <= 0:
            raise RuntimeError(f"stage {name} has invalid CPU/wall ratio")
        for field in ("checksum", "timed_sink"):
            if isinstance(row[field], bool) or not isinstance(row[field], (int, str)):
                raise RuntimeError(f"stage {name} has invalid {field}")
    return result


def validate_requested_workload(
    result: dict[str, Any],
    *,
    samples: int,
    seed: int,
    warmup: int,
    repeats: int,
    iterations: dict[str, int],
) -> None:
    corpus = result.get("corpus")
    if not isinstance(corpus, dict):
        raise RuntimeError("benchmark corpus identity is missing")
    if corpus.get("samples") != samples or corpus.get("seed") != seed:
        raise RuntimeError("benchmark corpus parameters do not match request")
    if samples == 256 and seed == 20260825:
        if corpus.get("checksum") != OFFICIAL_CORPUS_CHECKSUM:
            raise RuntimeError("official benchmark corpus checksum mismatch")
    if result.get("warmup_evaluations") != warmup:
        raise RuntimeError("benchmark warmup count does not match request")
    if result.get("repeats") != repeats:
        raise RuntimeError("benchmark repeat count does not match request")

    stages = result["stages"]
    for name, expected_evaluations in iterations.items():
        row = stages[name]
        if samples == 256 and seed == 20260825:
            if row.get("checksum") != OFFICIAL_STAGE_CHECKSUMS[name]:
                raise RuntimeError(f"official stage {name} checksum mismatch")
        if row.get("evaluations_per_repeat") != expected_evaluations:
            raise RuntimeError(f"stage {name} iteration count does not match request")
        for field in ("elapsed_ns", "process_cpu_ns"):
            values = row.get(field)
            if (
                not isinstance(values, list)
                or len(values) != repeats
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in values
                )
            ):
                raise RuntimeError(f"stage {name} has invalid raw {field}")

    batches = result.get("batches")
    if not isinstance(batches, list) or len(batches) != repeats * len(STAGES):
        raise RuntimeError("benchmark batch schedule has invalid length")
    for repeat in range(repeats):
        rows = [row for row in batches if row.get("repeat") == repeat]
        if len(rows) != len(STAGES):
            raise RuntimeError(f"benchmark repeat {repeat} has incomplete batches")
        if {row.get("order") for row in rows} != set(range(len(STAGES))):
            raise RuntimeError(f"benchmark repeat {repeat} has invalid stage order")
        if {row.get("stage") for row in rows} != set(STAGES):
            raise RuntimeError(f"benchmark repeat {repeat} has invalid stage set")
        for row in rows:
            stage = row["stage"]
            if row.get("evaluations") != iterations[stage]:
                raise RuntimeError(f"batch {stage} evaluation count does not match")
            for field in ("wall_ns", "process_cpu_ns"):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise RuntimeError(f"batch {stage} has invalid {field}")


def emit_result(result: dict[str, Any]) -> None:
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    compressed = zlib.compress(encoded, level=9)
    payload = base64.b64encode(compressed).decode("ascii")
    chunk_size = 6000
    chunks = [
        payload[index : index + chunk_size]
        for index in range(0, len(payload), chunk_size)
    ]
    for index, chunk in enumerate(chunks, start=1):
        print(f"{CHUNK_SENTINEL}{index}/{len(chunks)}:{chunk}", flush=True)
    print(
        END_SENTINEL
        + json.dumps(
            {
                "chunks": len(chunks),
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "json_sha256": hashlib.sha256(encoded).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--s1-iterations", type=int, required=True)
    parser.add_argument("--s2-iterations", type=int, required=True)
    parser.add_argument("--s3-iterations", type=int, required=True)
    parser.add_argument("--full-iterations", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        for label in ("benchmark", "model"):
            path = getattr(args, label)
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        for label in (
            "warmup",
            "repeats",
            "s1_iterations",
            "s2_iterations",
            "s3_iterations",
            "full_iterations",
            "samples",
        ):
            if getattr(args, label) <= 0:
                raise ValueError(f"--{label} must be positive")
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive")

        runner_sha256 = sha256_file(Path(__file__).resolve())
        if runner_sha256 != args.expected_runner_sha256:
            raise RuntimeError("remote runner SHA-256 mismatch")
        model_sha256 = sha256_file(args.model)
        if model_sha256 != args.expected_model_sha256:
            raise RuntimeError("staged model SHA-256 mismatch")

        host = host_metadata()
        cgroup_cpu_stat_before = parse_cpu_stat(read_optional_text(CGROUP_CPU_STAT))
        environment = os.environ.copy()
        environment["CHESS_NNUE_BACKEND"] = EXPECTED_BACKEND
        completed = subprocess.run(
            [
                str(args.benchmark),
                "--model",
                str(args.model),
                "--backend",
                EXPECTED_BACKEND,
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
                "--s1-iterations",
                str(args.s1_iterations),
                "--s2-iterations",
                str(args.s2_iterations),
                "--s3-iterations",
                str(args.s3_iterations),
                "--full-iterations",
                str(args.full_iterations),
                "--samples",
                str(args.samples),
                "--seed",
                str(args.seed),
            ],
            text=True,
            capture_output=True,
            env=environment,
            timeout=args.timeout_seconds,
            check=False,
        )
        cgroup_cpu_stat_after = parse_cpu_stat(read_optional_text(CGROUP_CPU_STAT))
        if completed.returncode != 0:
            raise RuntimeError(
                "benchmark failed: "
                f"returncode={completed.returncode} "
                f"stdout_tail={completed.stdout[-2000:]!r} "
                f"stderr_tail={completed.stderr[-2000:]!r}"
            )
        try:
            raw_result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"benchmark stdout is not one JSON object: {error}"
            ) from error
        result = validate_benchmark_result(
            raw_result,
            expected_model_sha256=model_sha256,
        )
        requested_iterations = {
            "s1_input_to_hidden2": args.s1_iterations,
            "s2_hidden2_to_hidden3": args.s2_iterations,
            "s3_hidden3_to_output": args.s3_iterations,
            "full": args.full_iterations,
        }
        validate_requested_workload(
            result,
            samples=args.samples,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            iterations=requested_iterations,
        )
        result["remote_identity"] = {
            "benchmark_sha256": sha256_file(args.benchmark),
            "model_sha256": model_sha256,
            "runner_sha256": runner_sha256,
        }
        result["host"] = host
        result["cgroup_cpu_stat"] = {
            "before": cgroup_cpu_stat_before,
            "after": cgroup_cpu_stat_after,
            "delta": cpu_stat_delta(
                cgroup_cpu_stat_before,
                cgroup_cpu_stat_after,
            ),
        }
        emit_result(result)
        return 0
    except Exception as error:
        print(
            ERROR_SENTINEL
            + json.dumps(
                {"error_type": type(error).__name__, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
