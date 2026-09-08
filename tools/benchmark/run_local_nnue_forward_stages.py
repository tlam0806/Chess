#!/usr/bin/env python3
"""Run protocol-v2 NNUE forward stages in three independent local processes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.benchmark.run_nnue_forward_stages_remote import (
        STAGES,
        validate_benchmark_result,
        validate_requested_workload,
    )
except ModuleNotFoundError:
    from run_nnue_forward_stages_remote import (  # type: ignore[no-redef]
        STAGES,
        validate_benchmark_result,
        validate_requested_workload,
    )


RUNS = 3
BACKEND = "neon"
EXPECTED_KERNEL = "arm_neon_dotprod_i8mm"
SAMPLES = 256
SEED = 20260825
WARMUP = 10_000
REPEATS = 12
ITERATIONS = {
    "s1_input_to_hidden2": 4_000_000,
    "s2_hidden2_to_hidden3": 20_000_000,
    "s3_hidden3_to_output": 100_000_000,
    "full": 3_000_000,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sysctl_value(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def local_platform_metadata() -> dict[str, Any]:
    uname = platform.uname()
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
            affinity = {"cpus": cpus, "count": len(cpus)}
        except OSError:
            affinity = None
    return {
        "uname": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "architecture": platform.machine(),
        "cpu": {
            "brand": sysctl_value("machdep.cpu.brand_string"),
            "model": sysctl_value("hw.model"),
            "dotprod": sysctl_value("hw.optional.arm.FEAT_DotProd"),
            "i8mm": sysctl_value("hw.optional.arm.FEAT_I8MM"),
        },
        "affinity": affinity,
    }


def benchmark_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.benchmark),
        "--model",
        str(args.model),
        "--backend",
        BACKEND,
        "--profile",
        "full",
        # Explicit overrides after --profile lock every declared v2 parameter.
        "--samples",
        str(SAMPLES),
        "--warmup",
        str(WARMUP),
        "--repeats",
        str(REPEATS),
        "--s1-iterations",
        str(ITERATIONS["s1_input_to_hidden2"]),
        "--s2-iterations",
        str(ITERATIONS["s2_hidden2_to_hidden3"]),
        "--s3-iterations",
        str(ITERATIONS["s3_hidden3_to_output"]),
        "--full-iterations",
        str(ITERATIONS["full"]),
        "--seed",
        str(SEED),
    ]


def run_one(
    args: argparse.Namespace,
    *,
    ordinal: int,
    output_dir: Path,
    model_sha256: str,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CHESS_NNUE_BACKEND"] = BACKEND
    completed = subprocess.run(
        benchmark_command(args),
        text=True,
        capture_output=True,
        env=environment,
        timeout=args.timeout_seconds,
        check=False,
    )
    (output_dir / f"local-forward-stages-run-{ordinal}.stdout.json").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output_dir / f"local-forward-stages-run-{ordinal}.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"local benchmark process {ordinal} failed with "
            f"exit {completed.returncode}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"local benchmark process {ordinal} did not emit one JSON object"
        ) from error
    validate_benchmark_result(
        result,
        expected_model_sha256=model_sha256,
        expected_backend=BACKEND,
        expected_kernel=EXPECTED_KERNEL,
    )
    validate_requested_workload(
        result,
        samples=SAMPLES,
        seed=SEED,
        warmup=WARMUP,
        repeats=REPEATS,
        iterations=ITERATIONS,
    )
    result["local_run"] = {
        "ordinal": ordinal,
        "independent_process": True,
        "source_label": args.source_label,
    }
    (output_dir / f"local-forward-stages-run-{ordinal}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def summarize(
    args: argparse.Namespace,
    results: Sequence[dict[str, Any]],
    *,
    benchmark_sha256: str,
    model_sha256: str,
    runner_sha256: str,
    validator_sha256: str,
) -> dict[str, Any]:
    identity = {
        (
            result["model_sha256"],
            result["kernel"],
            result["corpus"]["checksum"],
            result["correctness"]["checksum"],
        )
        for result in results
    }
    failures = []
    if len(identity) != 1:
        failures.append("model/kernel/corpus/output identity differs across processes")
    stages = {}
    for name in STAGES:
        wall = [
            float(result["stages"][name]["median_ns_per_evaluation"])
            for result in results
        ]
        process_cpu = [
            float(result["stages"][name]["median_process_cpu_ns_per_evaluation"])
            for result in results
        ]
        cpu_wall_ratios = [
            float(result["stages"][name]["median_cpu_wall_ratio"]) for result in results
        ]
        checksums = [result["stages"][name]["checksum"] for result in results]
        if len(set(checksums)) != 1:
            failures.append(f"stage {name} checksum differs across processes")
        stages[name] = {
            "process_wall_medians_ns_per_evaluation": wall,
            "median_of_process_wall_medians_ns_per_evaluation": statistics.median(wall),
            "process_cpu_medians_ns_per_evaluation": process_cpu,
            "median_of_process_cpu_medians_ns_per_evaluation": statistics.median(
                process_cpu
            ),
            "process_median_cpu_wall_ratios": cpu_wall_ratios,
            "checksums": checksums,
        }
    return {
        "schema_version": 2,
        "protocol": "phase_nnue_forward_stages_v2",
        "status": "valid" if not failures else "invalid",
        "backend_requested": BACKEND,
        "kernel": EXPECTED_KERNEL,
        "source_label": args.source_label,
        "independent_processes": RUNS,
        "parameters": {
            "samples": SAMPLES,
            "seed": SEED,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "iterations": ITERATIONS,
        },
        "identity": {
            "benchmark_sha256": benchmark_sha256,
            "model_sha256": model_sha256,
            "runner_sha256": runner_sha256,
            "validator_sha256": validator_sha256,
        },
        "platform": local_platform_metadata(),
        "stages": stages,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=repo_root / "build-lto/benchmark_phase_nnue_forward_stages",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.benchmark.is_file() or not args.model.is_file():
        raise SystemExit("benchmark binary or production model is missing")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    output_dir = args.output_dir or (
        Path("logs") / f"nnue_forward_stages_local_v2_{utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = output_dir / "snapshots"
    snapshots.mkdir()

    runner_path = Path(__file__).resolve()
    validator_path = runner_path.with_name("run_nnue_forward_stages_remote.py")
    runner_bytes = runner_path.read_bytes()
    validator_bytes = validator_path.read_bytes()
    (snapshots / "local-runner.py").write_bytes(runner_bytes)
    (snapshots / "protocol-validator.py").write_bytes(validator_bytes)
    benchmark_sha256 = sha256_file(args.benchmark)
    model_sha256 = sha256_file(args.model)
    runner_sha256 = sha256_bytes(runner_bytes)
    validator_sha256 = sha256_bytes(validator_bytes)
    request = {
        "schema_version": 2,
        "protocol": "phase_nnue_forward_stages_v2",
        "source_label": args.source_label,
        "independent_processes": RUNS,
        "parameters": {
            "samples": SAMPLES,
            "seed": SEED,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "iterations": ITERATIONS,
        },
        "identity": {
            "benchmark_sha256": benchmark_sha256,
            "model_sha256": model_sha256,
            "runner_sha256": runner_sha256,
            "validator_sha256": validator_sha256,
        },
        "platform": local_platform_metadata(),
    }
    (output_dir / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results = []
    for ordinal in range(1, RUNS + 1):
        print(f"starting independent local process {ordinal}/{RUNS}", flush=True)
        results.append(
            run_one(
                args,
                ordinal=ordinal,
                output_dir=output_dir,
                model_sha256=model_sha256,
            )
        )
    summary = summarize(
        args,
        results,
        benchmark_sha256=benchmark_sha256,
        model_sha256=model_sha256,
        runner_sha256=runner_sha256,
        validator_sha256=validator_sha256,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {output_dir / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())
