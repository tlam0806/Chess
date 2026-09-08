#!/usr/bin/env python3
"""Run the production NNUE forward-stage benchmark on isolated Heroku dynos."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import zlib
from pathlib import Path
from typing import Any, Sequence


CHUNK_SENTINEL = "CHESS_NNUE_FORWARD_STAGE_CHUNK="
END_SENTINEL = "CHESS_NNUE_FORWARD_STAGE_END="
ERROR_SENTINEL = "CHESS_NNUE_FORWARD_STAGE_ERROR="
EXPECTED_KERNEL = "x86_avx512vnni_256"
STAGES = (
    "s1_input_to_hidden2",
    "s2_hidden2_to_hidden3",
    "s3_hidden3_to_output",
    "full",
)
CANONICAL_PRODUCTION_APP = "stormy-garden-92984"
PROFILE_DEFAULTS = {
    "smoke": {
        "warmup": 1_000,
        "repeats": 3,
        "samples": 64,
        "iterations": {
            "s1": 100_000,
            "s2": 500_000,
            "s3": 2_000_000,
            "full": 100_000,
        },
    },
    "full": {
        "warmup": 10_000,
        "repeats": 12,
        "samples": 256,
        "iterations": {
            "s1": 4_000_000,
            "s2": 20_000_000,
            "s3": 100_000_000,
            "full": 3_000_000,
        },
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_checked(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=True,
    )


def require_idle_staging_app(app: str) -> None:
    completed = run_checked(("heroku", "ps", "--app", app, "--json"))
    dynos = json.loads(completed.stdout)
    if not isinstance(dynos, list) or dynos:
        count = len(dynos) if isinstance(dynos, list) else "unknown"
        raise RuntimeError(f"staging app is not idle ({count} active dynos)")


def force_formation_zero(app: str, process_type: str) -> None:
    run_checked(("heroku", "ps:scale", f"{process_type}=0", "--app", app))


def remote_command(
    args: argparse.Namespace,
    *,
    model_sha256: str,
    runner_sha256: str,
) -> list[str]:
    values = PROFILE_DEFAULTS[args.profile]
    return [
        "heroku",
        "run",
        "--app",
        args.app,
        "--type",
        args.process_type,
        "--size",
        "basic",
        "--no-tty",
        "--no-notify",
        "--no-launcher",
        "--exit-code",
        "--",
        "python3",
        args.remote_runner,
        "--benchmark",
        args.remote_benchmark,
        "--model",
        args.remote_model,
        "--expected-model-sha256",
        model_sha256,
        "--expected-runner-sha256",
        runner_sha256,
        "--warmup",
        str(values["warmup"]),
        "--repeats",
        str(values["repeats"]),
        "--s1-iterations",
        str(values["iterations"]["s1"]),
        "--s2-iterations",
        str(values["iterations"]["s2"]),
        "--s3-iterations",
        str(values["iterations"]["s3"]),
        "--full-iterations",
        str(values["iterations"]["full"]),
        "--samples",
        str(values["samples"]),
        "--seed",
        str(args.seed),
    ]


def run_one(
    args: argparse.Namespace,
    *,
    ordinal: int,
    output_dir: Path,
    model_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    require_idle_staging_app(args.app)
    command = remote_command(
        args,
        model_sha256=model_sha256,
        runner_sha256=runner_sha256,
    )
    console_path = output_dir / f"heroku-forward-stages-run-{ordinal}.console.log"
    chunks: dict[int, str] = {}
    expected_chunks: int | None = None
    end_metadata: dict[str, Any] | None = None
    remote_error: dict[str, Any] | None = None
    try:
        with console_path.open("w", encoding="utf-8") as console:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                console.write(line)
                console.flush()
                stripped = line.rstrip("\r\n")
                chunk_at = stripped.find(CHUNK_SENTINEL)
                end_at = stripped.find(END_SENTINEL)
                error_at = stripped.find(ERROR_SENTINEL)
                if chunk_at >= 0:
                    encoded = stripped[chunk_at + len(CHUNK_SENTINEL) :]
                    descriptor, value = encoded.split(":", 1)
                    index_text, count_text = descriptor.split("/", 1)
                    index = int(index_text)
                    count = int(count_text)
                    if expected_chunks is not None and expected_chunks != count:
                        raise RuntimeError("remote result changed chunk count")
                    expected_chunks = count
                    chunks[index] = value
                elif end_at >= 0:
                    end_metadata = json.loads(stripped[end_at + len(END_SENTINEL) :])
                elif error_at >= 0:
                    remote_error = json.loads(
                        stripped[error_at + len(ERROR_SENTINEL) :]
                    )
                    print(f"run {ordinal} remote error: {remote_error}", flush=True)
                elif len(stripped) < 500:
                    print(stripped, flush=True)
            return_code = process.wait()
    finally:
        force_formation_zero(args.app, args.process_type)

    if return_code != 0 or end_metadata is None or expected_chunks is None:
        detail = remote_error or "missing chunked result"
        raise RuntimeError(
            f"one-off {ordinal} failed (exit={return_code}): {detail}; "
            f"see {console_path}"
        )
    if set(chunks) != set(range(1, expected_chunks + 1)):
        raise RuntimeError(f"one-off {ordinal} returned incomplete chunks")
    compressed = base64.b64decode(
        "".join(chunks[index] for index in range(1, expected_chunks + 1)),
        validate=True,
    )
    if sha256_bytes(compressed) != end_metadata.get("compressed_sha256"):
        raise RuntimeError(f"one-off {ordinal} compressed hash mismatch")
    encoded = zlib.decompress(compressed)
    if sha256_bytes(encoded) != end_metadata.get("json_sha256"):
        raise RuntimeError(f"one-off {ordinal} JSON hash mismatch")
    result = json.loads(encoded)
    if result.get("schema_version") != 2:
        raise RuntimeError(f"one-off {ordinal} returned unexpected schema")
    if result.get("protocol") != "phase_nnue_forward_stages_v2":
        raise RuntimeError(f"one-off {ordinal} returned unexpected protocol")
    if result.get("status") != "valid":
        raise RuntimeError(f"one-off {ordinal} returned invalid status")
    if result.get("kernel") != EXPECTED_KERNEL:
        raise RuntimeError(f"one-off {ordinal} did not execute expected VNNI kernel")
    if result.get("model_sha256") != model_sha256:
        raise RuntimeError(f"one-off {ordinal} model SHA-256 mismatch")
    result["heroku_run"] = {
        "app": args.app,
        "dyno_ordinal": ordinal,
        "process_type": args.process_type,
        "profile": args.profile,
        "size": "basic",
        "source_label": args.source_label,
    }
    result_path = output_dir / f"heroku-forward-stages-run-{ordinal}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"run {ordinal}: saved {result_path}", flush=True)
    return result


def summarize(
    args: argparse.Namespace, results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    identities = {
        (
            result["model_sha256"],
            result["remote_identity"]["benchmark_sha256"],
            result["remote_identity"]["runner_sha256"],
            result["kernel"],
            result["corpus"]["checksum"],
            result["correctness"]["checksum"],
        )
        for result in results
    }
    failures = []
    if len(identities) != 1:
        failures.append(
            "binary/model/runner/kernel/corpus/output identity differs across dynos"
        )
    for ordinal, result in enumerate(results, start=1):
        if result.get("correctness", {}).get("status") != "pass":
            failures.append(f"run {ordinal} parity did not pass")

    stages = {}
    for name in STAGES:
        wall_medians = [
            float(result["stages"][name]["median_ns_per_evaluation"])
            for result in results
        ]
        process_cpu_medians = [
            float(result["stages"][name]["median_process_cpu_ns_per_evaluation"])
            for result in results
        ]
        cpu_wall_ratios = [
            float(result["stages"][name]["median_cpu_wall_ratio"]) for result in results
        ]
        stages[name] = {
            "dyno_wall_medians_ns_per_evaluation": wall_medians,
            "median_of_dyno_wall_medians_ns_per_evaluation": statistics.median(
                wall_medians
            ),
            "dyno_process_cpu_medians_ns_per_evaluation": process_cpu_medians,
            "median_of_dyno_process_cpu_medians_ns_per_evaluation": statistics.median(
                process_cpu_medians
            ),
            "dyno_median_cpu_wall_ratios": cpu_wall_ratios,
            "checksums": [result["stages"][name]["checksum"] for result in results],
        }
        if len({str(value) for value in stages[name]["checksums"]}) != 1:
            failures.append(f"stage {name} checksum differs across dynos")
    return {
        "schema_version": 2,
        "protocol": "phase_nnue_forward_stages_v2",
        "status": "valid" if not failures else "invalid",
        "backend_requested": "vnni",
        "kernel": EXPECTED_KERNEL,
        "profile": args.profile,
        "runs": len(results),
        "source_label": args.source_label,
        "design": {
            "dedicated_one_off_dynos": True,
            "dyno_size": "basic",
            "formation_outside_runs": 0,
            "lto_enabled": True,
            "production_app_refused": True,
        },
        "hosts": [
            {
                "host": result.get("host"),
                "cgroup_cpu_stat": result.get("cgroup_cpu_stat"),
            }
            for result in results
        ],
        "stages": stages,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    remote_root = "/app/chess-engine"
    model_rel = (
        "models/quantized_scale_grid/"
        "old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/"
        "phase_quantized_nnue.bin"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--production-app", default=CANONICAL_PRODUCTION_APP)
    parser.add_argument("--process-type", default="benchmark")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--local-model",
        type=Path,
        default=repo_root / model_rel,
    )
    parser.add_argument(
        "--local-remote-runner",
        type=Path,
        default=repo_root / "tools/benchmark/run_nnue_forward_stages_remote.py",
    )
    parser.add_argument(
        "--remote-runner",
        default=f"{remote_root}/tools/benchmark/run_nnue_forward_stages_remote.py",
    )
    parser.add_argument(
        "--remote-benchmark",
        default=f"{remote_root}/build-release/benchmark_phase_nnue_forward_stages",
    )
    parser.add_argument("--remote-model", default=f"{remote_root}/{model_rel}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    production_apps = {CANONICAL_PRODUCTION_APP, args.production_app}
    if args.app in production_apps:
        raise SystemExit(f"refusing production app: {args.app}")
    if os.environ.get("CHESS_HEROKU_STAGING_APP") != args.app:
        raise SystemExit("CHESS_HEROKU_STAGING_APP safety latch mismatch")
    if args.process_type != "benchmark":
        raise SystemExit("--process-type must be benchmark")
    runs = args.runs if args.runs is not None else (1 if args.profile == "smoke" else 3)
    if runs <= 0:
        raise SystemExit("--runs must be positive")
    if not args.local_model.is_file() or not args.local_remote_runner.is_file():
        raise SystemExit("local model or remote runner is missing")

    model_sha256 = sha256_file(args.local_model)
    runner_bytes = args.local_remote_runner.read_bytes()
    runner_sha256 = sha256_bytes(runner_bytes)
    local_runner_bytes = Path(__file__).resolve().read_bytes()
    output_dir = args.output_dir or (
        Path("logs") / f"nnue_forward_stages_heroku_{args.profile}_{utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "request.json").exists():
        raise SystemExit(f"output directory already contains a request: {output_dir}")
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(exist_ok=False)
    (snapshots / "local-runner.py").write_bytes(local_runner_bytes)
    (snapshots / "remote-runner.py").write_bytes(runner_bytes)
    request = {
        "schema_version": 2,
        "protocol": "phase_nnue_forward_stages_v2",
        "app": args.app,
        "profile": args.profile,
        "runs": runs,
        "size": "basic",
        "process_type": args.process_type,
        "source_label": args.source_label,
        "model_sha256": model_sha256,
        "local_runner_sha256": sha256_bytes(local_runner_bytes),
        "remote_runner_sha256": runner_sha256,
        "build": {"CHESS_ENABLE_LTO": True},
        "parameters": PROFILE_DEFAULTS[args.profile],
        "seed": args.seed,
    }
    (output_dir / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )

    force_formation_zero(args.app, args.process_type)
    results = []
    try:
        for ordinal in range(1, runs + 1):
            print(f"starting isolated Basic one-off {ordinal}/{runs}", flush=True)
            results.append(
                run_one(
                    args,
                    ordinal=ordinal,
                    output_dir=output_dir,
                    model_sha256=model_sha256,
                    runner_sha256=runner_sha256,
                )
            )
    finally:
        force_formation_zero(args.app, args.process_type)
    summary = summarize(args, results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"summary: {output_dir / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())
