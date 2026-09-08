#!/usr/bin/env python3
"""Run scalar/AVX2/VNNI on the same Heroku dynos with balanced ordering."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import random
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.analyze import compare_nnue_backend_benchmarks as comparator
    from tools.benchmark import run_heroku_uci_platform_benchmark as platform_runner
except ModuleNotFoundError:  # Direct execution adds only this category directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.analyze import compare_nnue_backend_benchmarks as comparator
    from tools.benchmark import run_heroku_uci_platform_benchmark as platform_runner


PROGRESS_SENTINEL = "CHESS_NNUE_MATRIX_PROGRESS="
CHUNK_SENTINEL = "CHESS_NNUE_MATRIX_CHUNK="
END_SENTINEL = "CHESS_NNUE_MATRIX_END="
ERROR_SENTINEL = "CHESS_NNUE_MATRIX_ERROR="
BACKENDS = ("scalar", "avx2", "vnni")
LATIN_SQUARE = (
    ("scalar", "avx2", "vnni"),
    ("avx2", "vnni", "scalar"),
    ("vnni", "scalar", "avx2"),
)
PROMOTION_DEPTHS = (7, 8)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def remote_command(
    args: argparse.Namespace,
    order: Sequence[str],
    ordinal: int,
    harness_sha256: str,
    spec_sha256: str,
    runner_sha256: str,
) -> list[str]:
    return [
        "heroku",
        "run",
        "--app",
        args.app,
        "--type",
        args.process_type,
        "--size",
        args.size,
        "--no-tty",
        "--no-notify",
        "--no-launcher",
        "--exit-code",
        "--",
        "python3",
        args.remote_matrix_runner,
        "--engine",
        args.remote_engine,
        "--test",
        args.remote_test,
        "--model",
        args.remote_model,
        "--parity",
        args.remote_parity,
        "--config",
        args.remote_config,
        "--spec",
        args.remote_spec,
        "--harness",
        args.remote_harness,
        "--engine-cwd",
        args.remote_engine_cwd,
        "--order",
        ",".join(order),
        "--profile",
        args.profile,
        "--host-label",
        f"heroku:{args.app}:{args.size}:matrix-{ordinal}",
        "--source-label",
        args.source_label,
        "--harness-sha256",
        harness_sha256,
        "--spec-sha256",
        spec_sha256,
        "--runner-sha256",
        runner_sha256,
    ]


def run_one(
    args: argparse.Namespace,
    order: Sequence[str],
    ordinal: int,
    output_dir: Path,
    harness_sha256: str,
    spec_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    command = remote_command(
        args, order, ordinal, harness_sha256, spec_sha256, runner_sha256
    )
    console_path = output_dir / f"heroku-matrix-run-{ordinal}.console.log"
    chunks: dict[int, str] = {}
    expected_chunks: int | None = None
    end_metadata: dict[str, Any] | None = None
    remote_error: dict[str, Any] | None = None
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
            progress_at = stripped.find(PROGRESS_SENTINEL)
            chunk_at = stripped.find(CHUNK_SENTINEL)
            end_at = stripped.find(END_SENTINEL)
            error_at = stripped.find(ERROR_SENTINEL)
            if progress_at >= 0:
                payload = json.loads(stripped[progress_at + len(PROGRESS_SENTINEL) :])
                print(
                    f"matrix {ordinal}: {json.dumps(payload, sort_keys=True)}",
                    flush=True,
                )
            elif chunk_at >= 0:
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
                remote_error = json.loads(stripped[error_at + len(ERROR_SENTINEL) :])
                print(f"matrix {ordinal} remote error: {remote_error}", flush=True)
            elif len(stripped) < 500:
                print(stripped, flush=True)
        return_code = process.wait()

    if return_code != 0 or end_metadata is None or expected_chunks is None:
        detail = remote_error or "missing chunked result"
        raise RuntimeError(
            f"matrix one-off {ordinal} failed (exit={return_code}): {detail}; "
            f"see {console_path}"
        )
    if set(chunks) != set(range(1, expected_chunks + 1)):
        raise RuntimeError(f"matrix one-off {ordinal} returned incomplete chunks")
    compressed = base64.b64decode(
        "".join(chunks[index] for index in range(1, expected_chunks + 1)),
        validate=True,
    )
    if sha256_bytes(compressed) != end_metadata["compressed_sha256"]:
        raise RuntimeError(f"matrix one-off {ordinal} compressed hash mismatch")
    encoded = zlib.decompress(compressed)
    if sha256_bytes(encoded) != end_metadata["json_sha256"]:
        raise RuntimeError(f"matrix one-off {ordinal} JSON hash mismatch")
    result = json.loads(encoded)
    if result.get("status") != "valid":
        raise RuntimeError(f"matrix one-off {ordinal} returned invalid status")
    if result.get("order") != list(order):
        raise RuntimeError(f"matrix one-off {ordinal} order mismatch")
    if result.get("auto_kernel") != comparator.EXPECTED_KERNEL["vnni"]:
        raise RuntimeError(
            f"matrix one-off {ordinal}: auto did not select VNNI: "
            f"{result.get('auto_kernel')!r}"
        )
    if result.get("fixed_depth_parity", {}).get("status") != "pass":
        raise RuntimeError(f"matrix one-off {ordinal}: backend parity failed")
    result["runner"] = {
        "app": args.app,
        "size": args.size,
        "process_type": args.process_type,
        "profile": args.profile,
        "ordinal": ordinal,
        "order": list(order),
    }
    result_path = output_dir / f"heroku-matrix-run-{ordinal}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"matrix {ordinal}: saved {result_path}", flush=True)
    return result


def paired_backend_speedups(
    all_runs: dict[str, list[dict[str, Any]]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    run_count = len(all_runs["scalar"])
    if run_count == 0 or any(len(runs) != run_count for runs in all_runs.values()):
        return {"status": "fail", "error": "backend run counts differ"}

    metrics = sorted(
        {
            (row["mode"], row["limit"])
            for run in all_runs["scalar"]
            for row in run["observations"]
        }
    )
    output = []
    for metric_index, (mode, limit) in enumerate(metrics):
        paired_runs: list[list[dict[str, dict[str, Any]]]] = []
        for run_index in range(run_count):
            by_backend: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
            for backend in BACKENDS:
                rows = [
                    row
                    for row in all_runs[backend][run_index]["observations"]
                    if row["mode"] == mode and row["limit"] == limit
                ]
                by_backend[backend] = {
                    (row["round"], row["position_id"]): row for row in rows
                }
            keys = set(by_backend["scalar"])
            if not keys or any(set(rows) != keys for rows in by_backend.values()):
                return {
                    "status": "fail",
                    "error": (
                        f"paired cases differ for {mode}={limit}, run {run_index + 1}"
                    ),
                }
            paired_runs.append(
                [
                    {backend: by_backend[backend][key] for backend in BACKENDS}
                    for key in sorted(keys)
                ]
            )

        def ratios(
            sampled: Sequence[Sequence[dict[str, dict[str, Any]]]],
        ) -> dict[str, float]:
            totals = {backend: {"nodes": 0, "elapsed_ns": 0} for backend in BACKENDS}
            for cases in sampled:
                for case in cases:
                    for backend in BACKENDS:
                        totals[backend]["nodes"] += int(case[backend]["nodes"])
                        totals[backend]["elapsed_ns"] += int(
                            case[backend]["elapsed_ns"]
                        )
            nps = {
                backend: values["nodes"] * 1_000_000_000.0 / values["elapsed_ns"]
                for backend, values in totals.items()
            }
            return {
                "avx2_over_scalar": nps["avx2"] / nps["scalar"],
                "vnni_over_scalar": nps["vnni"] / nps["scalar"],
                "vnni_over_avx2": nps["vnni"] / nps["avx2"],
            }

        point = ratios(paired_runs)
        rng = random.Random(seed + metric_index)
        draws = {name: [] for name in point}
        for _ in range(replicates):
            sampled_runs = []
            for _ in range(run_count):
                cases = paired_runs[rng.randrange(run_count)]
                sampled_runs.append([rng.choice(cases) for _ in cases])
            sampled_ratios = ratios(sampled_runs)
            for name, value in sampled_ratios.items():
                draws[name].append(value)
        output.append(
            {
                "mode": mode,
                "limit": limit,
                "paired_cases_per_backend": sum(len(cases) for cases in paired_runs),
                "ratios": {
                    name: {
                        "estimate": point[name],
                        "paired_bootstrap_interval95": [
                            platform_runner.percentile(values, 0.025),
                            platform_runner.percentile(values, 0.975),
                        ],
                    }
                    for name, values in draws.items()
                },
            }
        )
    return {
        "status": "pass",
        "replicates": replicates,
        "resampling": "dynos_then_matched_cases_within_dyno",
        "metrics": output,
    }


def full_promotion_gate(
    all_runs: dict[str, list[dict[str, Any]]],
    paired_speedups: dict[str, Any],
) -> dict[str, Any]:
    """Require every sampled dyno and the paired CI to favor SIMD."""
    failures: list[str] = []
    run_count = len(all_runs["scalar"])
    if run_count < len(LATIN_SQUARE):
        failures.append("fewer than three matched dynos")
    if any(len(runs) != run_count for runs in all_runs.values()):
        failures.append("backend run counts differ")

    per_dyno = []
    if not failures:
        for run_index in range(run_count):
            for depth in PROMOTION_DEPTHS:
                nps: dict[str, float] = {}
                for backend in BACKENDS:
                    rows = [
                        row
                        for row in all_runs[backend][run_index]["observations"]
                        if row["mode"] == "fixed_depth" and row["limit"] == depth
                    ]
                    if not rows:
                        failures.append(
                            f"missing depth {depth} for {backend} on dyno {run_index + 1}"
                        )
                        continue
                    elapsed_ns = sum(int(row["elapsed_ns"]) for row in rows)
                    nodes = sum(int(row["nodes"]) for row in rows)
                    if elapsed_ns <= 0 or nodes <= 0:
                        failures.append(
                            f"invalid depth {depth} totals for {backend} on dyno {run_index + 1}"
                        )
                        continue
                    nps[backend] = nodes * 1_000_000_000.0 / elapsed_ns
                if set(nps) != set(BACKENDS):
                    continue
                ratios = {
                    "avx2_over_scalar": nps["avx2"] / nps["scalar"],
                    "vnni_over_scalar": nps["vnni"] / nps["scalar"],
                    "vnni_over_avx2": nps["vnni"] / nps["avx2"],
                }
                per_dyno.append(
                    {
                        "dyno_ordinal": run_index + 1,
                        "depth": depth,
                        "nps": nps,
                        "ratios": ratios,
                    }
                )
                for name, value in ratios.items():
                    if value <= 1.0:
                        failures.append(
                            f"{name}={value:.6f} on dyno {run_index + 1}, depth {depth}"
                        )

    interval_checks = []
    if paired_speedups.get("status") != "pass":
        failures.append("paired bootstrap did not pass")
    else:
        metrics = {
            (row["mode"], row["limit"]): row
            for row in paired_speedups.get("metrics", [])
        }
        for depth in PROMOTION_DEPTHS:
            metric = metrics.get(("fixed_depth", depth))
            if metric is None:
                failures.append(f"missing paired bootstrap for depth {depth}")
                continue
            for name in (
                "avx2_over_scalar",
                "vnni_over_scalar",
                "vnni_over_avx2",
            ):
                ratio = metric.get("ratios", {}).get(name)
                interval = (
                    ratio.get("paired_bootstrap_interval95")
                    if isinstance(ratio, dict)
                    else None
                )
                if not isinstance(interval, list) or len(interval) != 2:
                    failures.append(f"missing {name} interval at depth {depth}")
                    continue
                lower = float(interval[0])
                interval_checks.append(
                    {"depth": depth, "ratio": name, "lower95": lower}
                )
                if lower <= 1.0:
                    failures.append(
                        f"{name} lower95={lower:.6f} at depth {depth}"
                    )

    return {
        "status": "pass" if not failures else "fail",
        "criterion": (
            "all per-dyno fixed-depth ratios and paired 95% interval lower "
            "bounds must exceed 1.0"
        ),
        "required_depths": list(PROMOTION_DEPTHS),
        "per_dyno": per_dyno,
        "interval_checks": interval_checks,
        "failures": failures,
    }


def write_aggregates(
    args: argparse.Namespace,
    matrix_results: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    all_runs: dict[str, list[dict[str, Any]]] = {backend: [] for backend in BACKENDS}
    for matrix in matrix_results:
        for backend in BACKENDS:
            result = matrix["results"][backend]
            result["runner"] = {
                **matrix["runner"],
                "backend": backend,
                "sequence": matrix["order"].index(backend),
            }
            all_runs[backend].append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for backend, runs in all_runs.items():
        backend_dir = output_dir / "backends" / backend
        backend_dir.mkdir(parents=True, exist_ok=False)
        for ordinal, result in enumerate(runs, start=1):
            (backend_dir / f"heroku-run-{ordinal}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        summary = platform_runner.aggregate_results(runs)
        (backend_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        summaries[backend] = summary

    mismatches = comparator.fixed_depth_signatures(all_runs)
    provenance = comparator.provenance_check(all_runs)
    paired_speedups = paired_backend_speedups(
        all_runs,
        replicates=10_000 if args.profile == "full" else 2_000,
        seed=20260824,
    )
    promotion_gate = (
        full_promotion_gate(all_runs, paired_speedups)
        if args.profile == "full"
        else {
            "status": "not_evaluated",
            "reason": "a complete three-dyno Latin-square run is required",
        }
    )
    comparison = {
        "schema_version": 1,
        "status": (
            "valid"
            if not mismatches
            and provenance["status"] == "pass"
            and paired_speedups["status"] == "pass"
            and all(summary["status"] == "valid" for summary in summaries.values())
            else "invalid"
        ),
        "design": {
            "same_dyno_backend_matrix": True,
            "orders": [matrix["order"] for matrix in matrix_results],
            "latin_square_complete": (
                len(matrix_results) >= len(LATIN_SQUARE)
                and len(matrix_results) % len(LATIN_SQUARE) == 0
            ),
            "timings_pooled_across_backends": False,
        },
        "backends": {
            backend: {
                "directory": str(output_dir / "backends" / backend),
                "runs": len(runs),
                "kernel": comparator.EXPECTED_KERNEL[backend],
            }
            for backend, runs in all_runs.items()
        },
        "fixed_depth_parity": {
            "status": "pass" if not mismatches else "fail",
            "mismatches": mismatches,
        },
        "provenance": provenance,
        "performance": comparator.performance_table(summaries),
        "paired_speedups": paired_speedups,
        "promotion_gate": promotion_gate,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "comparison.md").write_text(comparator.markdown(comparison))
    return comparison


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    remote_root = "/app/chess-engine"
    model_rel = (
        "models/quantized_scale_grid/"
        "old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--production-app", default="stormy-garden-92984")
    parser.add_argument("--size", default="basic")
    parser.add_argument("--process-type", default="benchmark")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--local-harness",
        type=Path,
        default=repo_root / "tools/benchmark/benchmark_uci_platform.py",
    )
    parser.add_argument(
        "--local-remote-runner",
        type=Path,
        default=repo_root / "tools/benchmark/run_nnue_backend_matrix_remote.py",
    )
    parser.add_argument(
        "--local-platform-runner",
        type=Path,
        default=repo_root / "tools/benchmark/run_heroku_uci_platform_benchmark.py",
    )
    parser.add_argument(
        "--local-comparator",
        type=Path,
        default=repo_root / "tools/analyze/compare_nnue_backend_benchmarks.py",
    )
    parser.add_argument(
        "--local-spec",
        type=Path,
        default=repo_root / "benchmarks/uci_platform_v1.json",
    )
    parser.add_argument(
        "--remote-engine", default=f"{remote_root}/build-release/uci_nnue_v41"
    )
    parser.add_argument(
        "--remote-test",
        default=f"{remote_root}/build-release/phase_quantized_nnue_tests",
    )
    parser.add_argument(
        "--remote-model", default=f"{remote_root}/{model_rel}phase_quantized_nnue.bin"
    )
    parser.add_argument(
        "--remote-parity",
        default=f"{remote_root}/{model_rel}phase_quantized_nnue_parity.tsv",
    )
    parser.add_argument(
        "--remote-config",
        default=f"{remote_root}/deploy/heroku/benchmark-config-v41.yml",
    )
    parser.add_argument(
        "--remote-spec", default=f"{remote_root}/benchmarks/uci_platform_v1.json"
    )
    parser.add_argument(
        "--remote-harness", default=f"{remote_root}/tools/benchmark/benchmark_uci_platform.py"
    )
    parser.add_argument(
        "--remote-matrix-runner",
        default=f"{remote_root}/tools/benchmark/run_nnue_backend_matrix_remote.py",
    )
    parser.add_argument("--remote-engine-cwd", default=remote_root)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.app == args.production_app:
        raise SystemExit(f"refusing production app: {args.app}")
    runs = args.runs if args.runs is not None else (1 if args.profile == "smoke" else 3)
    if runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.profile == "full" and runs % len(LATIN_SQUARE) != 0:
        raise SystemExit(
            "full --runs must be a multiple of 3 for a complete Latin square"
        )
    snapshot_sources = {
        "harness.snapshot.py": args.local_harness,
        "matrix_runner_remote.snapshot.py": args.local_remote_runner,
        "platform_aggregator.snapshot.py": args.local_platform_runner,
        "comparator.snapshot.py": args.local_comparator,
        "spec.snapshot.json": args.local_spec,
    }
    missing_snapshots = [
        str(path) for path in snapshot_sources.values() if not path.is_file()
    ]
    if missing_snapshots:
        raise SystemExit(f"local benchmark inputs are missing: {missing_snapshots}")

    harness_bytes = args.local_harness.read_bytes()
    local_runner_bytes = Path(__file__).resolve().read_bytes()
    remote_runner_bytes = args.local_remote_runner.read_bytes()
    platform_runner_bytes = args.local_platform_runner.read_bytes()
    comparator_bytes = args.local_comparator.read_bytes()
    spec_bytes = args.local_spec.read_bytes()
    harness_sha256 = sha256_bytes(harness_bytes)
    spec_sha256 = sha256_bytes(spec_bytes)
    runner_sha256 = sha256_bytes(
        b"\0".join(
            (
                local_runner_bytes,
                remote_runner_bytes,
                platform_runner_bytes,
                comparator_bytes,
            )
        )
    )
    output_dir = args.output_dir or (
        Path("logs") / f"nnue_simd_matrix_{args.profile}_{utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir()
    (snapshots_dir / "matrix_runner_local.snapshot.py").write_bytes(local_runner_bytes)
    for snapshot_name, source_path in snapshot_sources.items():
        (snapshots_dir / snapshot_name).write_bytes(source_path.read_bytes())
    (output_dir / "request.json").write_text(
        json.dumps(
            {
                "app": args.app,
                "size": args.size,
                "process_type": args.process_type,
                "profile": args.profile,
                "runs": runs,
                "source_label": args.source_label,
                "harness_sha256": harness_sha256,
                "runner_sha256": runner_sha256,
                "spec_sha256": spec_sha256,
                "snapshot_sha256": {
                    path.name: sha256_bytes(path.read_bytes())
                    for path in sorted(snapshots_dir.iterdir())
                },
                "orders": [list(LATIN_SQUARE[index % 3]) for index in range(runs)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    matrix_results = []
    for index in range(runs):
        ordinal = index + 1
        order = LATIN_SQUARE[index % len(LATIN_SQUARE)]
        print(
            f"starting same-dyno matrix {ordinal}/{runs}: {','.join(order)}",
            flush=True,
        )
        matrix_results.append(
            run_one(
                args,
                order,
                ordinal,
                output_dir,
                harness_sha256,
                spec_sha256,
                runner_sha256,
            )
        )
    comparison = write_aggregates(args, matrix_results, output_dir)
    print(f"comparison: {output_dir / 'comparison.md'}", flush=True)
    passed = comparison["status"] == "valid" and (
        args.profile != "full"
        or comparison.get("promotion_gate", {}).get("status") == "pass"
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
