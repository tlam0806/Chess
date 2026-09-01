#!/usr/bin/env python3
"""Run the UCI platform benchmark in isolated Heroku one-off dynos.

The benchmark program is streamed over stdin to the current release image, so
this runner neither deploys a new release nor restarts/scales the production
worker.  Results are intercepted locally instead of being left on the dyno's
ephemeral filesystem.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


RESULT_SENTINEL = "CHESS_BENCHMARK_RESULT="
ERROR_SENTINEL = "CHESS_BENCHMARK_ERROR="
PROGRESS_SENTINEL = "CHESS_BENCHMARK_PROGRESS="
EXPECTED_KERNEL_BY_BACKEND = {
    "scalar": "scalar",
    "avx2": "x86_avx2_exact",
    "vnni": "x86_avx512vnni_256",
}
EXPECTED_ACCUMULATOR_KERNEL_BY_BACKEND = {
    "portable": "portable",
    "avx2": "x86_avx2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def remote_command(
    args: argparse.Namespace,
    spec_base64: str,
    ordinal: int,
) -> list[str]:
    command = [
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
    ]
    remote_environment: list[str] = []
    if args.backend != "auto":
        remote_environment.append(f"CHESS_NNUE_BACKEND={args.backend}")
    if args.accumulator_backend != "auto":
        remote_environment.append(
            "CHESS_NNUE_ACCUMULATOR_BACKEND="
            f"{args.accumulator_backend}"
        )
    if remote_environment:
        # Heroku CLI accepts a semicolon-separated environment list, but
        # rejects repeated --env flags. Keep all selectors in one flag.
        command.extend(["--env", ";".join(remote_environment)])
    command.extend(
        [
            "--",
            "python",
            "-u",
            "-",
            "--engine",
            args.remote_engine,
            "--model",
            args.remote_model,
            "--config",
            args.remote_config,
            "--engine-cwd",
            args.remote_engine_cwd,
            "--spec-base64",
            spec_base64,
            "--host-label",
            f"heroku:{args.app}:{args.size}:oneoff-{ordinal}",
            "--source-label",
            args.source_label,
            "--harness-sha256",
            args.harness_sha256,
            "--runner-sha256",
            args.runner_sha256,
            "--output",
            "-",
        ]
    )
    if args.profile == "smoke":
        command.extend(
            [
                "--fixed-depth",
                "7",
                "--fixed-rounds",
                "1",
                "--movetime-ms",
                "1000",
                "--movetime-rounds",
                "1",
                "--warmup-depth",
                "4",
                "--warmup-positions",
                "2",
                "--bootstrap-replicates",
                "1000",
            ]
        )
    return command


def run_one(
    args: argparse.Namespace,
    harness_path: Path,
    spec_base64: str,
    ordinal: int,
    output_dir: Path,
) -> dict[str, Any]:
    command = remote_command(args, spec_base64, ordinal)
    console_path = output_dir / f"heroku-run-{ordinal}.console.log"
    result: dict[str, Any] | None = None
    remote_error: dict[str, Any] | None = None
    with (
        harness_path.open("r", encoding="utf-8") as harness,
        console_path.open("w", encoding="utf-8") as console,
    ):
        process = subprocess.Popen(
            command,
            stdin=harness,
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
            result_index = stripped.find(RESULT_SENTINEL)
            error_index = stripped.find(ERROR_SENTINEL)
            progress_index = stripped.find(PROGRESS_SENTINEL)
            if result_index >= 0:
                result = json.loads(stripped[result_index + len(RESULT_SENTINEL) :])
            elif error_index >= 0:
                remote_error = json.loads(stripped[error_index + len(ERROR_SENTINEL) :])
                print(
                    f"remote benchmark error: {remote_error.get('error')}",
                    flush=True,
                )
            elif progress_index >= 0:
                payload = json.loads(
                    stripped[progress_index + len(PROGRESS_SENTINEL) :]
                )
                if payload["phase"] == "warmup":
                    print(
                        f"run {ordinal}: warmup {payload['completed']}/{payload['total']}",
                        flush=True,
                    )
                else:
                    print(
                        "run {ordinal}: {mode}={limit} round {round}/{rounds} "
                        "nps={nps:,.0f}".format(ordinal=ordinal, **payload),
                        flush=True,
                    )
            else:
                # Keep normal Heroku lifecycle messages visible, but avoid
                # echoing the very long command line containing the suite.
                if len(stripped) < 500:
                    print(stripped, flush=True)
        return_code = process.wait()
    if result is None:
        detail = (
            remote_error.get("error") if remote_error else "missing result sentinel"
        )
        raise RuntimeError(
            f"Heroku run {ordinal} failed (exit={return_code}): {detail}; "
            f"see {console_path}"
        )
    result["runner"] = {
        "app": args.app,
        "size": args.size,
        "process_type": args.process_type,
        "profile": args.profile,
        "backend": args.backend,
        "accumulator_backend": args.accumulator_backend,
        "ordinal": ordinal,
    }
    result_path = output_dir / f"heroku-run-{ordinal}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"run {ordinal}: saved {result_path}", flush=True)
    if return_code != 0 or result.get("status") != "valid":
        raise RuntimeError(
            f"Heroku run {ordinal} returned exit={return_code}, "
            f"status={result.get('status')}; see {result_path}"
        )
    if args.backend in EXPECTED_KERNEL_BY_BACKEND:
        expected_kernel = EXPECTED_KERNEL_BY_BACKEND[args.backend]
        actual_kernel = result.get("uci", {}).get("handshake", {}).get("nnue_kernel")
        if actual_kernel != expected_kernel:
            raise RuntimeError(
                f"forced backend {args.backend!r} reported kernel "
                f"{actual_kernel!r}, expected {expected_kernel!r}; "
                f"see {result_path}"
            )
    if args.accumulator_backend in EXPECTED_ACCUMULATOR_KERNEL_BY_BACKEND:
        expected_kernel = EXPECTED_ACCUMULATOR_KERNEL_BY_BACKEND[
            args.accumulator_backend
        ]
        actual_kernel = result.get("uci", {}).get("handshake", {}).get(
            "nnue_accumulator_kernel"
        )
        if actual_kernel != expected_kernel:
            raise RuntimeError(
                "forced accumulator backend "
                f"{args.accumulator_backend!r} reported kernel "
                f"{actual_kernel!r}, expected {expected_kernel!r}; "
                f"see {result_path}"
            )
    return result


def cross_run_integrity(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str], set[tuple[Any, ...]]] = defaultdict(set)
    for result in results:
        for row in result["observations"]:
            if row["mode"] != "fixed_depth":
                continue
            grouped[(row["limit"], row["position_id"])].add(
                (
                    row["reported_depth"],
                    row["score_type"],
                    row["score_value"],
                    row["nodes"],
                    row["bestmove"],
                )
            )
    mismatches = [
        {
            "limit": limit,
            "position_id": position_id,
            "signatures": [list(value) for value in sorted(signatures)],
        }
        for (limit, position_id), signatures in sorted(grouped.items())
        if len(signatures) != 1
    ]
    engine_hashes = {result["provenance"]["engine_sha256"] for result in results}
    model_hashes = {result["provenance"]["model_sha256"] for result in results}
    spec_hashes = {result["provenance"]["spec_sha256"] for result in results}
    effective_spec_hashes = {
        result["provenance"].get("effective_spec_sha256") for result in results
    }
    config_hashes = {result["provenance"].get("config_sha256") for result in results}
    harness_hashes = {result["provenance"].get("harness_sha256") for result in results}
    runner_hashes = {result["provenance"].get("runner_sha256") for result in results}
    hashes_match = all(
        len(values) == 1 and None not in values
        for values in (
            engine_hashes,
            model_hashes,
            spec_hashes,
            effective_spec_hashes,
            config_hashes,
            harness_hashes,
            runner_hashes,
        )
    )
    config_valid = all(
        result["provenance"].get("config_validation", {}).get("status") == "match"
        for result in results
    )
    return {
        "status": (
            "pass" if not mismatches and hashes_match and config_valid else "fail"
        ),
        "fixed_depth_mismatches": mismatches,
        "engine_sha256": sorted(str(value) for value in engine_hashes),
        "model_sha256": sorted(str(value) for value in model_hashes),
        "spec_sha256": sorted(str(value) for value in spec_hashes),
        "effective_spec_sha256": sorted(str(value) for value in effective_spec_hashes),
        "config_sha256": sorted(str(value) for value in config_hashes),
        "harness_sha256": sorted(str(value) for value in harness_hashes),
        "runner_sha256": sorted(str(value) for value in runner_hashes),
        "hashes_match": hashes_match,
        "config_options_match": config_valid,
    }


def hierarchical_nps_ci(
    per_run_rounds: Sequence[Sequence[dict[str, Any]]],
    replicates: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        selected_runs = [rng.randrange(len(per_run_rounds)) for _ in per_run_rounds]
        nodes = 0
        elapsed_ns = 0
        for selected in selected_runs:
            rounds = per_run_rounds[selected]
            for _ in rounds:
                row = rng.choice(rounds)
                nodes += row["nodes"]
                elapsed_ns += row["elapsed_ns"]
        draws.append(nodes * 1_000_000_000.0 / elapsed_ns)
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs = sorted(
        {
            (summary["mode"], summary["limit"])
            for result in results
            for summary in result["summaries"]
        }
    )
    summaries = []
    for pair_index, (mode, limit) in enumerate(pairs):
        per_run = []
        per_run_rounds = []
        all_rows = []
        for result in results:
            summary = next(
                value
                for value in result["summaries"]
                if value["mode"] == mode and value["limit"] == limit
            )
            per_run.append(
                {
                    "ordinal": result["runner"]["ordinal"],
                    "aggregate_nps": summary["aggregate_nps"],
                    "round_nps_cv": summary["round_nps_cv"],
                    "total_nodes": summary["total_nodes"],
                    "total_elapsed_ns": summary["total_elapsed_ns"],
                }
            )
            per_run_rounds.append(summary["round_aggregates"])
            all_rows.extend(
                row
                for row in result["observations"]
                if row["mode"] == mode and row["limit"] == limit
            )
        total_nodes = sum(row["nodes"] for row in all_rows)
        total_elapsed_ns = sum(row["elapsed_ns"] for row in all_rows)
        position_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_rows:
            position_groups[row["position_id"]].append(row)
        per_position = []
        for position_id, position_rows in sorted(position_groups.items()):
            position_nodes = sum(row["nodes"] for row in position_rows)
            position_elapsed_ns = sum(row["elapsed_ns"] for row in position_rows)
            per_position.append(
                {
                    "position_id": position_id,
                    "aggregate_nps": (
                        position_nodes * 1_000_000_000.0 / position_elapsed_ns
                    ),
                    "nodes": position_nodes,
                    "elapsed_ns": position_elapsed_ns,
                }
            )
        position_nps = [row["aggregate_nps"] for row in per_position]
        summary: dict[str, Any] = {
            "mode": mode,
            "limit": limit,
            "runs": len(results),
            "observations": len(all_rows),
            "aggregate_nps": total_nodes * 1_000_000_000.0 / total_elapsed_ns,
            "descriptive_hierarchical_nps_interval95": hierarchical_nps_ci(
                per_run_rounds, 10000, 20260823 + pair_index
            ),
            "macro_median_position_nps": statistics.median(position_nps),
            "macro_geomean_position_nps": math.exp(
                statistics.fmean(math.log(value) for value in position_nps)
            ),
            "per_position": per_position,
            "per_run": per_run,
        }
        if mode == "movetime_ms":
            summary["median_completed_depth"] = statistics.median(
                row["reported_depth"] for row in all_rows
            )
            summary["median_nodes_per_move"] = statistics.median(
                row["nodes"] for row in all_rows
            )
            summary["wall_overshoot_ms_p95"] = percentile(
                [row["elapsed_ns"] / 1_000_000.0 - limit for row in all_rows],
                0.95,
            )
        summaries.append(summary)
    integrity = cross_run_integrity(results)
    warnings = sorted(
        {warning for result in results for warning in result.get("warnings", [])}
    )
    return {
        "schema_version": 1,
        "status": "valid" if integrity["status"] == "pass" else "invalid",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": len(results),
        "integrity": integrity,
        "warnings": warnings,
        "summaries": summaries,
    }


def markdown_report(
    args: argparse.Namespace,
    aggregate: dict[str, Any],
    output_dir: Path,
) -> str:
    lines = [
        "# Heroku UCI platform benchmark",
        "",
        f"- Status: **{aggregate['status']}**",
        f"- App/dyno: `{args.app}` / `{args.size}` one-off",
        "- Process/forward/accumulator: "
        f"`{args.process_type}` / `{args.backend}` / "
        f"`{args.accumulator_backend}`",
        f"- Profile: `{args.profile}`",
        f"- Independent dynos: {aggregate['runs']}",
        f"- Source label: `{args.source_label}`",
        "- Timing boundary: send `go` to receive `bestmove`, inside the dyno",
        "- Isolation: a fresh engine process for every measured position",
        f"- Run warnings: {len(aggregate.get('warnings', []))}",
        "",
        "| Mode | Limit | Aggregate NPS | Descriptive 95% resampling interval | Extra |",
        "|---|---:|---:|---:|---|",
    ]
    for summary in aggregate["summaries"]:
        ci = summary["descriptive_hierarchical_nps_interval95"]
        extra = ""
        if summary["mode"] == "movetime_ms":
            extra = (
                f"median depth {summary['median_completed_depth']}; "
                f"median nodes {summary['median_nodes_per_move']:,.0f}"
            )
        lines.append(
            "| {mode} | {limit} | {nps:,.0f} | [{low:,.0f}, {high:,.0f}] | {extra} |".format(
                mode=summary["mode"],
                limit=summary["limit"],
                nps=summary["aggregate_nps"],
                low=ci[0],
                high=ci[1],
                extra=extra,
            )
        )
    lines.extend(
        [
            "",
            "The interval is conditional on this fixed suite and three sampled dynos; it is not a Heroku fleet-wide confidence claim.",
            "",
            "This is an absolute Heroku baseline. It is not a Heroku/local speed ratio until the same suite passes fixed-depth parity on an idle local machine.",
            "",
            f"Raw artifacts: `{output_dir}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default="stormy-garden-92984")
    parser.add_argument("--size", default="basic")
    parser.add_argument("--process-type", default="worker")
    parser.add_argument(
        "--backend",
        choices=("auto", "scalar", "avx2", "vnni"),
        default="auto",
        help="force CHESS_NNUE_BACKEND and verify the reported UCI kernel",
    )
    parser.add_argument(
        "--accumulator-backend",
        choices=("auto", "portable", "avx2"),
        default="auto",
        help=(
            "force CHESS_NNUE_ACCUMULATOR_BACKEND and verify the reported "
            "UCI kernel"
        ),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--runs", type=int)
    parser.add_argument(
        "--harness", type=Path, default=repo_root / "tools/benchmark_uci_platform.py"
    )
    parser.add_argument(
        "--spec", type=Path, default=repo_root / "benchmarks/uci_platform_v1.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--source-label",
        default="heroku-current-release-unresolved",
    )
    parser.add_argument(
        "--remote-engine",
        default="/app/chess-engine/build-release/uci_nnue_v43",
    )
    parser.add_argument(
        "--remote-model",
        default=(
            "/app/chess-engine/models/quantized_scale_grid/"
            "old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/"
            "phase_quantized_nnue.bin"
        ),
    )
    parser.add_argument(
        "--remote-config", default="/app/lichess-bot/config-nnue-v43.yml"
    )
    parser.add_argument("--remote-engine-cwd", default="/app/chess-engine")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = args.runs if args.runs is not None else (1 if args.profile == "smoke" else 3)
    if runs <= 0:
        raise SystemExit("--runs must be positive")
    if not args.harness.is_file() or not args.spec.is_file():
        raise SystemExit("benchmark harness or spec is missing")
    harness_bytes = args.harness.read_bytes()
    runner_bytes = Path(__file__).resolve().read_bytes()
    spec_bytes = args.spec.read_bytes()
    args.harness_sha256 = sha256_bytes(harness_bytes)
    args.runner_sha256 = sha256_bytes(runner_bytes)
    output_dir = args.output_dir or (
        Path("logs") / f"uci_platform_{args.profile}_heroku_{utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "app": args.app,
        "size": args.size,
        "process_type": args.process_type,
        "profile": args.profile,
        "backend": args.backend,
        "accumulator_backend": args.accumulator_backend,
        "runs": runs,
        "source_label": args.source_label,
        "harness": str(args.harness),
        "harness_sha256": args.harness_sha256,
        "runner_sha256": args.runner_sha256,
        "spec": str(args.spec),
        "spec_sha256": sha256_bytes(spec_bytes),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )
    harness_snapshot = output_dir / "harness.snapshot.py"
    harness_snapshot.write_bytes(harness_bytes)
    (output_dir / "runner.snapshot.py").write_bytes(runner_bytes)
    (output_dir / "spec.snapshot.json").write_bytes(spec_bytes)
    spec_base64 = base64.b64encode(spec_bytes).decode("ascii")
    results = []
    print(f"artifacts: {output_dir}", flush=True)
    for ordinal in range(1, runs + 1):
        print(
            f"starting Heroku one-off {ordinal}/{runs} ({args.size}, {args.profile})",
            flush=True,
        )
        results.append(
            run_one(args, harness_snapshot, spec_base64, ordinal, output_dir)
        )
    aggregate = aggregate_results(results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    report_path = output_dir / "report.md"
    report_path.write_text(markdown_report(args, aggregate, output_dir))
    print(f"summary: {summary_path}", flush=True)
    print(f"report: {report_path}", flush=True)
    return 0 if aggregate["status"] == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())
