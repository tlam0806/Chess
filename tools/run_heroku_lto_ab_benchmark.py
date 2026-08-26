#!/usr/bin/env python3
"""Run and aggregate a paired full LTO/non-LTO benchmark on Heroku."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import subprocess
import zlib
from typing import Any, Sequence


PROGRESS_SENTINEL = "CHESS_LTO_AB_PROGRESS="
BENCHMARK_PROGRESS_SENTINEL = "CHESS_BENCHMARK_PROGRESS="
CHUNK_SENTINEL = "CHESS_LTO_AB_CHUNK="
END_SENTINEL = "CHESS_LTO_AB_END="
ERROR_SENTINEL = "CHESS_LTO_AB_ERROR="
VARIANTS = ("nonlto", "lto")
ABBA_ORDERS = (
    ("nonlto", "lto"),
    ("lto", "nonlto"),
    ("lto", "nonlto"),
    ("nonlto", "lto"),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
    root = args.remote_root.rstrip("/")
    model_dir = (
        f"{root}/models/quantized_scale_grid/"
        "old_score_huber200_lr_sweep_then_5ep_20260724_142758/best"
    )
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
        "python",
        "-u",
        "-",
        "--nonlto-engine",
        f"{root}/build-nonlto/uci_nnue_v41",
        "--lto-engine",
        f"{root}/build-lto/uci_nnue_v41",
        "--nonlto-test",
        f"{root}/build-nonlto/phase_quantized_nnue_tests",
        "--lto-test",
        f"{root}/build-lto/phase_quantized_nnue_tests",
        "--model",
        f"{model_dir}/phase_quantized_nnue.bin",
        "--parity",
        f"{model_dir}/phase_quantized_nnue_parity.tsv",
        "--config",
        f"{root}/deploy/heroku/benchmark-config-v41.yml",
        "--spec",
        f"{root}/benchmarks/uci_platform_v1.json",
        "--harness",
        f"{root}/tools/benchmark_uci_platform.py",
        "--engine-cwd",
        root,
        "--order",
        ",".join(order),
        "--profile",
        args.profile,
        "--host-label",
        f"heroku:{args.app}:{args.size}:lto-ab-{ordinal}",
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
    remote_script: bytes,
    order: Sequence[str],
    ordinal: int,
    output_dir: Path,
    harness_sha256: str,
    spec_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    command = remote_command(
        args,
        order,
        ordinal,
        harness_sha256,
        spec_sha256,
        runner_sha256,
    )
    console_path = output_dir / f"heroku-lto-ab-run-{ordinal}.console.log"
    chunks: dict[int, str] = {}
    chunk_count = None
    end_metadata = None
    remote_error = None
    current_variant = None
    with console_path.open("w", encoding="utf-8") as console:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdin is not None
        process.stdin.write(remote_script.decode("utf-8"))
        process.stdin.close()
        assert process.stdout is not None
        for line in process.stdout:
            console.write(line)
            console.flush()
            stripped = line.rstrip("\r\n")
            if PROGRESS_SENTINEL in stripped:
                payload = json.loads(stripped.split(PROGRESS_SENTINEL, 1)[1])
                current_variant = payload.get("variant", current_variant)
                print(f"run {ordinal}: {json.dumps(payload, sort_keys=True)}", flush=True)
            elif BENCHMARK_PROGRESS_SENTINEL in stripped:
                payload = json.loads(
                    stripped.split(BENCHMARK_PROGRESS_SENTINEL, 1)[1]
                )
                if payload.get("phase") == "measure":
                    print(
                        "run {run} {variant}: {mode}={limit} round "
                        "{round}/{rounds}, nps={nps:,.0f}".format(
                            run=ordinal,
                            variant=current_variant,
                            **payload,
                        ),
                        flush=True,
                    )
            elif CHUNK_SENTINEL in stripped:
                descriptor, encoded = stripped.split(CHUNK_SENTINEL, 1)[1].split(
                    ":", 1
                )
                index_text, count_text = descriptor.split("/", 1)
                index = int(index_text)
                count = int(count_text)
                if chunk_count is not None and chunk_count != count:
                    raise RuntimeError("remote result changed chunk count")
                chunk_count = count
                chunks[index] = encoded
            elif END_SENTINEL in stripped:
                end_metadata = json.loads(stripped.split(END_SENTINEL, 1)[1])
            elif ERROR_SENTINEL in stripped:
                remote_error = json.loads(stripped.split(ERROR_SENTINEL, 1)[1])
                print(f"run {ordinal} remote error: {remote_error}", flush=True)
        return_code = process.wait()

    if return_code != 0 or end_metadata is None or chunk_count is None:
        raise RuntimeError(
            f"LTO A/B run {ordinal} failed (exit={return_code}): "
            f"{remote_error or 'missing result'}; see {console_path}"
        )
    if set(chunks) != set(range(1, chunk_count + 1)):
        raise RuntimeError(f"LTO A/B run {ordinal} returned incomplete chunks")
    compressed = base64.b64decode(
        "".join(chunks[index] for index in range(1, chunk_count + 1)),
        validate=True,
    )
    if sha256_bytes(compressed) != end_metadata["compressed_sha256"]:
        raise RuntimeError("compressed result hash mismatch")
    encoded = zlib.decompress(compressed)
    if sha256_bytes(encoded) != end_metadata["json_sha256"]:
        raise RuntimeError("JSON result hash mismatch")
    result = json.loads(encoded)
    if result.get("status") != "valid" or result.get("order") != list(order):
        raise RuntimeError(f"invalid LTO A/B result in run {ordinal}")
    if result.get("fixed_depth_parity", {}).get("status") != "pass":
        raise RuntimeError(f"fixed-depth parity failed in run {ordinal}")
    result["runner"] = {
        "app": args.app,
        "size": args.size,
        "process_type": args.process_type,
        "profile": args.profile,
        "ordinal": ordinal,
        "order": list(order),
    }
    result_path = output_dir / f"heroku-lto-ab-run-{ordinal}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"run {ordinal}: saved {result_path}", flush=True)
    return result


def paired_analysis(
    runs: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    metrics = sorted(
        {
            (row["mode"], row["limit"])
            for run in runs
            for row in run["results"]["nonlto"]["observations"]
        }
    )
    output = []
    for metric_index, (mode, limit) in enumerate(metrics):
        paired_runs = []
        timed_warnings = []
        for run_index, run in enumerate(runs):
            variants = {}
            for variant in VARIANTS:
                rows = [
                    row
                    for row in run["results"][variant]["observations"]
                    if row["mode"] == mode and row["limit"] == limit
                ]
                variants[variant] = {
                    (row["round"], row["position_id"]): row for row in rows
                }
                timed_warnings.extend(
                    {
                        "run": run_index + 1,
                        "variant": variant,
                        "round": row["round"],
                        "position_id": row["position_id"],
                        "warnings": row.get("warnings", []),
                    }
                    for row in rows
                    if row.get("warnings")
                )
            keys = set(variants["nonlto"])
            if not keys or set(variants["lto"]) != keys:
                raise RuntimeError(f"paired cases differ for {mode}={limit}")
            paired_runs.append(
                [
                    {variant: variants[variant][key] for variant in VARIANTS}
                    for key in sorted(keys)
                ]
            )

        def calculate(sampled_runs: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
            totals = {
                variant: {"nodes": 0, "elapsed_ns": 0} for variant in VARIANTS
            }
            for cases in sampled_runs:
                for case in cases:
                    for variant in VARIANTS:
                        totals[variant]["nodes"] += int(case[variant]["nodes"])
                        totals[variant]["elapsed_ns"] += int(
                            case[variant]["elapsed_ns"]
                        )
            nps = {
                variant: values["nodes"] * 1_000_000_000.0 / values["elapsed_ns"]
                for variant, values in totals.items()
            }
            return {"nps": nps, "ratio": nps["lto"] / nps["nonlto"]}

        point = calculate(paired_runs)
        generator = random.Random(seed + metric_index)
        draws = []
        for _ in range(replicates):
            sampled_runs = []
            for _ in paired_runs:
                cases = paired_runs[generator.randrange(len(paired_runs))]
                sampled_runs.append([generator.choice(cases) for _ in cases])
            draws.append(calculate(sampled_runs)["ratio"])
        output.append(
            {
                "mode": mode,
                "limit": limit,
                "paired_cases_per_variant": sum(len(cases) for cases in paired_runs),
                "nps": point["nps"],
                "lto_over_nonlto": point["ratio"],
                "speedup_percent": (point["ratio"] - 1.0) * 100.0,
                "ratio_interval95": [
                    percentile(draws, 0.025),
                    percentile(draws, 0.975),
                ],
                "timed_warnings": timed_warnings,
            }
        )
    return {
        "method": "dynos_then_matched_cases_within_dyno paired bootstrap",
        "replicates": replicates,
        "seed": seed,
        "metrics": output,
    }


def build_report(summary: dict[str, Any], output_dir: Path) -> str:
    lines = [
        "# Heroku V41 LTO paired benchmark",
        "",
        f"Status: **{summary['status']}**",
        f"Promotion gate: **{summary['promotion_gate']['status']}**",
        "",
        "| Mode | Limit | non-LTO NPS | LTO NPS | Speedup | Ratio 95% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["analysis"]["metrics"]:
        lines.append(
            "| {mode} | {limit} | {base:,.0f} | {lto:,.0f} | "
            "{gain:+.2f}% | [{low:.4f}, {high:.4f}] |".format(
                mode=row["mode"],
                limit=row["limit"],
                base=row["nps"]["nonlto"],
                lto=row["nps"]["lto"],
                gain=row["speedup_percent"],
                low=row["ratio_interval95"][0],
                high=row["ratio_interval95"][1],
            )
        )
    lines.extend(["", f"Raw artifacts: `{output_dir}`", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--production-app", default="stormy-garden-92984")
    parser.add_argument("--size", default="basic")
    parser.add_argument("--process-type", default="benchmark")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--remote-script",
        type=Path,
        default=repo_root / "tools/run_lto_ab_remote.py",
    )
    parser.add_argument(
        "--harness",
        type=Path,
        default=repo_root / "tools/benchmark_uci_platform.py",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=repo_root / "benchmarks/uci_platform_v1.json",
    )
    parser.add_argument("--remote-root", default="/app/chess-engine")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.app == args.production_app:
        raise SystemExit(f"refusing production app: {args.app}")
    if args.runs != 4:
        raise SystemExit("paired ABBA benchmark requires exactly --runs 4")
    remote_script = args.remote_script.read_bytes()
    harness = args.harness.read_bytes()
    spec = args.spec.read_bytes()
    harness_sha256 = sha256_bytes(harness)
    spec_sha256 = sha256_bytes(spec)
    runner_sha256 = sha256_bytes(
        b"\0".join((Path(__file__).read_bytes(), remote_script, harness, spec))
    )
    output_dir = args.output_dir or Path(
        "logs"
    ) / f"nnue_lto_ab_{args.profile}_{utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = output_dir / "snapshots"
    snapshots.mkdir()
    (snapshots / "local-runner.py").write_bytes(Path(__file__).read_bytes())
    (snapshots / "remote-runner.py").write_bytes(remote_script)
    (snapshots / "harness.py").write_bytes(harness)
    (snapshots / "spec.json").write_bytes(spec)
    request = {
        "app": args.app,
        "size": args.size,
        "process_type": args.process_type,
        "profile": args.profile,
        "runs": args.runs,
        "orders": [list(order) for order in ABBA_ORDERS],
        "source_label": args.source_label,
        "harness_sha256": harness_sha256,
        "spec_sha256": spec_sha256,
        "runner_sha256": runner_sha256,
    }
    (output_dir / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )

    runs = []
    for index, order in enumerate(ABBA_ORDERS):
        ordinal = index + 1
        print(f"starting paired dyno {ordinal}/4: {','.join(order)}", flush=True)
        runs.append(
            run_one(
                args,
                remote_script,
                order,
                ordinal,
                output_dir,
                harness_sha256,
                spec_sha256,
                runner_sha256,
            )
        )

    identity_sets = {
        variant: {
            run["identities"][variant]["engine_sha256"] for run in runs
        }
        for variant in VARIANTS
    }
    analysis = paired_analysis(
        runs,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    gate_reasons = []
    if any(len(values) != 1 for values in identity_sets.values()):
        gate_reasons.append("engine identity changed across dynos")
    for row in analysis["metrics"]:
        if row["lto_over_nonlto"] <= 1.0:
            gate_reasons.append(f"LTO is not faster for {row['mode']}={row['limit']}")
        if row["ratio_interval95"][0] <= 1.0:
            gate_reasons.append(
                f"95% ratio interval includes 1 for {row['mode']}={row['limit']}"
            )
        if row["timed_warnings"]:
            gate_reasons.append(f"timing warnings for {row['mode']}={row['limit']}")
    summary = {
        "schema_version": 1,
        "status": "valid",
        "runs": len(runs),
        "orders": [list(order) for order in ABBA_ORDERS],
        "identity_sets": {key: sorted(value) for key, value in identity_sets.items()},
        "analysis": analysis,
        "promotion_gate": {
            "status": "pass" if not gate_reasons else "fail",
            "reasons": gate_reasons,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "report.md").write_text(build_report(summary, output_dir))
    print(f"summary: {output_dir / 'summary.json'}", flush=True)
    print(f"report: {output_dir / 'report.md'}", flush=True)
    return 0 if summary["promotion_gate"]["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
