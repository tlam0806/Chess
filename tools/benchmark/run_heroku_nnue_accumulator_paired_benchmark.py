#!/usr/bin/env python3
"""Benchmark portable and AVX2 NNUE accumulators on the same Heroku dynos."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import random
import shlex
import subprocess
import zlib
from pathlib import Path
from typing import Any, Sequence


BACKENDS = ("portable", "avx2")
ORDERS = (("portable", "avx2"), ("avx2", "portable"))
CHUNK_SENTINEL = "CHESS_ACC_PAIR_CHUNK="
END_SENTINEL = "CHESS_ACC_PAIR_END="
EXPECTED_FORWARD_KERNEL = "x86_avx512vnni_256"
EXPECTED_ACCUMULATOR_KERNEL = {
    "portable": "portable",
    "avx2": "x86_avx2",
}


REMOTE_AGGREGATOR = r'''
import base64
import hashlib
import json
import sys
import zlib

paths = {"portable": sys.argv[1], "avx2": sys.argv[2]}
order = sys.argv[3].split(",")
results = {name: json.load(open(path, encoding="utf-8")) for name, path in paths.items()}
expected_accumulator = {"portable": "portable", "avx2": "x86_avx2"}
expected_forward = "x86_avx512vnni_256"
for name, result in results.items():
    if result.get("status") != "valid":
        raise RuntimeError(f"{name} benchmark is invalid")
    handshake = result.get("uci", {}).get("handshake", {})
    if handshake.get("nnue_kernel") != expected_forward:
        raise RuntimeError(f"{name} forward kernel mismatch: {handshake}")
    if handshake.get("nnue_accumulator_kernel") != expected_accumulator[name]:
        raise RuntimeError(f"{name} accumulator kernel mismatch: {handshake}")

provenance_keys = (
    "engine_sha256", "model_sha256", "config_sha256", "spec_sha256",
    "effective_spec_sha256",
)
provenance = {
    key: results["portable"]["provenance"].get(key) for key in provenance_keys
}
for name in ("avx2",):
    for key, value in provenance.items():
        if results[name]["provenance"].get(key) != value:
            raise RuntimeError(f"provenance mismatch for {key}")

def fixed_signatures(result):
    grouped = {}
    for row in result["observations"]:
        if row["mode"] != "fixed_depth":
            continue
        key = (row["limit"], row["position_id"])
        grouped.setdefault(key, set()).add((
            row["reported_depth"], row["score_type"], row["score_value"],
            row["nodes"], row["bestmove"],
        ))
    return grouped

portable_signatures = fixed_signatures(results["portable"])
avx2_signatures = fixed_signatures(results["avx2"])
if portable_signatures != avx2_signatures:
    raise RuntimeError("fixed-depth search signatures differ")

metric_keys = sorted({
    (row["mode"], row["limit"])
    for result in results.values()
    for row in result["observations"]
})
metrics = []
for mode, limit in metric_keys:
    indexed = {}
    for name, result in results.items():
        rows = [
            row for row in result["observations"]
            if row["mode"] == mode and row["limit"] == limit
        ]
        indexed[name] = {
            (row["round"], row["position_id"]): row for row in rows
        }
    keys = set(indexed["portable"])
    if not keys or set(indexed["avx2"]) != keys:
        raise RuntimeError(f"paired cases differ for {mode}={limit}")
    cases = []
    totals = {name: {"nodes": 0, "elapsed_ns": 0} for name in paths}
    for round_index, position_id in sorted(keys):
        case = {"round": round_index, "position_id": position_id}
        for name in paths:
            row = indexed[name][(round_index, position_id)]
            nodes = int(row["nodes"])
            elapsed_ns = int(row["elapsed_ns"])
            case[name] = [nodes, elapsed_ns]
            totals[name]["nodes"] += nodes
            totals[name]["elapsed_ns"] += elapsed_ns
        cases.append(case)
    nps = {
        name: value["nodes"] * 1_000_000_000.0 / value["elapsed_ns"]
        for name, value in totals.items()
    }
    metrics.append({
        "mode": mode,
        "limit": limit,
        "cases": cases,
        "nps": nps,
        "avx2_over_portable": nps["avx2"] / nps["portable"],
    })

output = {
    "schema_version": 1,
    "status": "valid",
    "order": order,
    "fixed_depth_parity": "pass",
    "provenance": provenance,
    "handshakes": {
        name: results[name]["uci"]["handshake"] for name in paths
    },
    "machine": results["portable"].get("machine_before", {}),
    "metrics": metrics,
}
encoded = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
compressed = zlib.compress(encoded, 9)
payload = base64.b64encode(compressed).decode("ascii")
chunks = [payload[index:index + 6000] for index in range(0, len(payload), 6000)]
for index, chunk in enumerate(chunks, 1):
    print(f"CHESS_ACC_PAIR_CHUNK={index}/{len(chunks)}:{chunk}", flush=True)
print("CHESS_ACC_PAIR_END=" + json.dumps({
    "chunks": len(chunks),
    "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
    "json_sha256": hashlib.sha256(encoded).hexdigest(),
}, sort_keys=True, separators=(",", ":")), flush=True)
'''


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def benchmark_command(args: argparse.Namespace, backend: str, output: str) -> str:
    values = [
        "python3",
        args.remote_harness,
        "--engine",
        args.remote_engine,
        "--model",
        args.remote_model,
        "--config",
        args.remote_config,
        "--engine-cwd",
        args.remote_engine_cwd,
        "--spec",
        args.remote_spec,
        "--host-label",
        f"heroku:{args.app}:{args.size}:paired",
        "--source-label",
        args.source_label,
        "--output",
        output,
        "--fixed-depth",
        "7",
        "--fixed-depth",
        "8",
        "--fixed-rounds",
        str(args.fixed_rounds),
        "--movetime-ms",
        "500",
        "--movetime-ms",
        "1000",
        "--movetime-rounds",
        str(args.movetime_rounds),
        "--warmup-depth",
        "5",
        "--warmup-positions",
        "12",
        "--bootstrap-replicates",
        "2000",
    ]
    command = " ".join(shlex.quote(value) for value in values)
    return (
        "CHESS_NNUE_BACKEND=vnni "
        f"CHESS_NNUE_ACCUMULATOR_BACKEND={shlex.quote(backend)} {command}"
    )


def remote_script(args: argparse.Namespace, order: Sequence[str]) -> str:
    commands = ["set -eu"]
    for backend in order:
        commands.append(benchmark_command(args, backend, f"/tmp/{backend}.json"))
    aggregate = [
        "python3",
        "-c",
        REMOTE_AGGREGATOR,
        "/tmp/portable.json",
        "/tmp/avx2.json",
        ",".join(order),
    ]
    commands.append(" ".join(shlex.quote(value) for value in aggregate))
    return "\n".join(commands)


def run_one(
    args: argparse.Namespace,
    order: Sequence[str],
    ordinal: int,
    output_dir: Path,
) -> dict[str, Any]:
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
        "--",
        "bash",
        "-lc",
        remote_script(args, order),
    ]
    console_path = output_dir / f"paired-run-{ordinal}.console.log"
    chunks: dict[int, str] = {}
    chunk_count: int | None = None
    end: dict[str, Any] | None = None
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
            if stripped.startswith(CHUNK_SENTINEL):
                descriptor, value = stripped[len(CHUNK_SENTINEL) :].split(
                    ":", 1
                )
                index_text, count_text = descriptor.split("/", 1)
                chunks[int(index_text)] = value
                chunk_count = int(count_text)
            elif stripped.startswith(END_SENTINEL):
                end = json.loads(stripped[len(END_SENTINEL) :])
            elif stripped.startswith("CHESS_BENCHMARK_PROGRESS="):
                payload = json.loads(stripped.split("=", 1)[1])
                if payload.get("phase") == "measure":
                    print(
                        f"pair {ordinal}: {payload['mode']}={payload['limit']} "
                        f"round {payload['round']}/{payload['rounds']} "
                        f"nps={payload['nps']:,.0f}",
                        flush=True,
                    )
            elif len(stripped) < 500:
                print(stripped, flush=True)
        return_code = process.wait()
    if return_code != 0 or end is None or chunk_count is None:
        raise RuntimeError(
            f"paired dyno {ordinal} failed with exit {return_code}; see {console_path}"
        )
    if set(chunks) != set(range(1, chunk_count + 1)):
        raise RuntimeError(f"paired dyno {ordinal} returned incomplete chunks")
    compressed = base64.b64decode(
        "".join(chunks[index] for index in range(1, chunk_count + 1)),
        validate=True,
    )
    if sha256_bytes(compressed) != end["compressed_sha256"]:
        raise RuntimeError(f"paired dyno {ordinal} compressed hash mismatch")
    encoded = zlib.decompress(compressed)
    if sha256_bytes(encoded) != end["json_sha256"]:
        raise RuntimeError(f"paired dyno {ordinal} JSON hash mismatch")
    result = json.loads(encoded)
    if result.get("status") != "valid" or result.get("order") != list(order):
        raise RuntimeError(f"paired dyno {ordinal} returned invalid metadata")
    result["ordinal"] = ordinal
    result_path = output_dir / f"paired-run-{ordinal}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"pair {ordinal}: saved {result_path}", flush=True)
    return result


def aggregate(runs: Sequence[dict[str, Any]], replicates: int = 10_000) -> dict[str, Any]:
    metric_keys = sorted(
        {(metric["mode"], metric["limit"]) for run in runs for metric in run["metrics"]}
    )
    metrics = []
    for metric_index, (mode, limit) in enumerate(metric_keys):
        paired_runs = []
        per_dyno = []
        for run in runs:
            metric = next(
                row
                for row in run["metrics"]
                if row["mode"] == mode and row["limit"] == limit
            )
            paired_runs.append(metric["cases"])
            per_dyno.append(
                {
                    "ordinal": run["ordinal"],
                    "order": run["order"],
                    "ratio": metric["avx2_over_portable"],
                }
            )

        def ratio(sampled_runs: Sequence[Sequence[dict[str, Any]]]) -> float:
            totals = {
                backend: {"nodes": 0, "elapsed_ns": 0} for backend in BACKENDS
            }
            for cases in sampled_runs:
                for case in cases:
                    for backend in BACKENDS:
                        nodes, elapsed_ns = case[backend]
                        totals[backend]["nodes"] += nodes
                        totals[backend]["elapsed_ns"] += elapsed_ns
            nps = {
                backend: values["nodes"] * 1_000_000_000.0 / values["elapsed_ns"]
                for backend, values in totals.items()
            }
            return nps["avx2"] / nps["portable"]

        estimate = ratio(paired_runs)
        rng = random.Random(20260827 + metric_index)
        draws = []
        for _ in range(replicates):
            sampled_runs = []
            for _ in paired_runs:
                source = paired_runs[rng.randrange(len(paired_runs))]
                sampled_runs.append([rng.choice(source) for _ in source])
            draws.append(ratio(sampled_runs))
        metrics.append(
            {
                "mode": mode,
                "limit": limit,
                "paired_cases": sum(len(cases) for cases in paired_runs),
                "estimate": estimate,
                "interval95": [percentile(draws, 0.025), percentile(draws, 0.975)],
                "per_dyno": per_dyno,
            }
        )
    fixed = [row for row in metrics if row["mode"] == "fixed_depth"]
    failures = []
    for row in fixed:
        if row["interval95"][0] <= 1.0:
            failures.append(
                f"depth {row['limit']} lower95={row['interval95'][0]:.6f}"
            )
        for dyno in row["per_dyno"]:
            if dyno["ratio"] <= 1.0:
                failures.append(
                    f"depth {row['limit']} dyno {dyno['ordinal']}="
                    f"{dyno['ratio']:.6f}"
                )
    return {
        "schema_version": 1,
        "status": "valid",
        "runs": len(runs),
        "design": "same-dyno paired A/B with alternating order",
        "metrics": metrics,
        "promotion_gate": {
            "status": "pass" if not failures else "fail",
            "criterion": "fixed-depth lower95 and every paired dyno ratio exceed 1.0",
            "failures": failures,
        },
    }


def markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Heroku NNUE accumulator paired benchmark",
        "",
        "| mode | limit | AVX2 / portable | paired 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["metrics"]:
        lower, upper = row["interval95"]
        lines.append(
            f"| {row['mode']} | {row['limit']} | {row['estimate']:.4f}x | "
            f"[{lower:.4f}, {upper:.4f}] |"
        )
    lines.extend(
        [
            "",
            f"Promotion gate: **{comparison['promotion_gate']['status']}**",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    remote_root = "/app/chess-engine"
    model_root = (
        f"{remote_root}/models/quantized_scale_grid/"
        "old_score_huber200_lr_sweep_then_5ep_20260724_142758/best"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--production-app", default="stormy-garden-92984")
    parser.add_argument("--size", default="basic")
    parser.add_argument("--process-type", default="benchmark")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--fixed-rounds", type=int, default=4)
    parser.add_argument("--movetime-rounds", type=int, default=2)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--remote-engine", default=f"{remote_root}/build-release/uci_nnue_v41")
    parser.add_argument("--remote-model", default=f"{model_root}/phase_quantized_nnue.bin")
    parser.add_argument("--remote-config", default=f"{remote_root}/deploy/heroku/benchmark-config-v41.yml")
    parser.add_argument("--remote-spec", default=f"{remote_root}/benchmarks/uci_platform_v1.json")
    parser.add_argument("--remote-harness", default=f"{remote_root}/tools/benchmark/benchmark_uci_platform.py")
    parser.add_argument("--remote-engine-cwd", default=remote_root)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.app == args.production_app:
        raise SystemExit(f"refusing production app: {args.app}")
    if args.runs < 2 or args.fixed_rounds <= 0 or args.movetime_rounds <= 0:
        raise SystemExit("runs must be >=2 and round counts must be positive")
    output_dir = args.output_dir or (
        Path("logs") / f"nnue_accumulator_paired_{utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "app": args.app,
        "size": args.size,
        "runs": args.runs,
        "fixed_rounds": args.fixed_rounds,
        "movetime_rounds": args.movetime_rounds,
        "source_label": args.source_label,
        "orders": [list(ORDERS[index % 2]) for index in range(args.runs)],
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
    }
    (output_dir / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )
    runs = []
    for index in range(args.runs):
        order = ORDERS[index % 2]
        print(
            f"starting paired dyno {index + 1}/{args.runs}: {','.join(order)}",
            flush=True,
        )
        runs.append(run_one(args, order, index + 1, output_dir))
    comparison = aggregate(runs)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "comparison.md").write_text(markdown(comparison))
    print(f"comparison: {output_dir / 'comparison.md'}", flush=True)
    return 0 if comparison["promotion_gate"]["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
