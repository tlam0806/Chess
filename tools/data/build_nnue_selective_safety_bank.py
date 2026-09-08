#!/usr/bin/env python3
"""Mine pruning-sensitive positions and split them into core/sealed safety banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Probe:
    name: str
    args: tuple[str, ...]


def selective_args(
    *,
    enable_lmr: bool,
    lmr_base: float = 0.5,
    lmr_divisor: float = 2.6,
    lmr_min_depth: int = 4,
    lmr_min_move_index: int = 4,
    enable_null: bool,
    null_min_depth: int = 3,
    null_reduction: int = 2,
) -> tuple[str, ...]:
    return (
        "--lmr-base", str(lmr_base),
        "--lmr-divisor", str(lmr_divisor),
        "--lmr-min-depth", str(lmr_min_depth),
        "--lmr-min-move-index", str(lmr_min_move_index),
        "--null-min-depth", str(null_min_depth),
        "--null-reduction", str(null_reduction),
        *(() if enable_lmr else ("--disable-lmr",)),
        *(() if enable_null else ("--disable-null-move",)),
        "--disable-reverse-futility",
        "--disable-late-move-pruning",
    )


def default_probes() -> list[Probe]:
    return [
        Probe("lmr_fast", selective_args(
            enable_lmr=True, lmr_base=0.25, lmr_divisor=1.95,
            lmr_min_depth=3, lmr_min_move_index=8, enable_null=False)),
        Probe("lmr_balanced", selective_args(
            enable_lmr=True, lmr_base=0.45, lmr_divisor=2.45,
            lmr_min_depth=5, lmr_min_move_index=5, enable_null=False)),
        Probe("nmp_mild", selective_args(
            enable_lmr=False, enable_null=True,
            null_min_depth=5, null_reduction=1)),
        Probe("nmp_standard", selective_args(
            enable_lmr=False, enable_null=True,
            null_min_depth=3, null_reduction=2)),
        Probe("nmp_aggressive", selective_args(
            enable_lmr=False, enable_null=True,
            null_min_depth=3, null_reduction=3)),
        Probe("joint_fast", selective_args(
            enable_lmr=True, lmr_base=0.25, lmr_divisor=1.95,
            lmr_min_depth=3, lmr_min_move_index=8, enable_null=True,
            null_min_depth=3, null_reduction=2)),
        Probe("joint_balanced", selective_args(
            enable_lmr=True, lmr_base=0.45, lmr_divisor=2.45,
            lmr_min_depth=5, lmr_min_move_index=5, enable_null=True,
            null_min_depth=3, null_reduction=2)),
        Probe("joint_safe", selective_args(
            enable_lmr=True, lmr_base=0.65, lmr_divisor=2.9,
            lmr_min_depth=7, lmr_min_move_index=3, enable_null=True,
            null_min_depth=3, null_reduction=2)),
    ]


def parse_detail(line: str) -> dict | None:
    if not line.startswith("detail\t"):
        return None
    fields: dict[str, str] = {}
    for part in line.rstrip().split("\t")[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    required = {
        "category", "hash", "payload", "control_move", "candidate_move",
        "control_score", "candidate_strict_score", "regret",
    }
    if not required.issubset(fields):
        return None
    result: dict = dict(fields)
    for key in (
        "index", "static_target_cp", "control_score",
        "candidate_strict_score", "regret",
    ):
        result[key] = int(fields[key])
    for key in ("control_mate", "candidate_mate"):
        result[key] = fields.get(key, "0") == "1"
    return result


def classify(detail: dict) -> str:
    base = detail["control_score"]
    candidate = detail["candidate_strict_score"]
    if detail["candidate_mate"] and candidate < 0 and base > -100:
        return "self_mate"
    if base >= 500 and candidate < -100:
        return "win_to_loss"
    if base >= 500 and abs(candidate) <= 100:
        return "win_to_draw"
    return "near_miss"


def severity(detail: dict) -> tuple[int, int]:
    failure = classify(detail)
    order = {
        "self_mate": 3,
        "win_to_loss": 2,
        "win_to_draw": 1,
        "near_miss": 0,
    }
    return order[failure], detail["regret"]


def stable_fraction(seed: int, position_hash: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{position_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def read_seed_rows(paths: list[Path]) -> dict[str, dict]:
    seeded: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            fields = line.split("\t", 2)
            if len(fields) != 3:
                raise RuntimeError(f"bad seed safety row: {path}")
            category, position_hash, payload = fields
            seeded[position_hash] = {
                "category": category,
                "hash": position_hash,
                "payload": payload,
                "failure_type": "seeded_critical",
                "severity": [4, 0],
                "observations": [],
            }
    return seeded


def write_tsv(path: Path, entries: list[dict]) -> None:
    path.write_text("".join(
        f"{entry['category']}\t{entry['hash']}\t{entry['payload']}\n"
        for entry in entries
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, action="append", required=True)
    parser.add_argument("--seed-tsv", type=Path, action="append", default=[])
    parser.add_argument("--detail-threshold", type=int, default=250)
    parser.add_argument("--core-fraction", type=float, default=0.70)
    parser.add_argument("--max-core", type=int, default=100)
    parser.add_argument("--max-sealed", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.core_fraction < 1.0:
        raise SystemExit("--core-fraction must be between zero and one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_log = args.output_dir / "mining_runs.jsonl"
    completed: set[tuple[str, int, str]] = set()
    observations: list[dict] = []
    if args.resume and run_log.exists():
        for line in run_log.read_text().splitlines():
            record = json.loads(line)
            completed.add((
                record["dataset"], record["depth"], record["probe"]))
            observations.extend(record["details"])

    common = [
        str(args.binary),
        "--model", str(args.model),
        "--ranking-target-abs-cp", "1500",
        "--detail-threshold", str(args.detail_threshold),
    ]
    for dataset in args.dataset:
        for depth in args.depth:
            for probe in default_probes():
                key = (str(dataset), depth, probe.name)
                if key in completed:
                    continue
                command = [
                    *common,
                    "--dataset", str(dataset),
                    "--depth", str(depth),
                    *probe.args,
                ]
                process = subprocess.run(
                    command, text=True, capture_output=True)
                if process.returncode:
                    raise RuntimeError(
                        "safety mining failed\n"
                        f"command: {' '.join(command)}\n"
                        f"stderr: {process.stderr.strip()}")
                details: list[dict] = []
                for line in process.stderr.splitlines():
                    detail = parse_detail(line)
                    if detail is None:
                        continue
                    detail.update({
                        "dataset": str(dataset),
                        "depth": depth,
                        "probe": probe.name,
                        "failure_type": classify(detail),
                    })
                    details.append(detail)
                aggregate = json.loads(process.stdout)
                record = {
                    "dataset": str(dataset),
                    "depth": depth,
                    "probe": probe.name,
                    "aggregate": aggregate,
                    "details": details,
                }
                with run_log.open("a") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                observations.extend(details)
                print(
                    "mining_progress"
                    f" dataset={dataset.name}"
                    f" depth={depth}"
                    f" probe={probe.name}"
                    f" details={len(details)}"
                    f" critical={sum(classify(x) != 'near_miss' for x in details)}",
                    flush=True,
                )

    positions = read_seed_rows(args.seed_tsv)
    for observation in observations:
        position_hash = observation["hash"]
        entry = positions.setdefault(position_hash, {
            "category": observation["category"],
            "hash": position_hash,
            "payload": observation["payload"],
            "failure_type": observation["failure_type"],
            "severity": list(severity(observation)),
            "observations": [],
        })
        entry["observations"].append(observation)
        if severity(observation) > tuple(entry["severity"]):
            entry["severity"] = list(severity(observation))
            entry["failure_type"] = observation["failure_type"]

    ordered = sorted(
        positions.values(),
        key=lambda entry: (
            -entry["severity"][0],
            -entry["severity"][1],
            stable_fraction(args.seed, entry["hash"]),
        ),
    )
    critical = [
        entry for entry in ordered if entry["severity"][0] >= 1]
    near = [
        entry for entry in ordered if entry["severity"][0] == 0]
    core_pool: list[dict] = []
    sealed_pool: list[dict] = []
    for entry in ordered:
        seeded = entry["failure_type"] == "seeded_critical"
        in_core_hash_partition = (
            stable_fraction(args.seed, entry["hash"]) % 10_000
            < int(args.core_fraction * 10_000)
        )
        (core_pool if seeded or in_core_hash_partition else sealed_pool).append(
            entry)
    core_pool.sort(
        key=lambda entry: (-entry["severity"][0], -entry["severity"][1]))
    sealed_pool.sort(
        key=lambda entry: (-entry["severity"][0], -entry["severity"][1]))

    core = core_pool[:args.max_core]
    sealed = sealed_pool[:args.max_sealed]
    write_tsv(args.output_dir / "core.tsv", core)
    write_tsv(args.output_dir / "sealed.tsv", sealed)
    with (args.output_dir / "metadata.jsonl").open("w") as output:
        for split, entries in (("core", core), ("sealed", sealed)):
            for entry in entries:
                output.write(json.dumps({
                    **entry, "split": split,
                }, sort_keys=True) + "\n")
    manifest = {
        "seed": args.seed,
        "detail_threshold": args.detail_threshold,
        "core_fraction": args.core_fraction,
        "datasets": [str(path) for path in args.dataset],
        "depths": args.depth,
        "probes": [probe.name for probe in default_probes()],
        "unique_positions": len(positions),
        "critical_positions": len(critical),
        "near_miss_positions": len(near),
        "core": len(core),
        "sealed": len(sealed),
        "core_sha256": hashlib.sha256(
            (args.output_dir / "core.tsv").read_bytes()).hexdigest(),
        "sealed_sha256": hashlib.sha256(
            (args.output_dir / "sealed.tsv").read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "safety_bank_complete", **manifest},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
