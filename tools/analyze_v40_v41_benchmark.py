#!/usr/bin/env python3
"""Summarize paired V40/V41 repetition benchmark logs.

The benchmark emits one aggregate row per version and round.  This analyzer
keeps each round paired, computes pooled ratios, and resamples complete rounds
for descriptive confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Sequence


ROUND_RE = re.compile(
    r"^round=(?P<round>\d+) version=(?P<version>v40|v41) "
    r"nodes=(?P<nodes>\d+) us=(?P<us>\d+) "
)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_rounds(path: Path) -> list[dict[str, tuple[int, int]]]:
    by_round: dict[int, dict[str, tuple[int, int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROUND_RE.match(line)
        if match is None:
            continue
        round_index = int(match.group("round"))
        version = match.group("version")
        row = (int(match.group("nodes")), int(match.group("us")))
        if version in by_round.setdefault(round_index, {}):
            raise ValueError(f"duplicate {version} row for round {round_index}")
        by_round[round_index][version] = row

    if not by_round:
        raise ValueError(f"no benchmark rounds found in {path}")
    expected = list(range(max(by_round) + 1))
    if sorted(by_round) != expected:
        raise ValueError("round indices must be contiguous from zero")
    for round_index, versions in by_round.items():
        if set(versions) != {"v40", "v41"}:
            raise ValueError(f"round {round_index} is not a complete pair")
    return [by_round[index] for index in expected]


def summarize(
    rounds: Sequence[dict[str, tuple[int, int]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    def totals(sample: Sequence[dict[str, tuple[int, int]]], version: str) -> tuple[int, int]:
        return (
            sum(pair[version][0] for pair in sample),
            sum(pair[version][1] for pair in sample),
        )

    v40_nodes, v40_us = totals(rounds, "v40")
    v41_nodes, v41_us = totals(rounds, "v41")
    nps_ratio = (v41_nodes / v41_us) / (v40_nodes / v40_us)
    time_ratio = v41_us / v40_us

    rng = random.Random(seed)
    nps_ratios: list[float] = []
    time_ratios: list[float] = []
    for _ in range(replicates):
        sample = [rounds[rng.randrange(len(rounds))] for _ in rounds]
        sampled_v40_nodes, sampled_v40_us = totals(sample, "v40")
        sampled_v41_nodes, sampled_v41_us = totals(sample, "v41")
        nps_ratios.append(
            (sampled_v41_nodes / sampled_v41_us)
            / (sampled_v40_nodes / sampled_v40_us)
        )
        time_ratios.append(sampled_v41_us / sampled_v40_us)

    return {
        "rounds": len(rounds),
        "bootstrap_seed": seed,
        "v40": {
            "nodes": v40_nodes,
            "elapsed_us": v40_us,
            "nps": v40_nodes * 1_000_000.0 / v40_us,
        },
        "v41": {
            "nodes": v41_nodes,
            "elapsed_us": v41_us,
            "nps": v41_nodes * 1_000_000.0 / v41_us,
        },
        "node_ratio": v41_nodes / v40_nodes,
        "nps_ratio": nps_ratio,
        "nps_ratio_interval95": [
            percentile(nps_ratios, 0.025),
            percentile(nps_ratios, 0.975),
        ],
        "time_ratio": time_ratio,
        "time_ratio_interval95": [
            percentile(time_ratios, 0.025),
            percentile(time_ratios, 0.975),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_825)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")

    results = {}
    for index, path in enumerate(args.logs):
        results[path.name] = summarize(
            parse_rounds(path),
            replicates=args.bootstrap_replicates,
            seed=args.seed + index,
        )
    document = {
        "schema_version": 1,
        "bootstrap": {
            "method": "paired complete-round percentile bootstrap",
            "replicates": args.bootstrap_replicates,
            "seed_base": args.seed,
            "seed_scheme": "seed_base + zero-based input index",
            "interpretation": "conditional descriptive interval",
        },
        "results": results,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
