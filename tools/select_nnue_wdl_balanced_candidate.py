#!/usr/bin/env python3
"""Select a WDL-tuned balanced candidate under the old node budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FIELDS = (
    "lmr_base",
    "lmr_divisor",
    "lmr_min_depth",
    "lmr_min_move_index",
    "null_min_depth",
    "null_reduction",
)


def loss(entry: dict) -> float:
    result = entry["result"]
    if "objective_loss" in result:
        return float(result["objective_loss"])
    return float(result["mean_wdl_loss"])


def select_candidate(
    entries: list[dict],
    old_result: dict,
    node_tolerance: float,
) -> dict:
    valid = [
        entry for entry in entries
        if entry["result"].get("critical_mistakes", 0) == 0
    ]
    if not valid:
        raise ValueError("holdout frontier has no safe candidate")
    old_ratio = float(old_result["node_ratio"])
    node_cap = old_ratio * node_tolerance
    feasible = [
        entry for entry in valid
        if float(entry["result"]["node_ratio"]) <= node_cap
    ]
    if feasible:
        winner = min(
            feasible,
            key=lambda entry: (
                loss(entry),
                -float(entry["result"]["node_ratio"]),
            ),
        )
        selection_rule = "lowest_wdl_loss_under_node_cap"
    else:
        winner = min(
            valid,
            key=lambda entry: abs(
                math.log(float(entry["result"]["node_ratio"]) / old_ratio)),
        )
        selection_rule = "closest_node_ratio_no_candidate_under_cap"
    return {
        "selection_rule": selection_rule,
        "node_tolerance": node_tolerance,
        "old_node_ratio": old_ratio,
        "node_cap": node_cap,
        "old_result": old_result,
        "candidate": winner,
    }


def config_csv(config: dict) -> str:
    return ",".join(str(config[field]) for field in FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--old-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-tolerance", type=float, default=1.02)
    args = parser.parse_args()
    if args.node_tolerance < 1.0:
        raise ValueError("--node-tolerance must be at least 1.0")
    summary = json.loads(args.summary.read_text())
    old_result = json.loads(args.old_result.read_text())
    selected = select_candidate(
        summary["holdout"], old_result, args.node_tolerance)
    args.output.write_text(json.dumps(selected, indent=2) + "\n")
    print(config_csv(selected["candidate"]["config"]))


if __name__ == "__main__":
    main()
