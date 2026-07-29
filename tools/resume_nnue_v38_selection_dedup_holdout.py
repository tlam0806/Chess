#!/usr/bin/env python3
"""Resume missing V38 selection configs, deduplicate Pareto ties, then hold out."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_nnue_v38_selective import Config, append_json, evaluate, frontier


def config_of(record: dict) -> Config:
    return Config(**record["config"])


def representative_key(entry: dict) -> tuple:
    """Prefer safer tails, then the less aggressive config among exact ties."""
    result = entry["result"]
    config = config_of(entry)
    return (
        result.get("p95_root_regret", 0),
        result.get("above_100_cp_pct", 0),
        -result.get("ranking_move_agreement_pct", 0),
        -config.lmr_min_depth,
        -config.lmr_min_move_index,
        config.lmr_base,
        -config.lmr_divisor,
        -config.null_min_depth,
        config.null_reduction,
    )


def deduplicate_objectives(entries: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = {}
    for entry in entries:
        result = entry["result"]
        key = (result["node_ratio"], result["mean_root_regret"])
        groups.setdefault(key, []).append(entry)
    return [min(group, key=representative_key) for group in groups.values()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ranking-target-abs-cp", type=int, default=1500)
    parser.add_argument("--selection-depth", type=int, default=6)
    parser.add_argument("--holdout-depth", type=int, default=7)
    args = parser.parse_args()

    log_path = args.run_dir / "results.jsonl"
    existing = [
        json.loads(line) for line in log_path.read_text().splitlines()
    ] if log_path.exists() else []
    tune_records = [
        json.loads(line) for line in args.tune_log.read_text().splitlines()
        if json.loads(line).get("kind") == "tune"
    ]
    tune_entries = [
        {**record, "config_obj": config_of(record)}
        for record in tune_records
    ]
    candidates = frontier(tune_entries)
    selection_entries = [
        record for record in existing if record.get("kind") == "selection"
    ]
    selected = {config_of(record) for record in selection_entries}
    start = time.monotonic()

    for rank, tune_entry in enumerate(candidates):
        config = tune_entry["config_obj"]
        if config in selected:
            continue
        result = evaluate(
            args.binary,
            args.dataset_dir / "selection.tsv",
            args.model,
            args.selection_depth,
            config,
            args.ranking_target_abs_cp,
        )
        entry = {
            "kind": "selection",
            "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": config.__dict__,
            "result": result,
            "resumed": True,
        }
        append_json(log_path, entry)
        selection_entries.append(entry)
        selected.add(config)

    raw_frontier = frontier([
        {**entry, "config_obj": config_of(entry)}
        for entry in selection_entries
    ])
    deduplicated = deduplicate_objectives(raw_frontier)
    deduplicated.sort(key=lambda entry: entry["result"]["node_ratio"])
    holdout_entries = [
        record for record in existing if record.get("kind") == "holdout"
    ]
    held_out = {config_of(record) for record in holdout_entries}

    for rank, selection_entry in enumerate(deduplicated):
        config = config_of(selection_entry)
        if config in held_out:
            continue
        result = evaluate(
            args.binary,
            args.dataset_dir / "holdout.tsv",
            args.model,
            args.holdout_depth,
            config,
            args.ranking_target_abs_cp,
        )
        entry = {
            "kind": "holdout",
            "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": config.__dict__,
            "result": result,
            "deduplicated_selection_objective": True,
        }
        append_json(log_path, entry)
        holdout_entries.append(entry)
        held_out.add(config)

    summary = {
        "kind": "complete",
        "elapsed_sec": time.monotonic() - start,
        "tune_frontier_size": len(candidates),
        "selection_completed": len(selection_entries),
        "selection_frontier_size_raw": len(raw_frontier),
        "selection_frontier_size_deduplicated": len(deduplicated),
        "holdout": holdout_entries,
        "log": str(log_path),
    }
    append_json(log_path, summary)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
