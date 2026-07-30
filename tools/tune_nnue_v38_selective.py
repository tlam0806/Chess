#!/usr/bin/env python3
"""Pareto tuner for V38 LMR and null-move pruning."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    lmr_base: float = 0.5
    lmr_divisor: float = 2.6
    lmr_min_depth: int = 4
    lmr_min_move_index: int = 4
    null_min_depth: int = 3
    null_reduction: int = 2

    def args(self) -> list[str]:
        return [
            "--lmr-base", str(self.lmr_base),
            "--lmr-divisor", str(self.lmr_divisor),
            "--lmr-min-depth", str(self.lmr_min_depth),
            "--lmr-min-move-index", str(self.lmr_min_move_index),
            "--null-min-depth", str(self.null_min_depth),
            "--null-reduction", str(self.null_reduction),
        ]


def objective_loss(result: dict) -> float:
    if "objective_loss" in result:
        return result["objective_loss"]
    return result["mean_root_regret"]


def dominated(a: dict, b: dict) -> bool:
    """True when result a is dominated by result b."""
    return (
        b["node_ratio"] <= a["node_ratio"]
        and objective_loss(b) <= objective_loss(a)
        and (
            b["node_ratio"] < a["node_ratio"]
            or objective_loss(b) < objective_loss(a)
        )
    )


def frontier(entries: list[dict]) -> list[dict]:
    valid = [
        e for e in entries
        if e["result"].get(
            "critical_mistakes",
            e["result"].get("mate_mistakes", 0),
        ) == 0
    ]
    return [
        entry for entry in valid
        if not any(dominated(entry["result"], other["result"])
                   for other in valid if other is not entry)
    ]


def objective_group_key(entry: dict) -> tuple[float, float]:
    result = entry["result"]
    return result["node_ratio"], objective_loss(result)


def representative_key(entry: dict) -> tuple:
    """Prefer safer tails, then the less aggressive config among exact ties."""
    result = entry["result"]
    config = entry.get("config_obj") or Config(**entry["config"])
    return (
        result.get("p95_root_regret", 0),
        result.get("p95_wdl_loss", 0),
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
        groups.setdefault(objective_group_key(entry), []).append(entry)
    return [min(group, key=representative_key) for group in groups.values()]


def mutate(config: Config, rng: random.Random, changed_parameters: int = 1) -> Config:
    values = config.__dict__.copy()
    for name in rng.sample(list(values), changed_parameters):
        original = values[name]
        if name == "lmr_base":
            candidates = [
                round(min(1.5, max(0.0, original + delta)), 3)
                for delta in (-0.1, -0.05, 0.05, 0.1)
            ]
        elif name == "lmr_divisor":
            candidates = [
                round(min(5.0, max(1.2, original + delta)), 3)
                for delta in (-0.3, -0.15, 0.15, 0.3)
            ]
        elif name == "lmr_min_depth":
            candidates = [min(7, max(2, original + delta)) for delta in (-1, 1)]
        elif name == "lmr_min_move_index":
            candidates = [
                min(12, max(2, original + delta)) for delta in (-2, -1, 1, 2)
            ]
        elif name == "null_min_depth":
            candidates = [min(7, max(2, original + delta)) for delta in (-1, 1)]
        else:
            candidates = [min(4, max(1, original + delta)) for delta in (-1, 1)]
        values[name] = rng.choice([value for value in candidates if value != original])
    return Config(**values)


def random_config(rng: random.Random) -> Config:
    return Config(
        lmr_base=rng.choice(
            [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9]),
        lmr_divisor=rng.choice(
            [1.8, 2.15, 2.45, 2.75, 3.05, 3.35]),
        lmr_min_depth=rng.choice([2, 3, 4, 5, 6]),
        lmr_min_move_index=rng.choice([2, 3, 4, 5, 6, 8]),
        null_min_depth=rng.choice([2, 3, 4, 5]),
        null_reduction=rng.choice([1, 2, 3, 4]),
    )


def load_by_category(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        category = line.split("\t", 1)[0]
        groups.setdefault(category, []).append(line)
    return groups


def rotating_subset(groups: dict[str, list[str]], iteration: int, seed: int) -> list[str]:
    wanted = {"random": 140, "balanced": 30, "tactical": 20, "endgame": 10}
    selected: list[str] = []
    for category, count in wanted.items():
        rows = groups[category]
        order = list(range(len(rows)))
        random.Random(seed ^ iteration * 0x9E3779B1 ^ sum(map(ord, category))).shuffle(order)
        selected.extend(rows[index] for index in order[:count])
    random.Random(seed ^ iteration).shuffle(selected)
    return selected


def fixed_subset(
    groups: dict[str, list[str]], total: int, seed: int
) -> list[str]:
    weights = {"random": 70, "balanced": 15, "tactical": 10, "endgame": 5}
    counts = {
        category: total * weight // 100
        for category, weight in weights.items()
    }
    counts["random"] += total - sum(counts.values())
    selected: list[str] = []
    for category, count in counts.items():
        rows = groups[category].copy()
        random.Random(seed ^ sum(map(ord, category))).shuffle(rows)
        selected.extend(rows[:count])
    random.Random(seed).shuffle(selected)
    return selected


def thin_frontier(entries: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or len(entries) <= limit:
        return entries
    ordered = sorted(entries, key=lambda entry: entry["result"]["node_ratio"])
    indices = {
        round(index * (len(ordered) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [ordered[index] for index in sorted(indices)]


def evaluate(
    binary: Path, dataset: Path, model: Path, depth: int, config: Config,
    ranking_target_abs_cp: int, objective: str,
) -> dict:
    command = [
        str(binary), "--dataset", str(dataset), "--model", str(model),
        "--depth", str(depth),
        "--ranking-target-abs-cp", str(ranking_target_abs_cp),
        "--objective", objective,
        *config.args(),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"evaluation failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def append_json(path: Path, record: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=int, default=7200)
    parser.add_argument("--mutation-duration-sec", type=int)
    parser.add_argument("--tune-depth", type=int, default=6)
    parser.add_argument("--selection-depth", type=int, default=6)
    parser.add_argument("--holdout-depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--stage2-multi-param", action="store_true")
    parser.add_argument("--exclude-log", type=Path)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--selection-only-from-log", type=Path)
    parser.add_argument("--fixed-tune-size", type=int, default=0)
    parser.add_argument("--wide-mutations", action="store_true")
    parser.add_argument("--frontier-cap", type=int, default=0)
    parser.add_argument("--ranking-target-abs-cp", type=int, default=1500)
    parser.add_argument("--objective", choices=("cp", "wdl"), default="cp")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.run_dir / "current_subset.tsv"
    log_path = args.run_dir / "results.jsonl"
    groups = load_by_category(args.dataset_dir / "tune.tsv")
    rng = random.Random(args.seed)
    start = time.monotonic()
    entries: list[dict] = []
    seen: set[Config] = set()
    if args.exclude_log is not None:
        for line in args.exclude_log.read_text().splitlines():
            record = json.loads(line)
            if record.get("kind") == "tune":
                seen.add(Config(**record["config"]))
    current_frontier: list[dict] = []
    iteration = 0
    stage2_seeds = [
        Config(0.5, 2.45, 4, 4, 2, 3),
        Config(0.5, 2.45, 4, 5, 2, 3),
        Config(0.45, 2.45, 5, 5, 2, 3),
    ]
    wide_seeds = [
        Config(),
        *stage2_seeds,
        Config(0.5, 2.9, 3, 5, 2, 3),
        Config(0.65, 3.05, 3, 2, 2, 3),
        Config(0.55, 2.75, 3, 3, 2, 2),
    ]
    if args.fixed_tune_size > 0:
        subset_path.write_text("\n".join(
            fixed_subset(groups, args.fixed_tune_size, args.seed)
        ) + "\n")
    if args.selection_only_from_log is not None:
        for line in args.selection_only_from_log.read_text().splitlines():
            record = json.loads(line)
            if record.get("kind") != "tune":
                continue
            config = Config(**record["config"])
            entries.append({
                **record,
                "config_obj": config,
            })
        current_frontier = frontier(entries)
        iteration = len(entries)
    # Reserve roughly 20% for full selection and the one-shot holdout.
    mutation_duration_sec = (
        args.mutation_duration_sec
        if args.mutation_duration_sec is not None
        else args.duration_sec * 0.80
    )
    mutation_deadline = (
        start if args.selection_only_from_log is not None
        else start + mutation_duration_sec
    )
    while (
        time.monotonic() < mutation_deadline
        and (args.max_candidates <= 0 or iteration < args.max_candidates)
    ):
        seed_pool = wide_seeds if args.wide_mutations else stage2_seeds
        if args.wide_mutations and iteration < len(wide_seeds):
            config = wide_seeds[iteration]
            seen.discard(config)
        elif args.stage2_multi_param and iteration < len(stage2_seeds):
            config = stage2_seeds[iteration]
            seen.discard(config)
        else:
            parent_pool = seed_pool if (
                args.stage2_multi_param or args.wide_mutations
            ) else [Config()]
            source_roll = rng.random()
            if args.wide_mutations and source_roll >= 0.8:
                config = random_config(rng)
            else:
                parent = (
                    rng.choice(current_frontier)["config_obj"]
                    if current_frontier and source_roll < 0.6
                    else rng.choice(parent_pool)
                )
                config = mutate(
                    parent,
                    rng,
                    rng.choice([1, 2, 3]) if args.wide_mutations
                    else rng.choice([2, 3]) if args.stage2_multi_param
                    else 1,
                )
        if config in seen:
            continue
        seen.add(config)
        if args.fixed_tune_size <= 0:
            subset_path.write_text("\n".join(
                rotating_subset(groups, iteration, args.seed)) + "\n")
        result = evaluate(
            args.binary, subset_path, args.model, args.tune_depth, config,
            args.ranking_target_abs_cp, args.objective)
        entry = {
            "kind": "tune", "iteration": iteration,
            "elapsed_sec": time.monotonic() - start,
            "config": config.__dict__, "config_obj": config, "result": result,
        }
        entries.append(entry)
        current_frontier = frontier(entries)
        append_json(log_path, {
            key: value for key, value in entry.items() if key != "config_obj"
        })
        iteration += 1

    # Re-evaluate only the tune Pareto frontier on the fixed selection set.
    selection_entries: list[dict] = []
    selection_candidates = thin_frontier(
        current_frontier, args.frontier_cap)
    for rank, tune_entry in enumerate(selection_candidates):
        config = tune_entry["config_obj"]
        result = evaluate(
            args.binary, args.dataset_dir / "selection.tsv", args.model,
            args.selection_depth, config, args.ranking_target_abs_cp,
            args.objective)
        entry = {
            "kind": "selection", "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": config.__dict__, "config_obj": config, "result": result,
        }
        selection_entries.append(entry)
        append_json(log_path, {
            key: value for key, value in entry.items() if key != "config_obj"
        })

    raw_final_frontier = frontier(selection_entries)
    final_frontier = deduplicate_objectives(raw_final_frontier)
    final_frontier.sort(key=lambda entry: entry["result"]["node_ratio"])
    holdout_entries: list[dict] = []
    for rank, selection_entry in enumerate(final_frontier):
        config = selection_entry["config_obj"]
        result = evaluate(
            args.binary, args.dataset_dir / "holdout.tsv", args.model,
            args.holdout_depth, config, args.ranking_target_abs_cp,
            args.objective)
        entry = {
            "kind": "holdout", "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": config.__dict__, "result": result,
        }
        holdout_entries.append(entry)
        append_json(log_path, entry)
    summary = {
        "kind": "complete",
        "seed": args.seed,
        "duration_requested_sec": args.duration_sec,
        "objective": args.objective,
        "elapsed_sec": time.monotonic() - start,
        "mutations": iteration,
        "tune_frontier_size": len(current_frontier),
        "selection_candidates": len(selection_candidates),
        "selection_frontier_size_raw": len(raw_final_frontier),
        "selection_frontier_size": len(final_frontier),
        "holdout": holdout_entries,
        "log": str(log_path),
    }
    append_json(log_path, summary)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
