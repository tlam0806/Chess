#!/usr/bin/env python3
"""Joint Pareto tuner for V39 LMR, NMP, RFP and LMP."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    lmr_base: float
    lmr_divisor: float
    lmr_min_depth: int
    lmr_min_move_index: int
    null_min_depth: int
    null_reduction: int
    reverse_futility_max_depth: int
    reverse_futility_base_margin: int
    reverse_futility_margin_per_depth: int
    late_move_pruning_max_depth: int
    late_move_pruning_base: int
    late_move_pruning_depth_multiplier: int
    enable_reverse_futility: bool = True
    enable_late_move_pruning: bool = True

    def args(self) -> list[str]:
        return [
            "--lmr-base", str(self.lmr_base),
            "--lmr-divisor", str(self.lmr_divisor),
            "--lmr-min-depth", str(self.lmr_min_depth),
            "--lmr-min-move-index", str(self.lmr_min_move_index),
            "--null-min-depth", str(self.null_min_depth),
            "--null-reduction", str(self.null_reduction),
            ("--enable-reverse-futility"
             if self.enable_reverse_futility
             else "--disable-reverse-futility"),
            "--reverse-futility-max-depth",
            str(self.reverse_futility_max_depth),
            "--reverse-futility-base-margin",
            str(self.reverse_futility_base_margin),
            "--reverse-futility-margin-per-depth",
            str(self.reverse_futility_margin_per_depth),
            ("--enable-late-move-pruning"
             if self.enable_late_move_pruning
             else "--disable-late-move-pruning"),
            "--late-move-pruning-max-depth",
            str(self.late_move_pruning_max_depth),
            "--late-move-pruning-base", str(self.late_move_pruning_base),
            "--late-move-pruning-depth-multiplier",
            str(self.late_move_pruning_depth_multiplier),
        ]


FAST = Config(0.45, 2.9, 3, 6, 2, 3, 2, 175, 275, 3, 4, 2)
BALANCED = Config(0.45, 2.9, 3, 6, 2, 3, 2, 150, 150, 3, 7, 5)
BASELINE7 = Config(
    0.45, 2.9, 3, 6, 2, 3, 2, 250, 250, 2, 8, 4, False, False)

BLOCKS = {
    "lmr": (
        "lmr_base", "lmr_divisor", "lmr_min_depth",
        "lmr_min_move_index",
    ),
    "nmp": ("null_min_depth", "null_reduction"),
    "rfp": (
        "reverse_futility_max_depth", "reverse_futility_base_margin",
        "reverse_futility_margin_per_depth",
    ),
    "lmp": (
        "late_move_pruning_max_depth", "late_move_pruning_base",
        "late_move_pruning_depth_multiplier",
    ),
}
PARAMETERS = tuple(name for block in BLOCKS.values() for name in block)

GRIDS: dict[str, tuple[int | float, ...]] = {
    "lmr_base": tuple(round(value / 100, 2) for value in range(0, 101, 5)),
    "lmr_divisor": tuple(round(value / 100, 2)
                         for value in range(180, 351, 5)),
    "lmr_min_depth": tuple(range(3, 9)),
    "lmr_min_move_index": tuple(range(2, 13)),
    "null_min_depth": tuple(range(2, 8)),
    "null_reduction": tuple(range(1, 5)),
    "reverse_futility_max_depth": (1, 2, 3),
    "reverse_futility_base_margin": tuple(range(75, 501, 25)),
    "reverse_futility_margin_per_depth": tuple(range(100, 401, 25)),
    "late_move_pruning_max_depth": (2, 3, 4, 5),
    "late_move_pruning_base": tuple(range(0, 17)),
    "late_move_pruning_depth_multiplier": tuple(range(0, 9)),
}


def initial_configs() -> list[Config]:
    """Use the match winner plus diverse earlier LMR/NMP frontier points."""
    return [
        FAST,
        BALANCED,
        BASELINE7,
        replace(FAST, lmr_divisor=2.65, lmr_min_depth=5,
                lmr_min_move_index=5, null_min_depth=3,
                null_reduction=1),
        replace(BALANCED, lmr_base=0.5, lmr_divisor=2.55,
                lmr_min_depth=5, lmr_min_move_index=7,
                null_min_depth=5, null_reduction=2),
        replace(BALANCED, lmr_base=0.55, lmr_divisor=2.8,
                lmr_min_depth=5, lmr_min_move_index=8,
                null_min_depth=6, null_reduction=2),
    ]


def canonical(config: Config) -> Config:
    values = asdict(config)
    for name, grid in GRIDS.items():
        value = values[name]
        values[name] = min(grid, key=lambda item: abs(item - value))
    if values["reverse_futility_max_depth"] >= 3:
        values["reverse_futility_base_margin"] = max(
            300, values["reverse_futility_base_margin"])
        values["reverse_futility_margin_per_depth"] = max(
            250, values["reverse_futility_margin_per_depth"])
    if values["late_move_pruning_max_depth"] >= 4:
        values["late_move_pruning_base"] = max(
            8, values["late_move_pruning_base"])
        values["late_move_pruning_depth_multiplier"] = max(
            4, values["late_move_pruning_depth_multiplier"])
    return Config(**values)


def neighboring_value(name: str, original: int | float,
                      rng: random.Random) -> int | float:
    grid = GRIDS[name]
    index = grid.index(original)
    candidates = [
        grid[index + delta]
        for delta in (-2, -1, 1, 2)
        if 0 <= index + delta < len(grid)
    ]
    if not candidates:
        raise RuntimeError(f"no mutation available for {name}={original}")
    return rng.choice(candidates)


def mutate_parameters(config: Config, names: list[str],
                      rng: random.Random) -> Config:
    values = asdict(config)
    values["enable_reverse_futility"] = True
    values["enable_late_move_pruning"] = True
    for name in names:
        values[name] = neighboring_value(name, values[name], rng)
    return canonical(Config(**values))


def mutate(config: Config, mode: str, rng: random.Random) -> Config:
    if mode == "single":
        names = [rng.choice(PARAMETERS)]
    elif mode == "pair":
        first, second = rng.sample(list(BLOCKS), 2)
        names = [rng.choice(BLOCKS[first]), rng.choice(BLOCKS[second])]
    elif mode == "multi":
        block_count = rng.choice((3, 4))
        blocks = rng.sample(list(BLOCKS), block_count)
        names = [rng.choice(BLOCKS[block]) for block in blocks]
    else:
        raise ValueError(f"unknown mutation mode: {mode}")
    return mutate_parameters(config, names, rng)


def random_config(rng: random.Random) -> Config:
    return canonical(Config(**{
        name: rng.choice(grid) for name, grid in GRIDS.items()
    }))


def crossover(parents: list[Config], rng: random.Random) -> Config:
    if len(parents) < 2:
        return random_config(rng)
    first, second = rng.sample(parents, 2)
    values = asdict(first)
    inherited_second = 0
    for block in BLOCKS.values():
        if rng.random() < 0.5:
            inherited_second += 1
            for name in block:
                values[name] = getattr(second, name)
    if inherited_second in (0, len(BLOCKS)):
        block = rng.choice(list(BLOCKS.values()))
        source = second if inherited_second == 0 else first
        for name in block:
            values[name] = getattr(source, name)
    values["enable_reverse_futility"] = True
    values["enable_late_move_pruning"] = True
    return canonical(Config(**values))


def cp_loss(entry: dict) -> float:
    return float(entry["result"]["mean_root_regret"])


def wdl_loss(entry: dict) -> float:
    return float(entry["result"]["mean_wdl_loss"])


def dominated(entry: dict, other: dict, fixed_time: bool = False) -> bool:
    left, right = entry["result"], other["result"]
    metrics = [(wdl_loss(entry), wdl_loss(other)),
               (cp_loss(entry), cp_loss(other))]
    if not fixed_time:
        metrics.append((float(left["node_ratio"]),
                        float(right["node_ratio"])))
    return all(b <= a for a, b in metrics) and any(
        b < a for a, b in metrics)


def frontier(entries: list[dict], fixed_time: bool = False) -> list[dict]:
    return [
        entry for entry in entries
        if not any(dominated(entry, other, fixed_time)
                   for other in entries if other is not entry)
    ]


def representative_key(entry: dict) -> tuple:
    result = entry["result"]
    return (
        result.get("critical_mistakes", 0),
        result.get("p95_wdl_loss", 0.0),
        result.get("p95_root_regret", 0),
        -result.get("ranking_move_agreement_pct", 0.0),
    )


def epsilon_frontier(entries: list[dict],
                     fixed_time: bool = False) -> list[dict]:
    cells: dict[tuple[int, ...], list[dict]] = {}
    for entry in frontier(entries, fixed_time):
        result = entry["result"]
        key = (
            round(wdl_loss(entry) / 0.0001),
            round(cp_loss(entry) / 0.5),
        )
        if not fixed_time:
            key += (round(float(result["node_ratio"]) / 0.0025),)
        cells.setdefault(key, []).append(entry)
    representatives = [min(group, key=representative_key)
                       for group in cells.values()]
    return sorted(representatives,
                  key=lambda entry: entry["result"]["node_ratio"])


def unique_entries(entries: list[dict]) -> list[dict]:
    result: dict[Config, dict] = {}
    for entry in entries:
        config = entry["config_obj"]
        current = result.get(config)
        if current is None or representative_key(entry) < representative_key(current):
            result[config] = entry
    return list(result.values())


def preserve_config(sources: list[dict], entries: list[dict],
                    config: Config) -> list[dict]:
    preserved = next(
        (entry for entry in entries if entry["config_obj"] == config), None)
    if preserved is not None:
        sources.append(preserved)
    return unique_entries(sources)


def choose_parent(entries: list[dict], anchors: list[Config],
                  rng: random.Random) -> Config:
    parents = [entry["config_obj"] for entry in epsilon_frontier(entries)]
    roll = rng.random()
    if parents and roll < 0.60:
        return rng.choice(parents)
    if roll < 0.80:
        return FAST
    if roll < 0.90:
        return BALANCED
    return rng.choice(anchors[2:])


def next_candidate(entries: list[dict], anchors: list[Config],
                   seen: set[Config], rng: random.Random) -> tuple[Config, str]:
    for _ in range(500):
        roll = rng.random()
        if roll < 0.10:
            candidate, method = random_config(rng), "restart"
        elif roll < 0.20:
            parents = [entry["config_obj"]
                       for entry in epsilon_frontier(entries)]
            candidate, method = crossover(parents, rng), "crossover"
        else:
            parent = choose_parent(entries, anchors, rng)
            if roll < 0.55:
                mode = "single"
            elif roll < 0.85:
                mode = "pair"
            else:
                mode = "multi"
            candidate, method = mutate(parent, mode, rng), mode
        if candidate not in seen:
            return candidate, method
    while True:
        candidate = random_config(rng)
        if candidate not in seen:
            return candidate, "restart_fallback"


def load_by_category(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if line:
            groups.setdefault(line.split("\t", 1)[0], []).append(line)
    return groups


def fixed_subset(source: Path, total: int, seed: int) -> list[str]:
    groups = load_by_category(source)
    weights = {"random": 70, "balanced": 15, "tactical": 10, "endgame": 5}
    counts = {name: total * weight // 100 for name, weight in weights.items()}
    counts["random"] += total - sum(counts.values())
    rows: list[str] = []
    for category, count in counts.items():
        candidates = groups[category].copy()
        random.Random(seed ^ sum(map(ord, category))).shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(
                f"not enough {category}: need {count}, got {len(candidates)}")
        rows.extend(candidates[:count])
    random.Random(seed).shuffle(rows)
    return rows


def write_subset(path: Path, source: Path, total: int, seed: int) -> None:
    if not path.exists():
        path.write_text("\n".join(fixed_subset(source, total, seed)) + "\n")


def split_holdout(source: Path, main_path: Path, audit_path: Path,
                  audit_size: int, seed: int) -> None:
    if main_path.exists() and audit_path.exists():
        return
    all_rows = source.read_text().splitlines()
    audit = fixed_subset(source, audit_size, seed)
    audit_set = set(audit)
    main = [row for row in all_rows if row not in audit_set]
    if len(main) + len(audit) != len(all_rows):
        raise RuntimeError("holdout split lost or duplicated rows")
    main_path.write_text("\n".join(main) + "\n")
    audit_path.write_text("\n".join(audit) + "\n")


def evaluate(binary: Path, dataset: Path, model: Path, depth: int,
             config: Config, candidate_time_ms: int = 0) -> dict:
    command = [
        str(binary), "--dataset", str(dataset), "--model", str(model),
        "--depth", str(depth), "--ranking-target-abs-cp", "1500",
        "--objective", "wdl", "--include-all-in-objective", *config.args(),
    ]
    if candidate_time_ms:
        command.extend([
            "--candidate-time-ms", str(candidate_time_ms),
            "--candidate-max-depth", "64",
        ])
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "evaluation failed\n"
            f"command: {' '.join(command)}\n"
            f"stderr: {completed.stderr.strip()}")
    result = json.loads(completed.stdout)
    if result.get("ranking_count") != result.get("count"):
        raise RuntimeError("evaluator excluded positions from the objective")
    if int(result.get("candidate_time_ms", 0)) != candidate_time_ms:
        raise RuntimeError("evaluator did not honor candidate time mode")
    return result


def serializable(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key != "config_obj"}


def append_json(path: Path, entry: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(entry, sort_keys=True) + "\n")


def evaluate_stage(kind: str, sources: list[dict], dataset: Path,
                   binary: Path, model: Path, depth: int, log_path: Path,
                   start: float, candidate_time_ms: int = 0) -> list[dict]:
    entries: list[dict] = []
    for rank, source in enumerate(unique_entries(sources)):
        config = source["config_obj"]
        result = evaluate(binary, dataset, model, depth, config,
                          candidate_time_ms)
        entry = {
            "kind": kind, "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config), "config_obj": config,
            "result": result,
        }
        entries.append(entry)
        append_json(log_path, serializable(entry))
        print(
            f"{kind}_progress rank={rank + 1}/{len(unique_entries(sources))}"
            f" nodes={result['node_ratio']:.6f}"
            f" cp={result['mean_root_regret']:.4f}"
            f" wdl={result['mean_wdl_loss']:.9f}"
            f" mean_depth={result.get('candidate_mean_depth', depth):.3f}",
            flush=True,
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=int, default=28_800)
    parser.add_argument("--mutation-duration-sec", type=int, default=10_800)
    parser.add_argument("--max-candidates", type=int, default=180)
    parser.add_argument("--tune-size", type=int, default=1_500)
    parser.add_argument("--rotation-count", type=int, default=4)
    parser.add_argument("--probe-size", type=int, default=600)
    parser.add_argument("--time-selection-size", type=int, default=1_200)
    parser.add_argument("--audit-size", type=int, default=1_000)
    parser.add_argument("--tune-depth", type=int, default=6)
    parser.add_argument("--selection-depth", type=int, default=7)
    parser.add_argument("--audit-depth", type=int, default=8)
    parser.add_argument("--candidate-time-ms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "results.jsonl"
    tune_sets = []
    for index in range(args.rotation_count):
        path = args.run_dir / f"tune_gate_{index}.tsv"
        write_subset(path, args.dataset_dir / "tune.tsv", args.tune_size,
                     args.seed ^ (0x7100 + index))
        tune_sets.append(path)
    probe_set = args.run_dir / "selection_probe.tsv"
    time_selection_set = args.run_dir / "selection_time.tsv"
    holdout_main = args.run_dir / "holdout_main.tsv"
    audit_set = args.run_dir / "audit.tsv"
    write_subset(probe_set, args.dataset_dir / "selection.tsv",
                 args.probe_size, args.seed ^ 0x51EC7)
    write_subset(time_selection_set, args.dataset_dir / "selection.tsv",
                 args.time_selection_size, args.seed ^ 0x71AE)
    split_holdout(args.dataset_dir / "holdout.tsv", holdout_main, audit_set,
                  args.audit_size, args.seed ^ 0xA0D17)

    anchors = initial_configs()
    tune_entries: list[dict] = []
    seen: set[Config] = set()
    rng = random.Random(args.seed)
    start = time.monotonic()
    mutation_deadline = start + args.mutation_duration_sec
    iteration = 0
    pending = [(config, "anchor") for config in anchors]
    while (time.monotonic() < mutation_deadline
           and len(tune_entries) < args.max_candidates):
        if pending:
            config, method = pending.pop(0)
        else:
            config, method = next_candidate(
                tune_entries, anchors, seen, rng)
        if config in seen:
            continue
        seen.add(config)
        gate_index = iteration % len(tune_sets)
        result = evaluate(args.binary, tune_sets[gate_index], args.model,
                          args.tune_depth, config)
        entry = {
            "kind": "tune", "iteration": iteration,
            "mutation_method": method, "tune_gate": gate_index,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config), "config_obj": config,
            "result": result,
        }
        tune_entries.append(entry)
        append_json(log_path, serializable(entry))
        print(
            f"tune_progress iteration={iteration} method={method}"
            f" gate={gate_index} nodes={result['node_ratio']:.6f}"
            f" cp={result['mean_root_regret']:.4f}"
            f" wdl={result['mean_wdl_loss']:.9f}"
            f" frontier={len(epsilon_frontier(tune_entries))}",
            flush=True,
        )
        iteration += 1

    probe_sources = preserve_config(
        epsilon_frontier(tune_entries), tune_entries, FAST)
    probe_entries = evaluate_stage(
        "probe", probe_sources, probe_set, args.binary, args.model,
        args.selection_depth, log_path, start)
    selection_sources = preserve_config(
        epsilon_frontier(probe_entries), probe_entries, FAST)
    selection_entries = evaluate_stage(
        "selection", selection_sources, args.dataset_dir / "selection.tsv",
        args.binary, args.model, args.selection_depth, log_path, start)
    time_sources = preserve_config(
        epsilon_frontier(selection_entries), selection_entries, FAST)
    time_entries = evaluate_stage(
        "time_selection", time_sources, time_selection_set, args.binary,
        args.model, args.selection_depth, log_path, start,
        args.candidate_time_ms)
    holdout_sources = preserve_config(
        epsilon_frontier(time_entries, fixed_time=True), time_entries, FAST)
    holdout_entries = evaluate_stage(
        "time_holdout", unique_entries(holdout_sources), holdout_main,
        args.binary, args.model, args.selection_depth, log_path, start,
        args.candidate_time_ms)

    final_frontier = epsilon_frontier(holdout_entries, fixed_time=True)
    audit_sources = list(final_frontier)
    audit_sources = preserve_config(audit_sources, holdout_entries, FAST)
    audit_entries = evaluate_stage(
        "audit", unique_entries(audit_sources), audit_set, args.binary,
        args.model, args.audit_depth, log_path, start)

    summary = {
        "kind": "complete",
        "objective": "all-position WDL plus CP Pareto",
        "hard_safety": False,
        "fixed_time_selection": True,
        "candidate_time_ms": args.candidate_time_ms,
        "duration_requested_sec": args.duration_sec,
        "elapsed_sec": time.monotonic() - start,
        "seed": args.seed,
        "mutation_policy": {
            "single": 0.35, "cross_block_pair": 0.30,
            "cross_block_multi": 0.15, "crossover": 0.10,
            "random_restart": 0.10,
        },
        "rotating_tune_gates": len(tune_sets),
        "mutations": len(tune_entries),
        "tune_frontier_size": len(epsilon_frontier(tune_entries)),
        "probe_candidates": len(probe_entries),
        "selection_candidates": len(selection_entries),
        "time_selection_candidates": len(time_entries),
        "time_holdout_candidates": len(holdout_entries),
        "final_frontier_size": len(final_frontier),
        "time_holdout": [serializable(entry) for entry in holdout_entries],
        "audit": [serializable(entry) for entry in audit_entries],
        "log": str(log_path),
    }
    append_json(log_path, summary)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
