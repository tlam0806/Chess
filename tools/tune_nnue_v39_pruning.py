#!/usr/bin/env python3
"""Three-lineage Pareto tuner for V39 reverse-futility and late-move pruning."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class BaseProfile:
    lmr_base: float
    lmr_divisor: float
    lmr_min_depth: int
    lmr_min_move_index: int
    null_min_depth: int
    null_reduction: int


PROFILES = {
    "fast": BaseProfile(0.25, 1.95, 3, 8, 3, 2),
    "balanced": BaseProfile(0.45, 2.45, 5, 5, 3, 2),
    "safe": BaseProfile(0.65, 2.9, 7, 3, 3, 2),
}
PROFILE_ORDER = tuple(PROFILES)


@dataclass(frozen=True)
class Config:
    profile: str
    enable_reverse_futility: bool = True
    reverse_futility_max_depth: int = 3
    reverse_futility_base_margin: int = 100
    reverse_futility_margin_per_depth: int = 100
    enable_late_move_pruning: bool = True
    late_move_pruning_max_depth: int = 3
    late_move_pruning_base: int = 4
    late_move_pruning_depth_multiplier: int = 2

    def args(self) -> list[str]:
        base = PROFILES[self.profile]
        result = [
            "--lmr-base", str(base.lmr_base),
            "--lmr-divisor", str(base.lmr_divisor),
            "--lmr-min-depth", str(base.lmr_min_depth),
            "--lmr-min-move-index", str(base.lmr_min_move_index),
            "--null-min-depth", str(base.null_min_depth),
            "--null-reduction", str(base.null_reduction),
            (
                "--enable-reverse-futility"
                if self.enable_reverse_futility
                else "--disable-reverse-futility"
            ),
            "--reverse-futility-max-depth",
            str(self.reverse_futility_max_depth),
            "--reverse-futility-base-margin",
            str(self.reverse_futility_base_margin),
            "--reverse-futility-margin-per-depth",
            str(self.reverse_futility_margin_per_depth),
            (
                "--enable-late-move-pruning"
                if self.enable_late_move_pruning
                else "--disable-late-move-pruning"
            ),
            "--late-move-pruning-max-depth",
            str(self.late_move_pruning_max_depth),
            "--late-move-pruning-base",
            str(self.late_move_pruning_base),
            "--late-move-pruning-depth-multiplier",
            str(self.late_move_pruning_depth_multiplier),
        ]
        return result


def initial_configs() -> list[Config]:
    by_profile: dict[str, list[Config]] = {}
    for profile in PROFILE_ORDER:
        by_profile[profile] = [
            Config(
                profile,
                enable_reverse_futility=False,
                enable_late_move_pruning=False,
            ),
            Config(profile, enable_reverse_futility=False),
            Config(
                profile,
                reverse_futility_max_depth=2,
                reverse_futility_base_margin=175,
                reverse_futility_margin_per_depth=250,
                enable_late_move_pruning=False,
            ),
            Config(
                profile,
                reverse_futility_max_depth=2,
                reverse_futility_base_margin=175,
                reverse_futility_margin_per_depth=250,
                late_move_pruning_max_depth=4,
                late_move_pruning_base=2,
                late_move_pruning_depth_multiplier=3,
            ),
            Config(
                profile,
                reverse_futility_max_depth=1,
                reverse_futility_base_margin=125,
                reverse_futility_margin_per_depth=200,
                late_move_pruning_max_depth=3,
                late_move_pruning_base=4,
                late_move_pruning_depth_multiplier=3,
            ),
            Config(
                profile,
                reverse_futility_max_depth=3,
                reverse_futility_base_margin=350,
                reverse_futility_margin_per_depth=300,
                late_move_pruning_max_depth=3,
                late_move_pruning_base=6,
                late_move_pruning_depth_multiplier=4,
            ),
        ]
    return [
        by_profile[profile][anchor_index]
        for anchor_index in range(6)
        for profile in PROFILE_ORDER
    ]


def valid_result(entry: dict) -> bool:
    result = entry["result"]
    return (
        result.get("critical_mistakes", 0) == 0
        and result.get("safety_critical_mistakes", 0) == 0
        and result.get("self_mated", 0) == 0
    )


def dominated(a: dict, b: dict) -> bool:
    return (
        b["node_ratio"] <= a["node_ratio"]
        and b["mean_root_regret"] <= a["mean_root_regret"]
        and (
            b["node_ratio"] < a["node_ratio"]
            or b["mean_root_regret"] < a["mean_root_regret"]
        )
    )


def frontier(entries: list[dict]) -> list[dict]:
    valid = [entry for entry in entries if valid_result(entry)]
    return [
        entry
        for entry in valid
        if not any(
            dominated(entry["result"], other["result"])
            for other in valid
            if other is not entry
        )
    ]


def lineage_frontier(entries: list[dict], profile: str) -> list[dict]:
    return frontier([
        entry for entry in entries
        if entry["config_obj"].profile == profile
    ])


def safety_key(entry: dict) -> tuple:
    result = entry["result"]
    config = entry["config_obj"]
    return (
        result.get("p95_root_regret", 0),
        result.get("above_100_cp_pct", 0),
        -result.get("ranking_move_agreement_pct", 0),
        config.reverse_futility_max_depth,
        -config.reverse_futility_base_margin,
        -config.reverse_futility_margin_per_depth,
        config.late_move_pruning_max_depth,
        -config.late_move_pruning_base,
        -config.late_move_pruning_depth_multiplier,
    )


def epsilon_parent_pool(entries: list[dict], limit: int = 12) -> list[dict]:
    bins: dict[tuple[int, int], list[dict]] = {}
    for entry in frontier(entries):
        result = entry["result"]
        key = (
            round(result["node_ratio"] / 0.005),
            round(result["mean_root_regret"] / 0.25),
        )
        bins.setdefault(key, []).append(entry)
    representatives = [
        min(group, key=safety_key) for group in bins.values()
    ]
    representatives.sort(key=lambda item: item["result"]["node_ratio"])
    if len(representatives) <= limit:
        return representatives
    indices = {
        round(index * (len(representatives) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [representatives[index] for index in sorted(indices)]


def deduplicate_objectives(entries: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = {}
    for entry in entries:
        result = entry["result"]
        key = (result["node_ratio"], result["mean_root_regret"])
        groups.setdefault(key, []).append(entry)
    return [min(group, key=safety_key) for group in groups.values()]


def clamp(value: int, lower: int, upper: int) -> int:
    return min(upper, max(lower, value))


def mutate(config: Config, rng: random.Random) -> Config:
    values = asdict(config)
    values["enable_reverse_futility"] = True
    values["enable_late_move_pruning"] = True
    parameters = [
        "reverse_futility_max_depth",
        "reverse_futility_base_margin",
        "reverse_futility_margin_per_depth",
        "late_move_pruning_max_depth",
        "late_move_pruning_base",
        "late_move_pruning_depth_multiplier",
    ]
    roll = rng.random()
    changed = 1 if roll < 0.45 else 2 if roll < 0.80 else 3
    for name in rng.sample(parameters, changed):
        original = values[name]
        if name in (
            "reverse_futility_max_depth",
        ):
            candidates = [
                clamp(original + delta, 1, 3) for delta in (-1, 1)
            ]
        elif name == "late_move_pruning_max_depth":
            candidates = [
                clamp(original + delta, 2, 5) for delta in (-1, 1)
            ]
        elif name == "reverse_futility_base_margin":
            lower, upper = (
                (300, 500)
                if values["reverse_futility_max_depth"] >= 3
                else (75, 300)
            )
            candidates = [
                clamp(original + delta, lower, upper)
                for delta in (-50, -25, 25, 50)
            ]
        elif name == "reverse_futility_margin_per_depth":
            lower, upper = (
                (250, 400)
                if values["reverse_futility_max_depth"] >= 3
                else (100, 300)
            )
            candidates = [
                clamp(original + delta, lower, upper)
                for delta in (-50, -25, 25, 50)
            ]
        elif name == "late_move_pruning_base":
            candidates = [
                clamp(original + delta, 0, 12)
                for delta in (-2, -1, 1, 2)
            ]
        else:
            candidates = [
                clamp(original + delta, 0, 5) for delta in (-1, 1)
            ]
        candidates = sorted(set(
            candidate for candidate in candidates if candidate != original
        ))
        if candidates:
            values[name] = rng.choice(candidates)
    if values["reverse_futility_max_depth"] >= 3:
        values["reverse_futility_base_margin"] = clamp(
            values["reverse_futility_base_margin"], 300, 500)
        values["reverse_futility_margin_per_depth"] = clamp(
            values["reverse_futility_margin_per_depth"], 250, 400)
    else:
        values["reverse_futility_base_margin"] = clamp(
            values["reverse_futility_base_margin"], 75, 300)
        values["reverse_futility_margin_per_depth"] = clamp(
            values["reverse_futility_margin_per_depth"], 100, 300)
    return Config(**values)


def random_config(profile: str, rng: random.Random) -> Config:
    experimental_depth3 = rng.random() < 0.20
    return Config(
        profile=profile,
        reverse_futility_max_depth=(
            3 if experimental_depth3 else rng.choice([1, 2])
        ),
        reverse_futility_base_margin=(
            rng.randrange(300, 501, 25)
            if experimental_depth3
            else rng.randrange(75, 301, 25)
        ),
        reverse_futility_margin_per_depth=(
            rng.randrange(250, 401, 25)
            if experimental_depth3
            else rng.randrange(100, 301, 25)
        ),
        late_move_pruning_max_depth=rng.choice([2, 3, 4, 5]),
        late_move_pruning_base=rng.randrange(0, 13),
        late_move_pruning_depth_multiplier=rng.randrange(0, 6),
    )


def pruning_tuple(config: Config, profile: str) -> Config:
    return replace(config, profile=profile)


def load_by_category(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        category = line.split("\t", 1)[0]
        groups.setdefault(category, []).append(line)
    return groups


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
        if len(rows) < count:
            raise RuntimeError(
                f"not enough {category} rows: need {count}, got {len(rows)}"
            )
        selected.extend(rows[:count])
    random.Random(seed).shuffle(selected)
    return selected


def evaluate(
    binary: Path,
    dataset: Path,
    model: Path,
    depth: int,
    config: Config,
    ranking_target_abs_cp: int,
) -> dict:
    command = [
        str(binary),
        "--dataset", str(dataset),
        "--model", str(model),
        "--depth", str(depth),
        "--ranking-target-abs-cp", str(ranking_target_abs_cp),
        *config.args(),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "evaluation failed\n"
            f"command: {' '.join(command)}\n"
            f"stderr: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def regression_gate(
    binary: Path,
    dataset: Path,
    model: Path,
    depths: list[int],
    config: Config,
    ranking_target_abs_cp: int,
) -> tuple[bool, list[dict]]:
    results = [
        {
            "depth": depth,
            "result": evaluate(
                binary,
                dataset,
                model,
                depth,
                config,
                ranking_target_abs_cp,
            ),
        }
        for depth in depths
    ]
    passed = all(
        item["result"].get("critical_mistakes", 0) == 0
        and item["result"].get("safety_critical_mistakes", 0) == 0
        and item["result"].get("self_mated", 0) == 0
        for item in results
    )
    return passed, results


def append_json(path: Path, record: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def record_config(record: dict) -> Config:
    return Config(**record["config"])


def serializable(entry: dict) -> dict:
    return {
        key: value for key, value in entry.items()
        if key != "config_obj"
    }


def load_existing(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text().splitlines():
        record = json.loads(line)
        if record.get("kind") in {
            "regression_reject", "tune", "selection", "holdout"
        }:
            record["config_obj"] = record_config(record)
        records.append(record)
    return records


def next_candidate(
    profile: str,
    entries: list[dict],
    anchors: list[Config],
    seen: set[Config],
    rng: random.Random,
) -> Config:
    same_lineage = lineage_frontier(entries, profile)
    parent_pool = epsilon_parent_pool([
        entry for entry in same_lineage
        if (
            entry["config_obj"].enable_reverse_futility
            and entry["config_obj"].enable_late_move_pruning
        )
    ])
    anchor_pool = [
        anchor for anchor in anchors
        if (
            anchor.profile == profile
            and anchor.enable_reverse_futility
            and anchor.enable_late_move_pruning
        )
    ]
    transfer_pool = [
        entry["config_obj"]
        for other in PROFILE_ORDER
        if other != profile
        for entry in epsilon_parent_pool(lineage_frontier(entries, other))
        if (
            entry["config_obj"].enable_reverse_futility
            and entry["config_obj"].enable_late_move_pruning
        )
    ]

    for _ in range(200):
        source = rng.random()
        if source < 0.10:
            candidate = random_config(profile, rng)
        else:
            if source < 0.65 and parent_pool:
                parent = rng.choice(parent_pool)["config_obj"]
            elif source < 0.85 and anchor_pool:
                parent = rng.choice(anchor_pool)
            elif transfer_pool:
                parent = pruning_tuple(rng.choice(transfer_pool), profile)
            elif parent_pool:
                parent = rng.choice(parent_pool)["config_obj"]
            else:
                parent = Config(profile)
            candidate = mutate(parent, rng)
        if candidate not in seen:
            return candidate
    while True:
        candidate = random_config(profile, rng)
        if candidate not in seen:
            return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=int, default=28_800)
    parser.add_argument("--mutation-duration-sec", type=int, default=14_400)
    parser.add_argument("--max-candidates", type=int, default=180)
    parser.add_argument("--tune-size", type=int, default=2_000)
    parser.add_argument("--tune-depth", type=int, default=6)
    parser.add_argument("--selection-depth", type=int, default=6)
    parser.add_argument("--holdout-depth", type=int, default=7)
    parser.add_argument("--ranking-target-abs-cp", type=int, default=1500)
    parser.add_argument("--regression-dataset", type=Path)
    parser.add_argument(
        "--regression-depths", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--mutation-only", action="store_true")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "results.jsonl"
    tune_gate_path = args.run_dir / "tune_gate.tsv"
    if not tune_gate_path.exists():
        groups = load_by_category(args.dataset_dir / "tune.tsv")
        rows = fixed_subset(groups, args.tune_size, args.seed)
        tune_gate_path.write_text("\n".join(rows) + "\n")

    existing = load_existing(log_path)
    tune_entries = [
        record for record in existing if record.get("kind") == "tune"
    ]
    regression_rejects = [
        record for record in existing
        if record.get("kind") == "regression_reject"
    ]
    seen = {
        entry["config_obj"]
        for entry in tune_entries + regression_rejects
    }
    anchors = initial_configs()
    pending_anchors = [config for config in anchors if config not in seen]
    rng = random.Random(args.seed ^ len(seen))
    start = time.monotonic()
    mutation_deadline = start + args.mutation_duration_sec
    iteration = (
        max(
            (
                entry.get("iteration", -1)
                for entry in tune_entries + regression_rejects
            ),
            default=-1,
        )
        + 1
    )

    while (
        time.monotonic() < mutation_deadline
        and len(tune_entries) < args.max_candidates
    ):
        if pending_anchors:
            config = pending_anchors.pop(0)
        else:
            profile = PROFILE_ORDER[iteration % len(PROFILE_ORDER)]
            config = next_candidate(
                profile, tune_entries, anchors, seen, rng)
        seen.add(config)
        if args.regression_dataset is not None:
            passed, regression_results = regression_gate(
                args.binary,
                args.regression_dataset,
                args.model,
                args.regression_depths,
                config,
                args.ranking_target_abs_cp,
            )
            if not passed:
                failed_depths = [
                    item["depth"]
                    for item in regression_results
                    if item["result"].get("critical_mistakes", 0) > 0
                    or item["result"].get("safety_critical_mistakes", 0) > 0
                    or item["result"].get("self_mated", 0) > 0
                ]
                entry = {
                    "kind": "regression_reject",
                    "iteration": iteration,
                    "elapsed_sec": time.monotonic() - start,
                    "config": asdict(config),
                    "config_obj": config,
                    "failed_depths": failed_depths,
                    "regression_results": regression_results,
                }
                regression_rejects.append(entry)
                append_json(log_path, serializable(entry))
                print(
                    "regression_reject"
                    f" iteration={iteration}"
                    f" profile={config.profile}"
                    f" failed_depths={','.join(map(str, failed_depths))}",
                    flush=True,
                )
                iteration += 1
                continue
        result = evaluate(
            args.binary,
            tune_gate_path,
            args.model,
            args.tune_depth,
            config,
            args.ranking_target_abs_cp,
        )
        entry = {
            "kind": "tune",
            "iteration": iteration,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        tune_entries.append(entry)
        append_json(log_path, serializable(entry))
        sizes = {
            profile: len(lineage_frontier(tune_entries, profile))
            for profile in PROFILE_ORDER
        }
        print(
            "tune_progress"
            f" iteration={iteration}"
            f" profile={config.profile}"
            f" node_ratio={result['node_ratio']:.6f}"
            f" mean_regret={result['mean_root_regret']:.6f}"
            f" critical={result.get('critical_mistakes', 0)}"
            f" rfp_cutoffs={result.get('reverse_futility_cutoffs', 0)}"
            f" lmp_moves={result.get('late_move_pruned_moves', 0)}"
            f" frontier_fast={sizes['fast']}"
            f" frontier_balanced={sizes['balanced']}"
            f" frontier_safe={sizes['safe']}",
            flush=True,
        )
        iteration += 1

    if args.mutation_only:
        summary = {
            "kind": "mutation_complete",
            "elapsed_sec": time.monotonic() - start,
            "mutations": len(tune_entries),
            "regression_rejects": len(regression_rejects),
            "lineage_frontier_sizes": {
                profile: len(lineage_frontier(tune_entries, profile))
                for profile in PROFILE_ORDER
            },
            "log": str(log_path),
        }
        append_json(log_path, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return

    selection_candidates: list[dict] = []
    for profile in PROFILE_ORDER:
        selection_candidates.extend(lineage_frontier(tune_entries, profile))
    unique_candidates = {
        entry["config_obj"]: entry for entry in selection_candidates
    }
    selection_candidates = sorted(
        unique_candidates.values(),
        key=lambda item: (
            PROFILE_ORDER.index(item["config_obj"].profile),
            item["result"]["node_ratio"],
        ),
    )
    existing_selection = [
        record for record in existing if record.get("kind") == "selection"
    ]
    selected = {entry["config_obj"] for entry in existing_selection}
    selection_entries = existing_selection.copy()
    for rank, tune_entry in enumerate(selection_candidates):
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
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        selection_entries.append(entry)
        selected.add(config)
        append_json(log_path, serializable(entry))
        print(
            "selection_progress"
            f" rank={rank + 1}/{len(selection_candidates)}"
            f" profile={config.profile}"
            f" node_ratio={result['node_ratio']:.6f}"
            f" mean_regret={result['mean_root_regret']:.6f}",
            flush=True,
        )

    raw_selection_frontier = frontier(selection_entries)
    final_frontier = deduplicate_objectives(raw_selection_frontier)
    final_frontier.sort(key=lambda item: item["result"]["node_ratio"])
    existing_holdout = [
        record for record in existing if record.get("kind") == "holdout"
    ]
    held_out = {entry["config_obj"] for entry in existing_holdout}
    holdout_entries = existing_holdout.copy()
    for rank, selection_entry in enumerate(final_frontier):
        config = selection_entry["config_obj"]
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
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        holdout_entries.append(entry)
        held_out.add(config)
        append_json(log_path, serializable(entry))
        print(
            "holdout_progress"
            f" rank={rank + 1}/{len(final_frontier)}"
            f" profile={config.profile}"
            f" node_ratio={result['node_ratio']:.6f}"
            f" mean_regret={result['mean_root_regret']:.6f}",
            flush=True,
        )

    summary = {
        "kind": "complete",
        "seed": args.seed,
        "duration_requested_sec": args.duration_sec,
        "elapsed_sec": time.monotonic() - start,
        "mutations": len(tune_entries),
        "regression_rejects": len(regression_rejects),
        "lineage_frontier_sizes": {
            profile: len(lineage_frontier(tune_entries, profile))
            for profile in PROFILE_ORDER
        },
        "selection_candidates": len(selection_candidates),
        "selection_frontier_size_raw": len(raw_selection_frontier),
        "selection_frontier_size_deduplicated": len(final_frontier),
        "holdout": [serializable(entry) for entry in holdout_entries],
        "log": str(log_path),
    }
    append_json(log_path, summary)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
