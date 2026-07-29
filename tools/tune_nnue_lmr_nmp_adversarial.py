#!/usr/bin/env python3
"""Adversarial multi-stage Pareto tuner for V38 LMR and null-move pruning."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    enable_lmr: bool = True
    lmr_base: float = 0.5
    lmr_divisor: float = 2.6
    lmr_min_depth: int = 4
    lmr_min_move_index: int = 4
    enable_null_move: bool = True
    null_min_depth: int = 3
    null_reduction: int = 2

    @property
    def lineage(self) -> str:
        if self.enable_lmr and self.enable_null_move:
            return "joint"
        if self.enable_lmr:
            return "lmr"
        if self.enable_null_move:
            return "nmp"
        return "control"

    def args(self) -> list[str]:
        result = [
            "--lmr-base", str(self.lmr_base),
            "--lmr-divisor", str(self.lmr_divisor),
            "--lmr-min-depth", str(self.lmr_min_depth),
            "--lmr-min-move-index", str(self.lmr_min_move_index),
            "--null-min-depth", str(self.null_min_depth),
            "--null-reduction", str(self.null_reduction),
            "--disable-reverse-futility",
            "--disable-late-move-pruning",
        ]
        if not self.enable_lmr:
            result.append("--disable-lmr")
        if not self.enable_null_move:
            result.append("--disable-null-move")
        return result


LINEAGES = ("lmr", "nmp", "joint")
MUTATION_SCHEDULE = ("joint", "lmr", "joint", "nmp")


def initial_configs() -> list[Config]:
    control = Config(enable_lmr=False, enable_null_move=False)
    lmr = [
        Config(True, 0.25, 1.95, 3, 8, False, 3, 2),
        Config(True, 0.45, 2.45, 5, 5, False, 3, 2),
        Config(True, 0.65, 2.9, 7, 3, False, 3, 2),
    ]
    nmp = [
        Config(False, 0.5, 2.6, 4, 4, True, 3, 1),
        Config(False, 0.5, 2.6, 4, 4, True, 4, 2),
        Config(False, 0.5, 2.6, 4, 4, True, 5, 2),
        Config(False, 0.5, 2.6, 4, 4, True, 3, 3),
    ]
    joint = [
        replace(config, enable_null_move=True, null_min_depth=3,
                null_reduction=2)
        for config in lmr
    ]
    joint.extend([
        Config(True, 0.45, 2.45, 5, 5, True, 4, 1),
        Config(True, 0.55, 2.75, 4, 4, True, 5, 2),
    ])
    result = [control]
    for index in range(max(len(lmr), len(nmp), len(joint))):
        for group in (lmr, nmp, joint):
            if index < len(group):
                result.append(group[index])
    return result


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
        entry for entry in valid
        if not any(
            dominated(entry["result"], other["result"])
            for other in valid if other is not entry
        )
    ]


def lineage_frontier(entries: list[dict], lineage: str) -> list[dict]:
    return frontier([
        entry for entry in entries
        if entry["config_obj"].lineage == lineage
    ])


def safety_key(entry: dict) -> tuple:
    result = entry["result"]
    return (
        result.get("p95_root_regret", 0),
        result.get("above_100_cp_pct", 0),
        -result.get("ranking_move_agreement_pct", 0),
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


def thin_frontier(entries: list[dict], limit: int) -> list[dict]:
    ordered = sorted(
        deduplicate_objectives(entries),
        key=lambda entry: entry["result"]["node_ratio"])
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    indices = {
        round(index * (len(ordered) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [ordered[index] for index in sorted(indices)]


def clamp(value: int, lower: int, upper: int) -> int:
    return min(upper, max(lower, value))


def parameters_for(lineage: str) -> list[str]:
    lmr = [
        "lmr_base", "lmr_divisor",
        "lmr_min_depth", "lmr_min_move_index",
    ]
    nmp = ["null_min_depth", "null_reduction"]
    return lmr if lineage == "lmr" else nmp if lineage == "nmp" else lmr + nmp


def force_lineage(config: Config, lineage: str) -> Config:
    return replace(
        config,
        enable_lmr=lineage in {"lmr", "joint"},
        enable_null_move=lineage in {"nmp", "joint"},
    )


def mutate(
    config: Config, lineage: str, rng: random.Random
) -> Config:
    config = force_lineage(config, lineage)
    values = asdict(config)
    parameters = parameters_for(lineage)
    roll = rng.random()
    changed = min(
        len(parameters),
        1 if roll < 0.45 else 2 if roll < 0.80 else 3,
    )
    for name in rng.sample(parameters, changed):
        original = values[name]
        if name == "lmr_base":
            candidates = [
                round(min(1.0, max(0.0, original + delta)), 2)
                for delta in (-0.1, -0.05, 0.05, 0.1)
            ]
        elif name == "lmr_divisor":
            candidates = [
                round(min(3.5, max(1.8, original + delta)), 2)
                for delta in (-0.2, -0.1, 0.1, 0.2)
            ]
        elif name == "lmr_min_depth":
            candidates = [
                clamp(original + delta, 3, 8) for delta in (-1, 1)]
        elif name == "lmr_min_move_index":
            candidates = [
                clamp(original + delta, 2, 12)
                for delta in (-2, -1, 1, 2)
            ]
        elif name == "null_min_depth":
            candidates = [
                clamp(original + delta, 3, 7) for delta in (-1, 1)]
        else:
            candidates = [
                clamp(original + delta, 1, 3) for delta in (-1, 1)]
        choices = sorted({
            value for value in candidates if value != original
        })
        if choices:
            values[name] = rng.choice(choices)
    return Config(**values)


def random_config(lineage: str, rng: random.Random) -> Config:
    return Config(
        enable_lmr=lineage in {"lmr", "joint"},
        lmr_base=round(rng.randrange(0, 101, 5) / 100, 2),
        lmr_divisor=round(rng.randrange(180, 351, 5) / 100, 2),
        lmr_min_depth=rng.randrange(3, 9),
        lmr_min_move_index=rng.randrange(2, 13),
        enable_null_move=lineage in {"nmp", "joint"},
        null_min_depth=rng.randrange(3, 8),
        null_reduction=rng.randrange(1, 4),
    )


def combine_joint(
    lmr_entry: dict, nmp_entry: dict
) -> Config:
    lmr = lmr_entry["config_obj"]
    nmp = nmp_entry["config_obj"]
    return Config(
        True,
        lmr.lmr_base,
        lmr.lmr_divisor,
        lmr.lmr_min_depth,
        lmr.lmr_min_move_index,
        True,
        nmp.null_min_depth,
        nmp.null_reduction,
    )


def next_candidate(
    lineage: str,
    entries: list[dict],
    anchors: list[Config],
    seen: set[Config],
    rng: random.Random,
) -> Config:
    same_pool = epsilon_parent_pool(lineage_frontier(entries, lineage))
    anchor_pool = [
        config for config in anchors if config.lineage == lineage
    ]
    lmr_pool = epsilon_parent_pool(lineage_frontier(entries, "lmr"))
    nmp_pool = epsilon_parent_pool(lineage_frontier(entries, "nmp"))
    for _ in range(300):
        roll = rng.random()
        if roll < 0.15:
            candidate = random_config(lineage, rng)
        elif (
            lineage == "joint"
            and roll < 0.35
            and lmr_pool
            and nmp_pool
        ):
            candidate = mutate(
                combine_joint(
                    rng.choice(lmr_pool), rng.choice(nmp_pool)),
                lineage,
                rng,
            )
        else:
            if same_pool and roll < 0.75:
                parent = rng.choice(same_pool)["config_obj"]
            elif anchor_pool:
                parent = rng.choice(anchor_pool)
            else:
                parent = random_config(lineage, rng)
            candidate = mutate(parent, lineage, rng)
        if candidate not in seen:
            return candidate
    while True:
        candidate = random_config(lineage, rng)
        if candidate not in seen:
            return candidate


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
        rows = groups.get(category, []).copy()
        random.Random(seed ^ sum(map(ord, category))).shuffle(rows)
        if len(rows) < count:
            raise RuntimeError(
                f"not enough {category} rows: need {count}, got {len(rows)}")
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
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(
            "evaluation failed\n"
            f"command: {' '.join(command)}\n"
            f"stderr: {process.stderr.strip()}")
    return json.loads(process.stdout)


def evaluate_depths(
    binary: Path,
    dataset: Path,
    model: Path,
    depths: list[int],
    config: Config,
    ranking_target_abs_cp: int,
) -> list[dict]:
    return [
        {
            "depth": depth,
            "result": evaluate(
                binary, dataset, model, depth, config,
                ranking_target_abs_cp),
        }
        for depth in depths
    ]


def depths_are_safe(results: list[dict]) -> bool:
    return all(
        item["result"].get("critical_mistakes", 0) == 0
        and item["result"].get("safety_critical_mistakes", 0) == 0
        and item["result"].get("self_mated", 0) == 0
        for item in results
    )


def append_json(path: Path, record: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def serializable(entry: dict) -> dict:
    return {
        key: value for key, value in entry.items()
        if key != "config_obj"
    }


def record_config(record: dict) -> Config:
    return Config(**record["config"])


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    config_kinds = {
        "regression_reject", "tune", "adversarial_selection",
        "selection", "holdout7", "holdout8",
    }
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("kind") in config_kinds:
            record["config_obj"] = record_config(record)
        records.append(record)
    return records


def dataset_has_rows(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--safety-bank-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=int, default=28_800)
    parser.add_argument("--mutation-duration-sec", type=int, default=10_800)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--tune-size", type=int, default=2_000)
    parser.add_argument("--tune-depth", type=int, default=6)
    parser.add_argument(
        "--regression-depths", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--adversarial-depth", type=int, default=7)
    parser.add_argument("--selection-depth", type=int, default=7)
    parser.add_argument("--holdout-depth", type=int, default=7)
    parser.add_argument("--final-depth", type=int, default=8)
    parser.add_argument("--final-depth-cap", type=int, default=8)
    parser.add_argument("--ranking-target-abs-cp", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--mutation-only", action="store_true")
    args = parser.parse_args()

    core_path = args.safety_bank_dir / "core.tsv"
    sealed_path = args.safety_bank_dir / "sealed.tsv"
    if not dataset_has_rows(core_path):
        raise RuntimeError(f"empty safety core: {core_path}")
    if not dataset_has_rows(sealed_path):
        raise RuntimeError(f"empty sealed safety bank: {sealed_path}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "results.jsonl"
    tune_gate_path = args.run_dir / "tune_gate.tsv"
    if not tune_gate_path.exists():
        groups = load_by_category(args.dataset_dir / "tune.tsv")
        rows = fixed_subset(groups, args.tune_size, args.seed)
        tune_gate_path.write_text("\n".join(rows) + "\n")

    existing = load_existing(log_path)
    tune_entries = [
        entry for entry in existing if entry.get("kind") == "tune"]
    rejects = [
        entry for entry in existing
        if entry.get("kind") == "regression_reject"
    ]
    seen = {
        entry["config_obj"] for entry in tune_entries + rejects
    }
    anchors = initial_configs()
    pending_anchors = [config for config in anchors if config not in seen]
    attempted = tune_entries + rejects
    iteration = max(
        (entry.get("iteration", -1) for entry in attempted),
        default=-1,
    ) + 1
    rng = random.Random(args.seed ^ len(seen))
    start = time.monotonic()
    downstream_kinds = {
        "adversarial_selection", "selection", "holdout7", "holdout8",
        "complete",
    }
    downstream_started = any(
        entry.get("kind") in downstream_kinds for entry in existing)
    mutation_deadline = (
        start if downstream_started
        else start + args.mutation_duration_sec
    )

    while (
        time.monotonic() < mutation_deadline
        and len(tune_entries) < args.max_candidates
    ):
        if pending_anchors:
            config = pending_anchors.pop(0)
        else:
            lineage = MUTATION_SCHEDULE[
                iteration % len(MUTATION_SCHEDULE)]
            config = next_candidate(
                lineage, tune_entries, anchors, seen, rng)
        seen.add(config)
        regression_results = evaluate_depths(
            args.binary,
            core_path,
            args.model,
            args.regression_depths,
            config,
            args.ranking_target_abs_cp,
        )
        if not depths_are_safe(regression_results):
            failed_depths = [
                item["depth"] for item in regression_results
                if not depths_are_safe([item])
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
            rejects.append(entry)
            append_json(log_path, serializable(entry))
            print(
                "regression_reject"
                f" iteration={iteration}"
                f" lineage={config.lineage}"
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
            lineage: len(lineage_frontier(tune_entries, lineage))
            for lineage in LINEAGES
        }
        print(
            "tune_progress"
            f" iteration={iteration}"
            f" lineage={config.lineage}"
            f" node_ratio={result['node_ratio']:.6f}"
            f" mean_regret={result['mean_root_regret']:.6f}"
            f" critical={result.get('critical_mistakes', 0)}"
            f" frontier_lmr={sizes['lmr']}"
            f" frontier_nmp={sizes['nmp']}"
            f" frontier_joint={sizes['joint']}",
            flush=True,
        )
        iteration += 1

    if args.mutation_only:
        summary = {
            "kind": "mutation_complete",
            "elapsed_sec": time.monotonic() - start,
            "mutations": len(tune_entries),
            "regression_rejects": len(rejects),
            "lineage_frontier_sizes": {
                lineage: len(lineage_frontier(tune_entries, lineage))
                for lineage in LINEAGES
            },
        }
        append_json(log_path, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return

    candidate_pool: list[dict] = []
    controls = [
        entry for entry in tune_entries
        if entry["config_obj"].lineage == "control"
    ]
    candidate_pool.extend(controls)
    for lineage in LINEAGES:
        candidate_pool.extend(deduplicate_objectives(
            lineage_frontier(tune_entries, lineage)))
    candidate_pool = list({
        entry["config_obj"]: entry for entry in candidate_pool
    }.values())
    candidate_pool.sort(key=lambda entry: (
        ("control", *LINEAGES).index(entry["config_obj"].lineage),
        entry["result"]["node_ratio"],
    ))

    adversarial_entries = [
        entry for entry in existing
        if entry.get("kind") == "adversarial_selection"
    ]
    adversarial_done = {
        entry["config_obj"] for entry in adversarial_entries
    }
    for rank, tune_entry in enumerate(candidate_pool):
        config = tune_entry["config_obj"]
        if config in adversarial_done:
            continue
        result = evaluate(
            args.binary,
            sealed_path,
            args.model,
            args.adversarial_depth,
            config,
            args.ranking_target_abs_cp,
        )
        entry = {
            "kind": "adversarial_selection",
            "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        adversarial_entries.append(entry)
        adversarial_done.add(config)
        append_json(log_path, serializable(entry))
        print(
            "adversarial_progress"
            f" rank={rank + 1}/{len(candidate_pool)}"
            f" lineage={config.lineage}"
            f" critical={result.get('critical_mistakes', 0)}",
            flush=True,
        )

    adversarial_safe = {
        entry["config_obj"]
        for entry in adversarial_entries if valid_result(entry)
    }
    normal_candidates = [
        entry for entry in candidate_pool
        if entry["config_obj"] in adversarial_safe
    ]
    selection_entries = [
        entry for entry in existing if entry.get("kind") == "selection"
    ]
    selection_done = {
        entry["config_obj"] for entry in selection_entries
    }
    for rank, tune_entry in enumerate(normal_candidates):
        config = tune_entry["config_obj"]
        if config in selection_done:
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
        selection_done.add(config)
        append_json(log_path, serializable(entry))
        print(
            "selection_progress"
            f" rank={rank + 1}/{len(normal_candidates)}"
            f" lineage={config.lineage}"
            f" node_ratio={result['node_ratio']:.6f}"
            f" mean_regret={result['mean_root_regret']:.6f}"
            f" critical={result.get('critical_mistakes', 0)}",
            flush=True,
        )

    selection_frontier = deduplicate_objectives(frontier(selection_entries))
    selection_frontier.sort(
        key=lambda entry: entry["result"]["node_ratio"])
    holdout7_entries = [
        entry for entry in existing if entry.get("kind") == "holdout7"
    ]
    holdout7_done = {
        entry["config_obj"] for entry in holdout7_entries
    }
    for rank, selection_entry in enumerate(selection_frontier):
        config = selection_entry["config_obj"]
        if config in holdout7_done:
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
            "kind": "holdout7",
            "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        holdout7_entries.append(entry)
        holdout7_done.add(config)
        append_json(log_path, serializable(entry))
        print(
            "holdout7_progress"
            f" rank={rank + 1}/{len(selection_frontier)}"
            f" lineage={config.lineage}"
            f" node_ratio={result['node_ratio']:.6f}"
            f" mean_regret={result['mean_root_regret']:.6f}"
            f" critical={result.get('critical_mistakes', 0)}",
            flush=True,
        )

    final_candidates = thin_frontier(
        frontier(holdout7_entries), args.final_depth_cap)
    holdout8_entries = [
        entry for entry in existing if entry.get("kind") == "holdout8"
    ]
    holdout8_done = {
        entry["config_obj"] for entry in holdout8_entries
    }
    for rank, holdout7_entry in enumerate(final_candidates):
        config = holdout7_entry["config_obj"]
        if config in holdout8_done:
            continue
        result = evaluate(
            args.binary,
            args.dataset_dir / "holdout.tsv",
            args.model,
            args.final_depth,
            config,
            args.ranking_target_abs_cp,
        )
        entry = {
            "kind": "holdout8",
            "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        holdout8_entries.append(entry)
        holdout8_done.add(config)
        append_json(log_path, serializable(entry))
        print(
            "holdout8_progress"
            f" rank={rank + 1}/{len(final_candidates)}"
            f" lineage={config.lineage}"
            f" node_ratio={result['node_ratio']:.6f}"
            f" mean_regret={result['mean_root_regret']:.6f}"
            f" critical={result.get('critical_mistakes', 0)}",
            flush=True,
        )

    summary = {
        "kind": "complete",
        "seed": args.seed,
        "duration_requested_sec": args.duration_sec,
        "elapsed_sec": time.monotonic() - start,
        "mutations": len(tune_entries),
        "regression_rejects": len(rejects),
        "lineage_frontier_sizes": {
            lineage: len(lineage_frontier(tune_entries, lineage))
            for lineage in LINEAGES
        },
        "adversarial_candidates": len(candidate_pool),
        "adversarial_survivors": len(normal_candidates),
        "selection_frontier": len(selection_frontier),
        "holdout7_frontier": len(frontier(holdout7_entries)),
        "holdout8": [
            serializable(entry) for entry in holdout8_entries
        ],
        "log": str(log_path),
    }
    append_json(log_path, summary)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
