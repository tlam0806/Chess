#!/usr/bin/env python3
"""Pareto-tune V39 RFP/LMP on the fixed V38 config-7 baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


BASELINE_ARGS = (
    "--lmr-base", "0.45",
    "--lmr-divisor", "2.9",
    "--lmr-min-depth", "3",
    "--lmr-min-move-index", "6",
    "--null-min-depth", "2",
    "--null-reduction", "3",
)
LINEAGES = ("rfp", "lmp", "joint")


@dataclass(frozen=True)
class Config:
    lineage: str
    reverse_futility_max_depth: int = 1
    reverse_futility_base_margin: int = 250
    reverse_futility_margin_per_depth: int = 250
    late_move_pruning_max_depth: int = 2
    late_move_pruning_base: int = 8
    late_move_pruning_depth_multiplier: int = 4

    @property
    def enable_reverse_futility(self) -> bool:
        return self.lineage in ("rfp", "joint")

    @property
    def enable_late_move_pruning(self) -> bool:
        return self.lineage in ("lmp", "joint")

    def args(self) -> list[str]:
        return [
            *BASELINE_ARGS,
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


def initial_configs() -> list[Config]:
    return [
        Config("off"),
        Config("rfp", 1, 150, 150),
        Config("rfp", 1, 250, 250),
        Config("rfp", 2, 250, 250),
        Config("rfp", 2, 350, 350),
        Config("rfp", 3, 350, 300),
        Config("rfp", 3, 450, 400),
        Config("lmp", late_move_pruning_max_depth=2,
               late_move_pruning_base=4,
               late_move_pruning_depth_multiplier=2),
        Config("lmp", late_move_pruning_max_depth=2,
               late_move_pruning_base=8,
               late_move_pruning_depth_multiplier=4),
        Config("lmp", late_move_pruning_max_depth=3,
               late_move_pruning_base=10,
               late_move_pruning_depth_multiplier=5),
        Config("lmp", late_move_pruning_max_depth=4,
               late_move_pruning_base=12,
               late_move_pruning_depth_multiplier=6),
        Config("lmp", late_move_pruning_max_depth=3,
               late_move_pruning_base=16,
               late_move_pruning_depth_multiplier=8),
        Config("joint", 1, 250, 250, 2, 8, 4),
        Config("joint", 2, 300, 300, 3, 10, 5),
        Config("joint", 3, 400, 350, 4, 12, 6),
    ]


def objective_loss(entry: dict) -> float:
    return float(entry["result"]["objective_loss"])


def dominated(a: dict, b: dict) -> bool:
    ar, br = a["result"], b["result"]
    return (
        br["node_ratio"] <= ar["node_ratio"]
        and objective_loss(b) <= objective_loss(a)
        and (
            br["node_ratio"] < ar["node_ratio"]
            or objective_loss(b) < objective_loss(a)
        )
    )


def frontier(entries: list[dict]) -> list[dict]:
    """Keep critical positions in the WDL objective; never hard-reject them."""
    return [
        entry for entry in entries
        if not any(
            dominated(entry, other)
            for other in entries
            if other is not entry
        )
    ]


def lineage_frontier(entries: list[dict], lineage: str) -> list[dict]:
    return frontier([
        entry for entry in entries
        if entry["config_obj"].lineage == lineage
    ])


def representative_key(entry: dict) -> tuple:
    result = entry["result"]
    return (
        result.get("p95_wdl_loss", 0.0),
        result.get("critical_mistakes", 0),
        result.get("p95_root_regret", 0),
        -result.get("ranking_move_agreement_pct", 0.0),
    )


def deduplicate_objectives(entries: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = {}
    for entry in entries:
        key = (entry["result"]["node_ratio"], objective_loss(entry))
        groups.setdefault(key, []).append(entry)
    return [min(group, key=representative_key) for group in groups.values()]


def deduplicate_configs(entries: list[dict]) -> list[dict]:
    unique: dict[Config, dict] = {}
    for entry in entries:
        unique.setdefault(entry["config_obj"], entry)
    return list(unique.values())


def epsilon_frontier(entries: list[dict]) -> list[dict]:
    """Compress only differences below the resolution of this noisy metric."""
    groups: dict[tuple[int, int], list[dict]] = {}
    for entry in deduplicate_objectives(frontier(entries)):
        result = entry["result"]
        key = (
            round(float(result["node_ratio"]) / 0.0025),
            round(objective_loss(entry) / 0.0001),
        )
        groups.setdefault(key, []).append(entry)
    representatives = [min(group, key=representative_key)
                       for group in groups.values()]
    return sorted(representatives,
                  key=lambda entry: entry["result"]["node_ratio"])


def clamp(value: int, lower: int, upper: int) -> int:
    return min(upper, max(lower, value))


def canonical(config: Config) -> Config:
    values = asdict(config)
    lineage = values["lineage"]
    if lineage == "off":
        return Config("off")
    if lineage == "lmp":
        values["reverse_futility_max_depth"] = 1
        values["reverse_futility_base_margin"] = 250
        values["reverse_futility_margin_per_depth"] = 250
    if lineage == "rfp":
        values["late_move_pruning_max_depth"] = 2
        values["late_move_pruning_base"] = 8
        values["late_move_pruning_depth_multiplier"] = 4
    if values["reverse_futility_max_depth"] >= 3:
        values["reverse_futility_base_margin"] = clamp(
            values["reverse_futility_base_margin"], 300, 500)
        values["reverse_futility_margin_per_depth"] = clamp(
            values["reverse_futility_margin_per_depth"], 250, 400)
    else:
        values["reverse_futility_base_margin"] = clamp(
            values["reverse_futility_base_margin"], 150, 500)
        values["reverse_futility_margin_per_depth"] = clamp(
            values["reverse_futility_margin_per_depth"], 150, 400)
    if values["late_move_pruning_max_depth"] >= 4:
        values["late_move_pruning_base"] = clamp(
            values["late_move_pruning_base"], 8, 16)
        values["late_move_pruning_depth_multiplier"] = clamp(
            values["late_move_pruning_depth_multiplier"], 4, 8)
    else:
        values["late_move_pruning_base"] = clamp(
            values["late_move_pruning_base"], 4, 16)
        values["late_move_pruning_depth_multiplier"] = clamp(
            values["late_move_pruning_depth_multiplier"], 2, 8)
    return Config(**values)


def relevant_parameters(lineage: str) -> list[str]:
    rfp = [
        "reverse_futility_max_depth",
        "reverse_futility_base_margin",
        "reverse_futility_margin_per_depth",
    ]
    lmp = [
        "late_move_pruning_max_depth",
        "late_move_pruning_base",
        "late_move_pruning_depth_multiplier",
    ]
    return rfp if lineage == "rfp" else lmp if lineage == "lmp" else rfp + lmp


def mutate(config: Config, rng: random.Random) -> Config:
    values = asdict(config)
    parameters = relevant_parameters(config.lineage)
    roll = rng.random()
    changed = 1 if roll < 0.40 else 2 if roll < 0.80 else 3
    changed = min(changed, len(parameters))
    for name in rng.sample(parameters, changed):
        original = int(values[name])
        if name == "reverse_futility_max_depth":
            candidates = [clamp(original + delta, 1, 3) for delta in (-1, 1)]
        elif name in (
            "reverse_futility_base_margin",
            "reverse_futility_margin_per_depth",
        ):
            upper = 500 if name == "reverse_futility_base_margin" else 400
            candidates = [
                clamp(original + delta, 150, upper)
                for delta in (-50, -25, 25, 50)
            ]
        elif name == "late_move_pruning_max_depth":
            candidates = [clamp(original + delta, 2, 4) for delta in (-1, 1)]
        elif name == "late_move_pruning_base":
            candidates = [
                clamp(original + delta, 4, 16)
                for delta in (-2, -1, 1, 2)
            ]
        else:
            candidates = [
                clamp(original + delta, 2, 8)
                for delta in (-2, -1, 1, 2)
            ]
        candidates = sorted(set(value for value in candidates
                                if value != original))
        if candidates:
            values[name] = rng.choice(candidates)
    return canonical(Config(**values))


def random_config(lineage: str, rng: random.Random) -> Config:
    rfp_depth = rng.choices([1, 2, 3], weights=[4, 4, 1])[0]
    lmp_depth = rng.choices([2, 3, 4], weights=[4, 4, 1])[0]
    result = Config(
        lineage=lineage,
        reverse_futility_max_depth=rfp_depth,
        reverse_futility_base_margin=(
            rng.randrange(300, 501, 25)
            if rfp_depth == 3 else rng.randrange(150, 501, 25)
        ),
        reverse_futility_margin_per_depth=(
            rng.randrange(250, 401, 25)
            if rfp_depth == 3 else rng.randrange(150, 401, 25)
        ),
        late_move_pruning_max_depth=lmp_depth,
        late_move_pruning_base=(
            rng.randrange(8, 17) if lmp_depth == 4 else rng.randrange(4, 17)
        ),
        late_move_pruning_depth_multiplier=(
            rng.randrange(4, 9) if lmp_depth == 4 else rng.randrange(2, 9)
        ),
    )
    return canonical(result)


def crossover(entries: list[dict], rng: random.Random) -> Config | None:
    rfp = epsilon_frontier([
        entry for entry in entries
        if entry["config_obj"].lineage == "rfp"
    ])
    lmp = epsilon_frontier([
        entry for entry in entries
        if entry["config_obj"].lineage == "lmp"
    ])
    if not rfp or not lmp:
        return None
    r = rng.choice(rfp)["config_obj"]
    l = rng.choice(lmp)["config_obj"]
    return Config(
        "joint",
        r.reverse_futility_max_depth,
        r.reverse_futility_base_margin,
        r.reverse_futility_margin_per_depth,
        l.late_move_pruning_max_depth,
        l.late_move_pruning_base,
        l.late_move_pruning_depth_multiplier,
    )


def next_candidate(
    lineage: str,
    entries: list[dict],
    anchors: list[Config],
    seen: set[Config],
    rng: random.Random,
) -> Config:
    parents = epsilon_frontier([
        entry for entry in entries
        if entry["config_obj"].lineage == lineage
    ])
    lineage_anchors = [anchor for anchor in anchors
                       if anchor.lineage == lineage]
    for _ in range(300):
        roll = rng.random()
        if lineage == "joint" and roll < 0.15:
            candidate = crossover(entries, rng)
            if candidate is None:
                candidate = random_config(lineage, rng)
        elif roll < 0.35:
            candidate = random_config(lineage, rng)
        else:
            if parents and roll < 0.80:
                parent = rng.choice(parents)["config_obj"]
            elif lineage_anchors:
                parent = rng.choice(lineage_anchors)
            else:
                parent = random_config(lineage, rng)
            candidate = mutate(parent, rng)
        if candidate not in seen:
            return candidate
    while True:
        candidate = random_config(lineage, rng)
        if candidate not in seen:
            return candidate


def load_by_category(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if line:
            groups.setdefault(line.split("\t", 1)[0], []).append(line)
    return groups


def fixed_subset(groups: dict[str, list[str]], total: int, seed: int) -> list[str]:
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
    if path.exists():
        return
    rows = fixed_subset(load_by_category(source), total, seed)
    path.write_text("\n".join(rows) + "\n")


def split_holdout(
    source: Path, main_path: Path, audit_path: Path, audit_size: int, seed: int,
) -> None:
    if main_path.exists() and audit_path.exists():
        return
    all_rows = source.read_text().splitlines()
    audit = fixed_subset(load_by_category(source), audit_size, seed)
    audit_set = set(audit)
    main = [row for row in all_rows if row not in audit_set]
    if len(main) + len(audit) != len(all_rows):
        raise RuntimeError("holdout split lost or duplicated rows")
    main_path.write_text("\n".join(main) + "\n")
    audit_path.write_text("\n".join(audit) + "\n")


def evaluate(
    binary: Path, dataset: Path, model: Path, depth: int, config: Config,
) -> dict:
    command = [
        str(binary),
        "--dataset", str(dataset),
        "--model", str(model),
        "--depth", str(depth),
        "--ranking-target-abs-cp", "1500",
        "--objective", "wdl",
        "--include-all-in-objective",
        *config.args(),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "evaluation failed\n"
            f"command: {' '.join(command)}\n"
            f"stderr: {completed.stderr.strip()}")
    result = json.loads(completed.stdout)
    if result.get("objective") != "wdl" or not result.get(
        "include_all_in_objective", False
    ):
        raise RuntimeError("evaluator did not use all-position WDL objective")
    if result.get("ranking_count") != result.get("count"):
        raise RuntimeError("some positions were excluded from WDL objective")
    return result


def serializable(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key != "config_obj"}


def append_json(path: Path, record: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "config" in record:
            record["config_obj"] = Config(**record["config"])
        records.append(record)
    return records


def stage_candidates(entries: list[dict], include_lineages: bool) -> list[dict]:
    candidates = epsilon_frontier(entries)
    if include_lineages:
        for lineage in LINEAGES:
            candidates.extend(epsilon_frontier([
                entry for entry in entries
                if entry["config_obj"].lineage == lineage
            ]))
    baseline = next(
        (entry for entry in entries if entry["config_obj"].lineage == "off"),
        None,
    )
    if baseline is not None:
        candidates.append(baseline)
    return deduplicate_configs(candidates)


def knee(entries: list[dict]) -> dict:
    ordered = sorted(frontier(entries),
                     key=lambda entry: entry["result"]["node_ratio"])
    if len(ordered) <= 2:
        return ordered[len(ordered) // 2]
    x_min = float(ordered[0]["result"]["node_ratio"])
    x_max = float(ordered[-1]["result"]["node_ratio"])
    y_min = objective_loss(ordered[-1])
    y_max = objective_loss(ordered[0])
    if x_max == x_min or y_max == y_min:
        return ordered[len(ordered) // 2]
    return max(
        ordered,
        key=lambda entry: (
            1.0
            - (float(entry["result"]["node_ratio"]) - x_min) / (x_max - x_min)
            - (objective_loss(entry) - y_min) / (y_max - y_min)
        ) / math.sqrt(2.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=int, default=28_800)
    parser.add_argument("--mutation-duration-sec", type=int, default=12_600)
    parser.add_argument("--max-candidates", type=int, default=180)
    parser.add_argument("--tune-size", type=int, default=2_000)
    parser.add_argument("--selection-probe-size", type=int, default=500)
    parser.add_argument("--audit-size", type=int, default=1_000)
    parser.add_argument("--tune-depth", type=int, default=6)
    parser.add_argument("--selection-depth", type=int, default=7)
    parser.add_argument("--holdout-depth", type=int, default=7)
    parser.add_argument("--audit-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "results.jsonl"
    tune_gate = args.run_dir / "tune_gate.tsv"
    selection_probe = args.run_dir / "selection_probe.tsv"
    holdout_main = args.run_dir / "holdout_main.tsv"
    audit_set = args.run_dir / "audit.tsv"
    write_subset(
        tune_gate, args.dataset_dir / "tune.tsv", args.tune_size, args.seed)
    write_subset(
        selection_probe, args.dataset_dir / "selection.tsv",
        args.selection_probe_size, args.seed ^ 0x51EC7)
    split_holdout(
        args.dataset_dir / "holdout.tsv", holdout_main, audit_set,
        args.audit_size, args.seed ^ 0xA0D17)

    existing = load_existing(log_path)
    tune_entries = [record for record in existing if record.get("kind") == "tune"]
    anchors = initial_configs()
    seen = {entry["config_obj"] for entry in tune_entries}
    pending_anchors = [config for config in anchors if config not in seen]
    rng = random.Random(args.seed ^ len(seen))
    start = time.monotonic()
    mutation_deadline = start + args.mutation_duration_sec
    schedule = ("rfp", "lmp", "joint", "joint")
    iteration = max(
        (entry.get("iteration", -1) for entry in tune_entries), default=-1) + 1

    while (
        time.monotonic() < mutation_deadline
        and len(tune_entries) < args.max_candidates
    ):
        if pending_anchors:
            config = pending_anchors.pop(0)
        else:
            lineage = schedule[iteration % len(schedule)]
            config = next_candidate(lineage, tune_entries, anchors, seen, rng)
        seen.add(config)
        result = evaluate(
            args.binary, tune_gate, args.model, args.tune_depth, config)
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
        sizes = {lineage: len(lineage_frontier(tune_entries, lineage))
                 for lineage in LINEAGES}
        print(
            "tune_progress"
            f" iteration={iteration}"
            f" lineage={config.lineage}"
            f" nodes={result['node_ratio']:.6f}"
            f" wdl={result['objective_loss']:.9f}"
            f" critical={result.get('critical_mistakes', 0)}"
            f" rfp_cutoffs={result.get('reverse_futility_cutoffs', 0)}"
            f" lmp_moves={result.get('late_move_pruned_moves', 0)}"
            f" frontier_rfp={sizes['rfp']}"
            f" frontier_lmp={sizes['lmp']}"
            f" frontier_joint={sizes['joint']}",
            flush=True,
        )
        iteration += 1

    probe_sources = stage_candidates(tune_entries, include_lineages=True)
    probe_entries: list[dict] = []
    for rank, source in enumerate(probe_sources):
        config = source["config_obj"]
        result = evaluate(
            args.binary, selection_probe, args.model,
            args.selection_depth, config)
        entry = {
            "kind": "selection_probe",
            "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        probe_entries.append(entry)
        append_json(log_path, serializable(entry))
        print(
            "selection_probe_progress"
            f" rank={rank + 1}/{len(probe_sources)}"
            f" lineage={config.lineage}"
            f" nodes={result['node_ratio']:.6f}"
            f" wdl={result['objective_loss']:.9f}",
            flush=True,
        )

    selection_sources = stage_candidates(probe_entries, include_lineages=True)
    selection_entries: list[dict] = []
    for rank, source in enumerate(selection_sources):
        config = source["config_obj"]
        result = evaluate(
            args.binary, args.dataset_dir / "selection.tsv", args.model,
            args.selection_depth, config)
        entry = {
            "kind": "selection",
            "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        selection_entries.append(entry)
        append_json(log_path, serializable(entry))
        print(
            "selection_progress"
            f" rank={rank + 1}/{len(selection_sources)}"
            f" lineage={config.lineage}"
            f" nodes={result['node_ratio']:.6f}"
            f" wdl={result['objective_loss']:.9f}",
            flush=True,
        )

    holdout_sources = stage_candidates(selection_entries, include_lineages=True)
    holdout_entries: list[dict] = []
    for rank, source in enumerate(holdout_sources):
        config = source["config_obj"]
        result = evaluate(
            args.binary, holdout_main, args.model,
            args.holdout_depth, config)
        entry = {
            "kind": "holdout",
            "rank": rank,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        holdout_entries.append(entry)
        append_json(log_path, serializable(entry))
        print(
            "holdout_progress"
            f" rank={rank + 1}/{len(holdout_sources)}"
            f" lineage={config.lineage}"
            f" nodes={result['node_ratio']:.6f}"
            f" wdl={result['objective_loss']:.9f}",
            flush=True,
        )

    final_frontier = epsilon_frontier(holdout_entries)
    fast = min(final_frontier, key=lambda entry: entry["result"]["node_ratio"])
    safe = min(final_frontier, key=objective_loss)
    balanced = knee(final_frontier)
    baseline = next(entry for entry in holdout_entries
                    if entry["config_obj"].lineage == "off")
    audit_sources: list[tuple[str, dict]] = []
    used: set[Config] = set()
    for role, source in (
        ("fast", fast), ("balanced", balanced),
        ("safe", safe), ("baseline", baseline),
    ):
        if source["config_obj"] not in used:
            audit_sources.append((role, source))
            used.add(source["config_obj"])
    audit_entries: list[dict] = []
    for rank, (role, source) in enumerate(audit_sources):
        config = source["config_obj"]
        result = evaluate(
            args.binary, audit_set, args.model, args.audit_depth, config)
        entry = {
            "kind": "audit",
            "rank": rank,
            "role": role,
            "elapsed_sec": time.monotonic() - start,
            "config": asdict(config),
            "config_obj": config,
            "result": result,
        }
        audit_entries.append(entry)
        append_json(log_path, serializable(entry))
        print(
            "audit_progress"
            f" rank={rank + 1}/{len(audit_sources)}"
            f" role={role}"
            f" nodes={result['node_ratio']:.6f}"
            f" wdl={result['objective_loss']:.9f}",
            flush=True,
        )

    summary = {
        "kind": "complete",
        "seed": args.seed,
        "objective": "wdl",
        "include_all_in_objective": True,
        "hard_safety": False,
        "baseline_lmr_nmp": list(BASELINE_ARGS),
        "duration_requested_sec": args.duration_sec,
        "elapsed_sec": time.monotonic() - start,
        "mutations": len(tune_entries),
        "tune_frontier_size": len(frontier(tune_entries)),
        "selection_probe_candidates": len(probe_sources),
        "selection_candidates": len(selection_sources),
        "holdout_candidates": len(holdout_sources),
        "holdout_frontier_size": len(final_frontier),
        "holdout": [serializable(entry) for entry in holdout_entries],
        "audit": [serializable(entry) for entry in audit_entries],
        "log": str(log_path),
    }
    append_json(log_path, summary)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
