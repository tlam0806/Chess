#!/usr/bin/env python3
"""Exhaustive, resumable QSEE-threshold tuner for the V39/Fast baseline."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


THRESHOLDS: tuple[int | None, ...] = (
    None, -600, -500, -400, -350, -300, -250, -225, -200,
    -175, -150, -125, -100, -75, -50, -25, 0,
)

FAST_ARGS = (
    "--lmr-base", "0.45",
    "--lmr-divisor", "2.9",
    "--lmr-min-depth", "3",
    "--lmr-min-move-index", "6",
    "--null-min-depth", "2",
    "--null-reduction", "3",
    "--enable-reverse-futility",
    "--reverse-futility-max-depth", "2",
    "--reverse-futility-base-margin", "175",
    "--reverse-futility-margin-per-depth", "275",
    "--enable-late-move-pruning",
    "--late-move-pruning-max-depth", "3",
    "--late-move-pruning-base", "4",
    "--late-move-pruning-depth-multiplier", "2",
)


@dataclass(frozen=True)
class Config:
    threshold: int | None

    @property
    def label(self) -> str:
        return "off" if self.threshold is None else f"m{abs(self.threshold)}"

    def args(self) -> list[str]:
        if self.threshold is None:
            return [*FAST_ARGS, "--disable-qsearch-see-pruning"]
        return [
            *FAST_ARGS,
            "--enable-qsearch-see-pruning",
            "--qsearch-see-threshold", str(self.threshold),
        ]


CONFIGS = tuple(Config(value) for value in THRESHOLDS)


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


def evaluate(
    binary: Path,
    dataset: Path,
    model: Path,
    depth: int,
    config: Config,
    candidate_time_ms: int = 0,
) -> dict:
    command = [
        str(binary),
        "--dataset", str(dataset),
        "--model", str(model),
        "--depth", str(depth),
        "--objective", "wdl",
        "--include-all-in-objective",
        *config.args(),
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
        raise RuntimeError("evaluator excluded positions from WDL objective")
    if result.get("heuristics_reset_per_sample") is not True:
        raise RuntimeError("evaluator leaked search heuristics across samples")
    if int(result.get("candidate_time_ms", 0)) != candidate_time_ms:
        raise RuntimeError("evaluator did not honor candidate-time mode")
    return result


def append_json(path: Path, payload: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")


def load_completed(path: Path) -> dict[tuple[str, str], dict]:
    completed: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("kind") == "result":
            completed[(entry["stage"], entry["label"])] = entry
    return completed


def run_stage(
    stage: str,
    configs: list[Config],
    dataset: Path,
    depth: int,
    binary: Path,
    model: Path,
    results_path: Path,
    completed: dict[tuple[str, str], dict],
    start: float,
    workers: int,
    candidate_time_ms: int = 0,
) -> list[dict]:
    configs = sorted(set(configs), key=lambda item: THRESHOLDS.index(item.threshold))
    pending = [
        config for config in configs
        if (stage, config.label) not in completed
    ]

    def run(config: Config) -> tuple[Config, dict]:
        return config, evaluate(
            binary, dataset, model, depth, config, candidate_time_ms)

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, config): config for config in pending}
            for future in concurrent.futures.as_completed(futures):
                config, result = future.result()
                entry = {
                    "kind": "result",
                    "stage": stage,
                    "label": config.label,
                    "threshold": config.threshold,
                    "dataset": str(dataset),
                    "depth": depth,
                    "candidate_time_ms": candidate_time_ms,
                    "elapsed_sec": time.monotonic() - start,
                    "result": result,
                }
                append_json(results_path, entry)
                completed[(stage, config.label)] = entry
                print(
                    f"{stage}_progress label={config.label}"
                    f" nodes={result['node_ratio']:.6f}"
                    f" cp={result['mean_root_regret']:.4f}"
                    f" wdl={result['mean_wdl_loss']:.9f}"
                    f" depth={result.get('candidate_mean_depth', depth):.3f}"
                    f" critical={result['critical_mistakes']}",
                    flush=True,
                )
    return [completed[(stage, config.label)] for config in configs]


def dominates(left: dict, right: dict, fixed_time: bool = False) -> bool:
    """Return whether left is no worse than right and better in one metric."""
    a, b = left["result"], right["result"]
    if fixed_time:
        left_metrics = (
            float(a["mean_wdl_loss"]),
            float(a["mean_root_regret"]),
            -float(a["candidate_mean_depth"]),
        )
        right_metrics = (
            float(b["mean_wdl_loss"]),
            float(b["mean_root_regret"]),
            -float(b["candidate_mean_depth"]),
        )
    else:
        left_metrics = (
            float(a["mean_wdl_loss"]),
            float(a["node_ratio"]),
        )
        right_metrics = (
            float(b["mean_wdl_loss"]),
            float(b["node_ratio"]),
        )
    return all(x <= y for x, y in zip(left_metrics, right_metrics)) and any(
        x < y for x, y in zip(left_metrics, right_metrics))


def frontier(entries: list[dict], fixed_time: bool = False) -> list[dict]:
    return [
        entry for entry in entries
        if not any(
            dominates(other, entry, fixed_time)
            for other in entries if other is not entry
        )
    ]


def config_for(entry: dict) -> Config:
    return Config(entry["threshold"])


def expand_neighbors(entries: list[dict]) -> list[Config]:
    selected = {config_for(entry) for entry in entries}
    for entry in entries:
        index = THRESHOLDS.index(entry["threshold"])
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(THRESHOLDS):
                selected.add(Config(THRESHOLDS[neighbor]))
    return sorted(selected, key=lambda item: THRESHOLDS.index(item.threshold))


def profile(entries: list[dict]) -> list[dict]:
    return [
        {
            "label": entry["label"],
            "threshold": entry["threshold"],
            "result": entry["result"],
        }
        for entry in sorted(
            entries, key=lambda item: THRESHOLDS.index(item["threshold"]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate-size", type=int, default=3_000)
    parser.add_argument("--time-size", type=int, default=2_000)
    parser.add_argument("--gate-depth", type=int, default=6)
    parser.add_argument("--selection-depth", type=int, default=7)
    parser.add_argument("--holdout-depth", type=int, default=8)
    parser.add_argument("--candidate-time-ms", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    gate_path = args.run_dir / "gate.tsv"
    time_path = args.run_dir / "time_selection.tsv"
    write_subset(
        gate_path, args.dataset_dir / "tune.tsv",
        args.gate_size, args.seed ^ 0x6A7E)
    write_subset(
        time_path, args.dataset_dir / "selection.tsv",
        args.time_size, args.seed ^ 0x71AE)

    results_path = args.run_dir / "results.jsonl"
    completed = load_completed(results_path)
    start = time.monotonic()

    gate = run_stage(
        "gate", list(CONFIGS), gate_path, args.gate_depth,
        args.binary, args.model, results_path, completed, start, args.workers)
    gate_sources = expand_neighbors(frontier(gate))
    gate_sources.append(Config(None))

    selection = run_stage(
        "selection", gate_sources, args.dataset_dir / "selection.tsv",
        args.selection_depth, args.binary, args.model, results_path,
        completed, start, args.workers)
    selection_frontier = frontier(selection)
    time_sources = expand_neighbors(selection_frontier)
    time_sources.append(Config(None))

    time_selection = run_stage(
        "time_selection", time_sources, time_path, args.selection_depth,
        args.binary, args.model, results_path, completed, start, 1,
        args.candidate_time_ms)
    time_frontier = frontier(time_selection, fixed_time=True)

    holdout_sources = {
        *expand_neighbors(selection_frontier),
        *expand_neighbors(time_frontier),
        Config(None),
    }
    holdout = run_stage(
        "holdout", list(holdout_sources),
        args.dataset_dir / "holdout.tsv", args.holdout_depth,
        args.binary, args.model, results_path, completed, start, args.workers)
    holdout_frontier = frontier(holdout)

    safest = min(
        holdout_frontier,
        key=lambda entry: (
            entry["result"]["mean_wdl_loss"],
            entry["result"]["node_ratio"],
        ),
    )
    fastest = min(
        holdout_frontier,
        key=lambda entry: (
            entry["result"]["node_ratio"],
            entry["result"]["mean_wdl_loss"],
        ),
    )
    summary = {
        "kind": "complete",
        "objective": "all-position WDL/node Pareto",
        "hard_safety": False,
        "seed": args.seed,
        "threshold_grid": list(THRESHOLDS),
        "elapsed_sec": time.monotonic() - start,
        "gate": profile(gate),
        "selection": profile(selection),
        "time_selection": profile(time_selection),
        "holdout": profile(holdout),
        "holdout_frontier": profile(holdout_frontier),
        "safest_frontier": {
            "label": safest["label"],
            "threshold": safest["threshold"],
            "result": safest["result"],
        },
        "fastest_frontier": {
            "label": fastest["label"],
            "threshold": fastest["threshold"],
            "result": fastest["result"],
        },
        "requires_selfplay_promotion": True,
    }
    append_json(results_path, summary)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.run_dir / "DONE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
