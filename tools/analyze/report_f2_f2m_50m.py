#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    directory: Path
    summary: dict[str, object]
    errors: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired F2/F2M phase runs and an optional shared-L1 run"
    )
    parser.add_argument("--f2-dir", required=True, type=Path)
    parser.add_argument("--f2m-dir", required=True, type=Path)
    parser.add_argument("--shared-dir", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--append-poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def load_run(key: str, label: str, directory: Path) -> Run:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    errors = np.asarray(
        json.loads((directory / "ranking_errors.json").read_text(encoding="utf-8")),
        dtype=np.float64,
    )
    if summary.get("event") != "training_complete" or errors.ndim != 1:
        raise ValueError(f"incomplete run: {directory}")
    return Run(key, label, directory, summary, errors)


def wait_for_appended_shared_run(run_root: Path, poll_seconds: float) -> Path | None:
    request = run_root / "append_f2m_shared_l1.requested"
    if not request.exists():
        return None
    ready = run_root / "append_f2m_shared_l1.ready"
    failed = run_root / "append_f2m_shared_l1.failed"
    while not ready.exists():
        if failed.exists():
            detail = failed.read_text(encoding="utf-8").strip()
            raise RuntimeError(f"appended shared-L1 run failed: {detail}")
        time.sleep(poll_seconds)
    return run_root / "f2m_shared_l1"


def paired_ci(
    delta: np.ndarray,
    block_size: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float]:
    blocks = len(delta) // block_size
    if blocks == 0:
        raise ValueError("not enough ranking samples for block bootstrap")
    block_means = delta[: blocks * block_size].reshape(blocks, -1).mean(axis=1)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        bootstrap[index] = block_means[
            rng.integers(0, blocks, size=blocks)
        ].mean()
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def metric(summary: dict[str, object], split: str) -> dict[str, object]:
    value = summary[split]
    if not isinstance(value, dict):
        raise TypeError(f"invalid {split} metrics")
    return value


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.block_size <= 0:
        raise SystemExit("bootstrap and block sizes must be positive")
    if args.append_poll_seconds <= 0:
        raise SystemExit("append poll interval must be positive")

    run_root = args.f2_dir.parent
    run_specs = [
        ("f2", "F2 independent", args.f2_dir),
        ("f2m", "F2M independent", args.f2m_dir),
    ]
    shared_dir = args.shared_dir or wait_for_appended_shared_run(
        run_root, args.append_poll_seconds
    )
    if shared_dir is not None:
        run_specs.append(("f2m_shared_l1", "F2M shared L1", shared_dir))
    runs = [load_run(*spec) for spec in run_specs]

    ranking_samples = len(runs[0].errors)
    if any(len(run.errors) != ranking_samples for run in runs[1:]):
        raise SystemExit("ranking error vectors are not paired")
    if ranking_samples < args.block_size:
        raise SystemExit("not enough ranking samples for block bootstrap")

    # The training objective is the sealed selection Huber loss. Paired ranking
    # CP errors are an independent stability check, not a hidden winner switch.
    best = min(
        runs,
        key=lambda run: float(metric(run.summary, "selection")["objective_loss"]),
    )
    comparisons: dict[str, dict[str, object]] = {}
    for index, run in enumerate(runs):
        delta = run.errors - best.errors
        low, high = paired_ci(
            delta,
            args.block_size,
            args.bootstrap_samples,
            args.bootstrap_seed + index,
        )
        comparisons[run.key] = {
            "against": best.key,
            "paired_cp_mae_delta": float(delta.mean()),
            "paired_cp_mae_delta_95ci": [low, high],
        }

    f2 = runs[0]
    f2m = runs[1]
    legacy_delta = f2m.errors - f2.errors
    legacy_low, legacy_high = paired_ci(
        legacy_delta,
        args.block_size,
        args.bootstrap_samples,
        args.bootstrap_seed + 100,
    )
    if legacy_high < 0:
        legacy_verdict = "F2M lower paired CP error"
    elif legacy_low > 0:
        legacy_verdict = "F2 lower paired CP error"
    else:
        legacy_verdict = "inconclusive at 95% paired CI"

    models: dict[str, dict[str, object]] = {}
    for run in runs:
        models[run.key] = {
            "label": run.label,
            "directory": str(run.directory),
            "selection": metric(run.summary, "selection"),
            "ranking": metric(run.summary, "ranking"),
        }
    result = {
        "event": "phase_architecture_50m_comparison",
        "ranking_samples": ranking_samples,
        "winner_by_selection_huber": best.key,
        "models": models,
        "paired_ranking_cp_vs_selection_winner": comparisons,
        # Preserve the original pair fields for existing consumers.
        "paired_cp_mae_delta_f2m_minus_f2": float(legacy_delta.mean()),
        "paired_cp_mae_delta_95ci": [legacy_low, legacy_high],
        "selection_huber_delta_f2m_minus_f2": float(
            metric(f2m.summary, "selection")["objective_loss"]
            - metric(f2.summary, "selection")["objective_loss"]
        ),
        "verdict": (
            legacy_verdict
            if len(runs) == 2
            else f"selection Huber winner: {best.label}"
        ),
        "f2": {
            "selection": metric(f2.summary, "selection"),
            "ranking": metric(f2.summary, "ranking"),
        },
        "f2m": {
            "selection": metric(f2m.summary, "selection"),
            "ranking": metric(f2m.summary, "ranking"),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase architecture — paired 50M trial",
        "",
        f"Selection-Huber winner: **{best.label}**.",
        "",
        "Every model uses the same raw positions, order, seed, optimizer, and sealed validation records.",
        "",
        "| Model | Selection Huber | Selection CP MAE | Ranking Huber | Ranking CP MAE | Ranking WDL loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        selection = metric(run.summary, "selection")
        ranking = metric(run.summary, "ranking")
        lines.append(
            f"| {run.label} | {selection['objective_loss']:.8f} | "
            f"{selection['cp_mae']:.4f} | {ranking['objective_loss']:.8f} | "
            f"{ranking['cp_mae']:.4f} | {ranking['wdl_score_only_loss']:.8f} |"
        )

    lines.extend(
        [
            "",
            "## Paired ranking CP error vs selection winner",
            "",
            "A negative delta is better than the selection winner; a positive delta is worse.",
            "",
            "| Model | Delta | 95% block-bootstrap CI |",
            "|---|---:|---:|",
        ]
    )
    for run in runs:
        comparison = comparisons[run.key]
        low, high = comparison["paired_cp_mae_delta_95ci"]
        lines.append(
            f"| {run.label} | {comparison['paired_cp_mae_delta']:+.4f}cp | "
            f"[{low:+.4f}, {high:+.4f}] |"
        )

    lines.extend(
        [
            "",
            "## Per-phase ranking CP MAE",
            "",
            "| Phase | Samples | " + " | ".join(run.label for run in runs) + " |",
            "|---:|---:|" + "---:|" * len(runs),
        ]
    )
    phase_metrics = [metric(run.summary, "ranking")["phase"] for run in runs]
    for phase_index in range(8):
        rows = [phases[phase_index] for phases in phase_metrics]
        samples = int(rows[0]["samples"])
        if any(
            int(row["samples"]) != samples or int(row["phase"]) != phase_index
            for row in rows
        ):
            raise SystemExit("per-phase ranking samples are not paired")
        lines.append(
            f"| {phase_index} | {samples} | "
            + " | ".join(f"{float(row['cp_mae']):.4f}" for row in rows)
            + " |"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
