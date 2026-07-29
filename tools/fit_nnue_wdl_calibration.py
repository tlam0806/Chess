#!/usr/bin/env python3
"""Fit symmetric CP-to-WDL calibration models without game leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


FORMULAS: dict[str, tuple[str, ...]] = {
    "score_only": ("1",),
    "phase_linear": ("1", "phase"),
    "phase_quadratic": ("1", "phase", "phase2"),
    "ply_quadratic": ("1", "ply", "ply2"),
    "phase_ply_additive": ("1", "phase", "phase2", "ply", "ply2"),
    "phase_ply_interaction": (
        "1", "phase", "phase2", "ply", "ply2", "phase_ply",
    ),
    "phase_cubic_ply_interaction": (
        "1", "phase", "phase2", "phase3", "ply", "ply2", "phase_ply",
    ),
}


@dataclass(frozen=True)
class Sample:
    game_id: int
    opening_line: int
    cp: float
    phase: float
    ply: float
    outcome: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--regularizations", default="0,1e-6,1e-5,1e-4")
    parser.add_argument("--cp-cap", type=float, default=2000.0)
    return parser.parse_args()


def load_samples(path: Path, cp_cap: float) -> tuple[list[Sample], dict[str, int]]:
    completed: set[int] = set()
    raw_positions: list[dict] = []
    stats: Counter[str] = Counter()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") == "game":
                stats["game_records"] += 1
                if int(record.get("positions", 0)) > 0:
                    completed.add(int(record["game_id"]))
            elif record.get("kind") == "position":
                raw_positions.append(record)

    samples: list[Sample] = []
    seen: set[tuple[int, int]] = set()
    for record in raw_positions:
        game_id = int(record["game_id"])
        key = (game_id, int(record["game_ply"]))
        if game_id not in completed:
            stats["incomplete_positions"] += 1
            continue
        if key in seen:
            stats["duplicate_positions"] += 1
            continue
        seen.add(key)
        cp = float(record["teacher_cp"])
        if abs(cp) >= 90_000:
            stats["mate_positions"] += 1
            continue
        outcome = int(record["outcome"])
        if outcome not in (-1, 0, 1):
            stats["invalid_outcome"] += 1
            continue
        samples.append(
            Sample(
                game_id=game_id,
                opening_line=int(record["opening_line"]),
                cp=max(-cp_cap, min(cp_cap, cp)),
                phase=max(0.0, min(1.0, int(record["phase_index"]) / 7.0)),
                ply=max(0.0, min(1.0, int(record["ply"]) / 240.0)),
                outcome=outcome,
            )
        )
    stats["usable_positions"] = len(samples)
    stats["completed_games"] = len({sample.game_id for sample in samples})
    stats["openings"] = len({sample.opening_line for sample in samples})
    return samples, dict(stats)


def group_split(
    samples: list[Sample], seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    groups = sorted({sample.opening_line for sample in samples})
    if len(groups) < 12:
        raise ValueError(f"Need at least 12 completed opening groups, got {len(groups)}")
    random.Random(seed).shuffle(groups)
    train_end = max(1, round(0.70 * len(groups)))
    val_end = min(
        len(groups) - 1, train_end + max(1, round(0.15 * len(groups)))
    )
    group_sets = (
        set(groups[:train_end]),
        set(groups[train_end:val_end]),
        set(groups[val_end:]),
    )
    assert not (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    )
    return tuple(
        [sample for sample in samples if sample.opening_line in group_set]
        for group_set in group_sets
    )  # type: ignore[return-value]


def feature_tensor(samples: list[Sample], formula: str) -> torch.Tensor:
    rows = []
    for sample in samples:
        p, t = sample.phase, sample.ply
        values = {
            "1": 1.0,
            "phase": p,
            "phase2": p * p,
            "phase3": p * p * p,
            "ply": t,
            "ply2": t * t,
            "phase_ply": p * t,
        }
        rows.append([values[name] for name in FORMULAS[formula]])
    return torch.tensor(rows, dtype=torch.float64)


def tensors(samples: list[Sample], formula: str) -> tuple[torch.Tensor, ...]:
    x = feature_tensor(samples, formula)
    cp = torch.tensor([sample.cp for sample in samples], dtype=torch.float64)
    outcome = torch.tensor([sample.outcome for sample in samples], dtype=torch.int64)
    counts = Counter(sample.game_id for sample in samples)
    weights = torch.tensor(
        [1.0 / counts[sample.game_id] for sample in samples],
        dtype=torch.float64,
    )
    weights /= weights.sum()
    return x, cp, outcome, weights


class Calibrator(torch.nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.raw_a = torch.nn.Parameter(
            torch.zeros(feature_count, dtype=torch.float64)
        )
        self.raw_b = torch.nn.Parameter(
            torch.zeros(feature_count, dtype=torch.float64)
        )
        with torch.no_grad():
            self.raw_a[0] = 150.0
            self.raw_b[0] = 70.0

    def probabilities(self, x: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
        a = F.softplus(x @ self.raw_a)
        b = F.softplus(x @ self.raw_b) + 1e-6
        win = torch.sigmoid((cp - a) / b)
        loss = torch.sigmoid((-cp - a) / b)
        draw = torch.clamp(1.0 - win - loss, min=1e-12)
        probabilities = torch.stack((loss, draw, win), dim=1)
        return probabilities / probabilities.sum(dim=1, keepdim=True)


def weighted_nll(
    model: Calibrator,
    batch: tuple[torch.Tensor, ...],
    regularization: float = 0.0,
) -> torch.Tensor:
    x, cp, outcome, weights = batch
    probabilities = model.probabilities(x, cp)
    selected = probabilities[torch.arange(len(outcome)), outcome + 1]
    loss = -(weights * torch.log(torch.clamp(selected, min=1e-12))).sum()
    if regularization and x.shape[1] > 1:
        loss += regularization * (
            model.raw_a[1:].square().mean()
            + model.raw_b[1:].square().mean()
        )
    return loss


def fit(
    samples: list[Sample], formula: str, regularization: float, steps: int
) -> Calibrator:
    batch = tensors(samples, formula)
    model = Calibrator(len(FORMULAS[formula]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(steps):
        optimizer.zero_grad()
        loss = weighted_nll(model, batch, regularization)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        value = float(loss.detach())
        if math.isfinite(value) and value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError(f"Optimizer failed for {formula}, lambda={regularization}")
    model.load_state_dict(best_state)
    return model


def metrics(
    model: Calibrator, samples: list[Sample], formula: str
) -> dict[str, float]:
    x, cp, outcome, weights = tensors(samples, formula)
    with torch.no_grad():
        probabilities = model.probabilities(x, cp)
        selected = probabilities[torch.arange(len(outcome)), outcome + 1]
        nll = float(
            -(weights * torch.log(torch.clamp(selected, min=1e-12))).sum()
        )
        target = F.one_hot(outcome + 1, num_classes=3).to(torch.float64)
        brier = float(
            (weights * (probabilities - target).square().sum(dim=1)).sum()
        )
        predicted_score = probabilities[:, 2] + 0.5 * probabilities[:, 1]
        observed_score = (outcome.to(torch.float64) + 1.0) / 2.0
        ece = 0.0
        for lower in torch.linspace(0.0, 0.9, 10, dtype=torch.float64):
            upper = lower + 0.1
            mask = (predicted_score >= lower) & (
                predicted_score <= upper if upper >= 1.0 else predicted_score < upper
            )
            if mask.any():
                bin_weight = weights[mask].sum()
                predicted = (
                    weights[mask] * predicted_score[mask]
                ).sum() / bin_weight
                observed = (
                    weights[mask] * observed_score[mask]
                ).sum() / bin_weight
                ece += float(bin_weight * abs(predicted - observed))
    return {"nll": nll, "brier": brier, "expected_score_ece": ece}


def parameters(model: Calibrator) -> dict[str, list[float]]:
    return {
        "raw_a": [float(value) for value in model.raw_a.detach()],
        "raw_b": [float(value) for value in model.raw_b.detach()],
    }


def split_summary(samples: list[Sample]) -> dict[str, object]:
    return {
        "positions": len(samples),
        "games": len({sample.game_id for sample in samples}),
        "openings": len({sample.opening_line for sample in samples}),
        "outcomes": dict(sorted(Counter(sample.outcome for sample in samples).items())),
    }


def write_header(path: Path, winner: dict) -> None:
    features = ", ".join(f'"{name}"' for name in winner["feature_names"])
    raw_a = ", ".join(
        f"{value:.17g}" for value in winner["parameters"]["raw_a"]
    )
    raw_b = ", ".join(
        f"{value:.17g}" for value in winner["parameters"]["raw_b"]
    )
    path.write_text(
        f"""#pragma once
// Generated calibration coefficients. Phase and ply inputs are normalized to [0,1].
#include <array>
#include <cmath>

namespace chess::wdl_calibration {{
inline constexpr const char* formula = "{winner["formula"]}";
inline constexpr std::array<const char*, {len(winner["feature_names"])}> feature_names = {{{features}}};
inline constexpr std::array<double, {len(winner["feature_names"])}> raw_a = {{{raw_a}}};
inline constexpr std::array<double, {len(winner["feature_names"])}> raw_b = {{{raw_b}}};
inline double softplus(double x) {{
    return x > 40.0 ? x : std::log1p(std::exp(x));
}}
}}  // namespace chess::wdl_calibration
"""
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    samples, load_stats = load_samples(args.input, args.cp_cap)
    train, validation, test = group_split(samples, args.seed)
    regularizations = [
        float(value) for value in args.regularizations.split(",")
    ]
    comparisons = []
    for formula in FORMULAS:
        candidates = []
        for regularization in regularizations:
            model = fit(train, formula, regularization, args.steps)
            candidates.append(
                {
                    "formula": formula,
                    "feature_names": list(FORMULAS[formula]),
                    "regularization": regularization,
                    "parameters": parameters(model),
                    "train": metrics(model, train, formula),
                    "validation": metrics(model, validation, formula),
                }
            )
        comparisons.append(
            min(candidates, key=lambda result: result["validation"]["nll"])
        )

    selected = min(
        comparisons, key=lambda result: result["validation"]["nll"]
    )
    final_model = fit(
        train + validation,
        selected["formula"],
        selected["regularization"],
        args.steps,
    )
    winner = {
        "formula": selected["formula"],
        "feature_names": selected["feature_names"],
        "regularization": selected["regularization"],
        "parameters": parameters(final_model),
        "train_validation": metrics(
            final_model, train + validation, selected["formula"]
        ),
        "sealed_test": metrics(final_model, test, selected["formula"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "seed": args.seed,
        "cp_cap": args.cp_cap,
        "load_stats": load_stats,
        "splits": {
            "train": split_summary(train),
            "validation": split_summary(validation),
            "sealed_test": split_summary(test),
        },
    }
    (args.output_dir / "models.json").write_text(
        json.dumps({"metadata": metadata, "comparisons": comparisons}, indent=2)
        + "\n"
    )
    (args.output_dir / "winner.json").write_text(
        json.dumps({"metadata": metadata, "winner": winner}, indent=2) + "\n"
    )
    write_header(args.output_dir / "wdl_calibration.hpp", winner)

    rows = [
        "# NNUE CP to WDL calibration",
        "",
        "Formula selection used validation NLL only. The sealed test was opened once for the winner.",
        "",
        "| Formula | Features | L2 | Train NLL | Validation NLL | Validation Brier | Validation ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in comparisons:
        rows.append(
            f"| {result['formula']} | {len(result['feature_names'])} | "
            f"{result['regularization']:.1g} | {result['train']['nll']:.6f} | "
            f"{result['validation']['nll']:.6f} | "
            f"{result['validation']['brier']:.6f} | "
            f"{result['validation']['expected_score_ece']:.6f} |"
        )
    rows += [
        "",
        f"Winner: `{winner['formula']}`",
        "",
        f"- Sealed-test NLL: {winner['sealed_test']['nll']:.6f}",
        f"- Sealed-test Brier: {winner['sealed_test']['brier']:.6f}",
        f"- Sealed-test expected-score ECE: {winner['sealed_test']['expected_score_ece']:.6f}",
        f"- Usable positions: {len(samples)} from {load_stats['completed_games']} completed games",
        f"- Excluded mate-score positions: {load_stats.get('mate_positions', 0)}",
        "",
    ]
    (args.output_dir / "wdl_calibration_report.md").write_text("\n".join(rows))
    print(json.dumps({"winner": winner, "splits": metadata["splits"]}, indent=2))


if __name__ == "__main__":
    main()
