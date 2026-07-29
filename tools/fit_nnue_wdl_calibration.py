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


FORMULAS: dict[str, dict[str, tuple[str, ...]]] = {
    "symmetric_score": {
        "signed": ("z",),
        "draw": ("1", "abs_z"),
    },
    "symmetric_phase_ply": {
        "signed": ("z", "tanh_z_0p15", "tanh_z_0p4", "z_phase", "z_ply"),
        "draw": (
            "1",
            "abs_z",
            "abs_z2",
            "phase",
            "phase2",
            "ply",
            "ply2",
            "abs_z_phase",
            "abs_z_ply",
            "phase_ply",
        ),
    },
    "symmetric_spline_phase_ply": {
        "signed": (
            "z",
            "tanh_z_0p05",
            "tanh_z_0p15",
            "tanh_z_0p3",
            "tanh_z_0p5",
            "tanh_z_1",
            "tanh_z_1p5",
            "z_phase",
            "z_phase2",
            "z_ply",
            "z_ply2",
            "z_phase_ply",
        ),
        "draw": (
            "1",
            "abs_z",
            "abs_z2",
            "phase",
            "phase2",
            "phase3",
            "ply",
            "ply2",
            "abs_z_phase",
            "abs_z_ply",
            "phase_ply",
            "hinge_abs_z_0p05",
            "hinge_abs_z_0p15",
            "hinge_abs_z_0p3",
            "hinge_abs_z_0p5",
            "hinge_abs_z_1",
            "hinge_abs_z_1p5",
        ),
    },
}

CP_BUCKET_EDGES = (
    -math.inf,
    -1500.0,
    -1000.0,
    -500.0,
    -300.0,
    -150.0,
    -50.0,
    50.0,
    150.0,
    300.0,
    500.0,
    1000.0,
    1500.0,
    math.inf,
)


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
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Maximum strong-Wolfe LBFGS iterations per fit.",
    )
    parser.add_argument(
        "--regularizations",
        default="0,1e-5,1e-4,1e-3,1e-2",
        help="Comma-separated L2 strengths.",
    )
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


def game_equal_weights(samples: list[Sample]) -> torch.Tensor:
    counts = Counter(sample.game_id for sample in samples)
    weights = torch.tensor(
        [1.0 / counts[sample.game_id] for sample in samples],
        dtype=torch.float64,
    )
    return weights / weights.sum()


def feature_values(sample: Sample) -> dict[str, float]:
    z = sample.cp / 1000.0
    abs_z = abs(z)
    phase = sample.phase
    ply = sample.ply
    values = {
        "1": 1.0,
        "z": z,
        "abs_z": abs_z,
        "abs_z2": abs_z * abs_z,
        "phase": phase,
        "phase2": phase * phase,
        "phase3": phase * phase * phase,
        "ply": ply,
        "ply2": ply * ply,
        "z_phase": z * phase,
        "z_phase2": z * phase * phase,
        "z_ply": z * ply,
        "z_ply2": z * ply * ply,
        "z_phase_ply": z * phase * ply,
        "abs_z_phase": abs_z * phase,
        "abs_z_ply": abs_z * ply,
        "phase_ply": phase * ply,
    }
    for text, scale in (
        ("0p05", 0.05),
        ("0p15", 0.15),
        ("0p3", 0.3),
        ("0p4", 0.4),
        ("0p5", 0.5),
        ("1", 1.0),
        ("1p5", 1.5),
    ):
        values[f"tanh_z_{text}"] = math.tanh(z / scale)
        values[f"hinge_abs_z_{text}"] = max(0.0, abs_z - scale)
    return values


def feature_tensor(samples: list[Sample], names: tuple[str, ...]) -> torch.Tensor:
    return torch.tensor(
        [[feature_values(sample)[name] for name in names] for sample in samples],
        dtype=torch.float64,
    )


def tensors(samples: list[Sample], formula: str) -> tuple[torch.Tensor, ...]:
    definition = FORMULAS[formula]
    signed = feature_tensor(samples, definition["signed"])
    draw = feature_tensor(samples, definition["draw"])
    outcome = torch.tensor(
        [sample.outcome + 1 for sample in samples], dtype=torch.int64
    )
    return signed, draw, outcome, game_equal_weights(samples)


class SymmetricCalibrator(torch.nn.Module):
    """Convex symmetric three-class logit model.

    Loss and win logits are negatives of one another. Draw features are even in
    CP, while signed features are odd, so flipping the CP sign swaps W and L.
    """

    def __init__(self, signed_count: int, draw_count: int):
        super().__init__()
        self.signed_weights = torch.nn.Parameter(
            torch.zeros(signed_count, dtype=torch.float64)
        )
        self.draw_weights = torch.nn.Parameter(
            torch.zeros(draw_count, dtype=torch.float64)
        )
        with torch.no_grad():
            self.draw_weights[0] = 1.0

    def logits(
        self, signed_features: torch.Tensor, draw_features: torch.Tensor
    ) -> torch.Tensor:
        signed = signed_features @ self.signed_weights
        draw = draw_features @ self.draw_weights
        return torch.stack((-signed, draw, signed), dim=1)

    def probabilities(
        self, signed_features: torch.Tensor, draw_features: torch.Tensor
    ) -> torch.Tensor:
        return torch.softmax(self.logits(signed_features, draw_features), dim=1)


def weighted_nll(
    model: SymmetricCalibrator,
    batch: tuple[torch.Tensor, ...],
    regularization: float = 0.0,
) -> torch.Tensor:
    signed, draw, outcome, weights = batch
    log_probabilities = torch.log_softmax(model.logits(signed, draw), dim=1)
    loss = -(
        weights
        * log_probabilities[torch.arange(len(outcome)), outcome]
    ).sum()
    if regularization:
        draw_penalty = (
            model.draw_weights[1:].square().mean()
            if len(model.draw_weights) > 1
            else torch.zeros((), dtype=torch.float64)
        )
        loss += regularization * (
            model.signed_weights.square().mean() + draw_penalty
        )
    return loss


def fit(
    samples: list[Sample], formula: str, regularization: float, steps: int
) -> SymmetricCalibrator:
    batch = tensors(samples, formula)
    model = SymmetricCalibrator(batch[0].shape[1], batch[1].shape[1])
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=steps,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = weighted_nll(model, batch, regularization)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_loss = float(weighted_nll(model, batch, regularization).detach())
    if not math.isfinite(final_loss):
        raise RuntimeError(f"Optimizer failed for {formula}, lambda={regularization}")
    return model


def metrics_from_probabilities(
    probabilities: torch.Tensor,
    outcome: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        selected = probabilities[torch.arange(len(outcome)), outcome]
        nll = float(
            -(weights * torch.log(torch.clamp(selected, min=1e-12))).sum()
        )
        target = F.one_hot(outcome, num_classes=3).to(torch.float64)
        brier = float(
            (weights * (probabilities - target).square().sum(dim=1)).sum()
        )
        predicted_score = probabilities[:, 2] + 0.5 * probabilities[:, 1]
        observed_score = outcome.to(torch.float64) / 2.0
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


def metrics(
    model: SymmetricCalibrator, samples: list[Sample], formula: str
) -> dict[str, float]:
    signed, draw, outcome, weights = tensors(samples, formula)
    with torch.no_grad():
        probabilities = model.probabilities(signed, draw)
    return metrics_from_probabilities(probabilities, outcome, weights)


def constant_probabilities(samples: list[Sample]) -> torch.Tensor:
    weights = game_equal_weights(samples)
    outcome = torch.tensor(
        [sample.outcome + 1 for sample in samples], dtype=torch.int64
    )
    result = torch.zeros(3, dtype=torch.float64)
    for index in range(3):
        result[index] = weights[outcome == index].sum()
    return result


def constant_metrics(
    samples: list[Sample], probabilities: torch.Tensor
) -> dict[str, float]:
    outcome = torch.tensor(
        [sample.outcome + 1 for sample in samples], dtype=torch.int64
    )
    weights = game_equal_weights(samples)
    repeated = probabilities.unsqueeze(0).repeat(len(samples), 1)
    return metrics_from_probabilities(repeated, outcome, weights)


def cp_bucket(cp: float) -> int:
    for index, (lower, upper) in enumerate(
        zip(CP_BUCKET_EDGES, CP_BUCKET_EDGES[1:])
    ):
        if lower <= cp < upper:
            return index
    raise AssertionError("unreachable CP bucket")


def fit_bucket_baseline(samples: list[Sample]) -> list[list[float]]:
    weights = game_equal_weights(samples)
    global_probabilities = constant_probabilities(samples)
    effective_games = len({sample.game_id for sample in samples})
    smoothing = 1.0 / effective_games
    totals = torch.zeros(len(CP_BUCKET_EDGES) - 1, dtype=torch.float64)
    counts = torch.zeros((len(totals), 3), dtype=torch.float64)
    for sample, weight in zip(samples, weights):
        bucket = cp_bucket(sample.cp)
        totals[bucket] += weight
        counts[bucket, sample.outcome + 1] += weight
    result = []
    for bucket in range(len(totals)):
        probabilities = (
            counts[bucket] + smoothing * global_probabilities
        ) / (totals[bucket] + smoothing)
        result.append([float(value) for value in probabilities])
    return result


def bucket_metrics(
    samples: list[Sample], bucket_probabilities: list[list[float]]
) -> dict[str, float]:
    probabilities = torch.tensor(
        [bucket_probabilities[cp_bucket(sample.cp)] for sample in samples],
        dtype=torch.float64,
    )
    outcome = torch.tensor(
        [sample.outcome + 1 for sample in samples], dtype=torch.int64
    )
    return metrics_from_probabilities(
        probabilities, outcome, game_equal_weights(samples)
    )


def parameters(model: SymmetricCalibrator) -> dict[str, list[float]]:
    return {
        "signed_weights": [
            float(value) for value in model.signed_weights.detach()
        ],
        "draw_weights": [
            float(value) for value in model.draw_weights.detach()
        ],
    }


def split_summary(samples: list[Sample]) -> dict[str, object]:
    return {
        "positions": len(samples),
        "games": len({sample.game_id for sample in samples}),
        "openings": len({sample.opening_line for sample in samples}),
        "outcomes": dict(sorted(Counter(sample.outcome for sample in samples).items())),
    }


CPP_FEATURE_EXPRESSIONS = {
    "1": "1.0",
    "z": "z",
    "abs_z": "abs_z",
    "abs_z2": "abs_z * abs_z",
    "phase": "phase",
    "phase2": "phase * phase",
    "phase3": "phase * phase * phase",
    "ply": "ply",
    "ply2": "ply * ply",
    "z_phase": "z * phase",
    "z_phase2": "z * phase * phase",
    "z_ply": "z * ply",
    "z_ply2": "z * ply * ply",
    "z_phase_ply": "z * phase * ply",
    "abs_z_phase": "abs_z * phase",
    "abs_z_ply": "abs_z * ply",
    "phase_ply": "phase * ply",
}
for _text, _scale in (
    ("0p05", "0.05"),
    ("0p15", "0.15"),
    ("0p3", "0.3"),
    ("0p4", "0.4"),
    ("0p5", "0.5"),
    ("1", "1.0"),
    ("1p5", "1.5"),
):
    CPP_FEATURE_EXPRESSIONS[f"tanh_z_{_text}"] = f"std::tanh(z / {_scale})"
    CPP_FEATURE_EXPRESSIONS[f"hinge_abs_z_{_text}"] = (
        f"std::max(0.0, abs_z - {_scale})"
    )


def write_header(path: Path, winner: dict) -> None:
    signed_names = winner["signed_feature_names"]
    draw_names = winner["draw_feature_names"]
    signed_values = ", ".join(
        f"{value:.17g}" for value in winner["parameters"]["signed_weights"]
    )
    draw_values = ", ".join(
        f"{value:.17g}" for value in winner["parameters"]["draw_weights"]
    )
    signed_features = ", ".join(
        CPP_FEATURE_EXPRESSIONS[name] for name in signed_names
    )
    draw_features = ", ".join(
        CPP_FEATURE_EXPRESSIONS[name] for name in draw_names
    )
    path.write_text(
        f"""#pragma once
// Generated by fit_nnue_wdl_calibration.py.
#include <algorithm>
#include <array>
#include <cmath>

namespace chess::wdl_calibration {{
struct Wdl {{
    double loss;
    double draw;
    double win;
}};

inline constexpr const char* formula = "{winner["formula"]}";
inline constexpr std::array<double, {len(signed_names)}> signed_weights = {{{signed_values}}};
inline constexpr std::array<double, {len(draw_names)}> draw_weights = {{{draw_values}}};

inline Wdl probabilities(int cp, int phase_index, int absolute_ply) {{
    const double z = std::clamp(cp / 1000.0, -2.0, 2.0);
    const double abs_z = std::abs(z);
    const double phase = std::clamp(phase_index / 7.0, 0.0, 1.0);
    const double ply = std::clamp(absolute_ply / 240.0, 0.0, 1.0);
    const std::array<double, {len(signed_names)}> signed_features = {{{signed_features}}};
    const std::array<double, {len(draw_names)}> draw_features = {{{draw_features}}};
    double signed_logit = 0.0;
    double draw_logit = 0.0;
    for (std::size_t i = 0; i < signed_weights.size(); ++i) {{
        signed_logit += signed_weights[i] * signed_features[i];
    }}
    for (std::size_t i = 0; i < draw_weights.size(); ++i) {{
        draw_logit += draw_weights[i] * draw_features[i];
    }}
    const double maximum = std::max({{-signed_logit, draw_logit, signed_logit}});
    const double loss = std::exp(-signed_logit - maximum);
    const double draw = std::exp(draw_logit - maximum);
    const double win = std::exp(signed_logit - maximum);
    const double total = loss + draw + win;
    return {{loss / total, draw / total, win / total}};
}}

inline double expected_score(int cp, int phase_index, int absolute_ply) {{
    const Wdl wdl = probabilities(cp, phase_index, absolute_ply);
    return wdl.win + 0.5 * wdl.draw;
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

    empirical_train = constant_probabilities(train)
    bucket_train = fit_bucket_baseline(train)
    validation_baselines = {
        "uniform": constant_metrics(
            validation, torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
        ),
        "empirical_constant": constant_metrics(validation, empirical_train),
        "cp_bucket": bucket_metrics(validation, bucket_train),
    }

    comparisons = []
    for formula, definition in FORMULAS.items():
        candidates = []
        for regularization in regularizations:
            model = fit(train, formula, regularization, args.steps)
            candidates.append(
                {
                    "formula": formula,
                    "signed_feature_names": list(definition["signed"]),
                    "draw_feature_names": list(definition["draw"]),
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
    strongest_baseline = min(
        result["nll"] for result in validation_baselines.values()
    )
    if selected["validation"]["nll"] >= strongest_baseline:
        raise RuntimeError(
            "Best parametric formula failed the validation baseline gate: "
            f"{selected['validation']['nll']:.6f} >= {strongest_baseline:.6f}"
        )

    final_model = fit(
        train + validation,
        selected["formula"],
        selected["regularization"],
        args.steps,
    )
    empirical_train_validation = constant_probabilities(train + validation)
    bucket_train_validation = fit_bucket_baseline(train + validation)
    sealed_baselines = {
        "uniform": constant_metrics(
            test, torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
        ),
        "empirical_constant": constant_metrics(
            test, empirical_train_validation
        ),
        "cp_bucket": bucket_metrics(test, bucket_train_validation),
    }
    winner = {
        "formula": selected["formula"],
        "signed_feature_names": selected["signed_feature_names"],
        "draw_feature_names": selected["draw_feature_names"],
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
        "optimizer": {
            "name": "LBFGS",
            "line_search": "strong_wolfe",
            "max_iterations": args.steps,
        },
        "load_stats": load_stats,
        "splits": {
            "train": split_summary(train),
            "validation": split_summary(validation),
            "sealed_test": split_summary(test),
        },
        "validation_baselines": validation_baselines,
        "sealed_test_baselines": sealed_baselines,
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
        "Formula and L2 selection used validation NLL only. The sealed test was opened once for the winner.",
        "",
        "## Formula comparison",
        "",
        "| Formula | Signed/draw features | L2 | Train NLL | Validation NLL | Validation Brier | Validation ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in comparisons:
        rows.append(
            f"| {result['formula']} | "
            f"{len(result['signed_feature_names'])}/{len(result['draw_feature_names'])} | "
            f"{result['regularization']:.1g} | {result['train']['nll']:.6f} | "
            f"{result['validation']['nll']:.6f} | "
            f"{result['validation']['brier']:.6f} | "
            f"{result['validation']['expected_score_ece']:.6f} |"
        )
    rows += [
        "",
        "## Validation baselines",
        "",
        "| Baseline | NLL | Brier | ECE |",
        "|---|---:|---:|---:|",
    ]
    for name, result in validation_baselines.items():
        rows.append(
            f"| {name} | {result['nll']:.6f} | {result['brier']:.6f} | "
            f"{result['expected_score_ece']:.6f} |"
        )
    rows += [
        "",
        f"Winner: `{winner['formula']}`",
        "",
        f"- Sealed-test NLL: {winner['sealed_test']['nll']:.6f}",
        f"- Sealed-test Brier: {winner['sealed_test']['brier']:.6f}",
        f"- Sealed-test expected-score ECE: {winner['sealed_test']['expected_score_ece']:.6f}",
        f"- Sealed-test CP-bucket baseline NLL: {sealed_baselines['cp_bucket']['nll']:.6f}",
        f"- Sealed-test empirical-constant baseline NLL: {sealed_baselines['empirical_constant']['nll']:.6f}",
        f"- Usable positions: {len(samples)} from {load_stats['completed_games']} completed games",
        f"- Excluded mate-score positions: {load_stats.get('mate_positions', 0)}",
        "",
    ]
    (args.output_dir / "wdl_calibration_report.md").write_text("\n".join(rows))
    print(
        json.dumps(
            {
                "winner": winner,
                "validation_baselines": validation_baselines,
                "sealed_test_baselines": sealed_baselines,
                "splits": metadata["splits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
