#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.quantized_nnue_architectures import (  # noqa: E402
    QUANTIZATION_CONVENTION_LEGACY,
    QUANTIZED_ARCHITECTURES,
)
from chess_nnue.train_value import choose_device  # noqa: E402
from tools.analyze.evaluate_nnue_paired_comparison import (  # noqa: E402
    DEFAULT_BIN_EDGES,
    paired_block_bootstrap,
    parse_bin_edges,
)
from tools.analyze.evaluate_quantized_nnue_checkpoint import load_model  # noqa: E402
from tools.train.train_nnue_architecture import make_loader, validate_training_data  # noqa: E402
from tools.train.train_quantized_nnue_architecture import (  # noqa: E402
    collate_quantized_sparse_batch,
    regression_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an affine CP calibration on a deterministic selection slice and "
            "evaluate it on a disjoint holdout slice"
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=("auto", "cbin", "jsonl"), default="auto")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--selection-samples", type=int, default=200_000)
    parser.add_argument("--ranking-samples", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward-mode", choices=("auto", "float", "quantized"), default="auto")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-block-size", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--target-abs-cp-bins", type=parse_bin_edges, default=DEFAULT_BIN_EDGES)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def fit_ols_affine(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return a, b for target ~= a * prediction + b, minimizing squared error."""
    pred_centered = prediction - prediction.mean()
    denominator = float(np.dot(pred_centered, pred_centered))
    if denominator <= 0.0:
        raise ValueError("cannot calibrate a constant prediction")
    target_centered = target - target.mean()
    scale = float(np.dot(pred_centered, target_centered) / denominator)
    bias = float(target.mean() - scale * prediction.mean())
    return scale, bias


def fit_mae_affine(
    prediction: np.ndarray,
    target: np.ndarray,
    lower_scale: float = 0.0,
    upper_scale: float = 4.0,
    iterations: int = 80,
) -> tuple[float, float]:
    """Fit affine calibration for MAE; for each scale the optimal bias is a median."""
    if prediction.size == 0 or prediction.shape != target.shape:
        raise ValueError("prediction and target must be non-empty vectors with matching shapes")
    if not lower_scale < upper_scale or iterations <= 0:
        raise ValueError("invalid MAE affine search range")

    def objective(scale: float) -> tuple[float, float]:
        residual = target - scale * prediction
        bias = float(np.median(residual))
        return float(np.mean(np.abs(residual - bias))), bias

    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = lower_scale, upper_scale
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, _ = objective(x1)
    f2, _ = objective(x2)
    for _ in range(iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1, _ = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2, _ = objective(x2)
    scale = float((left + right) / 2.0)
    _, bias = objective(scale)
    return scale, bias


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - target
    target_centered = target - target.mean()
    pred_centered = prediction - prediction.mean()
    target_variance_sum = float(np.dot(target_centered, target_centered))
    pred_variance_sum = float(np.dot(pred_centered, pred_centered))
    covariance_sum = float(np.dot(target_centered, pred_centered))
    slope = covariance_sum / target_variance_sum if target_variance_sum > 0.0 else float("nan")
    intercept = float(prediction.mean() - slope * target.mean())
    pearson_denom = np.sqrt(target_variance_sum * pred_variance_sum)
    pearson = covariance_sum / pearson_denom if pearson_denom > 0.0 else float("nan")
    return {
        "samples": int(target.size),
        "cp_mae": float(np.mean(np.abs(error))),
        "cp_rmse": float(np.sqrt(np.mean(error * error))),
        "prediction_vs_target_slope": float(slope),
        "prediction_vs_target_intercept": intercept,
        "pearson": float(pearson),
        "mean_target": float(target.mean()),
        "mean_prediction": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


@torch.no_grad()
def collect_predictions(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data_format = validate_training_data(args.data, args.data_format)
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    architecture = str(checkpoint["architecture"])
    config = QUANTIZED_ARCHITECTURES[architecture]
    convention = str(
        checkpoint.get(
            "quantization_convention",
            checkpoint.get("quantization", {}).get("convention", QUANTIZATION_CONVENTION_LEGACY),
        )
    )
    forward_mode = (
        str(checkpoint.get("forward_mode", "quantized"))
        if args.forward_mode == "auto"
        else args.forward_mode
    )
    target_scale = float(checkpoint["target_scale"])
    hidden_scales = [int(value) for value in checkpoint["hidden_scales"]]
    output_scale = int(checkpoint["output_scale"])
    activation = str(checkpoint["activation"])
    total_samples = args.selection_samples + args.ranking_samples
    loader = make_loader(
        args.data,
        config.transform,
        data_format,
        args.split,
        100,
        98,
        99,
        total_samples,
        args.seed,
        args.batch_size,
        args.workers,
        0,
    )
    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    collected = 0
    model.eval()
    for batch in loader:
        features, offsets, _scores, normalized_target, _plies, _results = (
            collate_quantized_sparse_batch(
                batch,
                device=device,
                board_feature_count=config.board_feature_count,
                target_scale=target_scale,
            )
        )
        prediction = regression_forward(
            model,
            features,
            offsets,
            hidden_scales,
            output_scale,
            activation,
            convention,
            forward_mode,
        )
        target = normalized_target * target_scale
        take = min(int(target.numel()), total_samples - collected)
        prediction_chunks.append(prediction[:take].float().cpu().numpy())
        target_chunks.append(target[:take].float().cpu().numpy())
        collected += take
        if collected >= total_samples:
            break
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    if collected != total_samples:
        raise RuntimeError(f"requested {total_samples} samples but collected only {collected}")
    runtime = {
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "epoch": int(checkpoint["epoch"]),
        "activation": activation,
        "quantization_convention": convention,
        "forward_mode": forward_mode,
        "hidden_scales": hidden_scales,
        "output_scale": output_scale,
        "target_scale": target_scale,
    }
    return (
        np.concatenate(prediction_chunks).astype(np.float64, copy=False),
        np.concatenate(target_chunks).astype(np.float64, copy=False),
        runtime,
    )


def calibration_report(
    name: str,
    scale: float,
    bias: float,
    selection_prediction: np.ndarray,
    selection_target: np.ndarray,
    ranking_prediction: np.ndarray,
    ranking_target: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selection_calibrated = scale * selection_prediction + bias
    ranking_calibrated = scale * ranking_prediction + bias
    baseline_error = np.abs(ranking_prediction - ranking_target)
    calibrated_error = np.abs(ranking_calibrated - ranking_target)
    paired = paired_block_bootstrap(
        calibrated_error - baseline_error,
        args.bootstrap_replicates,
        args.bootstrap_block_size,
        args.confidence,
        np.random.default_rng(args.bootstrap_seed + (0 if name == "ols" else 1)),
    )
    bins: list[dict[str, Any]] = []
    abs_target = np.abs(ranking_target)
    edges = args.target_abs_cp_bins
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        mask = (abs_target >= lower) & (abs_target <= upper if index == len(edges) - 2 else abs_target < upper)
        if not np.any(mask):
            continue
        before = float(baseline_error[mask].mean())
        after = float(calibrated_error[mask].mean())
        bins.append(
            {
                "min_abs_target_cp": lower,
                "max_abs_target_cp": upper,
                "samples": int(mask.sum()),
                "before_cp_mae": before,
                "after_cp_mae": after,
                "improvement_cp": before - after,
            }
        )
    return {
        "method": name,
        "affine_scale": scale,
        "affine_bias_cp": bias,
        "selection_before": regression_metrics(selection_target, selection_prediction),
        "selection_after": regression_metrics(selection_target, selection_calibrated),
        "ranking_before": regression_metrics(ranking_target, ranking_prediction),
        "ranking_after": regression_metrics(ranking_target, ranking_calibrated),
        "paired_ranking": paired,
        "ranking_abs_target_bins": bins,
    }


def main() -> None:
    args = parse_args()
    if args.selection_samples <= 0 or args.ranking_samples <= 0:
        raise ValueError("selection/ranking sample counts must be positive")
    if args.batch_size <= 0 or args.workers < 0 or args.torch_threads <= 0:
        raise ValueError("invalid loader/thread settings")
    torch.set_num_threads(args.torch_threads)
    prediction, target, runtime = collect_predictions(args)
    split = args.selection_samples
    selection_prediction, ranking_prediction = prediction[:split], prediction[split:]
    selection_target, ranking_target = target[:split], target[split:]
    ols_scale, ols_bias = fit_ols_affine(selection_prediction, selection_target)
    mae_scale, mae_bias = fit_mae_affine(selection_prediction, selection_target)
    output = {
        "runtime": runtime,
        "data": str(args.data),
        "split": args.split,
        "selection_samples": args.selection_samples,
        "ranking_samples": args.ranking_samples,
        "selection_and_ranking_disjoint": True,
        "baseline_ranking": regression_metrics(ranking_target, ranking_prediction),
        "calibrations": [
            calibration_report(
                "ols", ols_scale, ols_bias,
                selection_prediction, selection_target,
                ranking_prediction, ranking_target, args,
            ),
            calibration_report(
                "mae", mae_scale, mae_bias,
                selection_prediction, selection_target,
                ranking_prediction, ranking_target, args,
            ),
        ],
    }
    rendered = json.dumps(output, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
