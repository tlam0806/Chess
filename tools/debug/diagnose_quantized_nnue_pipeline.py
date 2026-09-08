#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.quantized_nnue_architectures import QUANTIZED_ARCHITECTURES  # noqa: E402
from chess_nnue.train_value import choose_device  # noqa: E402
from tools.analyze.evaluate_quantized_nnue_checkpoint import load_model  # noqa: E402
from tools.train.train_nnue_architecture import make_loader, validate_training_data  # noqa: E402
from tools.train.train_quantized_nnue_architecture import collate_quantized_sparse_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed CP diagnostics for a quantized NNUE checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=("auto", "cbin", "jsonl"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument("--bins", default="0,100,300,600,1000,1600,2000")
    return parser.parse_args()


def bucket_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float | int]:
    count = target.numel()
    if count == 0:
        return {"samples": 0}
    error = prediction - target
    target_centered = target - target.mean()
    pred_centered = prediction - prediction.mean()
    target_var = (target_centered.square()).sum()
    covariance = (target_centered * pred_centered).sum()
    slope = covariance / target_var if target_var > 0 else torch.tensor(float("nan"))
    intercept = prediction.mean() - slope * target.mean()
    pearson_denom = torch.sqrt(target_var * pred_centered.square().sum())
    pearson = covariance / pearson_denom if pearson_denom > 0 else torch.tensor(float("nan"))
    nonzero = target != 0
    sign_accuracy = ((target[nonzero] * prediction[nonzero]) > 0).float().mean() if nonzero.any() else torch.tensor(float("nan"))
    result: dict[str, float | int] = {
        "samples": count,
        "model_mae": float(error.abs().mean()),
        "zero_mae": float(target.abs().mean()),
        "negated_mae": float((-prediction - target).abs().mean()),
        "mean_target": float(target.mean()),
        "mean_abs_target": float(target.abs().mean()),
        "mean_prediction": float(prediction.mean()),
        "mean_abs_prediction": float(prediction.abs().mean()),
        "prediction_std": float(prediction.std(unbiased=False)),
        "sign_accuracy": float(sign_accuracy),
        "pearson": float(pearson),
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
    }
    for threshold in (50, 100, 300, 600):
        result[f"fraction_abs_prediction_lt_{threshold}"] = float((prediction.abs() < threshold).float().mean())
    return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_samples <= 0 or args.workers < 0:
        raise ValueError("batch size/max samples must be positive and workers non-negative")
    edges = [float(value) for value in args.bins.split(",")]
    if len(edges) < 2 or edges[0] < 0 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("bin edges must be non-negative and strictly increasing")
    data_format = validate_training_data(args.data, args.data_format)
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = QUANTIZED_ARCHITECTURES[str(checkpoint["architecture"])]
    loader = make_loader(
        args.data, config.transform, data_format, args.split,
        args.split_mod, args.val_mod, args.test_mod, args.max_samples,
        20260717, args.batch_size, args.workers, 0,
    )
    target_chunks: list[torch.Tensor] = []
    prediction_chunks: list[torch.Tensor] = []
    raw_score_chunks: list[torch.Tensor] = []
    target_scale = float(checkpoint["target_scale"])
    for samples in loader:
        features, offsets, scores, targets, _plies, _results = collate_quantized_sparse_batch(
            samples, device, config.board_feature_count, target_scale
        )
        predictions = model(
            features, offsets,
            hidden_scales=[int(v) for v in checkpoint["hidden_scales"]],
            output_scale=int(checkpoint["output_scale"]),
            activation=str(checkpoint["activation"]),
            quantization_convention=str(checkpoint["quantization_convention"]),
        )
        raw_score_chunks.append(scores.cpu().double())
        target_chunks.append((targets * target_scale).cpu().double())
        prediction_chunks.append(predictions.cpu().double())
    target = torch.cat(target_chunks)
    prediction = torch.cat(prediction_chunks)
    raw_score = torch.cat(raw_score_chunks)
    rows = []
    abs_target = target.abs()
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        upper_mask = abs_target <= upper if index == len(edges) - 2 else abs_target < upper
        mask = (abs_target >= lower) & upper_mask
        row = {"min_abs_cp": lower, "max_abs_cp": upper, **bucket_metrics(target[mask], prediction[mask])}
        if mask.any():
            row.update({
                "positive_targets": int((target[mask] > 0).sum()),
                "negative_targets": int((target[mask] < 0).sum()),
                "exact_positive_2000": int((target[mask] == 2000).sum()),
                "exact_negative_2000": int((target[mask] == -2000).sum()),
                "raw_abs_cp_gt_2000": int(((raw_score[mask].abs() * (100.0 / 208.0)) > 2000).sum()),
            })
        rows.append(row)
    quantiles = torch.tensor([0, .01, .05, .25, .5, .75, .95, .99, 1.0], dtype=torch.double)
    output = {
        "checkpoint": str(args.checkpoint),
        "samples": target.numel(),
        "architecture": str(checkpoint["architecture"]),
        "activation": str(checkpoint["activation"]),
        "screlu_divisor": int(checkpoint.get("screlu_divisor", 255)),
        "target_scale": target_scale,
        "hidden_scales": checkpoint["hidden_scales"],
        "output_scale": checkpoint["output_scale"],
        "overall": bucket_metrics(target, prediction),
        "prediction_cp": {
            "min": float(prediction.min()), "max": float(prediction.max()),
            "mean": float(prediction.mean()), "std": float(prediction.std(unbiased=False)),
            "percentiles": [float(v) for v in torch.quantile(prediction, quantiles)],
        },
        "model_output_normalized": {
            "min": float(prediction.min() / target_scale), "max": float(prediction.max() / target_scale),
            "mean": float(prediction.mean() / target_scale), "std": float(prediction.std(unbiased=False) / target_scale),
        },
        "buckets": rows,
    }
    if not math.isfinite(float(output["overall"]["pearson"])):
        raise RuntimeError("non-finite overall correlation")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
