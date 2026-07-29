#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.quantized_nnue_architectures import (  # noqa: E402
    QUANTIZATION_CONVENTION_LEGACY,
    QUANTIZED_ARCHITECTURES,
)
from nn.train_value import choose_device  # noqa: E402
from tools.evaluate_quantized_nnue_checkpoint import load_model  # noqa: E402
from tools.train_nnue_architecture import make_loader, validate_training_data  # noqa: E402
from tools.train_quantized_nnue_architecture import (  # noqa: E402
    collate_quantized_sparse_batch,
    regression_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CP calibration on a fixed training or validation probe"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=("auto", "cbin", "jsonl"), default="auto")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward-mode", choices=("auto", "float", "quantized"), default="auto")
    parser.add_argument("--expected-epoch", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("sample/batch counts must be positive and workers non-negative")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    data_format = validate_training_data(args.data, args.data_format)
    model, checkpoint = load_model(args.checkpoint, device)
    epoch = int(checkpoint.get("epoch", -1))
    if args.expected_epoch is not None and epoch != args.expected_epoch:
        raise RuntimeError(
            f"checkpoint epoch changed while evaluating: expected {args.expected_epoch}, got {epoch}"
        )
    architecture = str(checkpoint["architecture"])
    config = QUANTIZED_ARCHITECTURES[architecture]
    hidden_scales = [int(value) for value in checkpoint["hidden_scales"]]
    output_scale = int(checkpoint["output_scale"])
    activation = str(checkpoint["activation"])
    convention = str(
        checkpoint.get(
            "quantization_convention",
            checkpoint.get("quantization", {}).get(
                "convention", QUANTIZATION_CONVENTION_LEGACY
            ),
        )
    )
    forward_mode = (
        str(checkpoint.get("forward_mode", "quantized"))
        if args.forward_mode == "auto"
        else args.forward_mode
    )
    target_scale = float(checkpoint["target_scale"])
    loader = make_loader(
        args.data,
        config.transform,
        data_format,
        args.split,
        100,
        98,
        99,
        args.max_samples,
        20260718,
        args.batch_size,
        args.workers,
        0,
    )

    samples = 0
    error_sum = 0.0
    sum_target = 0.0
    sum_prediction = 0.0
    sum_target2 = 0.0
    sum_target_prediction = 0.0
    low_count = 0
    low_error_sum = 0.0
    high_count = 0
    high_error_sum = 0.0
    for batch in loader:
        features, offsets, _scores, normalized_target, _plies, _results = (
            collate_quantized_sparse_batch(
                batch, device, config.board_feature_count, target_scale
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
        ).double()
        target = (normalized_target * target_scale).double()
        error = (prediction - target).abs()
        count = target.numel()
        samples += count
        error_sum += float(error.sum())
        sum_target += float(target.sum())
        sum_prediction += float(prediction.sum())
        sum_target2 += float((target * target).sum())
        sum_target_prediction += float((target * prediction).sum())
        low = target.abs() < 100.0
        high = target.abs() > 1000.0
        low_count += int(low.sum())
        high_count += int(high.sum())
        low_error_sum += float(error[low].sum())
        high_error_sum += float(error[high].sum())

    target_mean = sum_target / samples
    prediction_mean = sum_prediction / samples
    target_variance_sum = sum_target2 - samples * target_mean * target_mean
    covariance_sum = (
        sum_target_prediction - samples * target_mean * prediction_mean
    )
    slope = covariance_sum / target_variance_sum
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "data": str(args.data),
        "split": args.split,
        "forward_mode": forward_mode,
        "loss_type": checkpoint.get("loss_type"),
        "samples": samples,
        "cp_mae": error_sum / samples,
        "slope": slope,
        "intercept_cp": prediction_mean - slope * target_mean,
        "target_mean_cp": target_mean,
        "prediction_mean_cp": prediction_mean,
        "mae_abs_target_lt_100": low_error_sum / low_count,
        "samples_abs_target_lt_100": low_count,
        "mae_abs_target_gt_1000": high_error_sum / high_count,
        "samples_abs_target_gt_1000": high_count,
    }
    encoded = json.dumps(result, separators=(",", ":"))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
