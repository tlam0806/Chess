#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
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
from tools.diagnose_quantized_nnue_pipeline import bucket_metrics  # noqa: E402
from tools.evaluate_quantized_nnue_checkpoint import load_model  # noqa: E402
from tools.train_nnue_architecture import make_loader, validate_training_data  # noqa: E402
from tools.train_quantized_nnue_architecture import (  # noqa: E402
    collate_quantized_sparse_batch,
    measure_hidden_saturation,
)


def parse_edges(value: str) -> list[float]:
    edges = [float(part) for part in value.replace(",", " ").split()]
    if len(edges) < 2 or any(left >= right for left, right in zip(edges, edges[1:])):
        raise argparse.ArgumentTypeError("bucket edges must be strictly increasing")
    return edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare float and fake-quantized PyTorch inference on identical records"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=("auto", "cbin", "jsonl"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--signed-bins",
        type=parse_edges,
        default=parse_edges("-2000,-1600,-1000,-600,-300,-100,0,100,300,600,1000,1600,2000"),
    )
    parser.add_argument(
        "--prediction-histogram-bins",
        type=parse_edges,
        default=parse_edges("-4000,-2000,-1600,-1000,-600,-300,-100,0,100,300,600,1000,1600,2000,4000"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def signed_buckets(
    target: torch.Tensor,
    prediction: torch.Tensor,
    edges: list[float],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        upper_mask = target <= upper if index == len(edges) - 2 else target < upper
        mask = (target >= lower) & upper_mask
        if not mask.any():
            rows.append({"target_min": lower, "target_max": upper, "samples": 0})
            continue
        bucket_target = target[mask]
        bucket_prediction = prediction[mask]
        residual = bucket_prediction - bucket_target
        rows.append(
            {
                "target_min": lower,
                "target_max": upper,
                "samples": int(mask.sum()),
                "mean_target": float(bucket_target.mean()),
                "mean_prediction": float(bucket_prediction.mean()),
                "mean_residual": float(residual.mean()),
                "cp_mae": float(residual.abs().mean()),
                "prediction_std": float(bucket_prediction.std(unbiased=False)),
            }
        )
    return rows


def histogram(values: torch.Tensor, edges: list[float]) -> list[dict[str, float | int]]:
    counts = torch.histc(
        values.float(),
        bins=len(edges) - 1,
        min=float(edges[0]),
        max=float(edges[-1]),
    )
    # torch.histc uses equal-width bins, so use explicit masks for arbitrary edges.
    result: list[dict[str, float | int]] = []
    del counts
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        upper_mask = values <= upper if index == len(edges) - 2 else values < upper
        count = int(((values >= lower) & upper_mask).sum())
        result.append({"min": lower, "max": upper, "samples": count})
    return result


def mode_report(
    target: torch.Tensor,
    prediction: torch.Tensor,
    raw_output: torch.Tensor,
    signed_edges: list[float],
    histogram_edges: list[float],
) -> dict[str, object]:
    quantiles = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "overall": bucket_metrics(target, prediction),
        "signed_buckets": signed_buckets(target, prediction, signed_edges),
        "prediction": {
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "mean": float(prediction.mean()),
            "std": float(prediction.std(unbiased=False)),
            "quantiles": [float(value) for value in torch.quantile(prediction, quantiles)],
            "histogram": histogram(prediction, histogram_edges),
        },
        "raw_output_before_divide": {
            "min": float(raw_output.min()),
            "max": float(raw_output.max()),
            "mean": float(raw_output.mean()),
            "std": float(raw_output.std(unbiased=False)),
        },
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_samples <= 0 or args.workers < 0:
        raise ValueError("batch size/max samples must be positive and workers non-negative")
    device = choose_device(args.device)
    data_format = validate_training_data(args.data, args.data_format)
    model, checkpoint = load_model(args.checkpoint, device)
    model.eval()
    architecture = str(checkpoint["architecture"])
    config = QUANTIZED_ARCHITECTURES[architecture]
    hidden_scales = [int(value) for value in checkpoint["hidden_scales"]]
    output_scale = int(checkpoint["output_scale"])
    activation = str(checkpoint["activation"])
    convention = str(
        checkpoint.get(
            "quantization_convention",
            checkpoint.get("quantization", {}).get("convention", QUANTIZATION_CONVENTION_LEGACY),
        )
    )
    target_scale = float(checkpoint["target_scale"])

    def loader():
        return make_loader(
            args.data,
            config.transform,
            data_format,
            args.split,
            args.split_mod,
            args.val_mod,
            args.test_mod,
            args.max_samples,
            args.seed,
            args.batch_size,
            args.workers,
            0,
        )

    targets: list[torch.Tensor] = []
    quantized_predictions: list[torch.Tensor] = []
    float_predictions: list[torch.Tensor] = []
    quantized_positional_predictions: list[torch.Tensor] = []
    quantized_psqt_predictions: list[torch.Tensor] = []
    float_positional_predictions: list[torch.Tensor] = []
    float_psqt_predictions: list[torch.Tensor] = []
    quantized_raw: list[torch.Tensor] = []
    float_raw: list[torch.Tensor] = []
    for samples in loader():
        features, offsets, _scores, normalized_target, _plies, _results = (
            collate_quantized_sparse_batch(
                samples,
                device,
                config.board_feature_count,
                target_scale,
            )
        )
        quantized = model(
            features,
            offsets,
            hidden_scales,
            output_scale,
            activation,
            convention,
        )
        quantized_predivide = model(
            features,
            offsets,
            hidden_scales,
            1,
            activation,
            convention,
        )
        floating = model.forward_float(
            features,
            offsets,
            hidden_scales,
            output_scale,
            activation,
            convention,
        )
        if model.psqt is None:
            quantized_psqt = torch.zeros_like(quantized)
            float_psqt = torch.zeros_like(floating)
        else:
            quantized_psqt = torch.trunc(
                model.psqt_accumulator(features, offsets, quantized=True)
                / float(2 * model.psqt_weight_scale)
            )
            float_psqt = (
                model.psqt_accumulator(features, offsets, quantized=False)
                / float(2 * model.psqt_weight_scale)
            )
        quantized_positional = quantized - quantized_psqt
        float_positional = floating - float_psqt
        targets.append((normalized_target * target_scale).cpu().float())
        quantized_predictions.append(quantized.cpu().float())
        float_predictions.append(floating.cpu().float())
        quantized_positional_predictions.append(quantized_positional.cpu().float())
        quantized_psqt_predictions.append(quantized_psqt.cpu().float())
        float_positional_predictions.append(float_positional.cpu().float())
        float_psqt_predictions.append(float_psqt.cpu().float())
        quantized_raw.append(
            (quantized_predivide - quantized_psqt).cpu().float()
        )
        float_raw.append((float_positional * float(output_scale)).cpu().float())

    target = torch.cat(targets)
    quantized = torch.cat(quantized_predictions)
    floating = torch.cat(float_predictions)
    quantized_positional = torch.cat(quantized_positional_predictions)
    quantized_psqt = torch.cat(quantized_psqt_predictions)
    float_positional = torch.cat(float_positional_predictions)
    float_psqt = torch.cat(float_psqt_predictions)
    delta = quantized - floating
    with contextlib.redirect_stdout(io.StringIO()):
        saturation = measure_hidden_saturation(
            architecture,
            model,
            loader(),
            device,
            config.board_feature_count,
            target_scale,
            hidden_scales,
            math.ceil(args.max_samples / args.batch_size),
            activation,
            int(checkpoint.get("epoch", -1)),
            convention,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "samples": int(target.numel()),
        "architecture": architecture,
        "activation": activation,
        "quantization_convention": convention,
        "hidden_clip": int(checkpoint["hidden_clip"]),
        "screlu_divisor": int(checkpoint.get("screlu_divisor", 255)),
        "hidden_scales": hidden_scales,
        "output_scale": output_scale,
        "float_pytorch": mode_report(
            target,
            floating,
            torch.cat(float_raw),
            args.signed_bins,
            args.prediction_histogram_bins,
        ),
        "quantized_python": mode_report(
            target,
            quantized,
            torch.cat(quantized_raw),
            args.signed_bins,
            args.prediction_histogram_bins,
        ),
        "quantized_components": {
            "positional": mode_report(
                target,
                quantized_positional,
                torch.cat(quantized_raw),
                args.signed_bins,
                args.prediction_histogram_bins,
            ),
            "psqt": mode_report(
                target,
                quantized_psqt,
                quantized_psqt,
                args.signed_bins,
                args.prediction_histogram_bins,
            ),
        },
        "float_components": {
            "positional": mode_report(
                target,
                float_positional,
                torch.cat(float_raw),
                args.signed_bins,
                args.prediction_histogram_bins,
            ),
            "psqt": mode_report(
                target,
                float_psqt,
                float_psqt,
                args.signed_bins,
                args.prediction_histogram_bins,
            ),
        },
        "quantization_delta_cp": {
            "mean": float(delta.mean()),
            "mae": float(delta.abs().mean()),
            "max_abs": float(delta.abs().max()),
            "exact_match_rate": float((delta == 0).float().mean()),
        },
        "hidden_saturation": saturation,
    }
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
