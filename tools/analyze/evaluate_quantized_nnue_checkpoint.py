#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.quantized_nnue_architectures import (  # noqa: E402
    QUANTIZED_ARCHITECTURES,
    QUANTIZATION_CONVENTION_LEGACY,
    QuantizedSparseNnueArchitecture,
)
from chess_nnue.train_value import choose_device  # noqa: E402
from chess_nnue.training_targets import TRAINING_PIPELINE_VERSION  # noqa: E402
from tools.train.train_nnue_architecture import make_loader, validate_training_data  # noqa: E402
from tools.train.train_quantized_nnue_architecture import (  # noqa: E402
    collate_quantized_sparse_batch,
    evaluate_quantized_cp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one validation-selected quantized NNUE checkpoint on sealed test data"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--test-max-samples", type=int, default=500_000)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--target-abs-cp-min", type=float, default=None)
    parser.add_argument("--target-abs-cp-max", type=float, default=None)
    parser.add_argument(
        "--target-abs-cp-bins",
        type=str,
        default=None,
        help="Comma-separated absolute-CP bin edges; reports MAE for every adjacent range",
    )
    return parser.parse_args()


def require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"--{name.replace('_', '-')} must be positive")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[QuantizedSparseNnueArchitecture, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("pipeline_version") != TRAINING_PIPELINE_VERSION:
        raise ValueError("checkpoint has legacy or missing pipeline provenance")

    architecture = str(checkpoint.get("architecture"))
    if architecture not in QUANTIZED_ARCHITECTURES:
        raise ValueError(f"unknown checkpoint architecture: {architecture}")
    config = QUANTIZED_ARCHITECTURES[architecture]
    if tuple(checkpoint.get("hidden_sizes", ())) != config.hidden_sizes:
        raise ValueError("checkpoint hidden sizes do not match the current architecture")
    if int(checkpoint.get("feature_count", -1)) != config.feature_count:
        raise ValueError("checkpoint feature count does not match the current architecture")

    quantization = checkpoint.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("checkpoint is missing quantization metadata")
    psqt = checkpoint.get("psqt", {})
    if not isinstance(psqt, dict):
        raise ValueError("checkpoint PSQT metadata must be an object")
    use_psqt = bool(psqt.get("enabled", False))
    psqt_weight_scale_value = psqt.get("weight_scale")
    psqt_master_scale_value = psqt.get("master_scale_to_cp")
    psqt_weight_scale = int(
        psqt_weight_scale_value
        if use_psqt and psqt_weight_scale_value is not None
        else 16
    )
    psqt_master_scale_to_cp = float(
        psqt_master_scale_value
        if use_psqt and psqt_master_scale_value is not None
        else 1.0
    )
    if use_psqt:
        if int(psqt.get("buckets", -1)) != 8:
            raise ValueError("checkpoint must contain exactly 8 PSQT buckets")
        if psqt.get("weight_dtype") != "int32":
            raise ValueError("checkpoint PSQT weights are not int32")
        if psqt.get("perspective") != "stm_minus_opponent_div_2":
            raise ValueError("checkpoint PSQT perspective formula is incompatible")
        if psqt_weight_scale <= 0:
            raise ValueError("checkpoint PSQT weight scale must be positive")
        if psqt_master_scale_to_cp <= 0.0:
            raise ValueError("checkpoint PSQT master scale must be positive")
        master_unit = psqt.get("master_unit")
        if master_unit not in (None, "cp_over_target_scale"):
            raise ValueError(f"unsupported checkpoint PSQT master unit: {master_unit}")
    model = QuantizedSparseNnueArchitecture(
        config,
        hidden_clip=int(checkpoint["hidden_clip"]),
        feature_weight_scale=int(quantization["feature_weight_scale"]),
        linear_weight_scale=int(quantization["linear_weight_scale"]),
        output_weight_scale=int(quantization["output_weight_scale"]),
        screlu_divisor=int(checkpoint.get("screlu_divisor", 255)),
        use_psqt=use_psqt,
        psqt_weight_scale=psqt_weight_scale,
        psqt_master_scale_to_cp=psqt_master_scale_to_cp,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    hidden_scales = [int(value) for value in checkpoint["hidden_scales"]]
    if len(hidden_scales) != len(model.hidden_layers):
        raise ValueError("checkpoint hidden scale count does not match dense hidden layers")
    if any(value <= 0 for value in hidden_scales):
        raise ValueError("checkpoint contains a non-positive hidden scale")
    output_scale = int(checkpoint["output_scale"])
    if output_scale <= 0:
        raise ValueError("checkpoint contains a non-positive output scale")
    model.default_hidden_scales = hidden_scales
    model.default_output_scale = output_scale
    checkpoint["quantization_convention"] = str(
        checkpoint.get(
            "quantization_convention",
            quantization.get("convention", QUANTIZATION_CONVENTION_LEGACY),
        )
    )
    return model, checkpoint


@torch.no_grad()
def evaluate_filtered_cp_mae(
    model: QuantizedSparseNnueArchitecture,
    loader: Any,
    device: torch.device,
    board_feature_count: int,
    target_scale: float,
    hidden_scales: list[int],
    output_scale: int,
    activation: str,
    quantization_convention: str,
    target_abs_cp_min: float,
    target_abs_cp_max: float,
    max_batches: int | None,
) -> tuple[float, int, int]:
    model.eval()
    total_abs_cp = 0.0
    matched_samples = 0
    examined_samples = 0
    for batch_index, samples in enumerate(loader, 1):
        feature_indices, offsets, _scores, target_cp, _plies, _results = (
            collate_quantized_sparse_batch(
                samples,
                device=device,
                board_feature_count=board_feature_count,
                target_scale=target_scale,
            )
        )
        predictions = model(
            feature_indices,
            offsets,
            hidden_scales=hidden_scales,
            output_scale=output_scale,
            activation=activation,
            quantization_convention=quantization_convention,
        ) / float(target_scale)
        target_cp_values = target_cp * float(target_scale)
        mask = (target_cp_values.abs() >= target_abs_cp_min) & (
            target_cp_values.abs() <= target_abs_cp_max
        )
        examined_samples += int(target_cp.shape[0])
        matched_samples += int(mask.sum().item())
        total_abs_cp += float(
            ((predictions[mask] - target_cp[mask]).abs() * float(target_scale)).sum().cpu()
        )
        if max_batches is not None and batch_index >= max_batches:
            break
    if matched_samples == 0:
        raise RuntimeError("target CP filter matched no samples")
    return total_abs_cp / matched_samples, matched_samples, examined_samples


@torch.no_grad()
def evaluate_binned_cp_mae(
    model: QuantizedSparseNnueArchitecture,
    loader: Any,
    device: torch.device,
    board_feature_count: int,
    target_scale: float,
    hidden_scales: list[int],
    output_scale: int,
    activation: str,
    quantization_convention: str,
    bin_edges: list[float],
    max_batches: int | None,
) -> tuple[list[dict[str, float | int]], int]:
    model.eval()
    error_sums = [0.0] * (len(bin_edges) - 1)
    counts = [0] * (len(bin_edges) - 1)
    examined_samples = 0
    for batch_index, samples in enumerate(loader, 1):
        feature_indices, offsets, _scores, target_cp, _plies, _results = (
            collate_quantized_sparse_batch(
                samples,
                device=device,
                board_feature_count=board_feature_count,
                target_scale=target_scale,
            )
        )
        predictions = model(
            feature_indices,
            offsets,
            hidden_scales=hidden_scales,
            output_scale=output_scale,
            activation=activation,
            quantization_convention=quantization_convention,
        ) / float(target_scale)
        target_cp_values = target_cp * float(target_scale)
        abs_targets = target_cp_values.abs()
        abs_errors = (predictions - target_cp).abs() * float(target_scale)
        examined_samples += int(target_cp.shape[0])
        for index, (lower, upper) in enumerate(zip(bin_edges, bin_edges[1:])):
            upper_mask = abs_targets <= upper if index == len(counts) - 1 else abs_targets < upper
            mask = (abs_targets >= lower) & upper_mask
            counts[index] += int(mask.sum().item())
            error_sums[index] += float(abs_errors[mask].sum().cpu())
        if max_batches is not None and batch_index >= max_batches:
            break
    bins = [
        {
            "min_abs_cp": lower,
            "max_abs_cp": upper,
            "samples": count,
            "cp_mae": error_sum / count if count else float("nan"),
        }
        for lower, upper, count, error_sum in zip(
            bin_edges, bin_edges[1:], counts, error_sums
        )
    ]
    return bins, examined_samples


def main() -> None:
    args = parse_args()
    require_positive("batch_size", args.batch_size)
    require_positive("test_max_samples", args.test_max_samples)
    if args.workers < 0 or args.eval_progress_batches < 0:
        raise ValueError("worker/progress counts must be non-negative")
    if args.eval_max_batches is not None:
        require_positive("eval_max_batches", args.eval_max_batches)
    if args.split_mod <= 2:
        raise ValueError("--split-mod must be greater than 2")
    if not 0 <= args.val_mod < args.split_mod or not 0 <= args.test_mod < args.split_mod:
        raise ValueError("validation/test buckets must be inside --split-mod")
    if args.val_mod == args.test_mod:
        raise ValueError("validation and test buckets must differ")
    if (args.target_abs_cp_min is None) != (args.target_abs_cp_max is None):
        raise ValueError("both target CP filter bounds must be provided together")
    if args.target_abs_cp_bins is not None and args.target_abs_cp_min is not None:
        raise ValueError("target CP range and bins are mutually exclusive")
    if args.target_abs_cp_min is not None and (
        args.target_abs_cp_min < 0 or args.target_abs_cp_max < args.target_abs_cp_min
    ):
        raise ValueError("invalid target absolute CP range")
    bin_edges = None
    if args.target_abs_cp_bins is not None:
        bin_edges = [float(value) for value in args.target_abs_cp_bins.split(",")]
        if len(bin_edges) < 2 or bin_edges[0] < 0 or any(
            right <= left for left, right in zip(bin_edges, bin_edges[1:])
        ):
            raise ValueError("target absolute CP bin edges must be non-negative and increasing")
    args.data_format = validate_training_data(args.data, args.data_format)

    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    architecture = str(checkpoint["architecture"])
    config = QUANTIZED_ARCHITECTURES[architecture]
    test_loader = make_loader(
        args.data,
        config.transform,
        args.data_format,
        "test",
        args.split_mod,
        args.val_mod,
        args.test_mod,
        args.test_max_samples,
        args.seed,
        args.batch_size,
        args.workers,
        0,
    )

    selected_epoch = int(checkpoint["epoch"])
    hidden_scales = [int(value) for value in checkpoint["hidden_scales"]]
    output_scale = int(checkpoint["output_scale"])
    target_scale = float(checkpoint["target_scale"])
    score_lambda = float(checkpoint["score_lambda"])
    wdl_loss_exponent = float(checkpoint["wdl_loss_exponent"])
    activation = str(checkpoint["activation"])
    quantization_convention = str(checkpoint["quantization_convention"])
    started = time.monotonic()
    filtered = args.target_abs_cp_min is not None
    binned = bin_edges is not None
    cp_bins = None
    if binned:
        cp_bins, examined_samples = evaluate_binned_cp_mae(
            model,
            test_loader,
            device,
            config.board_feature_count,
            target_scale,
            hidden_scales,
            output_scale,
            activation,
            quantization_convention,
            bin_edges,
            args.eval_max_batches,
        )
        test_loss = None
        test_cp_bins = cp_bins
        test_cp = sum(
            float(item["cp_mae"]) * int(item["samples"])
            for item in cp_bins
            if int(item["samples"]) > 0
        ) / sum(int(item["samples"]) for item in cp_bins)
        test_samples = sum(int(item["samples"]) for item in cp_bins)
    elif filtered:
        test_cp, test_samples, examined_samples = evaluate_filtered_cp_mae(
            model,
            test_loader,
            device,
            config.board_feature_count,
            target_scale,
            hidden_scales,
            output_scale,
            activation,
            quantization_convention,
            args.target_abs_cp_min,
            args.target_abs_cp_max,
            args.eval_max_batches,
        )
        test_loss = None
        test_cp_bins = None
    else:
        test_loss, test_cp, test_samples, test_cp_bins = evaluate_quantized_cp(
            architecture,
            "test",
            model,
            test_loader,
            device,
            config.board_feature_count,
            target_scale,
            score_lambda,
            wdl_loss_exponent,
            hidden_scales,
            output_scale,
            args.eval_max_batches,
            selected_epoch,
            args.eval_progress_batches,
            started,
            activation,
            quantization_convention,
            float(checkpoint.get("cp_huber_delta", 200.0)),
        )
        examined_samples = test_samples
    if (test_loss is not None and not math.isfinite(test_loss)) or not math.isfinite(test_cp):
        raise RuntimeError("test evaluation produced a non-finite metric")
    print(
        json.dumps(
            {
                "event": "final_test",
                "arch": architecture,
                "activation": activation,
                "quantization_convention": quantization_convention,
                "checkpoint": str(args.checkpoint),
                "selected_epoch": selected_epoch,
                "best_val_loss": round(float(checkpoint["metrics"]["val_loss"]), 8),
                "hidden_scales": hidden_scales,
                "output_scale": output_scale,
                "test_samples": test_samples,
                "examined_samples": examined_samples,
                "target_abs_cp_range": (
                    [args.target_abs_cp_min, args.target_abs_cp_max] if filtered else None
                ),
                "target_abs_cp_bins": cp_bins if binned else None,
                "test_cp_bins": test_cp_bins,
                "test_loss": None if test_loss is None else round(test_loss, 8),
                "test_val_cp": round(test_cp, 4),
                "test_evaluated": True,
                "elapsed_sec": round(time.monotonic() - started, 3),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
