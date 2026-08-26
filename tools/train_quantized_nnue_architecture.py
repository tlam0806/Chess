#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.quantized_nnue_architectures import (  # noqa: E402
    ACTIVATION_CHOICES,
    ACTIVATION_RELU,
    FEATURE_WEIGHT_SCALE,
    HIDDEN_CLIP,
    LINEAR_WEIGHT_SCALE,
    OUTPUT_WEIGHT_SCALE,
    PSQT_WEIGHT_SCALE,
    SCRELU_DIVISOR,
    QUANTIZATION_CONVENTION_CHOICES,
    QUANTIZATION_CONVENTION_LEGACY,
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    QuantizedSparseNnueArchitecture,
    activation_kind,
    activation_numeric_clip,
    activation_output_scale,
    quantization_scale_audit,
    scale_from_positive_percentile,
)
from nn.training_targets import (  # noqa: E402
    DEFAULT_SCORE_LAMBDA,
    DEFAULT_WDL_LOSS_EXPONENT,
    TRAINING_PIPELINE_VERSION,
    is_value_none_target,
    stockfish_score_to_cp,
)
from nn.train_value import DEFAULT_TARGET_SCALE, choose_device  # noqa: E402
from nn.value_net import AUX_FEATURE_COUNT  # noqa: E402
from tools.train_nnue_architecture import (  # noqa: E402
    cp_huber_loss,
    git_provenance,
    make_loader,
    validate_training_data,
)


CP_MAE_BINS = (
    (0.0, 100.0),
    (100.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
    (1600.0, 2000.0),
)

FORWARD_MODES = ("quantized", "float")
CP_LOSS_TYPES = ("cp_huber", "cp_mse", "cp_mse_huber")


def regression_forward(
    model: QuantizedSparseNnueArchitecture,
    feature_indices: torch.Tensor,
    offsets: torch.Tensor,
    hidden_scales: list[int],
    output_scale: int,
    activation: str,
    quantization_convention: str,
    forward_mode: str,
) -> torch.Tensor:
    if forward_mode == "quantized":
        forward = model.forward
    elif forward_mode == "float":
        forward = model.forward_float
    else:
        raise ValueError(f"unknown forward mode: {forward_mode}")
    return forward(
        feature_indices,
        offsets,
        hidden_scales=hidden_scales,
        output_scale=output_scale,
        activation=activation,
        quantization_convention=quantization_convention,
    )


def cp_regression_loss(
    predictions: torch.Tensor,
    target_scores: torch.Tensor,
    target_scale: float,
    loss_type: str,
    cp_huber_delta: float,
    mse_mix_weight: float = 0.75,
) -> torch.Tensor:
    if loss_type == "cp_huber":
        return cp_huber_loss(
            predictions, target_scores, target_scale, cp_huber_delta
        )
    if loss_type == "cp_mse":
        target_cp = stockfish_score_to_cp(target_scores) / float(target_scale)
        return nn.functional.mse_loss(predictions, target_cp)
    if loss_type == "cp_mse_huber":
        if not 0.0 <= mse_mix_weight <= 1.0:
            raise ValueError("MSE mix weight must be in [0, 1]")
        target_cp = stockfish_score_to_cp(target_scores) / float(target_scale)
        mse = nn.functional.mse_loss(predictions, target_cp)
        huber = cp_huber_loss(
            predictions, target_scores, target_scale, cp_huber_delta
        )
        return mse_mix_weight * mse + (1.0 - mse_mix_weight) * huber
    raise ValueError(f"unknown CP loss type: {loss_type}")


def scheduled_learning_rate(
    schedule: str,
    step: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if schedule == "constant":
        return peak_lr
    if schedule != "cosine":
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    if total_steps <= 0 or step < 0 or peak_lr <= 0.0 or min_lr <= 0.0:
        raise ValueError("invalid cosine learning-rate parameters")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup steps must be in [0, total_steps)")
    if min_lr > peak_lr:
        raise ValueError("minimum learning rate cannot exceed peak learning rate")
    if warmup_steps > 0 and step < warmup_steps:
        fraction = float(step + 1) / float(warmup_steps)
        return min_lr + (peak_lr - min_lr) * fraction
    decay_count = total_steps - warmup_steps
    if decay_count == 1:
        return min_lr
    decay_steps = decay_count - 1
    decay_step = min(max(0, step - warmup_steps), decay_steps)
    progress = float(decay_step) / float(decay_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def validate_epoch_learning_rates(
    peak_lrs: list[float] | None,
    min_lrs: list[float] | None,
    epochs: int,
) -> None:
    if (peak_lrs is None) != (min_lrs is None):
        raise ValueError("--epoch-peak-lrs and --epoch-min-lrs must be used together")
    if peak_lrs is None or min_lrs is None:
        return
    if len(peak_lrs) != epochs or len(min_lrs) != epochs:
        raise ValueError("per-epoch learning-rate lists must contain exactly --epochs values")
    if any(value <= 0.0 for value in [*peak_lrs, *min_lrs]):
        raise ValueError("per-epoch learning rates must be positive")
    for epoch, (peak_lr, min_lr) in enumerate(zip(peak_lrs, min_lrs), 1):
        if min_lr > peak_lr:
            raise ValueError(
                f"epoch {epoch} minimum learning rate cannot exceed its peak"
            )


def sampled_quantile(
    values: torch.Tensor,
    quantile: float,
    max_values: int = 1_000_000,
) -> float:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("cannot compute a quantile of an empty tensor")
    if max_values <= 0:
        raise ValueError("max_values must be positive")
    stride = max(1, math.ceil(flat.numel() / max_values))
    sample = flat[::stride]
    return float(torch.quantile(sample.float(), quantile).item())


def collate_quantized_sparse_batch(
    samples: list[dict[str, Any]],
    device: torch.device,
    board_feature_count: int,
    target_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(is_value_none_target(sample["score"]) for sample in samples):
        raise ValueError("VALUE_NONE target cannot be used for training or evaluation")
    flat_features: list[int] = []
    offsets: list[int] = []
    cursor = 0
    scores: list[int] = []
    plies: list[int] = []
    results: list[int] = []
    for sample in samples:
        offsets.append(cursor)
        features = list(sample["features"])
        aux = sample["aux"]
        if len(aux) != AUX_FEATURE_COUNT:
            raise ValueError(f"expected {AUX_FEATURE_COUNT} aux values, got {len(aux)}")
        for index, value in enumerate(aux):
            if value:
                features.append(board_feature_count + index)
        features.sort()
        flat_features.extend(features)
        cursor += len(features)
        scores.append(int(sample["score"]))
        plies.append(int(sample["ply"]))
        results.append(int(sample["result"]))
    score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    return (
        torch.tensor(flat_features, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        score_tensor,
        stockfish_score_to_cp(score_tensor) / float(target_scale),
        torch.tensor(plies, dtype=torch.float32, device=device),
        torch.tensor(results, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def quantized_hidden_step(
    hidden: torch.Tensor,
    layer: nn.Linear,
    scale: int,
    clip: int,
    weight_scale: int,
    activation: str,
    layer_index: int,
    screlu_divisor: int,
    bias_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.clamp(
        torch.round(layer.weight * float(weight_scale)),
        -127,
        127,
    )
    if bias_scale is None:
        bias_scale = float(clip * weight_scale)
    bias = torch.round(layer.bias * float(bias_scale))
    acc = nn.functional.linear(hidden, weight, bias)
    numeric_clip = activation_numeric_clip(activation, layer_index, clip)
    clipped = torch.clamp(torch.trunc(acc / float(scale)), 0.0, float(numeric_clip))
    if activation_kind(activation, layer_index) == "screlu8":
        clipped = torch.trunc((clipped * clipped) / float(screlu_divisor))
    next_hidden = clipped
    return acc, next_hidden


@torch.no_grad()
def initial_hidden1_accumulator(
    model: QuantizedSparseNnueArchitecture,
    feature_indices: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return model.first_hidden_accumulator(feature_indices, offsets)


@torch.no_grad()
def initial_hidden1(
    model: QuantizedSparseNnueArchitecture,
    feature_indices: torch.Tensor,
    offsets: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    hidden = initial_hidden1_accumulator(model, feature_indices, offsets)
    return model._activate_hidden(hidden, activation, 0)


@torch.no_grad()
def calibrate_hidden_scales(
    arch: str,
    model: QuantizedSparseNnueArchitecture,
    loader: DataLoader,
    device: torch.device,
    board_feature_count: int,
    target_scale: float,
    max_batches: int | None,
    percentile: float,
    activation: str,
    epoch: int,
    quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
) -> tuple[list[int], list[dict[str, float]]]:
    if not model.hidden_layers:
        return [], []

    hidden_batches: list[torch.Tensor] = []
    for batch_index, samples in enumerate(loader, 1):
        feature_indices, offsets, _scores, _targets, _plies, _results = collate_quantized_sparse_batch(
            samples,
            device=device,
            board_feature_count=board_feature_count,
            target_scale=target_scale,
        )
        hidden_batches.append(initial_hidden1(model, feature_indices, offsets, activation).cpu())
        if max_batches is not None and batch_index >= max_batches:
            break
    if not hidden_batches:
        raise RuntimeError("calibration loader produced no samples")

    scales: list[int] = []
    stats: list[dict[str, float]] = []
    current_batches = hidden_batches
    activation_scale = activation_output_scale(
        float(model.feature_weight_scale),
        activation,
        0,
        model.screlu_divisor,
        model.hidden_clip,
        quantization_convention,
    )
    for layer_index, layer in enumerate(model.hidden_layers):
        acc_batches: list[torch.Tensor] = []
        next_batches: list[torch.Tensor] = []
        accumulator_scale = activation_scale * float(model.linear_weight_scale)
        bias_scale = (
            accumulator_scale
            if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
            else None
        )
        for hidden_cpu in current_batches:
            hidden = hidden_cpu.to(device)
            acc, _unused = quantized_hidden_step(
                hidden,
                layer,
                1,
                model.hidden_clip,
                model.linear_weight_scale,
                activation,
                layer_index + 1,
                model.screlu_divisor,
                bias_scale,
            )
            acc_batches.append(acc.cpu())
        all_acc = torch.cat([batch.reshape(-1) for batch in acc_batches])
        scale = scale_from_positive_percentile(all_acc, percentile, model.hidden_clip)
        scales.append(scale)
        positive = all_acc[all_acc > 0]
        saturation_threshold = float(scale * model.hidden_clip)
        saturation = float((positive > saturation_threshold).float().mean().item()) if positive.numel() else 0.0
        zero = float((all_acc <= 0).float().mean().item()) if all_acc.numel() else 0.0
        stats.append(
            {
                "layer": layer_index + 2,
                "scale": scale,
                "positive_count": float(positive.numel()),
                "zero_rate": zero,
                "positive_saturation_rate": saturation,
                "acc_mean": float(all_acc.float().mean().item()),
                "acc_p99": float(torch.quantile(all_acc.float(), 0.99).item()),
            }
        )

        for hidden_cpu in current_batches:
            hidden = hidden_cpu.to(device)
            _acc, next_hidden = quantized_hidden_step(
                hidden,
                layer,
                scale,
                model.hidden_clip,
                model.linear_weight_scale,
                activation,
                layer_index + 1,
                model.screlu_divisor,
                bias_scale,
            )
            next_batches.append(next_hidden.cpu())
        current_batches = next_batches
        activation_scale = activation_output_scale(
            accumulator_scale / float(scale),
            activation,
            layer_index + 1,
            model.screlu_divisor,
            model.hidden_clip,
            quantization_convention,
        )

    print(
        json.dumps(
            {
                "event": "calibrate_hidden_scales",
                "arch": arch,
                "activation": activation,
                "quantization_convention": quantization_convention,
                "epoch": epoch,
                "percentile": percentile,
                "scales": scales,
                "stats": stats,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return scales, stats


@torch.no_grad()
def measure_hidden_saturation(
    arch: str,
    model: QuantizedSparseNnueArchitecture,
    loader: DataLoader,
    device: torch.device,
    board_feature_count: int,
    target_scale: float,
    hidden_scales: list[int],
    max_batches: int | None,
    activation: str,
    epoch: int,
    quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
) -> list[dict[str, float]]:
    if not model.hidden_layers:
        return []

    first_acc_batches: list[torch.Tensor] = []
    hidden_batches: list[torch.Tensor] = []
    for batch_index, samples in enumerate(loader, 1):
        feature_indices, offsets, _scores, _targets, _plies, _results = collate_quantized_sparse_batch(
            samples,
            device=device,
            board_feature_count=board_feature_count,
            target_scale=target_scale,
        )
        first_acc_batches.append(
            initial_hidden1_accumulator(model, feature_indices, offsets).cpu()
        )
        hidden_batches.append(initial_hidden1(model, feature_indices, offsets, activation).cpu())
        if max_batches is not None and batch_index >= max_batches:
            break
    if not hidden_batches:
        raise RuntimeError("saturation loader produced no samples")

    first_acc_values = torch.cat([batch.reshape(-1) for batch in first_acc_batches])
    first_hidden_values = torch.cat([batch.reshape(-1) for batch in hidden_batches])
    stats: list[dict[str, float]] = [
        {
            "layer": 1.0,
            "scale": 1.0,
            "zero_rate": float((first_hidden_values <= 0).float().mean().item()),
            # Saturation must be measured before ScReLU. After x*x/divisor,
            # divisor=256 has a maximum output of 254 and would always report 0.
            "clip_rate": float(
                (
                    first_acc_values
                    >= activation_numeric_clip(activation, 0, model.hidden_clip)
                )
                .float()
                .mean()
                .item()
            ),
            "over_u8_rate": float(
                (first_acc_values >= model.hidden_clip).float().mean().item()
            ),
            "active_mean": float(first_hidden_values.float().mean().item()),
            "acc_mean": float(first_acc_values.float().mean().item()),
            "acc_p99": sampled_quantile(first_acc_values, 0.99),
            "acc_min": float(first_acc_values.min().item()),
            "acc_max": float(first_acc_values.max().item()),
        }
    ]
    current_batches = hidden_batches
    activation_scale = activation_output_scale(
        float(model.feature_weight_scale),
        activation,
        0,
        model.screlu_divisor,
        model.hidden_clip,
        quantization_convention,
    )
    for layer_index, (scale, layer) in enumerate(zip(hidden_scales, model.hidden_layers), 1):
        all_acc: list[torch.Tensor] = []
        all_hidden: list[torch.Tensor] = []
        next_batches: list[torch.Tensor] = []
        accumulator_scale = activation_scale * float(model.linear_weight_scale)
        bias_scale = (
            accumulator_scale
            if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
            else None
        )
        for hidden_cpu in current_batches:
            hidden = hidden_cpu.to(device)
            acc, next_hidden = quantized_hidden_step(
                hidden,
                layer,
                scale,
                model.hidden_clip,
                model.linear_weight_scale,
                activation,
                layer_index,
                model.screlu_divisor,
                bias_scale,
            )
            all_acc.append(acc.cpu())
            all_hidden.append(next_hidden.cpu())
            next_batches.append(next_hidden.cpu())

        acc_values = torch.cat([batch.reshape(-1) for batch in all_acc])
        hidden_values = torch.cat([batch.reshape(-1) for batch in all_hidden])
        stats.append(
            {
                "layer": float(layer_index + 1),
                "scale": float(scale),
                "zero_rate": float((hidden_values <= 0).float().mean().item()),
                "clip_rate": float(
                    (
                        acc_values
                        >= float(
                            scale
                            * activation_numeric_clip(
                                activation, layer_index, model.hidden_clip
                            )
                        )
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "over_u8_rate": float(
                    (acc_values >= float(scale * model.hidden_clip))
                    .float()
                    .mean()
                    .item()
                ),
                "active_mean": float(hidden_values.float().mean().item()),
                "acc_mean": float(acc_values.float().mean().item()),
                "acc_p99": sampled_quantile(acc_values, 0.99),
                "acc_min": float(acc_values.min().item()),
                "acc_max": float(acc_values.max().item()),
            }
        )
        current_batches = next_batches
        activation_scale = activation_output_scale(
            accumulator_scale / float(scale),
            activation,
            layer_index,
            model.screlu_divisor,
            model.hidden_clip,
            quantization_convention,
        )

    print(
        json.dumps(
            {
                "event": "hidden_saturation",
                "arch": arch,
                "activation": activation,
                "quantization_convention": quantization_convention,
                "epoch": epoch,
                "scales": hidden_scales,
                "stats": stats,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return stats


@torch.no_grad()
def calibrate_output_scale(
    model: QuantizedSparseNnueArchitecture,
    loader: DataLoader,
    device: torch.device,
    board_feature_count: int,
    target_scale: float,
    hidden_scales: list[int],
    max_batches: int | None,
    default_output_scale: int,
    activation: str,
    quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
) -> tuple[int, dict[str, float]]:
    raw_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    for batch_index, samples in enumerate(loader, 1):
        feature_indices, offsets, _scores, targets, _plies, _results = collate_quantized_sparse_batch(
            samples,
            device=device,
            board_feature_count=board_feature_count,
            target_scale=target_scale,
        )
        raw = model(
            feature_indices,
            offsets,
            hidden_scales=hidden_scales,
            output_scale=1,
            activation=activation,
            quantization_convention=quantization_convention,
        )
        raw_values.append(raw.detach().cpu().float())
        target_values.append((targets.detach().cpu().float() * float(target_scale)))
        if max_batches is not None and batch_index >= max_batches:
            break
    if not raw_values:
        raise RuntimeError("output calibration loader produced no samples")

    raw_all = torch.cat(raw_values)
    target_all = torch.cat(target_values)
    denom = float((raw_all * raw_all).sum().item())
    numer = float((raw_all * target_all).sum().item())
    if denom <= 0.0 or numer <= 0.0 or not math.isfinite(denom) or not math.isfinite(numer):
        return default_output_scale, {"reason": "fallback", "raw_rms": 0.0, "target_rms": 0.0}
    gain = numer / denom
    if gain <= 0.0 or not math.isfinite(gain):
        return default_output_scale, {"reason": "fallback", "raw_rms": 0.0, "target_rms": 0.0}
    output_scale = max(1, int(round(1.0 / gain)))
    return output_scale, {
        "reason": "least_squares",
        "gain": gain,
        "raw_rms": float(torch.sqrt((raw_all * raw_all).mean()).item()),
        "target_rms": float(torch.sqrt((target_all * target_all).mean()).item()),
    }


def run_train_epoch(
    arch: str,
    model: QuantizedSparseNnueArchitecture,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    board_feature_count: int,
    target_scale: float,
    hidden_scales: list[int],
    output_scale: int,
    score_lambda: float,
    wdl_loss_exponent: float,
    max_batches: int | None,
    epoch: int,
    progress_batches: int,
    epoch_start: float,
    activation: str,
    quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
    cp_huber_delta: float = 200.0,
    lr_schedule: str = "constant",
    lr_min: float = 1e-5,
    lr_warmup_steps: int = 0,
    lr_total_steps: int = 1,
    lr_step_offset: int = 0,
    forward_mode: str = "quantized",
    loss_type: str = "cp_huber",
    mse_mix_weight: float = 0.75,
    peak_lr: float | None = None,
) -> tuple[float, float, int, int]:
    model.train()
    total_loss = 0.0
    total_abs_cp = 0.0
    total_samples = 0
    completed_batches = 0
    if peak_lr is None:
        peak_lr = float(optimizer.param_groups[0]["lr"])
    for batch_index, samples in enumerate(loader, 1):
        learning_rate = scheduled_learning_rate(
            lr_schedule,
            lr_step_offset + batch_index - 1,
            lr_total_steps,
            peak_lr,
            lr_min,
            lr_warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group.get("lr_multiplier", 1.0))
        feature_indices, offsets, target_scores, target_cp, plies, results = collate_quantized_sparse_batch(
            samples,
            device=device,
            board_feature_count=board_feature_count,
            target_scale=target_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = regression_forward(
            model,
            feature_indices,
            offsets,
            hidden_scales,
            output_scale,
            activation,
            quantization_convention,
            forward_mode,
        ) / float(target_scale)
        loss = cp_regression_loss(
            predictions,
            target_scores,
            target_scale,
            loss_type,
            cp_huber_delta,
            mse_mix_weight,
        )
        loss.backward()
        optimizer.step()
        if forward_mode == "quantized":
            model.clamp_quantized_weights()

        batch_size = int(target_cp.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_abs_cp += float(((predictions.detach() - target_cp).abs() * target_scale).sum().cpu())
        total_samples += batch_size
        completed_batches = batch_index
        if progress_batches > 0 and batch_index % progress_batches == 0:
            elapsed = time.monotonic() - epoch_start
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "arch": arch,
                        "activation": activation,
                        "quantization_convention": quantization_convention,
                        "forward_mode": forward_mode,
                        "loss_type": loss_type,
                        "epoch": epoch,
                        "batch": batch_index,
                        "samples": total_samples,
                        "avg_loss": round(total_loss / max(1, total_samples), 8),
                        "avg_cp": round(total_abs_cp / max(1, total_samples), 4),
                        "lr": learning_rate,
                        "elapsed_sec": round(elapsed, 3),
                        "samples_per_sec": round(total_samples / max(elapsed, 1e-9), 2),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        if max_batches is not None and batch_index >= max_batches:
            break
    if total_samples == 0:
        raise RuntimeError("train split produced no samples")
    return total_loss / total_samples, total_abs_cp / total_samples, total_samples, completed_batches


@torch.no_grad()
def evaluate_quantized_cp(
    arch: str,
    split: str,
    model: QuantizedSparseNnueArchitecture,
    loader: DataLoader,
    device: torch.device,
    board_feature_count: int,
    target_scale: float,
    score_lambda: float,
    wdl_loss_exponent: float,
    hidden_scales: list[int],
    output_scale: int,
    max_batches: int | None,
    epoch: int,
    progress_batches: int,
    epoch_start: float,
    activation: str,
    quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
    cp_huber_delta: float = 200.0,
    forward_mode: str = "quantized",
    loss_type: str = "cp_huber",
    mse_mix_weight: float = 0.75,
) -> tuple[float, float, int, list[dict[str, float | int]]]:
    model.eval()
    total_loss = 0.0
    total_abs_cp = 0.0
    total_samples = 0
    bin_error_sums = [0.0] * len(CP_MAE_BINS)
    bin_counts = [0] * len(CP_MAE_BINS)
    sum_target_cp = 0.0
    sum_prediction_cp = 0.0
    sum_target_cp2 = 0.0
    sum_target_prediction_cp = 0.0
    for batch_index, samples in enumerate(loader, 1):
        feature_indices, offsets, target_scores, target_cp, plies, results = collate_quantized_sparse_batch(
            samples,
            device=device,
            board_feature_count=board_feature_count,
            target_scale=target_scale,
        )
        predictions = regression_forward(
            model,
            feature_indices,
            offsets,
            hidden_scales,
            output_scale,
            activation,
            quantization_convention,
            forward_mode,
        )
        predictions = predictions / float(target_scale)
        loss = cp_regression_loss(
            predictions,
            target_scores,
            target_scale,
            loss_type,
            cp_huber_delta,
            mse_mix_weight,
        )
        batch_size = int(target_cp.shape[0])
        error_cp = (predictions - target_cp).abs() * target_scale
        target_cp_magnitude = target_cp.abs() * target_scale
        target_cp_values = target_cp * float(target_scale)
        prediction_cp_values = predictions * float(target_scale)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_abs_cp += float(error_cp.sum().cpu())
        total_samples += batch_size
        sum_target_cp += float(target_cp_values.sum().cpu())
        sum_prediction_cp += float(prediction_cp_values.sum().cpu())
        sum_target_cp2 += float((target_cp_values * target_cp_values).sum().cpu())
        sum_target_prediction_cp += float(
            (target_cp_values * prediction_cp_values).sum().cpu()
        )
        for index, (lower, upper) in enumerate(CP_MAE_BINS):
            mask = target_cp_magnitude >= lower
            mask &= (
                target_cp_magnitude <= upper
                if index == len(CP_MAE_BINS) - 1
                else target_cp_magnitude < upper
            )
            count = int(mask.sum().item())
            if count:
                bin_counts[index] += count
                bin_error_sums[index] += float(error_cp[mask].sum().cpu())
        if progress_batches > 0 and batch_index % progress_batches == 0:
            elapsed = time.monotonic() - epoch_start
            print(
                json.dumps(
                    {
                        "event": "eval_progress",
                        "arch": arch,
                        "activation": activation,
                        "quantization_convention": quantization_convention,
                        "forward_mode": forward_mode,
                        "loss_type": loss_type,
                        "split": split,
                        "epoch": epoch,
                        "batch": batch_index,
                        "samples": total_samples,
                        "avg_loss": round(total_loss / max(1, total_samples), 8),
                        "avg_cp": round(total_abs_cp / max(1, total_samples), 4),
                        "elapsed_sec": round(elapsed, 3),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        if max_batches is not None and batch_index >= max_batches:
            break
    if total_samples == 0:
        raise RuntimeError("eval split produced no samples")
    bins = [
        {
            "min_abs_target_cp": lower,
            "max_abs_target_cp": upper,
            "samples": count,
            "cp_mae": error_sum / count if count else float("nan"),
        }
        for (lower, upper), count, error_sum in zip(
            CP_MAE_BINS, bin_counts, bin_error_sums
        )
    ]
    target_mean = sum_target_cp / total_samples
    prediction_mean = sum_prediction_cp / total_samples
    target_variance_sum = sum_target_cp2 - total_samples * target_mean * target_mean
    covariance_sum = (
        sum_target_prediction_cp
        - total_samples * target_mean * prediction_mean
    )
    slope = covariance_sum / target_variance_sum if target_variance_sum > 0.0 else float("nan")
    intercept = prediction_mean - slope * target_mean
    print(
        json.dumps(
            {
                "event": "calibration",
                "arch": arch,
                "split": split,
                "epoch": epoch,
                "forward_mode": forward_mode,
                "loss_type": loss_type,
                "samples": total_samples,
                "target_mean_cp": target_mean,
                "prediction_mean_cp": prediction_mean,
                "mean_residual_cp": prediction_mean - target_mean,
                "slope": slope,
                "intercept_cp": intercept,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return total_loss / total_samples, total_abs_cp / total_samples, total_samples, bins


def save_checkpoint(
    path: Path,
    model: QuantizedSparseNnueArchitecture,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    hidden_scales: list[int],
    output_scale: int,
    target_scale: float,
    activation: str,
    quantization_convention: str,
    score_lambda: float,
    wdl_loss_exponent: float,
    cp_huber_delta: float,
    provenance: dict[str, Any],
    forward_mode: str = "quantized",
    loss_type: str = "cp_huber",
    mse_mix_weight: float = 0.75,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "architecture": model.architecture,
            "architecture_config": asdict(model.config),
            "feature_count": model.feature_count,
            "board_feature_count": model.board_feature_count,
            "hidden_sizes": model.hidden_sizes,
            "hidden_clip": model.hidden_clip,
            "hidden_scales": hidden_scales,
            "output_scale": output_scale,
            "target_scale": target_scale,
            "activation": activation,
            "quantization_convention": quantization_convention,
            "screlu_divisor": model.screlu_divisor,
            "psqt": {
                "enabled": model.use_psqt,
                "buckets": 8 if model.use_psqt else 0,
                "bucket_formula": "clamp((piece_count - 1) // 4, 0, 7)",
                "weight_dtype": "int32" if model.use_psqt else None,
                "weight_scale": model.psqt_weight_scale if model.use_psqt else None,
                "master_unit": "cp_over_target_scale" if model.use_psqt else None,
                "master_scale_to_cp": (
                    model.psqt_master_scale_to_cp if model.use_psqt else None
                ),
                "perspective": "stm_minus_opponent_div_2",
            },
            "score_lambda": score_lambda,
            "target_encoding": "stockfish_raw_score_ply_result",
            "wdl_model": "nnue_pytorch_training_data_entry",
            "wdl_loss_exponent": wdl_loss_exponent,
            "forward_mode": forward_mode,
            "loss_type": loss_type,
            "mse_mix_weight": mse_mix_weight,
            "cp_huber_delta": cp_huber_delta,
            "provenance": provenance,
            "quantization": {
                "weight_dtype": "int8",
                "bias_dtype": "int32",
                "activation_dtype": "uint8",
                "convention": quantization_convention,
                "screlu_divisor": model.screlu_divisor,
                "feature_weight_scale": model.feature_weight_scale,
                "linear_weight_scale": model.linear_weight_scale,
                "output_weight_scale": model.output_weight_scale,
                "psqt_weight_dtype": "int32" if model.use_psqt else None,
                "psqt_weight_scale": model.psqt_weight_scale if model.use_psqt else None,
                "psqt_master_scale_to_cp": (
                    model.psqt_master_scale_to_cp if model.use_psqt else None
                ),
            },
        },
        path,
    )


def resolve_fixed_hidden_scales(
    fixed_hidden_scale: int | None,
    fixed_hidden_scales: list[int] | None,
    layer_count: int,
) -> list[int] | None:
    if fixed_hidden_scale is not None and fixed_hidden_scales is not None:
        raise ValueError("use only one of --fixed-hidden-scale and --fixed-hidden-scales")
    if fixed_hidden_scales is not None:
        if len(fixed_hidden_scales) != layer_count:
            raise ValueError(
                "--fixed-hidden-scales must provide exactly one value per dense hidden layer "
                f"(expected {layer_count}, got {len(fixed_hidden_scales)})"
            )
        return list(fixed_hidden_scales)
    if fixed_hidden_scale is not None:
        return [fixed_hidden_scale] * layer_count
    return None


@torch.no_grad()
def initialize_scale_clean_screlu_biases(
    model: QuantizedSparseNnueArchitecture,
    activation: str,
    hidden_scales: list[int] | None,
    quantization_convention: str,
    hidden_fraction: float,
    first_fraction: float,
) -> dict[str, Any] | None:
    if quantization_convention != QUANTIZATION_CONVENTION_SCALE_CLEAN:
        return None
    if activation == ACTIVATION_RELU or hidden_scales is None:
        return None
    if hidden_fraction == 0.0 and first_fraction == 0.0:
        return None

    model.hidden1_bias.fill_(first_fraction)
    activation_scale = activation_output_scale(
        float(model.feature_weight_scale),
        activation,
        0,
        model.screlu_divisor,
        model.hidden_clip,
        quantization_convention,
    )
    hidden_bias_values: list[float] = []
    target_code = float(model.hidden_clip) * hidden_fraction
    for layer_index, (scale, layer) in enumerate(
        zip(hidden_scales, model.hidden_layers), 1
    ):
        accumulator_scale = activation_scale * float(model.linear_weight_scale)
        bias_value = target_code * float(scale) / accumulator_scale
        layer.bias.fill_(bias_value)
        hidden_bias_values.append(bias_value)
        activation_scale = activation_output_scale(
            accumulator_scale / float(scale),
            activation,
            layer_index,
            model.screlu_divisor,
            model.hidden_clip,
            quantization_convention,
        )
    return {
        "event": "screlu_bias_initialization",
        "first_bias_value": first_fraction,
        "hidden_target_fraction": hidden_fraction,
        "hidden_target_code": target_code,
        "hidden_bias_values": hidden_bias_values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train quantized NNUE architecture with sparse aux features")
    parser.add_argument("--arch", required=True, choices=sorted(QUANTIZED_ARCHITECTURES))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Optional separate corpus for validation/test splits",
    )
    parser.add_argument(
        "--eval-data-format",
        choices=["auto", "jsonl", "cbin"],
        default="auto",
    )
    parser.add_argument(
        "--eval-data-all-records-for-val",
        action="store_true",
        help="treat an explicit validation corpus as pre-split and read every record",
    )
    parser.add_argument("--output-dir", default=Path("models/quantized_nnue_arch_sweep"), type=Path)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="resume model and optimizer state; --epochs remains the final epoch number",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta-loss", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--psqt-lr",
        type=float,
        default=None,
        help="optional PSQT optimizer LR; defaults to --lr",
    )
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--epoch-peak-lrs",
        type=float,
        nargs="+",
        default=None,
        help="optional peak LR for each epoch; enables an independent per-epoch schedule",
    )
    parser.add_argument(
        "--epoch-min-lrs",
        type=float,
        nargs="+",
        default=None,
        help="optional ending LR for each epoch; must pair with --epoch-peak-lrs",
    )
    parser.add_argument(
        "--lr-steps-per-epoch",
        type=int,
        default=None,
        help="override scheduler steps per epoch when a hash split changes the sample count",
    )
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lr-after-warmup", type=float, default=None)
    parser.add_argument("--lr-drop-patience", type=int, default=0)
    parser.add_argument("--lr-drop-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--target-scale", type=float, default=DEFAULT_TARGET_SCALE)
    parser.add_argument("--score-lambda", type=float, default=DEFAULT_SCORE_LAMBDA)
    parser.add_argument("--wdl-loss-exponent", type=float, default=DEFAULT_WDL_LOSS_EXPONENT)
    parser.add_argument("--cp-huber-delta", type=float, default=200.0)
    parser.add_argument("--forward-mode", choices=FORWARD_MODES, default="quantized")
    parser.add_argument("--loss-type", choices=CP_LOSS_TYPES, default="cp_huber")
    parser.add_argument("--mse-mix-weight", type=float, default=0.75)
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument("--train-max-samples", type=int, default=2_000_000)
    parser.add_argument("--val-max-samples", type=int, default=100_000)
    parser.add_argument("--test-max-samples", type=int, default=100_000)
    parser.add_argument(
        "--train-probe-max-samples",
        type=int,
        default=0,
        help="evaluate a fixed training subset each epoch to log train slope and MAE",
    )
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument(
        "--train-data-all-records",
        action="store_true",
        help="use every record in an already-separated training corpus",
    )
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--calibration-max-batches", type=int, default=50)
    parser.add_argument("--calibration-percentile", type=float, default=99.5)
    parser.add_argument("--feature-weight-scale", type=int, default=FEATURE_WEIGHT_SCALE)
    parser.add_argument("--linear-weight-scale", type=int, default=LINEAR_WEIGHT_SCALE)
    parser.add_argument("--output-weight-scale", type=int, default=OUTPUT_WEIGHT_SCALE)
    parser.add_argument("--psqt-weight-scale", type=int, default=PSQT_WEIGHT_SCALE)
    psqt_group = parser.add_mutually_exclusive_group()
    psqt_group.add_argument("--psqt", dest="use_psqt", action="store_true")
    psqt_group.add_argument("--no-psqt", dest="use_psqt", action="store_false")
    parser.set_defaults(use_psqt=True)
    parser.add_argument("--hidden-clip", type=int, default=HIDDEN_CLIP)
    parser.add_argument("--screlu-divisor", type=int, default=SCRELU_DIVISOR)
    parser.add_argument(
        "--quantization-convention",
        choices=QUANTIZATION_CONVENTION_CHOICES,
        default=QUANTIZATION_CONVENTION_LEGACY,
        help="integer arithmetic convention; legacy preserves old checkpoint behavior",
    )
    parser.add_argument("--fixed-hidden-scale", type=int, default=None)
    parser.add_argument("--fixed-hidden-scales", type=int, nargs="+", default=None)
    parser.add_argument("--fixed-output-scale", type=int, default=None)
    parser.add_argument(
        "--screlu-init-fraction",
        type=float,
        default=0.0,
        help="initialize dense hidden biases to this fraction of the SCReLU clip code",
    )
    parser.add_argument(
        "--screlu-first-bias-fraction",
        type=float,
        default=0.0,
        help="initialize the feature-transformer bias in clip-domain units",
    )
    parser.add_argument("--log-initial-saturation", action="store_true")
    parser.add_argument("--reject-initial-zero-rate", type=float, default=1.0)
    parser.add_argument("--reject-initial-clip-rate", type=float, default=1.0)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--activation", choices=ACTIVATION_CHOICES, default=ACTIVATION_RELU)
    parser.add_argument("--shuffle-block-size", type=int, default=1_000_000)
    parser.add_argument("--progress-batches", type=int, default=500)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs and --patience must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0 or args.eval_workers < 0 or args.torch_threads < 0:
        raise ValueError("worker/thread counts must be non-negative")
    if args.lr <= 0.0 or args.weight_decay < 0.0 or args.min_delta_loss < 0.0:
        raise ValueError("learning rate must be positive; decay/delta must be non-negative")
    if args.psqt_lr is not None and args.psqt_lr <= 0.0:
        raise ValueError("--psqt-lr must be positive")
    if args.psqt_lr is not None and not args.use_psqt:
        raise ValueError("--psqt-lr requires --psqt")
    if args.warmup_epochs < 0 or args.lr_drop_patience < 0 or args.lr_warmup_steps < 0:
        raise ValueError("warmup/plateau epoch counts must be non-negative")
    if args.lr_after_warmup is not None and args.lr_after_warmup <= 0.0:
        raise ValueError("--lr-after-warmup must be positive")
    if not 0.0 < args.lr_drop_factor <= 1.0 or args.min_lr <= 0.0:
        raise ValueError("--lr-drop-factor must be in (0, 1] and --min-lr must be positive")
    if args.lr_schedule == "cosine" and args.min_lr > args.lr:
        raise ValueError("--min-lr cannot exceed --lr for cosine scheduling")
    validate_epoch_learning_rates(
        args.epoch_peak_lrs,
        args.epoch_min_lrs,
        args.epochs,
    )
    if args.epoch_peak_lrs is not None and args.lr_schedule != "cosine":
        raise ValueError("per-epoch learning rates require --lr-schedule cosine")
    if args.lr_steps_per_epoch is not None and args.lr_steps_per_epoch <= 0:
        raise ValueError("--lr-steps-per-epoch must be positive")
    for name in ("train_max_samples", "val_max_samples", "test_max_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("train_max_batches", "eval_max_batches", "calibration_max_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.train_probe_max_samples < 0:
        raise ValueError("--train-probe-max-samples must be non-negative")
    if not 0.0 < args.calibration_percentile <= 100.0:
        raise ValueError("--calibration-percentile must be in (0, 100]")
    for name in ("feature_weight_scale", "linear_weight_scale", "output_weight_scale"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.hidden_clip <= 255:
        raise ValueError("--hidden-clip must be in [1, 255]")
    if args.screlu_divisor <= 0:
        raise ValueError("--screlu-divisor must be positive")
    if args.fixed_hidden_scale is not None and args.fixed_hidden_scale <= 0:
        raise ValueError("--fixed-hidden-scale must be positive")
    if args.fixed_hidden_scales is not None and any(
        value <= 0 for value in args.fixed_hidden_scales
    ):
        raise ValueError("--fixed-hidden-scales must contain positive values")
    if args.fixed_hidden_scale is not None and args.fixed_hidden_scales is not None:
        raise ValueError("use only one of --fixed-hidden-scale and --fixed-hidden-scales")
    if args.fixed_output_scale is not None and args.fixed_output_scale <= 0:
        raise ValueError("--fixed-output-scale must be positive")
    if args.lr_drop_patience > 0 and args.patience <= args.lr_drop_patience:
        raise ValueError("--patience must be greater than --lr-drop-patience")
    if args.shuffle_block_size < 0 or args.progress_batches < 0 or args.eval_progress_batches < 0:
        raise ValueError("shuffle/progress counts must be non-negative")
    if args.target_scale <= 0.0:
        raise ValueError("--target-scale must be positive")
    if not 0.0 <= args.score_lambda <= 1.0:
        raise ValueError("--score-lambda must be in [0, 1]")
    if args.wdl_loss_exponent <= 0.0:
        raise ValueError("--wdl-loss-exponent must be positive")
    if args.cp_huber_delta <= 0.0:
        raise ValueError("--cp-huber-delta must be positive")
    if not 0.0 <= args.mse_mix_weight <= 1.0:
        raise ValueError("--mse-mix-weight must be in [0, 1]")
    if not 0.0 <= args.screlu_init_fraction < 1.0:
        raise ValueError("--screlu-init-fraction must be in [0, 1)")
    if not 0.0 <= args.screlu_first_bias_fraction < 1.0:
        raise ValueError("--screlu-first-bias-fraction must be in [0, 1)")
    if not 0.0 < args.reject_initial_zero_rate <= 1.0:
        raise ValueError("--reject-initial-zero-rate must be in (0, 1]")
    if not 0.0 < args.reject_initial_clip_rate <= 1.0:
        raise ValueError("--reject-initial-clip-rate must be in (0, 1]")
    if args.split_mod <= 2:
        raise ValueError("--split-mod must be greater than 2")
    if not 0 <= args.val_mod < args.split_mod or not 0 <= args.test_mod < args.split_mod:
        raise ValueError("validation/test buckets must be inside --split-mod")
    if args.val_mod == args.test_mod:
        raise ValueError("validation and test buckets must differ")
    args.data_format = validate_training_data(args.data, args.data_format)
    eval_data = args.eval_data if args.eval_data is not None else args.data
    eval_data_format = validate_training_data(eval_data, args.eval_data_format)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = QUANTIZED_ARCHITECTURES[args.arch]
    device = choose_device(args.device)
    model = QuantizedSparseNnueArchitecture(
        config,
        hidden_clip=args.hidden_clip,
        feature_weight_scale=args.feature_weight_scale,
        linear_weight_scale=args.linear_weight_scale,
        output_weight_scale=args.output_weight_scale,
        screlu_divisor=args.screlu_divisor,
        use_psqt=args.use_psqt,
        psqt_weight_scale=args.psqt_weight_scale,
        psqt_master_scale_to_cp=args.target_scale,
    ).to(device)
    fixed_hidden_scales = resolve_fixed_hidden_scales(
        args.fixed_hidden_scale,
        args.fixed_hidden_scales,
        len(model.hidden_layers),
    )
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume_checkpoint}")
        resume_checkpoint = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if resume_checkpoint.get("architecture") != args.arch:
            raise ValueError("resume checkpoint architecture does not match --arch")
        if resume_checkpoint.get("activation") != args.activation:
            raise ValueError("resume checkpoint activation does not match --activation")
        if resume_checkpoint.get("quantization_convention") != args.quantization_convention:
            raise ValueError("resume checkpoint quantization convention does not match")
        checkpoint_psqt = resume_checkpoint.get("psqt", {})
        if bool(checkpoint_psqt.get("enabled", False)) != args.use_psqt:
            raise ValueError("resume checkpoint PSQT setting does not match --psqt/--no-psqt")
        if args.use_psqt and int(checkpoint_psqt.get("weight_scale", -1)) != args.psqt_weight_scale:
            raise ValueError("resume checkpoint PSQT weight scale does not match")
        if args.use_psqt and not math.isclose(
            float(checkpoint_psqt.get("master_scale_to_cp", 1.0)),
            float(args.target_scale),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("resume checkpoint PSQT master scale does not match target scale")
        checkpoint_hidden_scales = [int(value) for value in resume_checkpoint["hidden_scales"]]
        checkpoint_output_scale = int(resume_checkpoint["output_scale"])
        if fixed_hidden_scales is not None and checkpoint_hidden_scales != fixed_hidden_scales:
            raise ValueError("resume checkpoint hidden scales do not match fixed scales")
        if args.fixed_output_scale is not None and checkpoint_output_scale != args.fixed_output_scale:
            raise ValueError("resume checkpoint output scale does not match fixed scale")
        model.load_state_dict(resume_checkpoint["model_state"])
        screlu_initialization = None
    else:
        screlu_initialization = initialize_scale_clean_screlu_biases(
            model,
            args.activation,
            fixed_hidden_scales,
            args.quantization_convention,
            args.screlu_init_fraction,
            args.screlu_first_bias_fraction,
        )
    if model.psqt is None:
        optimizer_groups = [
            {
                "params": list(model.parameters()),
                "lr": args.lr,
                "lr_multiplier": 1.0,
                "group_name": "base",
            }
        ]
    else:
        psqt_lr = args.psqt_lr if args.psqt_lr is not None else args.lr
        psqt_parameter_ids = {id(parameter) for parameter in model.psqt.parameters()}
        base_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in psqt_parameter_ids
        ]
        optimizer_groups = [
            {
                "params": base_parameters,
                "lr": args.lr,
                "lr_multiplier": 1.0,
                "group_name": "base",
            },
            {
                "params": list(model.psqt.parameters()),
                "lr": psqt_lr,
                "lr_multiplier": psqt_lr / args.lr,
                "group_name": "psqt",
            },
        ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])

    train_loader = make_loader(
        args.data,
        config.transform,
        args.data_format,
        "all" if args.train_data_all_records else "train",
        args.split_mod,
        args.val_mod,
        args.test_mod,
        args.train_max_samples,
        args.seed,
        args.batch_size,
        args.workers,
        args.shuffle_block_size,
    )
    val_loader = make_loader(
        eval_data,
        config.transform,
        eval_data_format,
        "all" if args.eval_data_all_records_for_val else "val",
        args.split_mod,
        args.val_mod,
        args.test_mod,
        args.val_max_samples,
        args.seed,
        args.batch_size,
        args.eval_workers,
        0,
    )
    train_probe_loader = None
    if args.train_probe_max_samples > 0:
        train_probe_loader = make_loader(
            args.data,
            config.transform,
            args.data_format,
            "all" if args.train_data_all_records else "train",
            args.split_mod,
            args.val_mod,
            args.test_mod,
            args.train_probe_max_samples,
            args.seed,
            args.batch_size,
            args.eval_workers,
            0,
        )
    test_loader = None
    if not args.skip_final_test:
        test_loader = make_loader(
            eval_data,
            config.transform,
            eval_data_format,
            "test",
            args.split_mod,
            args.val_mod,
            args.test_mod,
            args.test_max_samples,
            args.seed,
            args.batch_size,
            args.eval_workers,
            0,
        )
    calibration_loader = make_loader(
        args.data,
        config.transform,
        args.data_format,
        "all" if args.train_data_all_records else "train",
        args.split_mod,
        args.val_mod,
        args.test_mod,
        args.batch_size * args.calibration_max_batches,
        args.seed,
        args.batch_size,
        args.eval_workers,
        0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_path = args.output_dir / f"quant_nnue_arch_{args.arch}_current.pt"
    best_path = args.output_dir / f"quant_nnue_arch_{args.arch}_best.pt"
    provenance = {
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "target_encoding": "stockfish_raw_score_ply_result",
        "score_to_cp": "clamp(100 * raw_score / 208, -2000, 2000)",
        "wdl_model": "nnue_pytorch_training_data_entry",
        "training_objective": args.loss_type,
        "cp_huber_delta": args.cp_huber_delta,
        "screlu_divisor": args.screlu_divisor,
        "quantization_convention": args.quantization_convention,
        "split_policy": "crc32(board_bytes + aux_bits_le16)",
        "scale_calibration_split": "train",
        **git_provenance(),
    }

    print(
        json.dumps(
            {
                "event": "start",
                "arch": args.arch,
                "activation": args.activation,
                "config": asdict(config),
                "device": str(device),
                "resume_checkpoint": (
                    str(args.resume_checkpoint) if args.resume_checkpoint is not None else None
                ),
                "resume_epoch": (
                    int(resume_checkpoint["epoch"]) if resume_checkpoint is not None else 0
                ),
                "data": str(args.data),
                "eval_data": str(eval_data),
                "batch_size": args.batch_size,
                "lr": args.lr,
                "psqt_lr": (
                    args.psqt_lr if args.psqt_lr is not None else args.lr
                ) if args.use_psqt else None,
                "lr_schedule": args.lr_schedule,
                "lr_warmup_steps": args.lr_warmup_steps,
                "epoch_peak_lrs": args.epoch_peak_lrs,
                "epoch_min_lrs": args.epoch_min_lrs,
                "lr_steps_per_epoch": args.lr_steps_per_epoch,
                "warmup_epochs": args.warmup_epochs,
                "lr_after_warmup": args.lr_after_warmup,
                "lr_drop_patience": args.lr_drop_patience,
                "lr_drop_factor": args.lr_drop_factor,
                "min_lr": args.min_lr,
                "train_max_samples": args.train_max_samples,
                "train_data_all_records": args.train_data_all_records,
                "val_max_samples": args.val_max_samples,
                "train_probe_max_samples": args.train_probe_max_samples,
                "test_max_samples": args.test_max_samples,
                "scale_calibration_split": "train",
                "scale_calibration_max_samples": (
                    args.batch_size * args.calibration_max_batches
                ),
                "target_scale": args.target_scale,
                "score_lambda": args.score_lambda,
                "wdl_loss_exponent": args.wdl_loss_exponent,
                "forward_mode": args.forward_mode,
                "loss_type": args.loss_type,
                "mse_mix_weight": args.mse_mix_weight,
                "cp_huber_delta": args.cp_huber_delta,
                "psqt": {
                    "enabled": args.use_psqt,
                    "buckets": 8 if args.use_psqt else 0,
                    "bucket_formula": "clamp((piece_count - 1) // 4, 0, 7)",
                    "weight_dtype": "int32" if args.use_psqt else None,
                    "weight_scale": args.psqt_weight_scale if args.use_psqt else None,
                    "master_unit": "cp_over_target_scale" if args.use_psqt else None,
                    "master_scale_to_cp": args.target_scale if args.use_psqt else None,
                    "perspective": "stm_minus_opponent_div_2",
                },
                "screlu_divisor": args.screlu_divisor,
                "workers": args.workers,
                "eval_workers": args.eval_workers,
                "quantization": {
                    "weight_dtype": "int8",
                    "clip": args.hidden_clip,
                    "feature_weight_scale": args.feature_weight_scale,
                    "linear_weight_scale": args.linear_weight_scale,
                    "output_weight_scale": args.output_weight_scale,
                    "psqt_weight_dtype": "int32" if args.use_psqt else None,
                    "psqt_weight_scale": args.psqt_weight_scale if args.use_psqt else None,
                    "psqt_master_scale_to_cp": args.target_scale if args.use_psqt else None,
                    "fixed_hidden_scale": args.fixed_hidden_scale,
                    "fixed_hidden_scales": fixed_hidden_scales,
                    "fixed_output_scale": args.fixed_output_scale,
                    "activation": args.activation,
                    "quantization_convention": args.quantization_convention,
                    "screlu_divisor": args.screlu_divisor,
                    "screlu_init_fraction": args.screlu_init_fraction,
                    "screlu_first_bias_fraction": args.screlu_first_bias_fraction,
                    "aux": "sparse",
                },
                "split": {
                    "split_mod": args.split_mod,
                    "val_mod": args.val_mod,
                    "test_mod": args.test_mod,
                },
                "provenance": provenance,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    if screlu_initialization is not None:
        print(json.dumps(screlu_initialization, separators=(",", ":")), flush=True)
    if args.log_initial_saturation and fixed_hidden_scales is not None:
        initial_saturation = measure_hidden_saturation(
            args.arch,
            model,
            calibration_loader,
            device,
            config.board_feature_count,
            args.target_scale,
            fixed_hidden_scales,
            args.calibration_max_batches,
            args.activation,
            0,
            args.quantization_convention,
        )
        rejection_reasons: list[str] = []
        for stat in initial_saturation:
            layer = int(float(stat["layer"]))
            zero_rate = float(stat["zero_rate"])
            clip_rate = float(stat["clip_rate"])
            if zero_rate >= args.reject_initial_zero_rate:
                rejection_reasons.append(
                    f"layer_{layer}_zero_rate={zero_rate:.6f}"
                )
            if clip_rate >= args.reject_initial_clip_rate:
                rejection_reasons.append(
                    f"layer_{layer}_clip_rate={clip_rate:.6f}"
                )
        if rejection_reasons:
            print(
                json.dumps(
                    {
                        "event": "config_rejected",
                        "arch": args.arch,
                        "activation": args.activation,
                        "hidden_scales": fixed_hidden_scales,
                        "output_scale": args.fixed_output_scale,
                        "reasons": rejection_reasons,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            print(
                json.dumps(
                    {
                        "event": "training_complete",
                        "arch": args.arch,
                        "activation": args.activation,
                        "selected_epoch": 0,
                        "hidden_scales": fixed_hidden_scales,
                        "output_scale": args.fixed_output_scale,
                        "test_evaluated": False,
                        "rejected": True,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            return

    if resume_checkpoint is not None:
        completed_epoch = int(resume_checkpoint["epoch"])
        if completed_epoch >= args.epochs:
            raise ValueError("resume checkpoint epoch must be smaller than --epochs")
        # Resuming in the same output directory is the normal long-run path.
        # Avoid copying current.pt onto itself, and preserve an earlier best.pt
        # instead of silently replacing it with the latest checkpoint.
        prior_best = None
        if best_path.is_file() and best_path.resolve() != args.resume_checkpoint.resolve():
            prior_best = torch.load(best_path, map_location=device, weights_only=False)
        selected_best = prior_best if prior_best is not None else resume_checkpoint
        best_val_loss = float(selected_best["metrics"]["val_loss"])
        best_epoch = int(selected_best["epoch"])
        best_hidden_scales = [int(value) for value in selected_best["hidden_scales"]]
        best_output_scale = int(selected_best["output_scale"])
        if prior_best is None and best_path.resolve() != args.resume_checkpoint.resolve():
            shutil.copy2(args.resume_checkpoint, best_path)
        if current_path.resolve() != args.resume_checkpoint.resolve():
            shutil.copy2(args.resume_checkpoint, current_path)
        print(
            json.dumps(
                {
                    "event": "resume",
                    "checkpoint": str(args.resume_checkpoint),
                    "completed_epoch": completed_epoch,
                    "best_val_loss": best_val_loss,
                    "hidden_scales": best_hidden_scales,
                    "output_scale": best_output_scale,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    else:
        best_val_loss = math.inf
        best_epoch = 0
        best_hidden_scales = model.default_hidden_scales
        best_output_scale = model.default_output_scale
    epochs_without_improvement = 0
    last_scale_audit_key: tuple[tuple[int, ...], int] | None = None
    if args.lr_steps_per_epoch is not None:
        steps_per_epoch = args.lr_steps_per_epoch
    elif args.train_max_batches is not None:
        steps_per_epoch = args.train_max_batches
    elif args.train_max_samples is not None:
        steps_per_epoch = math.ceil(args.train_max_samples / args.batch_size)
    else:
        if args.lr_schedule != "constant":
            raise ValueError("a finite train limit is required for cosine scheduling")
        steps_per_epoch = 1
    per_epoch_lr_schedule = args.epoch_peak_lrs is not None
    lr_total_steps = (
        steps_per_epoch if per_epoch_lr_schedule else steps_per_epoch * args.epochs
    )
    if args.lr_schedule == "cosine" and args.lr_warmup_steps >= lr_total_steps:
        raise ValueError("--lr-warmup-steps must be smaller than total optimizer steps")
    lr_steps_completed = (
        0 if per_epoch_lr_schedule else (
            int(resume_checkpoint["epoch"]) if resume_checkpoint is not None else 0
        ) * steps_per_epoch
    )

    completed_epoch = int(resume_checkpoint["epoch"]) if resume_checkpoint is not None else 0
    for epoch in range(completed_epoch + 1, args.epochs + 1):
        if (
            args.lr_after_warmup is not None
            and args.warmup_epochs > 0
            and epoch == args.warmup_epochs + 1
        ):
            for group in optimizer.param_groups:
                group["lr"] = args.lr_after_warmup * float(
                    group.get("lr_multiplier", 1.0)
                )
            print(
                json.dumps(
                    {
                        "event": "lr_update",
                        "arch": args.arch,
                        "epoch": epoch,
                        "reason": "after_warmup",
                        "lr": args.lr_after_warmup,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

        start = time.monotonic()
        train_hidden_scales = (
            fixed_hidden_scales
            if fixed_hidden_scales is not None
            else model.default_hidden_scales
        )
        train_output_scale = (
            args.fixed_output_scale
            if args.fixed_output_scale is not None
            else model.default_output_scale
        )
        model.default_hidden_scales = train_hidden_scales
        model.default_output_scale = train_output_scale
        if per_epoch_lr_schedule:
            assert args.epoch_peak_lrs is not None
            assert args.epoch_min_lrs is not None
            epoch_peak_lr = args.epoch_peak_lrs[epoch - 1]
            epoch_min_lr = args.epoch_min_lrs[epoch - 1]
            epoch_warmup_steps = args.lr_warmup_steps if epoch == 1 else 0
            epoch_lr_step_offset = 0
        else:
            epoch_peak_lr = (
                args.lr
                if args.lr_schedule == "cosine"
                else float(optimizer.param_groups[0]["lr"])
            )
            epoch_min_lr = args.min_lr
            epoch_warmup_steps = args.lr_warmup_steps
            epoch_lr_step_offset = lr_steps_completed
        train_loss, train_cp, train_samples, epoch_batches = run_train_epoch(
            args.arch,
            model,
            train_loader,
            optimizer,
            device,
            config.board_feature_count,
            args.target_scale,
            train_hidden_scales,
            train_output_scale,
            args.score_lambda,
            args.wdl_loss_exponent,
            args.train_max_batches,
            epoch,
            args.progress_batches,
            start,
            args.activation,
            args.quantization_convention,
            args.cp_huber_delta,
            args.lr_schedule,
            epoch_min_lr,
            epoch_warmup_steps,
            lr_total_steps,
            epoch_lr_step_offset,
            args.forward_mode,
            args.loss_type,
            args.mse_mix_weight,
            epoch_peak_lr,
        )
        if not per_epoch_lr_schedule:
            lr_steps_completed += epoch_batches
        if fixed_hidden_scales is not None:
            hidden_scales = fixed_hidden_scales
            hidden_scale_stats = measure_hidden_saturation(
                args.arch,
                model,
                val_loader,
                device,
                config.board_feature_count,
                args.target_scale,
                hidden_scales,
                args.calibration_max_batches,
                args.activation,
                epoch,
                args.quantization_convention,
            )
        else:
            hidden_scales, hidden_scale_stats = calibrate_hidden_scales(
                args.arch,
                model,
                calibration_loader,
                device,
                config.board_feature_count,
                args.target_scale,
                args.calibration_max_batches,
                args.calibration_percentile,
                args.activation,
                epoch,
                args.quantization_convention,
            )
            hidden_scale_stats = measure_hidden_saturation(
                args.arch,
                model,
                val_loader,
                device,
                config.board_feature_count,
                args.target_scale,
                hidden_scales,
                args.calibration_max_batches,
                args.activation,
                epoch,
                args.quantization_convention,
            )
        if args.fixed_output_scale is not None:
            output_scale = args.fixed_output_scale
            output_scale_stats = {"reason": "fixed", "scale": float(output_scale)}
        else:
            output_scale, output_scale_stats = calibrate_output_scale(
                model,
                calibration_loader,
                device,
                config.board_feature_count,
                args.target_scale,
                hidden_scales,
                args.calibration_max_batches,
                args.output_weight_scale,
                args.activation,
                args.quantization_convention,
            )
        model.default_hidden_scales = hidden_scales
        model.default_output_scale = output_scale
        scale_audit = quantization_scale_audit(
            model,
            args.activation,
            hidden_scales,
            output_scale,
            args.quantization_convention,
        )
        scale_audit_key = (tuple(int(value) for value in hidden_scales), int(output_scale))
        if scale_audit_key != last_scale_audit_key:
            print(
                json.dumps(
                    {"event": "scale_audit", "arch": args.arch, **scale_audit},
                    separators=(",", ":"),
                ),
                flush=True,
            )
            last_scale_audit_key = scale_audit_key
        val_loss, val_cp, val_samples, val_cp_bins = evaluate_quantized_cp(
            args.arch,
            "val",
            model,
            val_loader,
            device,
            config.board_feature_count,
            args.target_scale,
            args.score_lambda,
            args.wdl_loss_exponent,
            hidden_scales,
            output_scale,
            args.eval_max_batches,
            epoch,
            args.eval_progress_batches,
            start,
            args.activation,
            args.quantization_convention,
            args.cp_huber_delta,
            args.forward_mode,
            args.loss_type,
            args.mse_mix_weight,
        )
        train_probe_loss = None
        train_probe_cp = None
        train_probe_samples = 0
        train_probe_cp_bins = None
        if train_probe_loader is not None:
            (
                train_probe_loss,
                train_probe_cp,
                train_probe_samples,
                train_probe_cp_bins,
            ) = evaluate_quantized_cp(
                args.arch,
                "train_probe",
                model,
                train_probe_loader,
                device,
                config.board_feature_count,
                args.target_scale,
                args.score_lambda,
                args.wdl_loss_exponent,
                hidden_scales,
                output_scale,
                args.eval_max_batches,
                epoch,
                args.eval_progress_batches,
                start,
                args.activation,
                args.quantization_convention,
                args.cp_huber_delta,
                args.forward_mode,
                args.loss_type,
                args.mse_mix_weight,
            )
        elapsed = time.monotonic() - start
        current_lr = float(optimizer.param_groups[0]["lr"])
        current_psqt_lr = next(
            (
                float(group["lr"])
                for group in optimizer.param_groups
                if group.get("group_name") == "psqt"
            ),
            None,
        )
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_val_cp": train_cp,
            "val_cp": val_cp,
            "val_cp_bins": val_cp_bins,  # type: ignore[dict-item]
            "train_probe_loss": train_probe_loss,
            "train_probe_cp": train_probe_cp,
            "train_probe_samples": train_probe_samples,
            "train_probe_cp_bins": train_probe_cp_bins,
            "lr": current_lr,
            "psqt_lr": current_psqt_lr,
            "elapsed_sec": elapsed,
            "hidden_scale_stats": hidden_scale_stats,  # type: ignore[dict-item]
            "output_scale_stats": output_scale_stats,  # type: ignore[dict-item]
            "scale_audit": scale_audit,  # type: ignore[dict-item]
        }
        save_checkpoint(
            current_path,
            model,
            optimizer,
            epoch,
            metrics,
            hidden_scales,
            output_scale,
            args.target_scale,
            args.activation,
            args.quantization_convention,
            args.score_lambda,
            args.wdl_loss_exponent,
            args.cp_huber_delta,
            provenance,
            args.forward_mode,
            args.loss_type,
            args.mse_mix_weight,
        )
        improved = val_loss + args.min_delta_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_hidden_scales = hidden_scales
            best_output_scale = output_scale
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                metrics,
                hidden_scales,
                output_scale,
                args.target_scale,
                args.activation,
                args.quantization_convention,
                args.score_lambda,
                args.wdl_loss_exponent,
                args.cp_huber_delta,
                provenance,
                args.forward_mode,
                args.loss_type,
                args.mse_mix_weight,
            )
        else:
            epochs_without_improvement += 1

        print(
            json.dumps(
                {
                    "event": "epoch",
                    "arch": args.arch,
                    "activation": args.activation,
                    "epoch": epoch,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                    "train_probe_samples": train_probe_samples,
                    "train_loss": round(train_loss, 8),
                    "val_loss": round(val_loss, 8),
                    "train_val_cp": round(train_cp, 4),
                    "val_cp": round(val_cp, 4),
                    "train_probe_cp": (
                        round(train_probe_cp, 4) if train_probe_cp is not None else None
                    ),
                    "val_cp_bins": val_cp_bins,
                    "best_val_loss": round(best_val_loss, 8),
                    "best_epoch": best_epoch,
                    "lr": current_lr,
                    "psqt_lr": current_psqt_lr,
                    "train_hidden_scales": train_hidden_scales,
                    "train_output_scale": train_output_scale,
                    "hidden_scales": hidden_scales,
                    "output_scale": output_scale,
                    "improved": improved,
                    "no_improve_epochs": epochs_without_improvement,
                    "elapsed_sec": round(elapsed, 3),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        if (
            not improved
            and args.lr_drop_patience > 0
            and epochs_without_improvement > 0
            and epochs_without_improvement < args.patience
            and epochs_without_improvement % args.lr_drop_patience == 0
        ):
            old_lr = float(optimizer.param_groups[0]["lr"])
            new_lr = max(args.min_lr, old_lr * args.lr_drop_factor)
            if new_lr < old_lr:
                for group in optimizer.param_groups:
                    group["lr"] = new_lr * float(group.get("lr_multiplier", 1.0))
                print(
                    json.dumps(
                        {
                            "event": "lr_update",
                            "arch": args.arch,
                            "epoch": epoch,
                            "reason": "plateau",
                            "old_lr": old_lr,
                            "new_lr": new_lr,
                            "no_improve_epochs": epochs_without_improvement,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        if epochs_without_improvement >= args.patience:
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "arch": args.arch,
                        "epoch": epoch,
                        "best_val_loss": round(best_val_loss, 8),
                        "best_epoch": best_epoch,
                        "best_hidden_scales": best_hidden_scales,
                        "best_output_scale": best_output_scale,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            break

    if best_epoch == 0 or not best_path.exists():
        raise RuntimeError("training completed without a validation checkpoint")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    best_hidden_scales = [int(value) for value in best_checkpoint["hidden_scales"]]
    best_output_scale = int(best_checkpoint["output_scale"])
    model.default_hidden_scales = best_hidden_scales
    model.default_output_scale = best_output_scale
    if args.skip_final_test:
        print(
            json.dumps(
                {
                    "event": "training_complete",
                    "arch": args.arch,
                    "activation": args.activation,
                    "selected_epoch": best_epoch,
                    "best_val_loss": round(best_val_loss, 8),
                    "hidden_scales": best_hidden_scales,
                    "output_scale": best_output_scale,
                    "test_evaluated": False,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return
    assert test_loader is not None
    test_start = time.monotonic()
    test_loss, test_cp, test_samples, test_cp_bins = evaluate_quantized_cp(
        args.arch,
        "test",
        model,
        test_loader,
        device,
        config.board_feature_count,
        args.target_scale,
        args.score_lambda,
        args.wdl_loss_exponent,
        best_hidden_scales,
        best_output_scale,
        args.eval_max_batches,
        best_epoch,
        args.eval_progress_batches,
        test_start,
        args.activation,
        args.quantization_convention,
        args.cp_huber_delta,
        args.forward_mode,
        args.loss_type,
        args.mse_mix_weight,
    )
    print(
        json.dumps(
            {
                "event": "final_test",
                "arch": args.arch,
                "activation": args.activation,
                "selected_epoch": best_epoch,
                "best_val_loss": round(best_val_loss, 8),
                "hidden_scales": best_hidden_scales,
                "output_scale": best_output_scale,
                "test_samples": test_samples,
                "test_loss": round(test_loss, 8),
                "test_val_cp": round(test_cp, 4),
                "test_cp_bins": test_cp_bins,
                "elapsed_sec": round(time.monotonic() - test_start, 3),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
