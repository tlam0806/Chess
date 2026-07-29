#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.compact_board_data import (  # noqa: E402
    HEADER_SIZE,
    RECORD_SIZE,
    architecture_features_from_board,
    compact_paths,
    open_reader,
    unpack_aux,
    unpack_record,
    validate_header,
)
from nn.quantized_nnue_architectures import (  # noqa: E402
    ACTIVATION_SCRELU_RELU16_ALL,
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    PhaseStackQuantizedNnueArchitecture,
    QuantizedSparseNnueArchitecture,
    activation_kind,
    activation_numeric_clip,
    activation_output_scale,
    trunc_ste,
)
from nn.training_targets import (  # noqa: E402
    STOCKFISH_SCORE_CP_DENOMINATOR,
    STOCKFISH_SCORE_CP_NUMERATOR,
)
from nn.train_value import choose_device  # noqa: E402
from nn.value_net import AUX_FEATURE_COUNT  # noqa: E402


def raw_to_cp(value: torch.Tensor) -> torch.Tensor:
    return value * (STOCKFISH_SCORE_CP_NUMERATOR / STOCKFISH_SCORE_CP_DENOMINATOR)


def identity_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples


def worker_sample_limit(
    max_samples: int | None,
    worker_id: int,
    workers: int,
) -> int | None:
    """Split a global sample cap exactly across iterable-dataset workers."""
    if max_samples is None:
        return None
    quotient, remainder = divmod(max_samples, workers)
    return quotient + (1 if worker_id < remainder else 0)


class AlignedComponentDataset(IterableDataset):
    """Read PSQT and positional CBin shards in lockstep.

    Only bytes 34..35 (the score field) may differ. This assertion makes it
    impossible to silently train the two branches on different positions.
    """

    def __init__(
        self,
        psqt_path: Path,
        positional_path: Path,
        architecture: str,
        max_samples: int | None,
        skip_samples: int,
        seed: int,
        shuffle_block_size: int,
    ) -> None:
        super().__init__()
        self.psqt_path = psqt_path
        self.positional_path = positional_path
        self.architecture = architecture
        self.max_samples = max_samples
        self.skip_samples = skip_samples
        self.seed = seed
        self.shuffle_block_size = shuffle_block_size

    def _pairs(self) -> list[tuple[Path, Path]]:
        psqt = {path.name: path for path in compact_paths(self.psqt_path)}
        positional = {path.name: path for path in compact_paths(self.positional_path)}
        if psqt.keys() != positional.keys():
            missing_psqt = sorted(positional.keys() - psqt.keys())
            missing_positional = sorted(psqt.keys() - positional.keys())
            raise ValueError(
                f"component shard mismatch: missing_psqt={missing_psqt[:3]} "
                f"missing_positional={missing_positional[:3]}"
            )
        return [(psqt[name], positional[name]) for name in sorted(psqt)]

    @staticmethod
    def _same_position(left: bytes, right: bytes) -> bool:
        return left[:34] == right[:34] and left[36:] == right[36:]

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        if self.skip_samples and workers != 1:
            raise ValueError("skip_samples requires a single-worker evaluation loader")
        sample_limit = worker_sample_limit(self.max_samples, worker_id, workers)
        if sample_limit == 0:
            return

        pairs = self._pairs()
        rng = random.Random(self.seed)
        if self.shuffle_block_size > 0:
            rng.shuffle(pairs)
        block_records = max(1, self.shuffle_block_size or 250_000)
        block_bytes = block_records * RECORD_SIZE
        global_block = 0
        seen = 0
        yielded = 0

        for psqt_path, positional_path in pairs:
            psqt_stream, psqt_owner = open_reader(psqt_path)
            positional_stream, positional_owner = open_reader(positional_path)
            try:
                psqt_header = psqt_stream.read(HEADER_SIZE)
                positional_header = positional_stream.read(HEADER_SIZE)
                validate_header(psqt_header, psqt_path)
                validate_header(positional_header, positional_path)
                if psqt_header != positional_header:
                    raise ValueError(f"component headers differ for {psqt_path.name}")
                while True:
                    psqt_data = psqt_stream.read(block_bytes)
                    positional_data = positional_stream.read(block_bytes)
                    if not psqt_data and not positional_data:
                        break
                    if len(psqt_data) != len(positional_data):
                        raise ValueError(f"component shard lengths differ for {psqt_path.name}")
                    if len(psqt_data) % RECORD_SIZE:
                        raise ValueError(f"truncated component block in {psqt_path.name}")
                    assigned = global_block % workers
                    global_block += 1
                    if assigned != worker_id:
                        continue
                    offsets = list(range(0, len(psqt_data), RECORD_SIZE))
                    if self.shuffle_block_size > 0:
                        rng.shuffle(offsets)
                    for offset in offsets:
                        psqt_record = psqt_data[offset : offset + RECORD_SIZE]
                        positional_record = positional_data[offset : offset + RECORD_SIZE]
                        if not self._same_position(psqt_record, positional_record):
                            raise ValueError(
                                f"component positions differ in {psqt_path.name} "
                                f"at byte offset {offset}"
                            )
                        if seen < self.skip_samples:
                            seen += 1
                            continue
                        psqt_sample = unpack_record(psqt_record)
                        positional_sample = unpack_record(positional_record)
                        yield {
                            "features": architecture_features_from_board(
                                psqt_sample.board, self.architecture
                            ),
                            "aux": unpack_aux(psqt_sample.aux_bits),
                            "psqt_score": psqt_sample.score,
                            "positional_score": positional_sample.score,
                        }
                        yielded += 1
                        if sample_limit is not None and yielded >= sample_limit:
                            return
            finally:
                if psqt_owner is not None:
                    psqt_owner.close()  # type: ignore[attr-defined]
                else:
                    psqt_stream.close()
                if positional_owner is not None:
                    positional_owner.close()  # type: ignore[attr-defined]
                else:
                    positional_stream.close()


class TotalLabelDataset(IterableDataset):
    """Read one CBin corpus whose score field is the final training target."""

    def __init__(
        self,
        path: Path,
        architecture: str,
        max_samples: int | None,
        skip_samples: int,
        seed: int,
        shuffle_block_size: int,
    ) -> None:
        super().__init__()
        self.path = path
        self.architecture = architecture
        self.max_samples = max_samples
        self.skip_samples = skip_samples
        self.seed = seed
        self.shuffle_block_size = shuffle_block_size

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        if self.skip_samples and workers != 1:
            raise ValueError("skip_samples requires a single-worker evaluation loader")
        sample_limit = worker_sample_limit(self.max_samples, worker_id, workers)
        if sample_limit == 0:
            return

        paths = compact_paths(self.path)
        rng = random.Random(self.seed)
        if self.shuffle_block_size > 0:
            rng.shuffle(paths)
        block_records = max(1, self.shuffle_block_size or 250_000)
        block_bytes = block_records * RECORD_SIZE
        global_block = 0
        seen = 0
        yielded = 0

        for path in paths:
            stream, owner = open_reader(path)
            try:
                header = stream.read(HEADER_SIZE)
                validate_header(header, path)
                while True:
                    data = stream.read(block_bytes)
                    if not data:
                        break
                    if len(data) % RECORD_SIZE:
                        raise ValueError(f"truncated target block in {path.name}")
                    assigned = global_block % workers
                    global_block += 1
                    if assigned != worker_id:
                        continue
                    offsets = list(range(0, len(data), RECORD_SIZE))
                    if self.shuffle_block_size > 0:
                        rng.shuffle(offsets)
                    for offset in offsets:
                        if seen < self.skip_samples:
                            seen += 1
                            continue
                        sample = unpack_record(data[offset : offset + RECORD_SIZE])
                        yield {
                            "board": sample.board,
                            "features": architecture_features_from_board(
                                sample.board, self.architecture
                            ),
                            "aux": unpack_aux(sample.aux_bits),
                            # Reuse the positional slot so the existing collator stays
                            # allocation-compatible. In total-label mode this value is
                            # never interpreted as positional supervision.
                            "psqt_score": 0,
                            "positional_score": sample.score,
                        }
                        yielded += 1
                        if sample_limit is not None and yielded >= sample_limit:
                            return
            finally:
                if owner is not None:
                    owner.close()  # type: ignore[attr-defined]
                else:
                    stream.close()


def make_loader(
    psqt_path: Path,
    positional_path: Path,
    architecture: str,
    max_samples: int | None,
    skip_samples: int,
    seed: int,
    shuffle_block_size: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        AlignedComponentDataset(
            psqt_path,
            positional_path,
            architecture,
            max_samples,
            skip_samples,
            seed,
            shuffle_block_size,
        ),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=identity_collate,
        persistent_workers=workers > 0,
    )


def make_total_loader(
    path: Path,
    architecture: str,
    max_samples: int | None,
    skip_samples: int,
    seed: int,
    shuffle_block_size: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        TotalLabelDataset(
            path,
            architecture,
            max_samples,
            skip_samples,
            seed,
            shuffle_block_size,
        ),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=identity_collate,
        persistent_workers=workers > 0,
    )


def collate(
    samples: list[dict[str, Any]],
    board_feature_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[int] = []
    offsets: list[int] = []
    cursor = 0
    psqt_scores: list[int] = []
    positional_scores: list[int] = []
    for sample in samples:
        offsets.append(cursor)
        row = list(sample["features"])
        aux = sample["aux"]
        if len(aux) != AUX_FEATURE_COUNT:
            raise ValueError("bad auxiliary feature count")
        row.extend(
            board_feature_count + index
            for index, active in enumerate(aux)
            if active
        )
        row.sort()
        features.extend(row)
        cursor += len(row)
        psqt_scores.append(int(sample["psqt_score"]))
        positional_scores.append(int(sample["positional_score"]))
    return (
        torch.tensor(features, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        raw_to_cp(torch.tensor(psqt_scores, dtype=torch.float32, device=device)),
        raw_to_cp(
            torch.tensor(positional_scores, dtype=torch.float32, device=device)
        ),
    )


def component_huber(
    prediction_cp: torch.Tensor,
    target_cp: torch.Tensor,
    target_scale: float,
    delta_cp: float,
) -> torch.Tensor:
    return nn.functional.smooth_l1_loss(
        prediction_cp / target_scale,
        target_cp / target_scale,
        beta=delta_cp / target_scale,
    )


def stockfish_score_only_wdl_error(
    prediction_cp: torch.Tensor,
    target_cp: torch.Tensor,
    exponent: float,
    input_offset: float,
    output_offset: float,
    input_scaling: float,
    output_scaling: float,
) -> torch.Tensor:
    """Return the current nnue-pytorch score-only error per sample.

    CBin scores use Stockfish's internal score unit, while this trainer's
    forward path reports centipawns. Convert both sides back to raw units before
    applying the asymmetric win-probability transform used by nnue-pytorch.
    """

    cp_to_raw = STOCKFISH_SCORE_CP_DENOMINATOR / STOCKFISH_SCORE_CP_NUMERATOR
    prediction_raw = prediction_cp * cp_to_raw
    target_raw = target_cp * cp_to_raw
    q = (prediction_raw - input_offset) / input_scaling
    q_mirror = (-prediction_raw - input_offset) / input_scaling
    prediction_probability = 0.5 * (
        1.0 + torch.sigmoid(q) - torch.sigmoid(q_mirror)
    )
    score = (target_raw - output_offset) / output_scaling
    score_mirror = (-target_raw - output_offset) / output_scaling
    target_probability = 0.5 * (
        1.0 + torch.sigmoid(score) - torch.sigmoid(score_mirror)
    )
    return (target_probability - prediction_probability).abs().pow(exponent)


def regression_loss(
    prediction_cp: torch.Tensor,
    target_cp: torch.Tensor,
    target_scale: float,
    loss_type: str,
    delta_cp: float,
    wdl_exponent: float,
    wdl_input_offset: float,
    wdl_output_offset: float,
    wdl_input_scaling: float,
    wdl_output_scaling: float,
) -> torch.Tensor:
    if loss_type == "huber":
        return component_huber(
            prediction_cp, target_cp, target_scale, delta_cp
        )
    if loss_type == "mse":
        return nn.functional.mse_loss(
            prediction_cp / target_scale,
            target_cp / target_scale,
        )
    if loss_type == "wdl":
        return stockfish_score_only_wdl_error(
            prediction_cp,
            target_cp,
            wdl_exponent,
            wdl_input_offset,
            wdl_output_offset,
            wdl_input_scaling,
            wdl_output_scaling,
        ).mean()
    raise ValueError(f"unknown loss type: {loss_type}")


NnueModel = Union[
    PhaseStackQuantizedNnueArchitecture, QuantizedSparseNnueArchitecture
]


def dense_layer_count(model: NnueModel) -> int:
    if isinstance(model, PhaseStackQuantizedNnueArchitecture):
        return model.dense_layer_count
    return len(model.hidden_layers)


def model_forward_components(
    model: NnueModel,
    feature_indices: torch.Tensor,
    offsets: torch.Tensor,
    hidden_scales: list[int],
    output_scale: int,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, PhaseStackQuantizedNnueArchitecture):
        return model.forward_components(
            feature_indices,
            offsets,
            hidden_scales,
            output_scale,
            activation,
            QUANTIZATION_CONVENTION_SCALE_CLEAN,
        )
    total = model(
        feature_indices,
        offsets,
        hidden_scales=hidden_scales,
        output_scale=output_scale,
        activation=activation,
        quantization_convention=QUANTIZATION_CONVENTION_SCALE_CLEAN,
    )
    psqt = trunc_ste(
        model.psqt_accumulator(feature_indices, offsets)
        / float(2 * model.psqt_weight_scale)
    )
    return total - psqt, psqt


def initialize_biases(
    model: NnueModel,
    hidden_scales: list[int],
    activation: str,
    hidden_fraction: float,
    first_fraction: float,
) -> None:
    with torch.no_grad():
        model.hidden1_bias.fill_(first_fraction)
        activation_scale = activation_output_scale(
            float(model.feature_weight_scale),
            activation,
            0,
            model.screlu_divisor,
            model.hidden_clip,
            QUANTIZATION_CONVENTION_SCALE_CLEAN,
        )
        target_code = model.hidden_clip * hidden_fraction
        for layer_index, scale in enumerate(hidden_scales, 1):
            accumulator_scale = activation_scale * model.linear_weight_scale
            bias_value = target_code * scale / accumulator_scale
            stacks = (
                model.phase_hidden_layers
                if isinstance(model, PhaseStackQuantizedNnueArchitecture)
                else (model.hidden_layers,)
            )
            for layers in stacks:
                layers[layer_index - 1].bias.fill_(bias_value)
            activation_scale = activation_output_scale(
                accumulator_scale / scale,
                activation,
                layer_index,
                model.screlu_divisor,
                model.hidden_clip,
                QUANTIZATION_CONVENTION_SCALE_CLEAN,
            )


def scheduled_lr(
    step: int,
    total_steps: int,
    peak: float,
    minimum: float,
    warmup: int,
    warmup_start: float = 0.0,
) -> float:
    if warmup > 0 and step < warmup:
        progress = (step + 1) / warmup
        return warmup_start + (peak - warmup_start) * progress
    remaining = max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / remaining))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def train_epoch(
    model: NnueModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    hidden_scales: list[int],
    output_scale: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.train()
    start = time.monotonic()
    samples_total = 0
    loss_total = 0.0
    psqt_abs = 0.0
    positional_abs = 0.0
    expected_steps = math.ceil(args.train_samples / args.batch_size)
    for batch_index, samples in enumerate(loader):
        feature_indices, offsets, psqt_target, positional_target = collate(
            samples, model.board_feature_count, device
        )
        lr = scheduled_lr(
            batch_index,
            expected_steps,
            args.lr,
            args.min_lr,
            args.lr_warmup_steps,
            args.warmup_start_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_multiplier", 1.0)
        optimizer.zero_grad(set_to_none=True)
        positional_prediction, psqt_prediction = model_forward_components(
            model,
            feature_indices,
            offsets,
            hidden_scales,
            output_scale,
            args.activation,
        )
        if args.label_mode == "total":
            total_target = (
                positional_target
                if args.loss_type == "wdl"
                else torch.clamp(positional_target, -2000.0, 2000.0)
            )
            loss = regression_loss(
                psqt_prediction + positional_prediction,
                total_target,
                args.target_scale,
                args.loss_type,
                args.huber_delta,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            )
        elif args.objective == "separate":
            positional_loss = regression_loss(
                positional_prediction,
                positional_target,
                args.target_scale,
                args.loss_type,
                args.huber_delta,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            )
            psqt_loss = regression_loss(
                psqt_prediction,
                psqt_target,
                args.target_scale,
                args.loss_type,
                args.huber_delta,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            )
            loss = positional_loss + psqt_loss
        else:
            total_target = torch.clamp(
                psqt_target + positional_target, -2000.0, 2000.0
            )
            loss = regression_loss(
                psqt_prediction + positional_prediction,
                total_target,
                args.target_scale,
                args.loss_type,
                args.huber_delta,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            )
        loss.backward()
        optimizer.step()
        model.clamp_quantized_weights()

        count = offsets.numel()
        samples_total += count
        loss_total += float(loss.detach()) * count
        if args.label_mode == "components":
            psqt_abs += float((psqt_prediction.detach() - psqt_target).abs().sum())
            positional_abs += float(
                (positional_prediction.detach() - positional_target).abs().sum()
            )
        if args.progress_batches and (batch_index + 1) % args.progress_batches == 0:
            elapsed = time.monotonic() - start
            progress = {
                "event": "train_progress",
                "batch": batch_index + 1,
                "samples": samples_total,
                "loss": loss_total / samples_total,
                "lr": lr,
                "samples_per_sec": samples_total / elapsed,
            }
            if args.label_mode == "components":
                progress.update(
                    {
                        "psqt_cp_mae": psqt_abs / samples_total,
                        "positional_cp_mae": positional_abs / samples_total,
                    }
                )
            print(json.dumps(progress, separators=(",", ":")), flush=True)
    if samples_total != args.train_samples:
        raise RuntimeError(
            f"expected {args.train_samples} training samples, got {samples_total}"
        )
    metrics: dict[str, Any] = {
        "loss": loss_total / samples_total,
        "samples": float(samples_total),
        "elapsed_sec": time.monotonic() - start,
    }
    if args.label_mode == "components":
        metrics.update(
            {
                "psqt_cp_mae": psqt_abs / samples_total,
                "positional_cp_mae": positional_abs / samples_total,
            }
        )
    return metrics


@torch.no_grad()
def evaluate(
    model: NnueModel,
    loader: DataLoader,
    device: torch.device,
    hidden_scales: list[int],
    output_scale: int,
    args: argparse.Namespace,
    split: str,
) -> dict[str, Any]:
    model.eval()
    count = 0
    total_abs = 0.0
    psqt_abs = 0.0
    positional_abs = 0.0
    sum_target = 0.0
    sum_prediction = 0.0
    sum_target_sq = 0.0
    sum_target_prediction = 0.0
    phase_count = [0] * 8
    phase_abs = [0.0] * 8
    paired_errors: list[float] = []
    wdl_power_sum = 0.0
    objective_loss_sum = 0.0
    for samples in loader:
        feature_indices, offsets, psqt_target, positional_target = collate(
            samples, model.board_feature_count, device
        )
        positional_prediction, psqt_prediction = model_forward_components(
            model,
            feature_indices,
            offsets,
            hidden_scales,
            output_scale,
            args.activation,
        )
        raw_target_cp = (
            positional_target
            if args.label_mode == "total"
            else psqt_target + positional_target
        )
        target = torch.clamp(raw_target_cp, -2000.0, 2000.0)
        prediction = psqt_prediction + positional_prediction
        error = (prediction - target).abs()
        if args.label_mode == "total":
            objective_target = (
                raw_target_cp if args.loss_type == "wdl" else target
            )
            objective_loss = regression_loss(
                prediction,
                objective_target,
                args.target_scale,
                args.loss_type,
                args.huber_delta,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            )
        elif args.objective == "separate":
            objective_loss = regression_loss(
                positional_prediction,
                positional_target,
                args.target_scale,
                args.loss_type,
                args.huber_delta,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            ) + regression_loss(
                psqt_prediction,
                psqt_target,
                args.target_scale,
                args.loss_type,
                args.huber_delta,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            )
        else:
            objective_loss = regression_loss(
                prediction,
                target,
                args.target_scale,
                args.loss_type,
                args.huber_delta,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            )
        wdl_power_sum += float(
            stockfish_score_only_wdl_error(
                prediction,
                raw_target_cp,
                args.wdl_exponent,
                args.wdl_input_offset,
                args.wdl_output_offset,
                args.wdl_input_scaling,
                args.wdl_output_scaling,
            ).sum()
        )
        buckets = model.psqt_bucket_indices(feature_indices, offsets)
        batch = offsets.numel()
        count += batch
        total_abs += float(error.sum())
        objective_loss_sum += float(objective_loss) * batch
        if args.label_mode == "components":
            psqt_abs += float((psqt_prediction - psqt_target).abs().sum())
            positional_abs += float(
                (positional_prediction - positional_target).abs().sum()
            )
        sum_target += float(target.sum())
        sum_prediction += float(prediction.sum())
        sum_target_sq += float((target * target).sum())
        sum_target_prediction += float((target * prediction).sum())
        if split == "ranking":
            paired_errors.extend(error.cpu().tolist())
        for phase in range(8):
            mask = buckets == phase
            phase_count[phase] += int(mask.sum())
            phase_abs[phase] += float(error[mask].sum())
    if count == 0:
        raise RuntimeError(f"empty {split} loader")
    target_mean = sum_target / count
    prediction_mean = sum_prediction / count
    variance = sum_target_sq - count * target_mean * target_mean
    covariance = sum_target_prediction - count * target_mean * prediction_mean
    metrics: dict[str, Any] = {
        "split": split,
        "samples": count,
        "objective_loss": objective_loss_sum / count,
        "cp_mae": total_abs / count,
        "wdl_score_only_loss": wdl_power_sum / count,
        "slope": covariance / variance if variance > 0 else float("nan"),
        "intercept_cp": prediction_mean
        - (covariance / variance if variance > 0 else 0.0) * target_mean,
        "phase": [
            {
                "phase": phase,
                "samples": phase_count[phase],
                "cp_mae": (
                    phase_abs[phase] / phase_count[phase]
                    if phase_count[phase]
                    else None
                ),
            }
            for phase in range(8)
        ],
    }
    if args.label_mode == "components":
        metrics.update(
            {
                "psqt_cp_mae": psqt_abs / count,
                "positional_cp_mae": positional_abs / count,
            }
        )
    if split == "ranking":
        metrics["absolute_errors_cp"] = paired_errors
    return metrics


@torch.no_grad()
def saturation(
    model: NnueModel,
    loader: DataLoader,
    device: torch.device,
    hidden_scales: list[int],
    args: argparse.Namespace,
    max_batches: int,
) -> dict[str, Any]:
    model.eval()
    stack_count = 8 if isinstance(model, PhaseStackQuantizedNnueArchitecture) else 1
    totals = [[0, 0, 0] for _ in range(stack_count)]
    clipped = [[0, 0, 0] for _ in range(stack_count)]
    zeros = [[0, 0, 0] for _ in range(stack_count)]
    for batch_index, samples in enumerate(loader):
        feature_indices, offsets, _psqt, _positional = collate(
            samples, model.board_feature_count, device
        )
        buckets = model.psqt_bucket_indices(feature_indices, offsets)
        first_acc = model.first_hidden_accumulator(feature_indices, offsets)
        first = model._activate_hidden(first_acc, args.activation, 0)
        first_clip = activation_numeric_clip(args.activation, 0, model.hidden_clip)
        for phase in range(stack_count):
            selected = (
                torch.nonzero(buckets == phase, as_tuple=False).squeeze(1)
                if stack_count == 8
                else torch.arange(offsets.numel(), device=offsets.device)
            )
            if selected.numel() == 0:
                continue
            accumulator = first_acc.index_select(0, selected)
            hidden = first.index_select(0, selected)
            totals[phase][0] += accumulator.numel()
            clipped[phase][0] += int((accumulator >= first_clip).sum())
            zeros[phase][0] += int((hidden <= 0).sum())
            activation_scale = activation_output_scale(
                float(model.feature_weight_scale),
                args.activation,
                0,
                model.screlu_divisor,
                model.hidden_clip,
                QUANTIZATION_CONVENTION_SCALE_CLEAN,
            )
            layers = (
                model.phase_hidden_layers[phase]
                if isinstance(model, PhaseStackQuantizedNnueArchitecture)
                else model.hidden_layers
            )
            for layer_index, (scale, layer) in enumerate(zip(hidden_scales, layers), 1):
                accumulator_scale = activation_scale * model.linear_weight_scale
                accumulator = nn.functional.linear(
                    hidden,
                    model.quantized_layer_weight(layer),
                    model.quantized_layer_bias(layer, accumulator_scale),
                )
                numeric_clip = activation_numeric_clip(
                    args.activation, layer_index, model.hidden_clip
                )
                hidden = model._activate_hidden(
                    torch.trunc(accumulator / scale), args.activation, layer_index
                )
                totals[phase][layer_index] += accumulator.numel()
                clipped[phase][layer_index] += int(
                    (accumulator >= scale * numeric_clip).sum()
                )
                zeros[phase][layer_index] += int((hidden <= 0).sum())
                activation_scale = activation_output_scale(
                    accumulator_scale / scale,
                    args.activation,
                    layer_index,
                    model.screlu_divisor,
                    model.hidden_clip,
                    QUANTIZATION_CONVENTION_SCALE_CLEAN,
                )
        if batch_index + 1 >= max_batches:
            break
    return {
        "phase": [
            {
                "phase": phase,
                "layers": [
                    {
                        "layer": layer + 1,
                        "clip_rate": clipped[phase][layer]
                        / max(1, totals[phase][layer]),
                        "zero_rate": zeros[phase][layer]
                        / max(1, totals[phase][layer]),
                        "values": totals[phase][layer],
                    }
                    for layer in range(3)
                ],
            }
            for phase in range(stack_count)
        ]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a phase NNUE from either total or component labels"
    )
    parser.add_argument("--total-data", type=Path)
    parser.add_argument("--eval-total-data", type=Path)
    parser.add_argument("--psqt-data", type=Path)
    parser.add_argument("--positional-data", type=Path)
    parser.add_argument("--eval-psqt-data", type=Path)
    parser.add_argument("--eval-positional-data", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arch", default="F2", choices=sorted(QUANTIZED_ARCHITECTURES))
    parser.add_argument("--activation", default=ACTIVATION_SCRELU_RELU16_ALL)
    parser.add_argument("--phase-stacks", type=int, choices=(1, 8), default=8)
    parser.add_argument("--objective", choices=("total", "separate"), default="separate")
    parser.add_argument("--loss-type", choices=("huber", "mse", "wdl"), default="huber")
    parser.add_argument("--hidden-scales", required=True, type=int, nargs="+")
    parser.add_argument("--output-scale", required=True, type=int)
    parser.add_argument("--hidden-clip", type=int, default=181)
    parser.add_argument("--screlu-divisor", type=int, default=128)
    parser.add_argument("--feature-weight-scale", type=int, default=181)
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--psqt-weight-scale", type=int, default=16)
    parser.add_argument("--target-scale", type=float, default=1000.0)
    parser.add_argument("--train-samples", type=int, default=50_000_000)
    parser.add_argument("--selection-samples", type=int, default=500_000)
    parser.add_argument("--ranking-samples", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--shuffle-block-size", type=int, default=250_000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--psqt-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=610)
    parser.add_argument("--warmup-start-lr", type=float, default=0.0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=200.0)
    parser.add_argument("--wdl-exponent", type=float, default=2.5)
    parser.add_argument("--wdl-input-offset", type=float, default=270.0)
    parser.add_argument("--wdl-output-offset", type=float, default=270.0)
    parser.add_argument("--wdl-input-scaling", type=float, default=340.0)
    parser.add_argument("--wdl-output-scaling", type=float, default=380.0)
    parser.add_argument("--hidden-bias-fraction", type=float, default=0.25)
    parser.add_argument("--first-bias-fraction", type=float, default=0.1)
    parser.add_argument("--saturation-batches", type=int, default=20)
    parser.add_argument("--progress-batches", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_paths = (args.total_data, args.eval_total_data)
    component_paths = (
        args.psqt_data,
        args.positional_data,
        args.eval_psqt_data,
        args.eval_positional_data,
    )
    has_total = any(path is not None for path in total_paths)
    has_components = any(path is not None for path in component_paths)
    if has_total == has_components:
        raise ValueError(
            "provide either --total-data/--eval-total-data or all four component paths"
        )
    if has_total:
        if any(path is None for path in total_paths):
            raise ValueError("total-label mode requires both total data paths")
        if args.objective != "total":
            raise ValueError("total-label mode requires --objective total")
        args.label_mode = "total"
        paths = total_paths
    else:
        if any(path is None for path in component_paths):
            raise ValueError("component-label mode requires all four component paths")
        args.label_mode = "components"
        paths = component_paths
    if any(scale <= 0 for scale in args.hidden_scales) or args.output_scale <= 0:
        raise ValueError("scales must be positive")
    if args.wdl_exponent <= 0:
        raise ValueError("WDL exponent must be positive")
    if args.wdl_input_scaling <= 0 or args.wdl_output_scaling <= 0:
        raise ValueError("WDL scaling values must be positive")
    if args.min_lr < 0 or args.warmup_start_lr < 0:
        raise ValueError("learning rates must be non-negative")
    if args.min_lr > args.lr or args.warmup_start_lr > args.lr:
        raise ValueError("minimum and warmup-start learning rates must not exceed peak LR")
    if args.resume_optimizer and args.resume_checkpoint is None:
        raise ValueError("--resume-optimizer requires --resume-checkpoint")
    if args.evaluate_only and args.resume_checkpoint is None:
        raise ValueError("--evaluate-only requires --resume-checkpoint")
    if args.loss_type == "wdl" and (
        args.label_mode != "total" or args.objective != "total"
    ):
        raise ValueError("WDL score-only loss requires total-label/total-objective mode")
    for path in paths:
        assert path is not None
        if not path.exists():
            raise FileNotFoundError(path)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    config = QUANTIZED_ARCHITECTURES[args.arch]
    model_class = (
        PhaseStackQuantizedNnueArchitecture
        if args.phase_stacks == 8
        else QuantizedSparseNnueArchitecture
    )
    model = model_class(
        config,
        hidden_clip=args.hidden_clip,
        feature_weight_scale=args.feature_weight_scale,
        linear_weight_scale=args.linear_weight_scale,
        output_weight_scale=args.output_weight_scale,
        screlu_divisor=args.screlu_divisor,
        psqt_weight_scale=args.psqt_weight_scale,
        psqt_master_scale_to_cp=args.target_scale,
        use_psqt=True,
    ).to(device)
    if len(args.hidden_scales) != dense_layer_count(model):
        raise ValueError(
            f"expected {dense_layer_count(model)} hidden scales, got {args.hidden_scales}"
        )
    initialize_biases(
        model,
        args.hidden_scales,
        args.activation,
        args.hidden_bias_fraction,
        args.first_bias_fraction,
    )
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        resume_checkpoint = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        expected_metadata = {
            "architecture": args.arch,
            "phase_stacks": args.phase_stacks,
            "component_supervision": args.objective,
            "label_mode": args.label_mode,
            "loss_type": args.loss_type,
            "hidden_scales": args.hidden_scales,
            "output_scale": args.output_scale,
            "activation": args.activation,
            "hidden_clip": args.hidden_clip,
            "screlu_divisor": args.screlu_divisor,
            "feature_weight_scale": args.feature_weight_scale,
            "linear_weight_scale": args.linear_weight_scale,
            "output_weight_scale": args.output_weight_scale,
            "psqt_weight_scale": args.psqt_weight_scale,
            "target_scale": args.target_scale,
        }
        for key, expected in expected_metadata.items():
            actual = resume_checkpoint.get(key)
            if actual != expected:
                raise ValueError(
                    f"resume checkpoint {key} mismatch: "
                    f"expected={expected!r} actual={actual!r}"
                )
        if args.loss_type == "huber" and resume_checkpoint.get("huber_delta") != args.huber_delta:
            raise ValueError("resume checkpoint Huber delta mismatch")
        model.load_state_dict(resume_checkpoint["model_state"])
    assert model.psqt is not None
    psqt_ids = {id(parameter) for parameter in model.psqt.parameters()}
    base = [parameter for parameter in model.parameters() if id(parameter) not in psqt_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base, "lr_multiplier": 1.0},
            {
                "params": list(model.psqt.parameters()),
                "lr_multiplier": args.psqt_lr / args.lr,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.resume_optimizer:
        assert resume_checkpoint is not None
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        expected_multipliers = (1.0, args.psqt_lr / args.lr)
        if len(optimizer.param_groups) != len(expected_multipliers):
            raise ValueError("resume optimizer parameter-group count mismatch")
        for group, multiplier in zip(optimizer.param_groups, expected_multipliers):
            group["lr_multiplier"] = multiplier
            group["weight_decay"] = args.weight_decay

    loader_factory = make_total_loader if args.label_mode == "total" else make_loader
    if args.label_mode == "total":
        assert args.total_data is not None and args.eval_total_data is not None
        train_loader = loader_factory(
            args.total_data, config.transform, args.train_samples, 0, args.seed,
            args.shuffle_block_size, args.batch_size, args.workers,
        )
        selection_loader = loader_factory(
            args.eval_total_data, config.transform, args.selection_samples, 0,
            args.seed, 0, args.batch_size, 0,
        )
        ranking_loader = loader_factory(
            args.eval_total_data, config.transform, args.ranking_samples,
            args.selection_samples, args.seed, 0, args.batch_size, 0,
        )
        saturation_loader = loader_factory(
            args.eval_total_data, config.transform,
            args.batch_size * args.saturation_batches, 0, args.seed, 0,
            args.batch_size, 0,
        )
    else:
        assert all(path is not None for path in component_paths)
        train_loader = loader_factory(
            args.psqt_data, args.positional_data, config.transform,
            args.train_samples, 0, args.seed, args.shuffle_block_size,
            args.batch_size, args.workers,
        )
        selection_loader = loader_factory(
            args.eval_psqt_data, args.eval_positional_data, config.transform,
            args.selection_samples, 0, args.seed, 0, args.batch_size, 0,
        )
        ranking_loader = loader_factory(
            args.eval_psqt_data, args.eval_positional_data, config.transform,
            args.ranking_samples, args.selection_samples, args.seed, 0,
            args.batch_size, 0,
        )
        saturation_loader = loader_factory(
            args.eval_psqt_data, args.eval_positional_data, config.transform,
            args.batch_size * args.saturation_batches, 0, args.seed, 0,
            args.batch_size, 0,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "start",
                "architecture": f"shared_transformer_{args.phase_stacks}_phase_stacks",
                "objective": args.objective,
                "label_mode": args.label_mode,
                "loss_type": args.loss_type,
                "huber_delta": args.huber_delta if args.loss_type == "huber" else None,
                "wdl": {
                    "exponent": args.wdl_exponent,
                    "input_offset": args.wdl_input_offset,
                    "output_offset": args.wdl_output_offset,
                    "input_scaling": args.wdl_input_scaling,
                    "output_scaling": args.wdl_output_scaling,
                },
                "arch": args.arch,
                "hidden_scales": args.hidden_scales,
                "output_scale": args.output_scale,
                "train_samples": args.train_samples,
                "resume_checkpoint": (
                    str(args.resume_checkpoint)
                    if args.resume_checkpoint is not None
                    else None
                ),
                "resume_optimizer": args.resume_optimizer,
                "phase_formula": "clamp((piece_count - 1) // 4, 0, 7)",
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    if args.evaluate_only:
        train_metrics = {
            "loss": None,
            "samples": 0.0,
            "elapsed_sec": 0.0,
        }
    else:
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.hidden_scales,
            args.output_scale,
            args,
        )
    selection_metrics = evaluate(
        model,
        selection_loader,
        device,
        args.hidden_scales,
        args.output_scale,
        args,
        "selection",
    )
    ranking_metrics = evaluate(
        model,
        ranking_loader,
        device,
        args.hidden_scales,
        args.output_scale,
        args,
        "ranking",
    )
    saturation_metrics = saturation(
        model,
        saturation_loader,
        device,
        args.hidden_scales,
        args,
        args.saturation_batches,
    )
    ranking_errors = ranking_metrics.pop("absolute_errors_cp")
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "architecture": args.arch,
        "phase_stacks": args.phase_stacks,
        "component_supervision": args.objective,
        "label_mode": args.label_mode,
        "loss_type": args.loss_type,
        "huber_delta": args.huber_delta if args.loss_type == "huber" else None,
        "wdl_exponent": args.wdl_exponent,
        "wdl_input_offset": args.wdl_input_offset,
        "wdl_output_offset": args.wdl_output_offset,
        "wdl_input_scaling": args.wdl_input_scaling,
        "wdl_output_scaling": args.wdl_output_scaling,
        "hidden_sizes": config.hidden_sizes,
        "hidden_scales": args.hidden_scales,
        "output_scale": args.output_scale,
        "activation": args.activation,
        "quantization_convention": QUANTIZATION_CONVENTION_SCALE_CLEAN,
        "hidden_clip": args.hidden_clip,
        "screlu_divisor": args.screlu_divisor,
        "feature_weight_scale": args.feature_weight_scale,
        "linear_weight_scale": args.linear_weight_scale,
        "output_weight_scale": args.output_weight_scale,
        "psqt_weight_scale": args.psqt_weight_scale,
        "target_scale": args.target_scale,
        "metrics": {
            "train": train_metrics,
            "selection": selection_metrics,
            "ranking": ranking_metrics,
            "saturation": saturation_metrics,
        },
        "args": vars(args),
    }
    torch.save(checkpoint, args.output_dir / "phase_component_best.pt")
    (args.output_dir / "ranking_errors.json").write_text(
        json.dumps(ranking_errors, separators=(",", ":")), encoding="utf-8"
    )
    summary = {
        "event": "training_complete",
        "train": train_metrics,
        "selection": selection_metrics,
        "ranking": ranking_metrics,
        "saturation": saturation_metrics,
        "checkpoint": str(args.output_dir / "phase_component_best.pt"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
