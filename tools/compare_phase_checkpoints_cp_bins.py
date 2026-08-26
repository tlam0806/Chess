#!/usr/bin/env python3
"""Compare two phase checkpoints on identical CP-range validation samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.compact_board_data import (  # noqa: E402
    canonical_architecture_input,
    compact_paths,
    iter_compact_samples,
    position_split_bucket,
)
from nn.quantized_nnue_architectures import QUANTIZED_ARCHITECTURES  # noqa: E402
from nn.training_targets import is_value_none_target  # noqa: E402
from tools.evaluate_phase_checkpoint_cp_huber import load_phase_model  # noqa: E402
from tools.train_nnue_architecture import identity_collate, worker_sample_limit  # noqa: E402
from tools.train_phase_component_nnue import collate, model_forward_components  # noqa: E402


CP_BINS = (
    (0.0, 50.0),
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 1500.0),
    (1500.0, 2000.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", required=True, type=Path)
    parser.add_argument("--checkpoint-b", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split-arch", choices=("F2", "F2M"), default="F2M")
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--huber-delta", type=float, default=200.0)
    parser.add_argument("--progress-batches", type=int, default=20)
    return parser.parse_args()


class PairedSplitDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        transform_a: str,
        transform_b: str,
        split_transform: str,
        split_mod: int,
        val_mod: int,
        max_samples: int,
    ) -> None:
        super().__init__()
        self.path = path
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.split_transform = split_transform
        self.split_mod = split_mod
        self.val_mod = val_mod
        self.max_samples = max_samples

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        limit = worker_sample_limit(self.max_samples, worker_id, workers)
        yielded = 0
        paths = compact_paths(self.path)
        for path_index, path in enumerate(paths):
            if len(paths) >= workers and path_index % workers != worker_id:
                continue
            for sample in iter_compact_samples(path):
                if is_value_none_target(sample.score):
                    continue
                split_board, split_aux, _features, _aux = canonical_architecture_input(
                    sample.board, sample.aux_bits, self.split_transform
                )
                if position_split_bucket(
                    split_board, split_aux, self.split_mod
                ) != self.val_mod:
                    continue
                _board, _bits, features_a, aux_a = canonical_architecture_input(
                    sample.board, sample.aux_bits, self.transform_a
                )
                _board, _bits, features_b, aux_b = canonical_architecture_input(
                    sample.board, sample.aux_bits, self.transform_b
                )
                yield {
                    "features_a": features_a,
                    "aux_a": aux_a,
                    "features_b": features_b,
                    "aux_b": aux_b,
                    "score": sample.score,
                    "ply": sample.ply,
                    "result": sample.result,
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def model_batch(samples: list[dict[str, Any]], suffix: str) -> list[dict[str, Any]]:
    return [
        {
            "features": sample[f"features_{suffix}"],
            "aux": sample[f"aux_{suffix}"],
            "score": sample["score"],
            "ply": sample["ply"],
            "result": sample["result"],
        }
        for sample in samples
    ]


def empty_stats() -> dict[str, Any]:
    return {
        "loss_sum": 0.0,
        "mae_sum": 0.0,
        "bins": [
            {"samples": 0, "loss_sum": 0.0, "mae_sum": 0.0}
            for _ in CP_BINS
        ],
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")
    model_a, checkpoint_a = load_phase_model(args.checkpoint_a, device)
    model_b, checkpoint_b = load_phase_model(args.checkpoint_b, device)
    config_a = QUANTIZED_ARCHITECTURES[str(checkpoint_a["architecture"])]
    config_b = QUANTIZED_ARCHITECTURES[str(checkpoint_b["architecture"])]
    split_config = QUANTIZED_ARCHITECTURES[args.split_arch]
    loader = DataLoader(
        PairedSplitDataset(
            args.data,
            config_a.transform,
            config_b.transform,
            split_config.transform,
            args.split_mod,
            args.val_mod,
            args.max_samples,
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=identity_collate,
        persistent_workers=args.workers > 0,
        pin_memory=False,
    )
    model_a.eval()
    model_b.eval()
    stats = [empty_stats(), empty_stats()]
    checkpoints = (checkpoint_a, checkpoint_b)
    models = (model_a, model_b)
    started = time.monotonic()
    total_samples = 0

    for batch_index, samples in enumerate(loader, 1):
        predictions = []
        target = None
        for index, (model, checkpoint, suffix) in enumerate(
            zip(models, checkpoints, ("a", "b"))
        ):
            feature_indices, offsets, _psqt, raw_target = collate(
                model_batch(samples, suffix), model.board_feature_count, device
            )
            positional, psqt = model_forward_components(
                model,
                feature_indices,
                offsets,
                [int(value) for value in checkpoint["hidden_scales"]],
                int(checkpoint["output_scale"]),
                str(checkpoint["activation"]),
            )
            predictions.append(positional + psqt)
            if target is None:
                target = torch.clamp(raw_target, -2000.0, 2000.0)
        assert target is not None
        target_abs = target.abs()
        batch_size = int(target.numel())
        total_samples += batch_size
        for model_stats, prediction, checkpoint in zip(
            stats, predictions, checkpoints
        ):
            target_scale = float(checkpoint["target_scale"])
            losses = F.smooth_l1_loss(
                prediction / target_scale,
                target / target_scale,
                beta=args.huber_delta / target_scale,
                reduction="none",
            )
            errors = (prediction - target).abs()
            model_stats["loss_sum"] += float(losses.sum())
            model_stats["mae_sum"] += float(errors.sum())
            for bin_index, (lower, upper) in enumerate(CP_BINS):
                mask = target_abs >= lower
                mask &= target_abs <= upper if upper == 2000.0 else target_abs < upper
                count = int(mask.sum())
                if count:
                    item = model_stats["bins"][bin_index]
                    item["samples"] += count
                    item["loss_sum"] += float(losses[mask].sum())
                    item["mae_sum"] += float(errors[mask].sum())
        if args.progress_batches and batch_index % args.progress_batches == 0:
            print(json.dumps({
                "event": "progress",
                "samples": total_samples,
                "elapsed_sec": time.monotonic() - started,
            }, separators=(",", ":")), flush=True)

    results = []
    for checkpoint_path, checkpoint, model_stats in zip(
        (args.checkpoint_a, args.checkpoint_b), checkpoints, stats
    ):
        results.append({
            "checkpoint": str(checkpoint_path),
            "architecture": str(checkpoint["architecture"]),
            "loss": model_stats["loss_sum"] / total_samples,
            "cp_mae": model_stats["mae_sum"] / total_samples,
            "bins": [
                {
                    "min_abs_target_cp": lower,
                    "max_abs_target_cp": upper,
                    "samples": item["samples"],
                    "loss": item["loss_sum"] / item["samples"],
                    "cp_mae": item["mae_sum"] / item["samples"],
                }
                for (lower, upper), item in zip(CP_BINS, model_stats["bins"])
            ],
        })
    print(json.dumps({
        "event": "cp_bin_comparison",
        "samples": total_samples,
        "elapsed_sec": time.monotonic() - started,
        "split_arch": args.split_arch,
        "results": results,
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
