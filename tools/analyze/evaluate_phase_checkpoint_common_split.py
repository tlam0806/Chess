#!/usr/bin/env python3
"""Evaluate a phase checkpoint on another architecture's canonical split.

This is primarily for comparing the production F2 checkpoint with F2M
training: positions are selected with F2M's mirror-canonical CRC32 key while
features are still expanded according to the checkpoint's own architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.compact_board_data import (  # noqa: E402
    canonical_architecture_input,
    compact_paths,
    iter_compact_samples,
    position_split_bucket,
)
from chess_nnue.quantized_nnue_architectures import QUANTIZED_ARCHITECTURES  # noqa: E402
from chess_nnue.training_targets import is_value_none_target  # noqa: E402
from chess_nnue.train_value import choose_device  # noqa: E402
from tools.analyze.evaluate_phase_checkpoint_cp_huber import load_phase_model  # noqa: E402
from tools.train.train_nnue_architecture import (  # noqa: E402
    identity_collate,
    validate_training_data,
    worker_sample_limit,
)
from tools.train.train_phase_component_nnue import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split-arch", choices=("F2", "F2M"), default="F2M")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-samples", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument("--huber-delta", type=float, default=200.0)
    return parser.parse_args()


class CommonSplitDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        feature_transform: str,
        split_transform: str,
        split: str,
        split_mod: int,
        val_mod: int,
        test_mod: int,
        max_samples: int,
    ) -> None:
        super().__init__()
        self.path = path
        self.feature_transform = feature_transform
        self.split_transform = split_transform
        self.split = split
        self.split_mod = split_mod
        self.val_mod = val_mod
        self.test_mod = test_mod
        self.max_samples = max_samples

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        limit = worker_sample_limit(self.max_samples, worker_id, workers)
        wanted_bucket = self.val_mod if self.split == "val" else self.test_mod
        yielded = 0
        paths = compact_paths(self.path)

        # Match the current trainer exactly: one owner per compressed shard,
        # then an equal accepted-sample quota per worker.
        for path_index, path in enumerate(paths):
            if len(paths) >= workers and path_index % workers != worker_id:
                continue
            for sample in iter_compact_samples(path):
                if is_value_none_target(sample.score):
                    continue
                split_board, split_aux_bits, _split_features, _split_aux = (
                    canonical_architecture_input(
                        sample.board, sample.aux_bits, self.split_transform
                    )
                )
                if position_split_bucket(
                    split_board, split_aux_bits, self.split_mod
                ) != wanted_bucket:
                    continue
                _board, _aux_bits, features, aux = canonical_architecture_input(
                    sample.board, sample.aux_bits, self.feature_transform
                )
                yield {
                    "features": features,
                    "aux": aux,
                    "score": sample.score,
                    "ply": sample.ply,
                    "result": sample.result,
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("invalid sample, batch, or worker count")
    validate_training_data(args.data, "cbin")
    torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    model, checkpoint = load_phase_model(args.checkpoint, device)
    model_arch = str(checkpoint["architecture"])
    model_config = QUANTIZED_ARCHITECTURES[model_arch]
    split_config = QUANTIZED_ARCHITECTURES[args.split_arch]
    loader = DataLoader(
        CommonSplitDataset(
            args.data,
            model_config.transform,
            split_config.transform,
            args.split,
            args.split_mod,
            args.val_mod,
            args.test_mod,
            args.max_samples,
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=identity_collate,
        persistent_workers=args.workers > 0,
        pin_memory=False,
    )
    eval_args = SimpleNamespace(
        activation=str(checkpoint["activation"]),
        label_mode="total",
        objective="total",
        loss_type="huber",
        target_scale=float(checkpoint["target_scale"]),
        huber_delta=args.huber_delta,
        wdl_exponent=float(checkpoint.get("wdl_exponent", 2.5)),
        wdl_input_offset=float(checkpoint.get("wdl_input_offset", 270.0)),
        wdl_output_offset=float(checkpoint.get("wdl_output_offset", 270.0)),
        wdl_input_scaling=float(checkpoint.get("wdl_input_scaling", 340.0)),
        wdl_output_scaling=float(checkpoint.get("wdl_output_scaling", 380.0)),
    )
    started = time.monotonic()
    metrics = evaluate(
        model,
        loader,
        device,
        [int(value) for value in checkpoint["hidden_scales"]],
        int(checkpoint["output_scale"]),
        eval_args,
        args.split,
    )
    print(json.dumps({
        "event": "common_split_result",
        "checkpoint": str(args.checkpoint),
        "model_arch": model_arch,
        "split_arch": args.split_arch,
        "split": args.split,
        "split_mod": args.split_mod,
        "val_mod": args.val_mod,
        "test_mod": args.test_mod,
        "elapsed_sec": time.monotonic() - started,
        **metrics,
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
