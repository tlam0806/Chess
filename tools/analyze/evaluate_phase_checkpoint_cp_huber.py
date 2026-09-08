#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.quantized_nnue_architectures import (  # noqa: E402
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    PhaseStackQuantizedNnueArchitecture,
)
from chess_nnue.compact_board_data import (  # noqa: E402
    HEADER_SIZE,
    RECORD_SIZE,
    canonical_architecture_input,
    compact_paths,
    open_reader,
    position_split_bucket,
    unpack_record,
    uses_horizontal_mirror,
    validate_header,
)
from chess_nnue.training_targets import is_value_none_target  # noqa: E402
from chess_nnue.train_value import choose_device  # noqa: E402
from tools.train.train_nnue_architecture import (  # noqa: E402
    identity_collate,
    make_loader,
    validate_training_data,
    worker_sample_limit,
)
from tools.train.train_quantized_nnue_architecture import evaluate_quantized_cp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a legacy eight-phase NNUE checkpoint with the current "
            "CP-Huber validation pipeline"
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=("auto", "cbin", "jsonl"), default="auto")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-samples", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--progress-batches", type=int, default=100)
    parser.add_argument("--cp-huber-delta", type=float, default=200.0)
    return parser.parse_args()


def load_phase_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[PhaseStackQuantizedNnueArchitecture, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = str(checkpoint.get("architecture"))
    if architecture not in QUANTIZED_ARCHITECTURES:
        raise ValueError(f"unknown checkpoint architecture: {architecture}")
    if int(checkpoint.get("phase_stacks", -1)) != 8:
        raise ValueError("checkpoint is not an eight-phase model")
    convention = str(checkpoint.get("quantization_convention"))
    if convention != QUANTIZATION_CONVENTION_SCALE_CLEAN:
        raise ValueError(f"unsupported quantization convention: {convention}")

    config = QUANTIZED_ARCHITECTURES[architecture]
    if tuple(checkpoint.get("hidden_sizes", ())) != config.hidden_sizes:
        raise ValueError("checkpoint hidden sizes do not match architecture")
    model = PhaseStackQuantizedNnueArchitecture(
        config,
        phase_layout=str(checkpoint.get("phase_layout", "independent")),
        hidden_clip=int(checkpoint["hidden_clip"]),
        feature_weight_scale=int(checkpoint["feature_weight_scale"]),
        linear_weight_scale=int(checkpoint["linear_weight_scale"]),
        output_weight_scale=int(checkpoint["output_weight_scale"]),
        screlu_divisor=int(checkpoint["screlu_divisor"]),
        psqt_weight_scale=int(checkpoint["psqt_weight_scale"]),
        psqt_master_scale_to_cp=float(checkpoint["target_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


class ExactCompactSplitDataset(IterableDataset):
    """Match the trainer's per-worker split while filtering before feature expansion."""

    def __init__(
        self,
        path: Path,
        architecture: str,
        split: str,
        split_mod: int,
        val_mod: int,
        test_mod: int,
        max_samples: int,
    ) -> None:
        super().__init__()
        self.path = path
        self.architecture = architecture
        self.split = split
        self.split_mod = split_mod
        self.val_mod = val_mod
        self.test_mod = test_mod
        self.max_samples = max_samples

    def _belongs(self, bucket: int) -> bool:
        return bucket == (self.val_mod if self.split == "val" else self.test_mod)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        sample_limit = worker_sample_limit(self.max_samples, worker_id, workers)
        yielded = 0
        global_index = 0
        block_records = 250_000
        block_bytes = block_records * RECORD_SIZE
        mirrored = uses_horizontal_mirror(self.architecture)

        for path in compact_paths(self.path):
            stream, owner = open_reader(path)
            try:
                validate_header(stream.read(HEADER_SIZE), path)
                while True:
                    data = stream.read(block_bytes)
                    if not data:
                        break
                    if len(data) % RECORD_SIZE:
                        raise ValueError(f"truncated compact block in {path}")
                    records = len(data) // RECORD_SIZE
                    first = (worker_id - global_index) % workers
                    for local_index in range(first, records, workers):
                        offset = local_index * RECORD_SIZE
                        sample = unpack_record(data[offset : offset + RECORD_SIZE])
                        if is_value_none_target(sample.score):
                            continue
                        if mirrored:
                            canonical_board, canonical_aux_bits, features, aux = (
                                canonical_architecture_input(
                                    sample.board, sample.aux_bits, self.architecture
                                )
                            )
                            bucket = position_split_bucket(
                                canonical_board, canonical_aux_bits, self.split_mod
                            )
                        else:
                            bucket = position_split_bucket(
                                sample.board, sample.aux_bits, self.split_mod
                            )
                            if not self._belongs(bucket):
                                continue
                            _board, _aux_bits, features, aux = canonical_architecture_input(
                                sample.board, sample.aux_bits, self.architecture
                            )
                        if not self._belongs(bucket):
                            continue
                        yield {
                            "features": features,
                            "aux": aux,
                            "score": sample.score,
                            "ply": sample.ply,
                            "result": sample.result,
                        }
                        yielded += 1
                        if sample_limit is not None and yielded >= sample_limit:
                            return
                    global_index += records
            finally:
                if owner is not None:
                    owner.close()  # type: ignore[attr-defined]
                else:
                    stream.close()


def make_exact_loader(
    args: argparse.Namespace, architecture: str, data_format: str
) -> DataLoader:
    if data_format != "cbin":
        return make_loader(
            args.data,
            architecture,
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
    return DataLoader(
        ExactCompactSplitDataset(
            args.data,
            architecture,
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
    )


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0 or args.cp_huber_delta <= 0:
        raise ValueError("sample count, batch size, and Huber delta must be positive")
    if args.workers < 0 or args.torch_threads <= 0:
        raise ValueError("workers must be non-negative and torch threads positive")

    data_format = validate_training_data(args.data, args.data_format)
    torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    model, checkpoint = load_phase_model(args.checkpoint, device)
    architecture = str(checkpoint["architecture"])
    config = QUANTIZED_ARCHITECTURES[architecture]
    loader = make_exact_loader(args, config.transform, data_format)
    started = time.monotonic()
    loss, cp_mae, samples, bins = evaluate_quantized_cp(
        architecture,
        args.split,
        model,
        loader,
        device,
        config.board_feature_count,
        float(checkpoint["target_scale"]),
        1.0,
        2.6,
        [int(value) for value in checkpoint["hidden_scales"]],
        int(checkpoint["output_scale"]),
        None,
        0,
        args.progress_batches,
        started,
        str(checkpoint["activation"]),
        str(checkpoint["quantization_convention"]),
        args.cp_huber_delta,
        "quantized",
        "cp_huber",
        0.75,
    )
    print(
        json.dumps(
            {
                "event": "common_cp_huber_result",
                "checkpoint": str(args.checkpoint),
                "data": str(args.data),
                "split": args.split,
                "split_policy": (
                    f"crc32(board+aux) mod {args.split_mod}; "
                    f"val={args.val_mod}; test={args.test_mod}"
                ),
                "samples": samples,
                "cp_huber_delta": args.cp_huber_delta,
                "cp_huber_loss": loss,
                "cp_mae": cp_mae,
                "cp_mae_bins": bins,
                "elapsed_sec": time.monotonic() - started,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
