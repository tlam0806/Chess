from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.compact_board_data import (
    HEADER_SIZE,
    RECORD_SIZE,
    canonical_architecture_input,
    compact_paths,
    iter_compact_samples,
    open_reader,
    pack_aux,
    pack_board_from_raw_features,
    position_split_bucket,
    unpack_record,
    validate_compact_dataset,
    validate_header,
)
from chess_nnue.nnue_architectures import ARCHITECTURES, SparseNnueArchitecture
from chess_nnue.training_targets import (
    DEFAULT_SCORE_LAMBDA,
    DEFAULT_WDL_LOSS_EXPONENT,
    TRAINING_PIPELINE_VERSION,
    is_value_none_target,
    stockfish_score_to_cp,
    stockfish_wdl_loss,
)
from chess_nnue.train_value import DEFAULT_TARGET_SCALE, choose_device
from chess_nnue.value_net import AUX_FEATURE_COUNT, make_batch


def worker_sample_limit(
    max_samples: int | None,
    worker_id: int,
    num_workers: int,
) -> int | None:
    if max_samples is None:
        return None
    workers = max(1, num_workers)
    base, remainder = divmod(max_samples, workers)
    return base + (1 if worker_id < remainder else 0)


def identity_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples


class JsonlSplitDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        architecture: str,
        split: str,
        split_mod: int,
        val_mod: int,
        test_mod: int,
        max_samples: int | None,
        seed: int,
    ) -> None:
        super().__init__()
        self.path = path
        self.architecture = architecture
        self.split = split
        self.split_mod = split_mod
        self.val_mod = val_mod
        self.test_mod = test_mod
        self.max_samples = max_samples
        self.seed = seed

    def _bucket_belongs_to_split(self, bucket: int) -> bool:
        if self.split == "all":
            return True
        if self.split == "test":
            return bucket == self.test_mod
        if self.split == "val":
            return bucket == self.val_mod
        if self.split == "train":
            return bucket != self.test_mod and bucket != self.val_mod
        raise ValueError(f"unknown split: {self.split}")

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        max_samples = worker_sample_limit(self.max_samples, worker_id, num_workers)
        yielded = 0

        with self.path.open("r", encoding="utf-8") as file:
            for line_index, line in enumerate(file):
                if line_index % num_workers != worker_id:
                    continue
                line = line.strip()
                if not line:
                    continue

                sample = json.loads(line)
                raw_features = sample.get("features")
                aux = sample.get("aux")
                score = sample.get("score")
                ply = sample.get("ply")
                result = sample.get("result")
                if not raw_features or aux is None or score is None or ply is None or result is None:
                    continue
                if len(aux) != AUX_FEATURE_COUNT:
                    continue
                board = pack_board_from_raw_features([int(value) for value in raw_features])
                aux_bits = pack_aux([int(value) for value in aux])
                canonical_board, canonical_aux_bits, features, canonical_aux = (
                    canonical_architecture_input(board, aux_bits, self.architecture)
                )
                bucket = position_split_bucket(
                    canonical_board, canonical_aux_bits, self.split_mod
                )
                if not self._bucket_belongs_to_split(bucket):
                    continue
                score_value = int(score)
                ply_value = int(ply)
                result_value = int(result)
                if is_value_none_target(score_value):
                    continue
                if not 0 <= ply_value <= 0x3FFF or result_value not in (-1, 0, 1):
                    continue

                yield {
                    "features": features,
                    "aux": canonical_aux,
                    "score": score_value,
                    "ply": ply_value,
                    "result": result_value,
                }
                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    break


class CompactSplitDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        architecture: str,
        split: str,
        split_mod: int,
        val_mod: int,
        test_mod: int,
        max_samples: int | None,
        seed: int,
        shuffle_block_size: int,
        cache_unshuffled: bool = False,
    ) -> None:
        super().__init__()
        self.path = path
        self.architecture = architecture
        self.split = split
        self.split_mod = split_mod
        self.val_mod = val_mod
        self.test_mod = test_mod
        self.max_samples = max_samples
        self.seed = seed
        self.shuffle_block_size = shuffle_block_size
        self.cache_unshuffled = cache_unshuffled
        self.iteration = 0
        # Evaluation splits are deterministic and revisited after every epoch.
        # Keep only the compact 40-byte samples in each persistent DataLoader
        # worker after its first scan. Caching expanded Python feature lists
        # would use an order of magnitude more RAM.
        self._cached_unshuffled_samples: list[Any] | None = None

    def _bucket_belongs_to_split(self, bucket: int) -> bool:
        if self.split == "all":
            return True
        if self.split == "test":
            return bucket == self.test_mod
        if self.split == "val":
            return bucket == self.val_mod
        if self.split == "train":
            return bucket != self.test_mod and bucket != self.val_mod
        raise ValueError(f"unknown split: {self.split}")

    def _sample_from_compact(self, sample: Any) -> dict[str, Any] | None:
        if is_value_none_target(sample.score):
            return None
        canonical_board, canonical_aux_bits, features, aux = (
            canonical_architecture_input(
                sample.board, sample.aux_bits, self.architecture
            )
        )
        bucket = position_split_bucket(
            canonical_board, canonical_aux_bits, self.split_mod
        )
        if not self._bucket_belongs_to_split(bucket):
            return None
        return {
            "features": features,
            "aux": aux,
            "score": sample.score,
            "ply": sample.ply,
            "result": sample.result,
        }

    def _sample_from_record(self, record: bytes) -> dict[str, Any] | None:
        return self._sample_from_compact(unpack_record(record))

    def _compact_paths(self) -> list[Path]:
        return compact_paths(self.path)

    def _iter_stream_blocks(self, path: Path, base_index: int, block_size: int) -> Iterable[tuple[int, bytes]]:
        stream, owner = open_reader(path)
        try:
            header = stream.read(HEADER_SIZE)
            validate_header(header, path)
            block_bytes = block_size * RECORD_SIZE
            current_base = base_index
            while True:
                data = stream.read(block_bytes)
                if not data:
                    break
                if len(data) % RECORD_SIZE != 0:
                    raise ValueError(f"truncated compact block in {path}")
                yield current_base, data
                current_base += len(data) // RECORD_SIZE
        finally:
            if owner is not None:
                owner.close()  # type: ignore[attr-defined]
            else:
                stream.close()

    def _iter_shuffled_compact(self) -> Iterable[tuple[int, bytes]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        block_size = max(1, self.shuffle_block_size)
        iteration = self.iteration
        self.iteration += 1
        shard_rng = random.Random(self.seed + iteration * 1_000_003)
        worker_rng = random.Random(self.seed + iteration * 1_000_003 + worker_id * 97_409)

        if self.path.is_dir():
            shard_paths = self._compact_paths()
            shard_ids = list(range(len(shard_paths)))
            # A pre-split training corpus is exposed as ``all``. It must still
            # be shuffled when the caller requested shuffled blocks; otherwise
            # every epoch sees the corpus in identical on-disk order.
            if self.split in {"train", "all"}:
                shard_rng.shuffle(shard_ids)

            # Assign complete compressed shards whenever there are enough of
            # them. The old block-level assignment made every DataLoader worker
            # open and decompress every shard, only to discard blocks assigned
            # to another worker. With 32 workers that multiplied decompression
            # and network-volume traffic by roughly 32x. Whole-shard ownership
            # keeps the streams disjoint while retaining deterministic shuffles.
            if len(shard_paths) >= num_workers:
                for shard_id in shard_ids[worker_id::num_workers]:
                    path = shard_paths[shard_id]
                    base_index = shard_id * block_size
                    for block_base, block_data in self._iter_stream_blocks(
                        path, base_index, block_size
                    ):
                        offsets = list(range(0, len(block_data), RECORD_SIZE))
                        if self.split in {"train", "all"}:
                            worker_rng.shuffle(offsets)
                        for offset in offsets:
                            yield (
                                block_base + offset // RECORD_SIZE,
                                block_data[offset : offset + RECORD_SIZE],
                            )
                return

            # Tiny test/single-shard corpora still need multiple workers to
            # contribute to a global sample cap, so retain block partitioning
            # for that narrow case even though compressed streams overlap.
            global_block_id = 0
            for shard_id in shard_ids:
                path = shard_paths[shard_id]
                base_index = shard_id * block_size
                for block_base, block_data in self._iter_stream_blocks(path, base_index, block_size):
                    assigned_worker = global_block_id % num_workers
                    global_block_id += 1
                    if assigned_worker != worker_id:
                        continue
                    offsets = list(range(0, len(block_data), RECORD_SIZE))
                    if self.split in {"train", "all"}:
                        worker_rng.shuffle(offsets)
                    for offset in offsets:
                        yield block_base + offset // RECORD_SIZE, block_data[offset : offset + RECORD_SIZE]
            return

        for block_id, (block_base, block_data) in enumerate(
            self._iter_stream_blocks(self.path, 0, block_size)
        ):
            if block_id % num_workers != worker_id:
                continue
            offsets = list(range(0, len(block_data), RECORD_SIZE))
            if self.split in {"train", "all"}:
                worker_rng.shuffle(offsets)
            for offset in offsets:
                yield block_base + offset // RECORD_SIZE, block_data[offset : offset + RECORD_SIZE]

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        max_samples = worker_sample_limit(self.max_samples, worker_id, num_workers)
        yielded = 0

        if self.shuffle_block_size > 0:
            iterator = self._iter_shuffled_compact()
            for _record_index, record in iterator:
                converted = self._sample_from_record(record)
                if converted is None:
                    continue
                yield converted
                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    break
            return

        if self.cache_unshuffled and self._cached_unshuffled_samples is not None:
            for sample in self._cached_unshuffled_samples:
                converted = self._sample_from_compact(sample)
                if converted is None:
                    raise RuntimeError("cached compact sample no longer belongs to its split")
                yield converted
            return

        paths = self._compact_paths()
        cached_samples: list[Any] | None = [] if self.cache_unshuffled else None
        shard_partition = self.path.is_dir() and len(paths) >= num_workers
        for path_index, path in enumerate(paths):
            # As in the shuffled path, give complete compressed shards to one
            # worker. Record-level modulo partitioning made every eval worker
            # decompress every shard, multiplying validation I/O by num_workers.
            if shard_partition and path_index % num_workers != worker_id:
                continue
            base_index = path_index * max(1, self.shuffle_block_size or 1_000_000)
            for local_index, sample in enumerate(iter_compact_samples(path)):
                record_index = base_index + local_index
                if not shard_partition and record_index % num_workers != worker_id:
                    continue
                if is_value_none_target(sample.score):
                    continue
                canonical_board, canonical_aux_bits, features, aux = (
                    canonical_architecture_input(
                        sample.board, sample.aux_bits, self.architecture
                    )
                )
                bucket = position_split_bucket(
                    canonical_board, canonical_aux_bits, self.split_mod
                )
                if not self._bucket_belongs_to_split(bucket):
                    continue

                if cached_samples is not None:
                    cached_samples.append(sample)
                yield {
                    "features": features,
                    "aux": aux,
                    "score": sample.score,
                    "ply": sample.ply,
                    "result": sample.result,
                }
                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    break
            if max_samples is not None and yielded >= max_samples:
                break
        if cached_samples is not None:
            self._cached_unshuffled_samples = cached_samples


def collate_sparse_batch(
    samples: list[dict[str, Any]],
    device: torch.device,
    target_scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if any(is_value_none_target(sample["score"]) for sample in samples):
        raise ValueError("VALUE_NONE target cannot be used for training or evaluation")
    batch_features = [sample["features"] for sample in samples]
    batch_aux = [sample["aux"] for sample in samples]
    scores = torch.tensor([sample["score"] for sample in samples], dtype=torch.float32, device=device)
    target_cp = stockfish_score_to_cp(scores) / float(target_scale)
    plies = torch.tensor([sample["ply"] for sample in samples], dtype=torch.float32, device=device)
    results = torch.tensor([sample["result"] for sample in samples], dtype=torch.float32, device=device)
    feature_indices, offsets, aux = make_batch(batch_features, batch_aux, device=device)
    return feature_indices, offsets, aux, scores, target_cp, plies, results


def make_loader(
    path: Path,
    architecture: str,
    data_format: str,
    split: str,
    split_mod: int,
    val_mod: int,
    test_mod: int,
    max_samples: int | None,
    seed: int,
    batch_size: int,
    workers: int,
    shuffle_block_size: int,
    cache_unshuffled: bool = False,
) -> DataLoader:
    if data_format == "auto":
        data_format = "cbin" if path.is_dir() or path.name.endswith(".cbin") or path.name.endswith(".cbin.zst") else "jsonl"
    if data_format == "jsonl":
        dataset = JsonlSplitDataset(
            path=path,
            architecture=architecture,
            split=split,
            split_mod=split_mod,
            val_mod=val_mod,
            test_mod=test_mod,
            max_samples=max_samples,
            seed=seed,
        )
    elif data_format == "cbin":
        dataset = CompactSplitDataset(
            path=path,
            architecture=architecture,
            split=split,
            split_mod=split_mod,
            val_mod=val_mod,
            test_mod=test_mod,
            max_samples=max_samples,
            seed=seed,
            shuffle_block_size=shuffle_block_size,
            cache_unshuffled=cache_unshuffled,
        )
    else:
        raise ValueError(f"unknown data format: {data_format}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=identity_collate,
        pin_memory=False,
        persistent_workers=workers > 0,
    )


def validate_training_data(path: Path, data_format: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"training data does not exist: {path}")
    resolved = data_format
    if resolved == "auto":
        resolved = (
            "cbin"
            if path.is_dir() or path.name.endswith(".cbin") or path.name.endswith(".cbin.zst")
            else "jsonl"
        )
    if resolved == "cbin":
        validate_compact_dataset(path)
    elif path.is_dir():
        raise ValueError("JSONL training data must be a file, not a directory")
    return resolved


def wdl_loss(
    predictions: torch.Tensor,
    target_scores: torch.Tensor,
    plies: torch.Tensor,
    results: torch.Tensor,
    target_scale: float,
    score_lambda: float = DEFAULT_SCORE_LAMBDA,
    exponent: float = DEFAULT_WDL_LOSS_EXPONENT,
) -> torch.Tensor:
    if target_scale <= 0.0:
        raise ValueError("target_scale must be positive")
    return stockfish_wdl_loss(
        prediction_cp=predictions * float(target_scale),
        target_score=target_scores,
        ply=plies,
        result=results,
        score_lambda=score_lambda,
        exponent=exponent,
    )


def cp_huber_loss(
    predictions: torch.Tensor,
    target_scores: torch.Tensor,
    target_scale: float,
    delta_cp: float = 200.0,
) -> torch.Tensor:
    """Smooth-L1 in CP space, normalized to keep gradients comparable across target scales."""
    if target_scale <= 0.0 or delta_cp <= 0.0:
        raise ValueError("target_scale and delta_cp must be positive")
    target_cp = stockfish_score_to_cp(target_scores)
    return F.smooth_l1_loss(
        predictions,
        target_cp / float(target_scale),
        beta=delta_cp / float(target_scale),
    )


def run_train_epoch(
    arch: str,
    model: SparseNnueArchitecture,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_scale: float,
    score_lambda: float,
    wdl_loss_exponent: float,
    max_batches: int | None,
    epoch: int,
    progress_batches: int,
    epoch_start: float,
    cp_huber_delta: float = 200.0,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_abs_cp = 0.0
    total_samples = 0

    for batch_index, samples in enumerate(loader, 1):
        feature_indices, offsets, aux, target_scores, target_cp, plies, results = collate_sparse_batch(
            samples,
            device=device,
            target_scale=target_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = model(feature_indices, offsets, aux)
        loss = cp_huber_loss(predictions, target_scores, target_scale, cp_huber_delta)
        loss.backward()
        optimizer.step()

        batch_size = target_cp.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_abs_cp += float(((predictions.detach() - target_cp).abs() * target_scale).sum().cpu())
        total_samples += batch_size
        if progress_batches > 0 and batch_index % progress_batches == 0:
            elapsed = time.monotonic() - epoch_start
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "arch": arch,
                        "epoch": epoch,
                        "batch": batch_index,
                        "samples": total_samples,
                        "avg_loss": round(total_loss / max(1, total_samples), 8),
                        "avg_cp": round(total_abs_cp / max(1, total_samples), 4),
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
    return total_loss / total_samples, total_abs_cp / total_samples, total_samples


@torch.no_grad()
def evaluate_cp(
    arch: str,
    split: str,
    model: SparseNnueArchitecture,
    loader: DataLoader,
    device: torch.device,
    target_scale: float,
    score_lambda: float,
    wdl_loss_exponent: float,
    max_batches: int | None,
    epoch: int,
    progress_batches: int,
    epoch_start: float,
    cp_huber_delta: float = 200.0,
) -> tuple[float, float, int]:
    model.eval()
    total_loss = 0.0
    total_abs_cp = 0.0
    total_samples = 0

    for batch_index, samples in enumerate(loader, 1):
        feature_indices, offsets, aux, target_scores, target_cp, plies, results = collate_sparse_batch(
            samples,
            device=device,
            target_scale=target_scale,
        )
        predictions = model(feature_indices, offsets, aux)
        loss = cp_huber_loss(predictions, target_scores, target_scale, cp_huber_delta)
        batch_size = target_cp.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_abs_cp += float(((predictions - target_cp).abs() * target_scale).sum().cpu())
        total_samples += batch_size
        if progress_batches > 0 and batch_index % progress_batches == 0:
            elapsed = time.monotonic() - epoch_start
            print(
                json.dumps(
                    {
                        "event": "eval_progress",
                        "arch": arch,
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
    return total_loss / total_samples, total_abs_cp / total_samples, total_samples


def save_checkpoint(
    path: Path,
    model: SparseNnueArchitecture,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    target_scale: float,
    score_lambda: float,
    wdl_loss_exponent: float,
    cp_huber_delta: float,
    provenance: dict[str, Any],
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
            "aux_feature_count": model.aux_feature_count,
            "hidden1_size": model.hidden1_size,
            "hidden2_size": model.hidden2_size,
            "target_scale": target_scale,
            "score_lambda": score_lambda,
            "target_encoding": "stockfish_raw_score_ply_result",
            "wdl_model": "nnue_pytorch_training_data_entry",
            "wdl_loss_exponent": wdl_loss_exponent,
            "loss_type": "cp_huber",
            "cp_huber_delta": cp_huber_delta,
            "provenance": provenance,
        },
        path,
    )


def git_provenance() -> dict[str, Any]:
    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = run_git("rev-parse", "HEAD")
        dirty = bool(run_git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
        dirty = True
    return {"git_revision": revision, "git_dirty": dirty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one NNUE architecture on JSONL sparse data")
    parser.add_argument("--arch", required=True, choices=sorted(ARCHITECTURES))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--output-dir", default=Path("models/nnue_arch_sweep"), type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta-loss", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--target-scale", type=float, default=DEFAULT_TARGET_SCALE)
    parser.add_argument("--score-lambda", type=float, default=DEFAULT_SCORE_LAMBDA)
    parser.add_argument("--wdl-loss-exponent", type=float, default=DEFAULT_WDL_LOSS_EXPONENT)
    parser.add_argument("--cp-huber-delta", type=float, default=200.0)
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=50_000)
    parser.add_argument("--test-max-samples", type=int, default=50_000)
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument(
        "--shuffle-block-size",
        type=int,
        default=1_000_000,
        help="For compact data, shuffle train records inside blocks of this many records. "
        "If --data is a shard directory, train also shuffles shard order. Use 0 to disable.",
    )
    parser.add_argument(
        "--progress-batches",
        type=int,
        default=1000,
        help="Print JSON progress every N train batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--eval-progress-batches",
        type=int,
        default=0,
        help="Print JSON progress every N eval batches. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs and --patience must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0 or args.torch_threads < 0:
        raise ValueError("worker/thread counts must be non-negative")
    if args.lr <= 0.0 or args.weight_decay < 0.0 or args.min_delta_loss < 0.0:
        raise ValueError("learning rate must be positive; decay/delta must be non-negative")
    for name in ("train_max_samples", "val_max_samples", "test_max_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("train_max_batches", "eval_max_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
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
    if args.split_mod <= 2:
        raise ValueError("--split-mod must be greater than 2")
    if not 0 <= args.val_mod < args.split_mod or not 0 <= args.test_mod < args.split_mod:
        raise ValueError("validation/test buckets must be inside --split-mod")
    if args.val_mod == args.test_mod:
        raise ValueError("validation and test buckets must differ")
    args.data_format = validate_training_data(args.data, args.data_format)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = ARCHITECTURES[args.arch]
    device = choose_device(args.device)
    model = SparseNnueArchitecture(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(
        args.data,
        config.transform,
        args.data_format,
        "train",
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
        args.data,
        config.transform,
        args.data_format,
        "val",
        args.split_mod,
        args.val_mod,
        args.test_mod,
        args.val_max_samples,
        args.seed,
        args.batch_size,
        args.workers,
        0,
    )
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_path = args.output_dir / f"nnue_arch_{args.arch}_current.pt"
    best_path = args.output_dir / f"nnue_arch_{args.arch}_best.pt"
    provenance = {
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "target_encoding": "stockfish_raw_score_ply_result",
        "score_to_cp": "clamp(100 * raw_score / 208, -2000, 2000)",
        "wdl_model": "nnue_pytorch_training_data_entry",
        "training_objective": "cp_huber",
        "cp_huber_delta": args.cp_huber_delta,
        "split_policy": "crc32(board_bytes + aux_bits_le16)",
        **git_provenance(),
    }

    print(
        json.dumps(
            {
                "event": "start",
                "arch": args.arch,
                "config": asdict(config),
                "device": str(device),
                "data": str(args.data),
                "batch_size": args.batch_size,
                "target_scale": args.target_scale,
                "score_lambda": args.score_lambda,
                "wdl_loss_exponent": args.wdl_loss_exponent,
                "loss_type": "cp_huber",
                "cp_huber_delta": args.cp_huber_delta,
                "split": {
                    "split_mod": args.split_mod,
                    "val_mod": args.val_mod,
                    "test_mod": args.test_mod,
                },
                "shuffle_block_size": args.shuffle_block_size,
                "train_max_samples": args.train_max_samples,
                "val_max_samples": args.val_max_samples,
                "test_max_samples": args.test_max_samples,
                "provenance": provenance,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        train_loss, train_cp, train_samples = run_train_epoch(
            args.arch,
            model,
            train_loader,
            optimizer,
            device,
            args.target_scale,
            args.score_lambda,
            args.wdl_loss_exponent,
            args.train_max_batches,
            epoch,
            args.progress_batches,
            start,
            args.cp_huber_delta,
        )
        val_loss, val_cp, val_samples = evaluate_cp(
            args.arch,
            "val",
            model,
            val_loader,
            device,
            args.target_scale,
            args.score_lambda,
            args.wdl_loss_exponent,
            args.eval_max_batches,
            epoch,
            args.eval_progress_batches,
            start,
            args.cp_huber_delta,
        )
        elapsed = time.monotonic() - start

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_val_cp": train_cp,
            "val_cp": val_cp,
            "elapsed_sec": elapsed,
        }
        save_checkpoint(
            current_path,
            model,
            optimizer,
            epoch,
            metrics,
            args.target_scale,
            args.score_lambda,
            args.wdl_loss_exponent,
            args.cp_huber_delta,
            provenance,
        )

        improved = val_loss + args.min_delta_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                metrics,
                args.target_scale,
                args.score_lambda,
                args.wdl_loss_exponent,
                args.cp_huber_delta,
                provenance,
            )
        else:
            epochs_without_improvement += 1

        print(
            json.dumps(
                {
                    "event": "epoch",
                    "arch": args.arch,
                    "epoch": epoch,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                    "train_loss": round(train_loss, 8),
                    "val_loss": round(val_loss, 8),
                    "train_val_cp": round(train_cp, 4),
                    "val_cp": round(val_cp, 4),
                    "best_val_loss": round(best_val_loss, 8),
                    "best_epoch": best_epoch,
                    "improved": improved,
                    "no_improve_epochs": epochs_without_improvement,
                    "elapsed_sec": round(elapsed, 3),
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
    test_start = time.monotonic()
    test_loss, test_cp, test_samples = evaluate_cp(
        args.arch,
        "test",
        model,
        test_loader,
        device,
        args.target_scale,
        args.score_lambda,
        args.wdl_loss_exponent,
        args.eval_max_batches,
        best_epoch,
        args.eval_progress_batches,
        test_start,
        args.cp_huber_delta,
    )
    print(
        json.dumps(
            {
                "event": "final_test",
                "arch": args.arch,
                "selected_epoch": best_epoch,
                "best_val_loss": round(best_val_loss, 8),
                "test_samples": test_samples,
                "test_loss": round(test_loss, 8),
                "test_val_cp": round(test_cp, 4),
                "elapsed_sec": round(time.monotonic() - test_start, 3),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
