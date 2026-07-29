#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.training_targets import stockfish_score_to_cp  # noqa: E402
from nn.quantized_nnue_architectures import QUANTIZED_ARCHITECTURES  # noqa: E402
from tools.train_nnue_architecture import make_loader, validate_training_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure absolute target-CP distribution")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--shuffle-block-size", type=int, default=1_000_000)
    parser.add_argument("--bins", default="0,100,200,400,800,1200,1600,2000")
    args = parser.parse_args()

    edges = [float(value) for value in args.bins.split(",")]
    if len(edges) < 2 or any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("bin edges must be strictly increasing")
    data_format = validate_training_data(args.data, args.data_format)
    loader = make_loader(
        args.data,
        QUANTIZED_ARCHITECTURES["F2"].transform,
        data_format,
        args.split,
        100,
        98,
        99,
        args.max_samples,
        args.seed,
        args.batch_size,
        args.workers,
        args.shuffle_block_size if args.split == "train" else 0,
    )

    counts = [0] * (len(edges) - 1)
    abs_cp_sums = [0.0] * (len(edges) - 1)
    total = 0
    raw_score_min = None
    raw_score_max = None
    unclamped_cp_min = None
    unclamped_cp_max = None
    clamped_samples = 0
    for samples in loader:
        scores = torch.tensor([sample["score"] for sample in samples], dtype=torch.float32)
        raw_score_min = min(raw_score_min if raw_score_min is not None else float("inf"), float(scores.min()))
        raw_score_max = max(raw_score_max if raw_score_max is not None else float("-inf"), float(scores.max()))
        unclamped_cp = scores * (100.0 / 208.0)
        unclamped_cp_min = min(
            unclamped_cp_min if unclamped_cp_min is not None else float("inf"),
            float(unclamped_cp.min()),
        )
        unclamped_cp_max = max(
            unclamped_cp_max if unclamped_cp_max is not None else float("-inf"),
            float(unclamped_cp.max()),
        )
        clamped_samples += int((unclamped_cp.abs() > 2000.0).sum().item())
        abs_cp = stockfish_score_to_cp(scores).abs()
        total += int(abs_cp.numel())
        for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
            upper_mask = abs_cp <= upper if index == len(counts) - 1 else abs_cp < upper
            mask = (abs_cp >= lower) & upper_mask
            counts[index] += int(mask.sum().item())
            abs_cp_sums[index] += float(abs_cp[mask].sum().item())

    print(json.dumps({
        "split": args.split,
        "samples": total,
        "raw_score_range": [raw_score_min, raw_score_max],
        "unclamped_cp_range": [unclamped_cp_min, unclamped_cp_max],
        "clamped_samples": clamped_samples,
        "clamped_ratio": clamped_samples / total,
        "bins": [
            {
                "min_abs_cp": lower,
                "max_abs_cp": upper,
                "samples": count,
                "ratio": count / total,
                "mean_abs_cp": cp_sum / count if count else None,
            }
            for lower, upper, count, cp_sum in zip(edges, edges[1:], counts, abs_cp_sums)
        ],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
