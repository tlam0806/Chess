#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.horizontal_mirror_checkpoint import convert_f2_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold a full F2 phase checkpoint into horizontal-mirror F2M"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symmetric-reference-output", type=Path)
    parser.add_argument(
        "--fold",
        choices=("average", "canonical"),
        default="average",
        help="average mirror pairs or preserve the canonical a-d row",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    symmetric, mirror = convert_f2_checkpoint(checkpoint, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mirror, args.output)
    if args.symmetric_reference_output is not None:
        args.symmetric_reference_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(symmetric, args.symmetric_reference_output)
    print(
        f"source={args.source} output={args.output} architecture=F2M "
        f"fold={args.fold} feature_rows="
        f"{mirror['model_state']['feature_weights.weight'].shape[0]}"
    )


if __name__ == "__main__":
    main()
