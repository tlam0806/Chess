#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.compact_board_data import (
    HEADER_SIZE,
    RECORD_SIZE,
    compact_paths,
    open_reader,
    validate_header,
)


DEFAULT_BINS = "0:100,100:300,300:600,600:1000,1000:1600,1600:2000,2000:5000"


def parse_bins(specification: str) -> list[tuple[float, float]]:
    bins: list[tuple[float, float]] = []
    for item in specification.split(","):
        lower, upper = (float(value) for value in item.split(":"))
        if lower < 0.0 or upper <= lower:
            raise ValueError(f"invalid bin: {item}")
        bins.append((lower, upper))
    if any(left[1] != right[0] for left, right in zip(bins, bins[1:])):
        raise ValueError("bins must be contiguous")
    return bins


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze raw and clipped CP distribution in CHSCBIN2 data")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--bins", default=DEFAULT_BINS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--target-clip", type=float, default=2000.0)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.target_clip <= 0.0:
        raise ValueError("--target-clip must be positive")

    bins = parse_bins(args.bins)
    counts = [0] * len(bins)
    samples = 0
    signed_sum = 0.0
    absolute_sum = 0.0
    squared_sum = 0.0
    outside_bins = 0
    for path in compact_paths(args.data):
        stream, owner = open_reader(path)
        try:
            validate_header(stream.read(HEADER_SIZE), path)
            while args.max_samples is None or samples < args.max_samples:
                record = stream.read(RECORD_SIZE)
                if not record:
                    break
                if len(record) != RECORD_SIZE:
                    raise RuntimeError(f"truncated record in {path}")
                raw_score = struct.unpack_from("<h", record, 34)[0]
                raw_cp = raw_score * 100.0 / 208.0
                target_cp = max(-args.target_clip, min(args.target_clip, raw_cp))
                magnitude = abs(raw_cp)
                matched = False
                for index, (lower, upper) in enumerate(bins):
                    if magnitude >= lower and (
                        magnitude < upper
                        or (index == len(bins) - 1 and magnitude == upper)
                    ):
                        counts[index] += 1
                        matched = True
                        break
                if not matched:
                    outside_bins += 1
                samples += 1
                signed_sum += target_cp
                absolute_sum += abs(target_cp)
                squared_sum += target_cp * target_cp
            if args.max_samples is not None and samples >= args.max_samples:
                break
        finally:
            if owner is not None:
                owner.close()  # type: ignore[attr-defined]
            else:
                stream.close()
    if samples == 0:
        raise RuntimeError("dataset contained no records")
    result = {
        "data": str(args.data),
        "samples": samples,
        "target_clip": args.target_clip,
        "zero_predictor_cp_mae": absolute_sum / samples,
        "zero_predictor_cp_rmse": math.sqrt(squared_sum / samples),
        "target_mean_cp": signed_sum / samples,
        "raw_cp_bins": [
            {
                "min_abs_cp": lower,
                "max_abs_cp": upper,
                "samples": count,
                "ratio": count / samples,
            }
            for (lower, upper), count in zip(bins, counts)
        ],
        "outside_bins": outside_bins,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
