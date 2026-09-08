#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.compact_board_data import RECORD_SIZE, open_writer, write_header  # noqa: E402
from tools.data.build_stratified_cp_corpus import parse_bins  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix an exact corpus from existing raw bucket record files")
    parser.add_argument("--bucket-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bins", required=True)
    parser.add_argument("--records-per-shard", type=int, default=1_000_000)
    parser.add_argument("--zstd-level", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    bins = parse_bins(args.bins)
    total = sum(quota for _, _, quota in bins)
    if total % args.records_per_shard:
        raise ValueError("total records must divide evenly into shards")
    if args.output_dir.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite output or manifest")
    paths = [args.bucket_dir / f"bucket_{index}.records" for index in range(len(bins))]
    for path, (_, _, quota) in zip(paths, bins):
        available = path.stat().st_size // RECORD_SIZE
        if available < quota:
            raise RuntimeError(f"{path} has {available} records, needs {quota}")

    # The requested ratios use 250k as their greatest common unit.
    unit = 250_000
    if any(quota % unit for _, _, quota in bins):
        raise ValueError("all quotas must be multiples of 250,000")
    schedule_template: list[int] = []
    for index, (_, _, quota) in enumerate(bins):
        schedule_template.extend([index] * (quota // unit))
    if args.records_per_shard % len(schedule_template):
        raise ValueError("records per shard must divide into exact ratio blocks")

    readers = [path.open("rb") for path in paths]
    rng = random.Random(args.seed)
    digest = hashlib.sha256()
    args.output_dir.mkdir(parents=True)
    written = [0] * len(bins)
    try:
        for shard_index in range(total // args.records_per_shard):
            output = args.output_dir / f"part_{shard_index:05d}.cbin.zst"
            stream, owner = open_writer(output, args.zstd_level)
            try:
                write_header(stream)
                for _ in range(args.records_per_shard // len(schedule_template)):
                    schedule = schedule_template.copy()
                    rng.shuffle(schedule)
                    for index in schedule:
                        record = readers[index].read(RECORD_SIZE)
                        if len(record) != RECORD_SIZE:
                            raise RuntimeError(f"bucket {index} ended early")
                        stream.write(record)
                        written[index] += 1
            finally:
                if owner is not None:
                    owner.close()  # type: ignore[attr-defined]
                else:
                    stream.close()
            with output.open("rb") as compressed:
                while block := compressed.read(8 * 1024 * 1024):
                    digest.update(block)
            print(f"write shard={shard_index + 1} records={(shard_index + 1) * args.records_per_shard}", flush=True)
    finally:
        for reader in readers:
            reader.close()
    expected = [quota for _, _, quota in bins]
    if written != expected:
        raise RuntimeError(f"wrong bucket counts: {written} != {expected}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bucket_directory": str(args.bucket_dir),
        "selection": {
            "score_to_cp": "100 * raw_score / 208 (no target above 2000 CP)",
            "seed": args.seed,
            "bins": [
                {"min_abs_cp": lo, "max_abs_cp": hi, "records": quota, "ratio": quota / total}
                for lo, hi, quota in bins
            ],
            "oversampling": False,
        },
        "output": {
            "directory": str(args.output_dir),
            "records": total,
            "shards": total // args.records_per_shard,
            "records_per_shard": args.records_per_shard,
            "compressed_sha256": digest.hexdigest(),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
