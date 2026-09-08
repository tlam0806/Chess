#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.compact_board_data import (  # noqa: E402
    HEADER_SIZE,
    RECORD_SIZE,
    compact_paths,
    open_reader,
    open_writer,
    position_split_bucket,
    validate_header,
    write_header,
)


DEFAULT_BINS = "0:100:2500000,100:300:2500000,300:600:2000000,600:1000:1500000,1000:1600:1000000,1600:2000:500000"


def parse_bins(specification: str) -> list[tuple[int, int, int]]:
    bins = []
    for item in specification.split(","):
        lower, upper, quota = (int(value) for value in item.split(":"))
        if lower < 0 or upper <= lower or quota <= 0:
            raise ValueError(f"invalid bin: {item}")
        bins.append((lower, upper, quota))
    if any(left[1] != right[0] for left, right in zip(bins, bins[1:])):
        raise ValueError("bins must be contiguous")
    return bins


def cp_bin(score: int, bins: list[tuple[int, int, int]]) -> int | None:
    # Compare abs(score) * 100 / 208 without floating-point rounding. Unlike
    # target clamping, values above the final upper bound are excluded.
    scaled = abs(score) * 100
    for index, (lower, upper, _quota) in enumerate(bins):
        if scaled >= lower * 208 and (
            scaled < upper * 208
            or (index == len(bins) - 1 and scaled == upper * 208)
        ):
            return index
    return None


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CP-stratified CHSCBIN2 corpus")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bins", default=DEFAULT_BINS)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--records-per-shard", type=int, default=1_000_000)
    parser.add_argument("--zstd-level", type=int, default=6)
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument(
        "--repeat-underfilled-bins",
        action="store_true",
        help="fill an underfilled rare bucket by cycling its selected records",
    )
    args = parser.parse_args()

    bins = parse_bins(args.bins)
    total_quota = sum(quota for _lower, _upper, quota in bins)
    if total_quota % args.records_per_shard:
        raise SystemExit("total quota must be divisible by records per shard")
    if args.output_dir.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite output directory or manifest")

    source_is_stdin = str(args.source) == "-"
    paths: list[Path | None] = [None] if source_is_stdin else list(compact_paths(args.source))
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    temporary = Path(f"{args.output_dir}.tmp")
    temporary.mkdir(parents=True)
    bucket_paths = [temporary / f"bucket_{index}.records" for index in range(len(bins))]
    bucket_streams = [path.open("wb") for path in bucket_paths]
    selected = [0] * len(bins)
    scanned = 0
    train_eligible = 0
    shards_scanned = 0
    try:
        for source_path in paths:
            if source_path is None:
                stream, owner = sys.stdin.buffer, None
                source_label = Path("<stdin>")
            else:
                stream, owner = open_reader(source_path)
                source_label = source_path
            try:
                validate_header(stream.read(HEADER_SIZE), source_label)
                while True:
                    record = stream.read(RECORD_SIZE)
                    if not record:
                        break
                    if len(record) != RECORD_SIZE:
                        raise RuntimeError(f"truncated record in {source_path}")
                    scanned += 1
                    if scanned % 5_000_000 == 0:
                        print(
                            f"scan records={scanned} selected={selected}",
                            flush=True,
                        )
                    aux_bits = struct.unpack_from("<H", record, 32)[0]
                    if position_split_bucket(record[:32], aux_bits, args.split_mod) in (
                        args.val_mod,
                        args.test_mod,
                    ):
                        continue
                    train_eligible += 1
                    score = struct.unpack_from("<h", record, 34)[0]
                    index = cp_bin(score, bins)
                    if index is None or selected[index] >= bins[index][2]:
                        continue
                    bucket_streams[index].write(record)
                    selected[index] += 1
            finally:
                if owner is not None:
                    owner.close()  # type: ignore[attr-defined]
                elif source_path is not None:
                    stream.close()
            shards_scanned += 1
            print(
                f"scan shards={shards_scanned}/{len(paths)} records={scanned} "
                f"selected={selected}",
                flush=True,
            )
            if all(count == quota for count, (_lower, _upper, quota) in zip(selected, bins)):
                break
    finally:
        for stream in bucket_streams:
            stream.close()

    unique_selected = selected.copy()
    missing = [quota - count for count, (_lower, _upper, quota) in zip(selected, bins)]
    if any(missing) and not args.repeat_underfilled_bins:
        shutil.rmtree(temporary)
        raise RuntimeError(f"source exhausted before quotas were filled: missing={missing}")
    if any(missing):
        for index, count_missing in enumerate(missing):
            if count_missing <= 0:
                continue
            if selected[index] == 0:
                shutil.rmtree(temporary)
                raise RuntimeError(f"cannot repeat empty bucket {index}")
            original = bucket_paths[index].read_bytes()
            if len(original) != selected[index] * RECORD_SIZE:
                shutil.rmtree(temporary)
                raise RuntimeError(f"bad temporary bucket size for bucket {index}")
            with bucket_paths[index].open("ab") as stream:
                full_repeats, remainder = divmod(count_missing, selected[index])
                for _ in range(full_repeats):
                    stream.write(original)
                stream.write(original[: remainder * RECORD_SIZE])
            selected[index] += count_missing

    bucket_readers = [path.open("rb") for path in bucket_paths]
    local_mix = []
    quota_gcd = math.gcd(*(quota for _lower, _upper, quota in bins))
    for index, (_lower, _upper, quota) in enumerate(bins):
        local_mix.extend([index] * (quota // quota_gcd))
    written = [0] * len(bins)
    args.output_dir.mkdir(parents=True)
    try:
        total_written = 0
        for shard_index in range(total_quota // args.records_per_shard):
            output = args.output_dir / f"part_{shard_index:05d}.cbin.zst"
            out, owner = open_writer(output, args.zstd_level)
            try:
                write_header(out)
                for _ in range(args.records_per_shard // len(local_mix)):
                    schedule = local_mix.copy()
                    rng.shuffle(schedule)
                    for index in schedule:
                        record = bucket_readers[index].read(RECORD_SIZE)
                        if len(record) != RECORD_SIZE:
                            raise RuntimeError(f"bucket {index} ended early")
                        out.write(record)
                        written[index] += 1
                        total_written += 1
            finally:
                if owner is not None:
                    owner.close()  # type: ignore[attr-defined]
                else:
                    out.close()
            print(f"write shard={shard_index} total={total_written}", flush=True)
    finally:
        for stream in bucket_readers:
            stream.close()

    if written != selected:
        raise RuntimeError(f"written quotas differ: selected={selected} written={written}")
    shutil.rmtree(temporary)
    output_paths = sorted(args.output_dir.glob("*.cbin.zst"))
    manifest = {
        "format": {
            "magic": "CHSCBIN2",
            "version": 2,
            "record_size_bytes": RECORD_SIZE,
            "target_encoding": "stockfish_raw_score_ply_result",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.source), "shards_considered": len(paths)},
        "selection": {
            "seed": args.seed,
            "score_to_cp_for_selection": "100 * raw_score / 208 (unclamped)",
            "split_policy": "crc32(board_bytes + aux_bits_le16)",
            "split_mod": args.split_mod,
            "excluded_buckets": [args.val_mod, args.test_mod],
            "source_shards_scanned": shards_scanned,
            "source_records_scanned": scanned,
            "train_eligible_records_scanned": train_eligible,
            "bins": [
                {
                    "min_abs_cp": lower,
                    "max_abs_cp": upper,
                    "records": quota,
                    "unique_source_records": unique_selected[index],
                    "repeated_records": quota - unique_selected[index],
                    "effective_repeat_factor": quota / unique_selected[index],
                    "ratio": quota / total_quota,
                }
                for index, (lower, upper, quota) in enumerate(bins)
            ],
            "mixing": (
                f"random permutation of exact {len(local_mix)}-record ratio blocks"
            ),
            "above_final_bin_excluded": True,
        },
        "output": {
            "directory": str(args.output_dir),
            "shards": len(output_paths),
            "records_per_shard": args.records_per_shard,
            "records": total_quota,
            "compressed_sha256": sha256_files(output_paths),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"done records={total_quota} manifest={args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
