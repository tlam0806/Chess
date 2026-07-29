#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.compact_board_data import (  # noqa: E402
    HEADER_SIZE, RECORD_SIZE, compact_paths, open_reader, open_writer, position_split_bucket,
    validate_header, write_header,
)


def records(path: Path):
    stream, owner = open_reader(path)
    try:
        validate_header(stream.read(HEADER_SIZE), path)
        while record := stream.read(RECORD_SIZE):
            if len(record) != RECORD_SIZE:
                raise RuntimeError(f"truncated record in {path}")
            yield record
    finally:
        if owner is not None:
            owner.close()  # type: ignore[attr-defined]
        else:
            stream.close()


def score(record: bytes) -> int:
    return struct.unpack_from("<h", record, 34)[0]


def _raw_records(path: Path):
    with path.open("rb") as stream:
        while record := stream.read(RECORD_SIZE):
            if len(record) != RECORD_SIZE:
                raise RuntimeError(f"truncated raw bucket file: {path}")
            yield record


def _stdin_cbin_records():
    validate_header(sys.stdin.buffer.read(HEADER_SIZE), Path("<stdin>"))
    while record := sys.stdin.buffer.read(RECORD_SIZE):
        if len(record) != RECORD_SIZE:
            raise RuntimeError("truncated CBIN stream")
        yield record


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace the final validation bucket with a clamped/unclamped mix")
    parser.add_argument("--base-validation", required=True, type=Path)
    parser.add_argument("--unclamped-bucket-records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--final-count", type=int, default=51_862)
    parser.add_argument("--unclamped-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--unclamped-require-split-bucket", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite output or manifest")
    unclamped_quota = round(args.final_count * args.unclamped_ratio)
    clamped_quota = args.final_count - unclamped_quota
    kept: list[bytes] = []
    clamped: list[bytes] = []
    base_final_unclamped = 0
    for record in records(args.base_validation):
        scaled = abs(score(record)) * 100
        if scaled < 1600 * 208:
            kept.append(record)
        elif scaled > 2000 * 208 and len(clamped) < clamped_quota:
            clamped.append(record)
        elif scaled <= 2000 * 208:
            base_final_unclamped += 1
    if len(clamped) != clamped_quota:
        raise RuntimeError(f"base validation only supplied {len(clamped)} clamped records")
    unclamped: list[bytes] = []
    # This raw bucket was produced with the strict 1600 <= abs(CP) <= 2000 rule.
    source_paths = (
        compact_paths(args.unclamped_bucket_records)
        if args.unclamped_bucket_records.is_dir()
        or str(args.unclamped_bucket_records).endswith((".cbin", ".cbin.zst"))
        else []
    )
    if str(args.unclamped_bucket_records) == "-":
        source_records = _stdin_cbin_records()
    elif source_paths:
        source_records = (record for path in source_paths for record in records(path))
    else:
        source_records = _raw_records(args.unclamped_bucket_records)
    for record in source_records:
            if len(unclamped) >= unclamped_quota:
                break
            if args.unclamped_require_split_bucket is not None:
                aux_bits = struct.unpack_from("<H", record, 32)[0]
                if position_split_bucket(record[:32], aux_bits, 100) != args.unclamped_require_split_bucket:
                    continue
            scaled = abs(score(record)) * 100
            if 1600 * 208 <= scaled <= 2000 * 208:
                unclamped.append(record)
    if len(unclamped) != unclamped_quota:
        raise RuntimeError(f"unclamped source only supplied {len(unclamped)} records")
    output_records = kept + unclamped + clamped
    random.Random(args.seed).shuffle(output_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out, owner = open_writer(args.output, 6)
    try:
        write_header(out)
        for record in output_records:
            out.write(record)
    finally:
        if owner is not None:
            owner.close()  # type: ignore[attr-defined]
        else:
            out.close()
    manifest = {
        "base_validation": str(args.base_validation),
        "unclamped_source": str(args.unclamped_bucket_records),
        "records": len(output_records),
        "records_below_1600": len(kept),
        "final_bucket": {
            "records": args.final_count,
            "unclamped": len(unclamped),
            "clamped": len(clamped),
            "unclamped_ratio": len(unclamped) / args.final_count,
            "clamped_ratio": len(clamped) / args.final_count,
            "discarded_base_unclamped": base_final_unclamped,
        },
        "pre_split": True,
        "seed": args.seed,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
