#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.compact_board_data import (
    HEADER_SIZE,
    RECORD_SIZE,
    open_reader,
    open_writer,
    validate_header,
    write_header,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a .cbin(.zst) file into record-count shards.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--records-per-shard", type=int, default=1_000_000)
    parser.add_argument("--zstd-level", type=int, default=6)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.records_per_shard <= 0:
        raise SystemExit("--records-per-shard must be positive")
    if args.output_dir.exists():
        existing = list(args.output_dir.glob("*.cbin")) + list(args.output_dir.glob("*.cbin.zst"))
        if existing and not args.force:
            raise SystemExit(f"{args.output_dir} already contains compact shards; use --force")
        if args.force:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stream, owner = open_reader(args.input)
    shard_index = 0
    total_records = 0
    records_per_read = args.records_per_shard
    bytes_per_read = records_per_read * RECORD_SIZE
    try:
        header = stream.read(HEADER_SIZE)
        validate_header(header, args.input)

        while True:
            if args.max_records is not None:
                remaining = args.max_records - total_records
                if remaining <= 0:
                    break
                read_bytes = min(bytes_per_read, remaining * RECORD_SIZE)
            else:
                read_bytes = bytes_per_read

            data = stream.read(read_bytes)
            if not data:
                break
            if len(data) % RECORD_SIZE != 0:
                raise SystemExit(f"truncated compact record block in {args.input}")

            output = args.output_dir / f"part_{shard_index:05d}.cbin.zst"
            out_stream, out_owner = open_writer(output, args.zstd_level)
            try:
                write_header(out_stream)
                out_stream.write(data)
            finally:
                if out_owner is not None:
                    out_owner.close()  # type: ignore[attr-defined]
                else:
                    out_stream.close()

            records = len(data) // RECORD_SIZE
            total_records += records
            print(
                f"shard={shard_index} records={records} total={total_records} path={output}",
                flush=True,
            )
            shard_index += 1
    finally:
        if owner is not None:
            owner.close()  # type: ignore[attr-defined]
        else:
            stream.close()

    print(f"done shards={shard_index} records={total_records}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
