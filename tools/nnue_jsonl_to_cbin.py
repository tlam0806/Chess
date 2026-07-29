#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.compact_board_data import (  # noqa: E402
    HEADER_SIZE,
    RECORD_SIZE,
    open_writer,
    pack_aux,
    pack_board_from_raw_features,
    pack_record,
    write_header,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sparse NNUE JSONL samples to compact side-to-move board records"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Use .zst suffix for zstd output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    parser.add_argument("--zstd-level", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.progress_interval <= 0:
        raise SystemExit("--progress-interval must be positive")

    written = 0
    malformed = 0
    stream, owner = open_writer(args.output, args.zstd_level)
    try:
        write_header(stream)
        with args.input.open("r", encoding="utf-8") as input_file:
            for line_index, line in enumerate(input_file, 1):
                if args.limit is not None and written >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    board = pack_board_from_raw_features(sample["features"])
                    aux_bits = pack_aux([int(value) for value in sample["aux"]])
                    score = int(sample["score"])
                    ply = int(sample["ply"])
                    result = int(sample["result"])
                    stream.write(pack_record(board, aux_bits, score, ply, result))
                    written += 1
                except (KeyError, TypeError, ValueError) as error:
                    malformed += 1
                    if malformed <= 10:
                        print(
                            f"malformed line {line_index}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                if written and written % args.progress_interval == 0:
                    print(
                        f"written={written} malformed={malformed}",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        if owner is not None:
            owner.close()  # type: ignore[attr-defined]
        else:
            stream.close()

    raw_bytes = HEADER_SIZE + written * RECORD_SIZE
    print(
        f"done written={written} malformed={malformed} raw_bytes={raw_bytes} output={args.output}",
        flush=True,
    )
    return 0 if malformed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
