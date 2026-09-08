#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.compact_board_data import HEADER_SIZE, compact_paths, open_reader, validate_header  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge CHSCBIN2 shard payloads into one CHSCBIN2 stdout stream"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--buffer-size", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()

    paths = compact_paths(args.input)
    output = sys.stdout.buffer
    first_header: bytes | None = None
    total_bytes = 0
    for index, path in enumerate(paths):
        stream, owner = open_reader(path)
        try:
            header = stream.read(HEADER_SIZE)
            validate_header(header, path)
            if first_header is None:
                first_header = header
                output.write(header)
            elif header != first_header:
                raise RuntimeError(f"mixed CHSCBIN2 headers at {path}")
            copied = shutil.copyfileobj(stream, output, length=args.buffer_size)
            if copied is not None:
                total_bytes += copied
        finally:
            if owner is not None:
                owner.close()  # type: ignore[attr-defined]
            else:
                stream.close()
        print(f"streamed={index + 1}/{len(paths)} shard={path.name}", file=sys.stderr)
    output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
