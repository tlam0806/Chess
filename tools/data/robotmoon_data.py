#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROBOTMOON_PAGE = "https://robotmoon.com/nnue-training-data/"


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    url: str
    note: str


SOURCES = [
    Source(
        "T80 2024",
        "huggingface",
        "https://huggingface.co/datasets/linrock/test80-2024/tree/main",
        "Recent Stockfish training data group; individual files are multi-GB.",
    ),
    Source(
        "T80 2023",
        "huggingface",
        "https://huggingface.co/datasets/linrock/test80-2023/tree/main",
        "Older Stockfish training data group.",
    ),
    Source(
        "T78 2022",
        "huggingface",
        "https://huggingface.co/datasets/linrock/test78/tree/main",
        "Older Stockfish training data group.",
    ),
    Source(
        "binpack format",
        "docs",
        "https://github.com/official-stockfish/Stockfish/blob/tools/docs/binpack.md",
        "Format reference; not this repo's JSONL format.",
    ),
    Source(
        "nnue-pytorch",
        "trainer",
        "https://github.com/official-stockfish/nnue-pytorch",
        "Upstream trainer that consumes Stockfish binpack data.",
    ),
]


def run_curl(args: list[str]) -> bytes:
    command = ["curl", "-L", "--fail", "--silent", "--show-error", *args]
    try:
        return subprocess.check_output(command)
    except FileNotFoundError as error:
        raise SystemExit("curl is required for this tool") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"curl failed with exit code {error.returncode}") from error


def hf_api_url(repo: str, path: str | None = None) -> str:
    encoded_repo = quote(repo, safe="/")
    if path:
        return f"https://huggingface.co/api/datasets/{encoded_repo}/tree/main/{quote(path, safe='/')}?recursive=true"
    return f"https://huggingface.co/api/datasets/{encoded_repo}/tree/main?recursive=true"


def hf_resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{quote(repo, safe='/')}/resolve/main/{quote(path, safe='/')}"


def load_hf_tree(repo: str, path: str | None) -> list[dict[str, Any]]:
    payload = run_curl([hf_api_url(repo, path)])
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"unexpected HuggingFace API response for {repo}")
    return data


def command_sources(_: argparse.Namespace) -> int:
    print(f"RobotMoon page: {ROBOTMOON_PAGE}")
    for source in SOURCES:
        print(f"{source.kind:12} {source.name:18} {source.url}")
        print(f"{'':12} {'':18} {source.note}")
    return 0


def command_list_hf(args: argparse.Namespace) -> int:
    entries = load_hf_tree(args.repo, args.path)
    rows: list[tuple[int, str]] = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        path = str(entry.get("path", ""))
        if args.suffix and not path.endswith(args.suffix):
            continue
        size = int(entry.get("size", 0))
        rows.append((size, path))

    rows.sort(key=lambda item: item[1])
    for size, path in rows:
        print(f"{size:14d}  {path}")
    print(f"files={len(rows)}")
    return 0


def command_download_hf(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    url = hf_resolve_url(args.repo, args.path)

    curl_args = []
    if args.max_bytes is not None:
        if args.max_bytes <= 0:
            raise SystemExit("--max-bytes must be positive")
        curl_args.extend(["--range", f"0-{args.max_bytes - 1}"])
    curl_args.extend(["--output", str(output), url])

    if args.dry_run:
        print("curl -L --fail --silent --show-error " + " ".join(curl_args))
        return 0

    run_curl(curl_args)
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    return 0


def command_inspect_zst(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"missing input: {path}")

    zstd = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert zstd.stdout is not None
    header = zstd.stdout.read(args.bytes)
    zstd.kill()
    _, stderr = zstd.communicate()

    if stderr and not header:
        raise SystemExit(stderr.decode("utf-8", errors="replace").strip())

    hex_bytes = " ".join(f"{byte:02x}" for byte in header)
    ascii_bytes = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in header)
    print(f"bytes={len(header)}")
    print(f"hex={hex_bytes}")
    print(f"ascii={ascii_bytes}")
    print(f"binpack_magic={header.startswith(b'BINP')}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect/download RobotMoon NNUE binpack data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sources = subparsers.add_parser("sources", help="List known RobotMoon-related sources")
    sources.set_defaults(func=command_sources)

    list_hf = subparsers.add_parser("list-hf", help="List files in a HuggingFace dataset")
    list_hf.add_argument("--repo", required=True, help="Dataset repo, e.g. linrock/test80-2024")
    list_hf.add_argument("--path", default=None, help="Optional path inside dataset")
    list_hf.add_argument("--suffix", default=".binpack.zst", help="Filter by suffix")
    list_hf.set_defaults(func=command_list_hf)

    download_hf = subparsers.add_parser("download-hf", help="Download a HuggingFace dataset file")
    download_hf.add_argument("--repo", required=True, help="Dataset repo, e.g. linrock/test80-2024")
    download_hf.add_argument("--path", required=True, help="File path inside dataset")
    download_hf.add_argument("--output", required=True, help="Local output path")
    download_hf.add_argument("--max-bytes", type=int, default=None, help="Optional byte limit")
    download_hf.add_argument("--dry-run", action="store_true", help="Print curl command only")
    download_hf.set_defaults(func=command_download_hf)

    inspect_zst = subparsers.add_parser("inspect-zst", help="Inspect the first decompressed bytes")
    inspect_zst.add_argument("--input", required=True, help="Local .zst file")
    inspect_zst.add_argument("--bytes", type=int, default=32, help="Number of decompressed bytes")
    inspect_zst.set_defaults(func=command_inspect_zst)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
