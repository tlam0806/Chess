#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def label_one(
    source: Path,
    destination: Path,
    labeler: Path,
    expected_records: int,
    zstd_level: int,
    label_component: str,
) -> dict[str, object]:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)

    decoder = subprocess.Popen(
        ["zstd", "-q", "-dc", str(source)],
        stdout=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    label_process = subprocess.Popen(
        [
            str(labeler),
            "--input",
            "-",
            "--output",
            "-",
            "--expected-records",
            str(expected_records),
            "--progress-interval",
            "0",
            "--label-component",
            label_component,
        ],
        stdin=decoder.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    decoder.stdout.close()
    assert label_process.stdout is not None
    encoder = subprocess.Popen(
        [
            "zstd",
            "-q",
            "-T1",
            f"-{zstd_level}",
            "-f",
            "-o",
            str(temporary),
            "-",
        ],
        stdin=label_process.stdout,
    )
    label_process.stdout.close()

    label_stderr = label_process.stderr.read() if label_process.stderr is not None else b""
    label_code = label_process.wait()
    decode_code = decoder.wait()
    encode_code = encoder.wait()
    if decode_code != 0 or label_code != 0 or encode_code != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"failed {source.name}: decode={decode_code} label={label_code} "
            f"encode={encode_code}\n{label_stderr.decode(errors='replace')}"
        )
    temporary.replace(destination)
    return {
        "source": str(source),
        "output": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "labeler_log": label_stderr.decode(errors="replace").strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Relabel CHSCBIN2 shards with one Stockfish static-NNUE component"
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--labeler", type=Path, default=Path("build/stockfish_static_nnue_labeler")
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--records-per-shard", type=int, default=1_000_000)
    parser.add_argument("--zstd-level", type=int, default=6)
    parser.add_argument(
        "--label-component",
        choices=("total", "psqt", "positional"),
        default="total",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stockfish-commit", required=True)
    parser.add_argument("--network", required=True, type=Path)
    args = parser.parse_args()

    if not args.labeler.exists():
        raise SystemExit(f"missing labeler: {args.labeler}")
    if not args.network.exists():
        raise SystemExit(f"missing Stockfish NNUE: {args.network}")
    sources = sorted(args.source_dir.glob("*.cbin.zst"))
    if not sources:
        raise SystemExit(f"no .cbin.zst shards in {args.source_dir}")
    if args.output_dir.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite output directory or manifest")

    args.output_dir.mkdir(parents=True)
    results: list[dict[str, object]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    label_one,
                    source,
                    args.output_dir / source.name,
                    args.labeler,
                    args.records_per_shard,
                    args.zstd_level,
                    args.label_component,
                ): source
                for source in sources
            }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(
                    f"completed={completed}/{len(sources)} shard={Path(str(result['source'])).name}",
                    flush=True,
                )
    except BaseException:
        for path in args.output_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        raise

    results.sort(key=lambda item: str(item["source"]))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "CHSCBIN2",
        "records": len(results) * args.records_per_shard,
        "source_directory": str(args.source_dir),
        "output_directory": str(args.output_dir),
        "shards": len(results),
        "records_per_shard": args.records_per_shard,
        "label": {
            "teacher": "official-stockfish-static-nnue",
            "stockfish_commit": args.stockfish_commit,
            "network_path": str(args.network),
            "network_sha256": sha256(args.network),
            "value": {
                "total": "raw PSQT + positional",
                "psqt": "raw PSQT",
                "positional": "raw positional",
            }[args.label_component],
            "perspective": "side-to-move",
            "excluded": ["positions in check", "positions without a legal source move"],
            "not_applied": [
                "UCI WDL-to-CP conversion",
                "optimism",
                "rule50 scaling",
                "correction history",
                "search",
            ],
        },
        "labeler": {
            "path": str(args.labeler),
            "sha256": sha256(args.labeler),
            "workers": args.workers,
        },
        "outputs": results,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"done records={manifest['records']} manifest={args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
