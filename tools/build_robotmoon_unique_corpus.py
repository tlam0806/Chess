#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.compact_board_data import HEADER_SIZE, RECORD_SIZE, open_reader, validate_header


CONVERSION_SUMMARY = re.compile(
    r"done converted=(?P<converted>\d+) read=(?P<read>\d+) "
    r"value_none_filtered=(?P<value_none>\d+) "
    r"(?:in_check_filtered=\d+ )?"
    r"duplicate_or_bloom_filtered=(?P<duplicates>\d+) bytes=(?P<bytes>\d+)"
)
DEDUP_BYTES = re.compile(r"dedup_filter_bytes=(?P<bytes>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, validate, and shard a unique CHSCBIN2 RobotMoon corpus."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--records", type=int, default=500_000_000)
    parser.add_argument("--records-per-shard", type=int, default=1_000_000)
    parser.add_argument("--dedup-bits-per-record", type=int, default=16)
    parser.add_argument("--zstd-level", type=int, default=6)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("build/robotmoon_binpack_to_cbin"),
    )
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path("build/validate_robotmoon_cbin"),
    )
    parser.add_argument("--keep-monolith", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def compact_payload_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    payload_bytes = 0
    stream, owner = open_reader(path)
    try:
        header = stream.read(HEADER_SIZE)
        validate_header(header, path)
        while block := stream.read(8 * 1024 * 1024):
            payload_bytes += len(block)
            digest.update(block)
    finally:
        if owner is not None:
            owner.close()  # type: ignore[attr-defined]
        else:
            stream.close()
    if payload_bytes % RECORD_SIZE != 0:
        raise RuntimeError(f"truncated compact payload in {path}")
    return payload_bytes // RECORD_SIZE, digest.hexdigest()


def sharded_payload_digest(shard_directory: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    records = 0
    shards = sorted(shard_directory.glob("*.cbin.zst"))
    if not shards:
        raise RuntimeError(f"no shards found in {shard_directory}")
    for shard_index, shard in enumerate(shards):
        stream, owner = open_reader(shard)
        payload_bytes = 0
        try:
            header = stream.read(HEADER_SIZE)
            validate_header(header, shard)
            while block := stream.read(8 * 1024 * 1024):
                payload_bytes += len(block)
                digest.update(block)
        finally:
            if owner is not None:
                owner.close()  # type: ignore[attr-defined]
            else:
                stream.close()
        if payload_bytes % RECORD_SIZE != 0:
            raise RuntimeError(f"truncated compact payload in {shard}")
        records += payload_bytes // RECORD_SIZE
        if (shard_index + 1) % 25 == 0 or shard_index + 1 == len(shards):
            print(
                f"verified_shards={shard_index + 1}/{len(shards)} records={records}",
                flush=True,
            )
    return len(shards), records, digest.hexdigest()


def run_conversion(command: list[str], log_path: Path) -> dict[str, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stderr is None:
            raise RuntimeError("failed to capture converter stderr")
        for line in process.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            log.write(line)
            log.flush()
            lines.append(line.rstrip())
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)

    summary_match = None
    dedup_bytes = None
    for line in lines:
        if match := CONVERSION_SUMMARY.fullmatch(line):
            summary_match = match
        if match := DEDUP_BYTES.fullmatch(line):
            dedup_bytes = int(match.group("bytes"))
    if summary_match is None or dedup_bytes is None:
        raise RuntimeError("converter did not emit complete summary statistics")
    result = {name: int(value) for name, value in summary_match.groupdict().items()}
    result["dedup_filter_bytes"] = dedup_bytes
    return result


def validate_monolith(
    monolith: Path,
    validator: Path,
    expected_records: int,
    spool_directory: Path,
    validation_path: Path,
) -> dict[str, object]:
    if spool_directory.exists():
        shutil.rmtree(spool_directory)
    validation_path.unlink(missing_ok=True)
    decompressor = subprocess.Popen(
        ["zstd", "-q", "-dc", str(monolith)],
        stdout=subprocess.PIPE,
    )
    if decompressor.stdout is None:
        raise RuntimeError("failed to open monolith decompressor")
    with validation_path.open("w", encoding="utf-8") as validation_output:
        validator_process = subprocess.Popen(
            [
                str(validator),
                "--input",
                "-",
                "--spool-directory",
                str(spool_directory),
                "--expected-records",
                str(expected_records),
                "--progress-interval",
                "10000000",
            ],
            stdin=decompressor.stdout,
            stdout=validation_output,
        )
        decompressor.stdout.close()
        validator_code = validator_process.wait()
    decompressor_code = decompressor.wait()
    if decompressor_code != 0:
        raise RuntimeError(f"monolith decompression failed with exit code {decompressor_code}")
    if validator_code != 0:
        raise RuntimeError(f"independent validation failed with exit code {validator_code}")
    return json.loads(validation_path.read_text(encoding="utf-8"))


def git_revision() -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False
    ).returncode != 0
    return revision, dirty


def main() -> int:
    args = parse_args()
    if args.records <= 0 or args.records_per_shard <= 0:
        raise SystemExit("record counts must be positive")
    if args.dedup_bits_per_record <= 0:
        raise SystemExit("--dedup-bits-per-record must be positive")
    for path, label in (
        (args.source, "source"),
        (args.converter, "converter"),
        (args.validator, "validator"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    prefix = args.output_prefix
    monolith = prefix.with_suffix(".cbin.zst")
    shard_directory = Path(f"{prefix}_shards_1m")
    temporary_shards = Path(f"{shard_directory}.tmp")
    manifest_path = prefix.with_suffix(".manifest.json")
    conversion_log = Path(f"{prefix}.conversion.log")
    validation_path = Path(f"{prefix}.validation.json")
    spool_directory = Path(f"{prefix}.validation_spool")
    for path in (monolith, shard_directory, temporary_shards, manifest_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    print("stage=source_hash", flush=True)
    source_sha256 = sha256_file(args.source)
    print(f"source_sha256={source_sha256}", flush=True)

    print("stage=convert", flush=True)
    conversion = run_conversion(
        [
            sys.executable,
            "tools/robotmoon_binpack_zst_to_cbin.py",
            "--input",
            str(args.source),
            "--output",
            str(monolith),
            "--converter",
            str(args.converter),
            "--limit",
            str(args.records),
            "--progress-interval",
            "10000000",
            "--zstd-level",
            str(args.zstd_level),
            "--deduplicate-positions",
            "--dedup-expected-records",
            str(args.records),
            "--dedup-bits-per-record",
            str(args.dedup_bits_per_record),
            "--require-limit-reached",
            "--require-legal-move",
        ],
        conversion_log,
    )
    if conversion["converted"] != args.records:
        raise RuntimeError("converter summary did not reach requested record count")
    subprocess.run(["zstd", "-q", "-t", str(monolith)], check=True)
    monolithic_sha256 = sha256_file(monolith)

    print("stage=independent_validation", flush=True)
    validation = validate_monolith(
        monolith,
        args.validator,
        args.records,
        spool_directory,
        validation_path,
    )
    if validation.get("fingerprint_duplicates") != 0:
        raise RuntimeError("independent validator found duplicate model inputs")

    print("stage=shard", flush=True)
    subprocess.run(
        [
            sys.executable,
            "tools/shard_cbin.py",
            "--input",
            str(monolith),
            "--output-dir",
            str(temporary_shards),
            "--records-per-shard",
            str(args.records_per_shard),
            "--zstd-level",
            str(args.zstd_level),
        ],
        check=True,
    )

    print("stage=payload_comparison", flush=True)
    monolith_records, monolith_payload_sha256 = compact_payload_digest(monolith)
    shard_count, shard_records, shard_payload_sha256 = sharded_payload_digest(temporary_shards)
    if monolith_records != args.records or shard_records != args.records:
        raise RuntimeError("record count changed during sharding")
    if shard_payload_sha256 != monolith_payload_sha256:
        raise RuntimeError("shard payload does not match monolithic payload")
    temporary_shards.replace(shard_directory)

    revision, dirty = git_revision()
    manifest = {
        "format": {
            "magic": "CHSCBIN2",
            "version": 2,
            "record_size_bytes": RECORD_SIZE,
            "target_encoding": "stockfish_raw_score_ply_result",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(args.source),
            "bytes": args.source.stat().st_size,
            "sha256": source_sha256,
        },
        "conversion": {
            "git_revision": revision,
            "git_dirty": dirty,
            "converter_binary_sha256": sha256_file(args.converter),
            "require_legal_move": True,
            "require_limit_reached": True,
            "requested_unique_model_inputs": args.records,
            "input_records_read": conversion["read"],
            "value_none_filtered": conversion["value_none"],
            "duplicate_or_bloom_filtered": conversion["duplicates"],
            "dedup": {
                "key": "normalized_board_plus_aux",
                "algorithm": "blocked_bloom_512bit_8probes",
                "bits_per_requested_record": args.dedup_bits_per_record,
                "filter_bytes": conversion["dedup_filter_bytes"],
            },
        },
        "output": {
            "monolithic_path": str(monolith),
            "monolithic_removed_after_validation": not args.keep_monolith,
            "monolithic_sha256": monolithic_sha256,
            "shard_directory": str(shard_directory),
            "shards": shard_count,
            "records_per_shard": args.records_per_shard,
            "records": shard_records,
            "uncompressed_bytes": HEADER_SIZE + shard_records * RECORD_SIZE,
            "payload_sha256": shard_payload_sha256,
        },
        "validated_distribution": validation,
        "validation": {
            "source_sha256": "pass",
            "output_zstd_test": "pass",
            "all_records_scanned": True,
            "piece_codes_and_kings": "pass",
            "aux_and_en_passant_bits": "pass",
            "value_none_absent": "pass",
            "ply_and_result_range": "pass",
            "independent_128bit_fingerprint_duplicates": 0,
            "shard_payload_matches_monolithic": "pass",
        },
    }
    temporary_manifest = manifest_path.with_suffix(".manifest.tmp.json")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    if not args.keep_monolith:
        monolith.unlink()
    print(f"done manifest={manifest_path} shards={shard_directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
