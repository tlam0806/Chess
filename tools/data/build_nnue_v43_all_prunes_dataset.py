#!/usr/bin/env python3
"""Build the fresh, sealed corpus for the final V43 joint-prune tune.

The source population contains up to four positions per Lichess game.  This
builder first removes every *game* used by the earlier V42/V43 aspiration
experiment, then deterministically keeps one position per remaining game and
assigns whole games to four exact-size splits.

The collector's ``model_key`` is already the canonical F2M key (the horizontal
mirror is folded into the same 128-bit key), so deduplicating that value is
both a position-hash and mirror-hash dedupe.  The output keeps the explicit,
zero-based absolute ply required by the WDL calibration.

The current population database was collected with in-check roots disabled.
This builder does not repeat that filter: a legal in-check FEN is retained and
counted.  Consequently a future collector replay can add such roots without a
dataset-format change, while the manifest honestly records zero checked roots
for the existing database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import chess


SCHEMA_VERSION = 3
DATASET_FORMAT = "nnue-v43-all-prunes-game-disjoint-v3"
OLD_DATASET_FORMAT = "nnue-v42-lichess-game-disjoint-v2"
DEFAULT_COUNTS = {
    "tune": 8_000,
    "selection": 4_000,
    "aspiration_refresh": 4_000,
    "holdout": 4_000,
}
DEFAULT_DATABASE = Path(
    "data/lichess_bot_2300_2600_population_v1/collector.sqlite3")
DEFAULT_EXCLUDE_DATASET = Path(
    "data/nnue_v42_aspiration_lichess_20260830")

# These constants reproduce the V42 builder exactly.  V43 aspiration reused
# that same dataset, so reconstructing these three selections accounts for
# both experiments without double-counting them.
OLD_SPLIT_BUCKETS = 10_000
OLD_SPLIT_LIMITS = {"tune": 5_000, "selection": 7_500, "holdout": 10_000}
OLD_SPLITS = ("tune", "selection", "holdout")


@dataclass(frozen=True)
class PositionRow:
    model_key: str
    fen: str
    game_id: str
    quarter: int
    source_ply: int
    ply: int
    phase: int
    player: str
    player_elo: int
    in_check: bool


@dataclass(frozen=True)
class Exclusions:
    game_ids: frozenset[str]
    model_keys: frozenset[str]
    datasets: tuple[dict[str, object], ...]


def digest_int(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fen_absolute_ply(fen: str) -> int:
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ("w", "b"):
        raise ValueError(f"invalid six-field FEN: {fen}")
    fullmove = int(fields[5])
    if fullmove < 1:
        raise ValueError(f"invalid FEN fullmove number: {fen}")
    return 2 * (fullmove - 1) + (fields[1] == "b")


def logical_source_hash(rows: Iterable[PositionRow]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.model_key):
        digest.update(json.dumps(
            {
                "fen": row.fen,
                "game_id": row.game_id,
                "model_key": row.model_key,
                "phase": row.phase,
                "player": row.player,
                "player_elo": row.player_elo,
                "ply": row.ply,
                "quarter": row.quarter,
                "source_ply": row.source_ply,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_rows(database: Path) -> list[PositionRow]:
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        columns = {
            str(row[1]) for row in connection.execute(
                "PRAGMA table_info(positions)")
        }
        required = {
            "model_key", "fen", "game_id", "quarter", "ply", "phase",
            "player", "player_elo",
        }
        if not required <= columns:
            raise RuntimeError(
                f"positions table is missing columns: {sorted(required - columns)}")

        result: list[PositionRow] = []
        for raw in connection.execute(
            """
            SELECT hex(model_key),fen,game_id,quarter,ply,phase,player,player_elo
            FROM positions
            """
        ):
            model_key = str(raw[0]).lower()
            fen = str(raw[1])
            board = chess.Board(fen)
            if not board.is_valid():
                raise RuntimeError(f"illegal source FEN for {model_key}: {fen}")
            ply = fen_absolute_ply(fen)
            row = PositionRow(
                model_key=model_key,
                fen=fen,
                game_id=str(raw[2]),
                quarter=int(raw[3]),
                source_ply=int(raw[4]),
                ply=ply,
                phase=int(raw[5]),
                player=str(raw[6]),
                player_elo=int(raw[7]),
                in_check=board.is_check(),
            )
            if len(row.model_key) != 32:
                raise RuntimeError(f"invalid canonical model key: {row.model_key}")
            if not 0 <= row.quarter < 4:
                raise RuntimeError(f"invalid quarter for {row.model_key}")
            if not 0 <= row.phase < 8:
                raise RuntimeError(f"invalid phase for {row.model_key}")
            if not 0 <= row.source_ply <= 0x3FFF:
                raise RuntimeError(f"invalid source ply for {row.model_key}")
            if row.source_ply not in (row.ply, row.ply + 1):
                raise RuntimeError(
                    f"FEN/metadata ply mismatch for {row.model_key}: "
                    f"fen={row.ply} metadata={row.source_ply}")
            result.append(row)

        offsets = {row.source_ply - row.ply for row in result}
        if len(offsets) > 1:
            raise RuntimeError(
                f"source mixes incompatible ply conventions: {sorted(offsets)}")
        keys = [row.model_key for row in result]
        if len(set(keys)) != len(keys):
            raise RuntimeError(
                "source contains duplicate canonical position/mirror keys")
        return result
    finally:
        connection.close()


def choose_one_per_game(rows: Iterable[PositionRow], seed: int) -> list[PositionRow]:
    grouped: dict[str, list[PositionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.game_id].append(row)
    return [
        min(
            candidates,
            key=lambda row: (
                digest_int(seed, "pick", game_id, row.model_key),
                row.model_key,
            ),
        )
        for game_id, candidates in grouped.items()
    ]


def old_split_for_game(game_id: str, seed: int) -> str:
    bucket = digest_int(seed, "split", game_id) % OLD_SPLIT_BUCKETS
    if bucket < OLD_SPLIT_LIMITS["tune"]:
        return "tune"
    if bucket < OLD_SPLIT_LIMITS["selection"]:
        return "selection"
    return "holdout"


def reconstruct_old_splits(
    rows: list[PositionRow], seed: int, counts: Mapping[str, int]
) -> dict[str, list[PositionRow]]:
    chosen = choose_one_per_game(rows, seed)
    selected: dict[str, list[PositionRow]] = {}
    for split in OLD_SPLITS:
        candidates = [
            row for row in chosen
            if old_split_for_game(row.game_id, seed) == split
        ]
        candidates.sort(key=lambda row: (
            digest_int(seed, "sample", split, row.model_key), row.model_key))
        count = int(counts[split])
        if len(candidates) < count:
            raise RuntimeError(
                f"cannot reconstruct old {split}: {len(candidates)} < {count}")
        selected[split] = candidates[:count]
    return selected


def parse_dataset_keys(dataset_dir: Path, split: str) -> set[str]:
    path = dataset_dir / f"{split}.tsv"
    if not path.is_file():
        raise RuntimeError(f"missing prior dataset split: {path}")
    keys: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line:
            continue
        fields = raw_line.split("\t", 3)
        if len(fields) != 4:
            raise RuntimeError(f"malformed {path}:{line_number}")
        key = fields[1].lower()
        if len(key) != 32 or any(ch not in "0123456789abcdef" for ch in key):
            raise RuntimeError(f"invalid model key at {path}:{line_number}")
        if key in keys:
            raise RuntimeError(f"duplicate model key in {path}: {key}")
        keys.add(key)
    return keys


def collect_exclusions(
    rows: list[PositionRow], dataset_dirs: Iterable[Path]
) -> Exclusions:
    by_key = {row.model_key: row for row in rows}
    source_hash = logical_source_hash(rows)
    all_games: set[str] = set()
    all_keys: set[str] = set()
    provenance: list[dict[str, object]] = []
    seen_dirs: set[Path] = set()

    for raw_dir in dataset_dirs:
        dataset_dir = raw_dir.resolve(strict=True)
        if dataset_dir in seen_dirs:
            continue
        seen_dirs.add(dataset_dir)
        manifest_path = dataset_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"cannot read prior dataset manifest {manifest_path}: {error}") from error
        if (
            manifest.get("format") != OLD_DATASET_FORMAT
            or manifest.get("schema_version") != 2
        ):
            raise RuntimeError(
                f"unsupported prior dataset schema: {manifest_path}")
        if manifest.get("source", {}).get("logical_sha256") != source_hash:
            raise RuntimeError(
                "prior dataset was not built from this exact logical source; "
                "refusing an unverifiable game exclusion")

        counts = {
            split: int(manifest.get("splits", {}).get(split, {}).get("count", 0))
            for split in OLD_SPLITS
        }
        if any(count <= 0 for count in counts.values()):
            raise RuntimeError(f"invalid split counts in {manifest_path}")
        seed = int(manifest["seed"])
        reconstructed = reconstruct_old_splits(rows, seed, counts)
        dataset_keys: set[str] = set()
        for split in OLD_SPLITS:
            actual = parse_dataset_keys(dataset_dir, split)
            expected = {row.model_key for row in reconstructed[split]}
            if actual != expected:
                missing = sorted(expected - actual)[:3]
                extra = sorted(actual - expected)[:3]
                raise RuntimeError(
                    f"prior {split} does not match its seed/count reconstruction: "
                    f"missing={missing} extra={extra}")
            dataset_keys.update(actual)

        missing_from_source = sorted(dataset_keys - by_key.keys())
        if missing_from_source:
            raise RuntimeError(
                f"prior keys missing from source: {missing_from_source[:3]}")
        dataset_games = {by_key[key].game_id for key in dataset_keys}
        if len(dataset_games) != len(dataset_keys):
            raise RuntimeError(
                "prior game-disjoint dataset unexpectedly has multiple rows per game")
        all_keys.update(dataset_keys)
        all_games.update(dataset_games)
        provenance.append({
            "directory": str(dataset_dir),
            "manifest_sha256": file_sha256(manifest_path),
            "format": OLD_DATASET_FORMAT,
            "seed": seed,
            "position_keys": len(dataset_keys),
            "game_ids": len(dataset_games),
            "reconstruction_verified": True,
        })

    if not provenance:
        raise RuntimeError("at least one prior V42/V43 dataset must be excluded")
    return Exclusions(
        game_ids=frozenset(all_games),
        model_keys=frozenset(all_keys),
        datasets=tuple(provenance),
    )


def select_exact_splits(
    rows: list[PositionRow], counts: Mapping[str, int], seed: int
) -> dict[str, list[PositionRow]]:
    required = sum(int(counts[split]) for split in DEFAULT_COUNTS)
    if len(rows) < required:
        raise RuntimeError(
            f"not enough fresh one-per-game rows: {len(rows)} < {required}")
    ordered = sorted(rows, key=lambda row: (
        digest_int(seed, "fresh-sample", row.game_id, row.model_key),
        row.game_id,
        row.model_key,
    ))
    selected: dict[str, list[PositionRow]] = {}
    offset = 0
    for split in DEFAULT_COUNTS:
        count = int(counts[split])
        selected[split] = ordered[offset:offset + count]
        offset += count
    return selected


def row_text(row: PositionRow) -> str:
    return f"phase{row.phase}\t{row.model_key}\t{row.ply}\t{row.fen}"


def split_stats(rows: list[PositionRow]) -> dict[str, object]:
    plies = sorted(row.ply for row in rows)
    return {
        "count": len(rows),
        "games": len({row.game_id for row in rows}),
        "canonical_position_and_mirror_keys": len({row.model_key for row in rows}),
        "players": len({row.player for row in rows}),
        "checked_root_count": sum(row.in_check for row in rows),
        "phase_counts": dict(sorted(Counter(row.phase for row in rows).items())),
        "quarter_counts": dict(sorted(Counter(row.quarter for row in rows).items())),
        "ply": {
            "min": plies[0],
            "median": plies[len(plies) // 2],
            "mean": sum(plies) / len(plies),
            "max": plies[-1],
        },
    }


def atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(data)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_dataset(
    database: Path,
    output_dir: Path,
    exclude_dataset_dirs: Iterable[Path],
    counts: Mapping[str, int],
    seed: int,
) -> dict[str, object]:
    database = database.resolve(strict=True)
    rows = load_rows(database)
    if not rows:
        raise RuntimeError("source database contains no positions")
    exclusions = collect_exclusions(rows, exclude_dataset_dirs)

    fresh_rows = [
        row for row in rows
        if row.game_id not in exclusions.game_ids
        and row.model_key not in exclusions.model_keys
    ]
    chosen = choose_one_per_game(fresh_rows, seed)
    selected = select_exact_splits(chosen, counts, seed)

    game_sets = {
        split: {row.game_id for row in split_rows}
        for split, split_rows in selected.items()
    }
    key_sets = {
        split: {row.model_key for row in split_rows}
        for split, split_rows in selected.items()
    }
    split_names = tuple(DEFAULT_COUNTS)
    for left_index, left in enumerate(split_names):
        if game_sets[left] & exclusions.game_ids:
            raise RuntimeError(f"excluded game leaked into {left}")
        if key_sets[left] & exclusions.model_keys:
            raise RuntimeError(f"excluded position/mirror key leaked into {left}")
        for right in split_names[left_index + 1:]:
            if game_sets[left] & game_sets[right]:
                raise RuntimeError(f"game leakage between {left} and {right}")
            if key_sets[left] & key_sets[right]:
                raise RuntimeError(f"position/mirror leakage between {left} and {right}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, object]] = {}
    for split, split_rows in selected.items():
        shuffled = list(split_rows)
        random.Random(digest_int(seed, "output", split)).shuffle(shuffled)
        path = output_dir / f"{split}.tsv"
        payload = ("\n".join(row_text(row) for row in shuffled) + "\n").encode()
        atomic_write(path, payload)
        split_files[split] = {
            "file": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    all_selected = [row for split in split_names for row in selected[split]]
    source_offsets = {row.source_ply - row.ply for row in rows}
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "format": DATASET_FORMAT,
        "seed": seed,
        "source": {
            "database": str(database),
            "logical_sha256": logical_source_hash(rows),
            "positions": len(rows),
            "games_with_any_collected_position": len({row.game_id for row in rows}),
            "players": len({row.player for row in rows}),
            "elo": {
                "min": min(row.player_elo for row in rows),
                "max": max(row.player_elo for row in rows),
            },
            "stored_ply_offset_from_fen": next(iter(source_offsets)),
            "checked_root_count": sum(row.in_check for row in rows),
            "limitation": (
                "The current population DB was collected with in-check roots "
                "disabled, so checked_root_count is expected to be 0. The "
                "builder accepts and retains legal in-check roots when a future "
                "collector replay supplies them."
            ),
        },
        "exclusions": {
            "datasets": list(exclusions.datasets),
            "unique_game_ids": len(exclusions.game_ids),
            "canonical_position_and_mirror_keys": len(exclusions.model_keys),
            "selected_overlap_game_ids": len(
                {row.game_id for row in all_selected} & exclusions.game_ids),
            "selected_overlap_position_or_mirror_keys": len(
                {row.model_key for row in all_selected} & exclusions.model_keys),
        },
        "selection_policy": {
            "split_unit": "game_id",
            "positions_per_game": 1,
            "sampling": (
                "uniform deterministic SHA-256 order over fresh games; exact "
                "contiguous split counts; no phase reweighting"
            ),
            "phase_distribution": "natural (not stratified or reweighted)",
            "position_dedupe": (
                "canonical F2M model_key; folds horizontal mirrors together"
            ),
            "category": "phase_index",
            "output_ply": "zero-based absolute ply derived from six-field FEN",
            "legal_in_check_roots": "retained without special filtering",
        },
        "splits": {
            split: split_stats(split_rows)
            for split, split_rows in selected.items()
        },
        "split_files": split_files,
        "overlap": {
            "games_between_new_splits": 0,
            "position_or_mirror_keys_between_new_splits": 0,
            "games_with_prior_v42_v43": 0,
            "position_or_mirror_keys_with_prior_v42_v43": 0,
        },
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    # The manifest is the commit marker.  A consumer must reject a directory
    # whose recorded split digests do not match the files.
    atomic_write(output_dir / "manifest.json", manifest_payload)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--exclude-dataset-dir",
        action="append",
        type=Path,
        help=(
            "Prior V42/V43 sealed dataset to exclude. Defaults to the shared "
            "V42/V43 aspiration corpus. May be repeated."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    for split, default in DEFAULT_COUNTS.items():
        parser.add_argument(f"--{split.replace('_', '-')}-count", type=int, default=default)
    args = parser.parse_args()
    if args.exclude_dataset_dir is None:
        args.exclude_dataset_dir = [DEFAULT_EXCLUDE_DATASET]
    for split in DEFAULT_COUNTS:
        if getattr(args, f"{split}_count") <= 0:
            parser.error(f"--{split.replace('_', '-')}-count must be positive")
    return args


def main() -> None:
    args = parse_args()
    counts = {
        split: getattr(args, f"{split}_count") for split in DEFAULT_COUNTS
    }
    manifest = build_dataset(
        args.database,
        args.output_dir,
        args.exclude_dataset_dir,
        counts,
        args.seed,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
