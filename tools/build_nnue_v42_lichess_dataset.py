#!/usr/bin/env python3
"""Build sealed V42 tune splits from the production-like Lichess BOT corpus.

The source collector can retain up to four positions per game.  This builder
first assigns whole games to a split and then keeps exactly one deterministic
position from each game.  Consequently no game can contribute to two splits,
and the rows used by the tuner are independent at game granularity.

TSV schema (v2): category, canonical position key, absolute ply, FEN.
The explicit ply is required by the phase/ply WDL calibration; it must not be
reconstructed from a normalized training-board FEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 2
DEFAULT_COUNTS = {"tune": 8_000, "selection": 4_000, "holdout": 4_000}
SPLIT_BUCKETS = 10_000
SPLIT_LIMITS = {"tune": 5_000, "selection": 7_500, "holdout": 10_000}


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


def digest_int(*parts: object) -> int:
    text = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def split_for_game(game_id: str, seed: int) -> str:
    bucket = digest_int(seed, "split", game_id) % SPLIT_BUCKETS
    if bucket < SPLIT_LIMITS["tune"]:
        return "tune"
    if bucket < SPLIT_LIMITS["selection"]:
        return "selection"
    return "holdout"


def fen_absolute_ply(fen: str) -> int:
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ("w", "b"):
        raise ValueError(f"invalid six-field FEN: {fen}")
    fullmove = int(fields[5])
    if fullmove < 1:
        raise ValueError(f"invalid FEN fullmove number: {fen}")
    return 2 * (fullmove - 1) + (fields[1] == "b")


def logical_source_hash(rows: list[PositionRow]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.model_key):
        digest.update(
            json.dumps(
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
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_rows(database: Path) -> list[PositionRow]:
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(positions)")
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
            fen_ply = fen_absolute_ply(str(raw[1]))
            row = PositionRow(
                model_key=str(raw[0]).lower(),
                fen=str(raw[1]),
                game_id=str(raw[2]),
                quarter=int(raw[3]),
                source_ply=int(raw[4]),
                # The collector historically stored the index of the move
                # about to be played (one based).  WDL calibration instead
                # uses the absolute position ply (zero based).  The six-field
                # FEN is authoritative for that value.
                ply=fen_ply,
                phase=int(raw[5]),
                player=str(raw[6]),
                player_elo=int(raw[7]),
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
        return result
    finally:
        connection.close()


def choose_one_per_game(rows: list[PositionRow], seed: int) -> list[PositionRow]:
    grouped: dict[str, list[PositionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.game_id].append(row)
    chosen: list[PositionRow] = []
    for game_id, candidates in grouped.items():
        chosen.append(min(
            candidates,
            key=lambda row: (
                digest_int(seed, "pick", game_id, row.model_key),
                row.model_key,
            ),
        ))
    return chosen


def select_split(
    rows: list[PositionRow], split: str, count: int, seed: int
) -> list[PositionRow]:
    candidates = [
        row for row in rows if split_for_game(row.game_id, seed) == split
    ]
    if len(candidates) < count:
        raise RuntimeError(
            f"not enough one-per-game rows for {split}: "
            f"{len(candidates)} < {count}")
    candidates.sort(key=lambda row: (
        digest_int(seed, "sample", split, row.model_key), row.model_key))
    return candidates[:count]


def row_text(row: PositionRow) -> str:
    return f"phase{row.phase}\t{row.model_key}\t{row.ply}\t{row.fen}"


def split_stats(rows: list[PositionRow]) -> dict[str, object]:
    plies = sorted(row.ply for row in rows)
    return {
        "count": len(rows),
        "games": len({row.game_id for row in rows}),
        "players": len({row.player for row in rows}),
        "phase_counts": dict(sorted(Counter(row.phase for row in rows).items())),
        "quarter_counts": dict(sorted(Counter(row.quarter for row in rows).items())),
        "ply": {
            "min": plies[0],
            "median": plies[len(plies) // 2],
            "mean": sum(plies) / len(plies),
            "max": plies[-1],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    for split, default in DEFAULT_COUNTS.items():
        parser.add_argument(f"--{split}-count", type=int, default=default)
    args = parser.parse_args()
    for split in DEFAULT_COUNTS:
        if getattr(args, f"{split}_count") <= 0:
            parser.error(f"--{split}-count must be positive")
    return args


def main() -> None:
    args = parse_args()
    database = args.database.resolve(strict=True)
    rows = load_rows(database)
    if not rows:
        raise RuntimeError("source database contains no positions")
    if len({row.model_key for row in rows}) != len(rows):
        raise RuntimeError("source contains duplicate canonical model keys")
    chosen = choose_one_per_game(rows, args.seed)

    selected: dict[str, list[PositionRow]] = {}
    for split in DEFAULT_COUNTS:
        selected[split] = select_split(
            chosen, split, getattr(args, f"{split}_count"), args.seed)
    game_sets = {
        split: {row.game_id for row in split_rows}
        for split, split_rows in selected.items()
    }
    for left in DEFAULT_COUNTS:
        for right in DEFAULT_COUNTS:
            if left < right and game_sets[left] & game_sets[right]:
                raise RuntimeError(f"game leakage between {left} and {right}")
    key_sets = {
        split: {row.model_key for row in split_rows}
        for split, split_rows in selected.items()
    }
    for left in DEFAULT_COUNTS:
        for right in DEFAULT_COUNTS:
            if left < right and key_sets[left] & key_sets[right]:
                raise RuntimeError(f"position leakage between {left} and {right}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in selected.items():
        shuffled = list(split_rows)
        random.Random(digest_int(args.seed, "output", split)).shuffle(shuffled)
        (args.output_dir / f"{split}.tsv").write_text(
            "\n".join(row_text(row) for row in shuffled) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": "nnue-v42-lichess-game-disjoint-v2",
        "seed": args.seed,
        "source": {
            "database": str(database),
            "logical_sha256": logical_source_hash(rows),
            "positions": len(rows),
            "games": len({row.game_id for row in rows}),
            "players": len({row.player for row in rows}),
            "elo": {
                "min": min(row.player_elo for row in rows),
                "max": max(row.player_elo for row in rows),
            },
            "stored_ply_offset_from_fen": next(iter({
                row.source_ply - row.ply for row in rows
            })),
        },
        "selection_policy": {
            "split_unit": "game_id",
            "positions_per_game": 1,
            "split_buckets": {
                "tune": [0, SPLIT_LIMITS["tune"]],
                "selection": [SPLIT_LIMITS["tune"], SPLIT_LIMITS["selection"]],
                "holdout": [SPLIT_LIMITS["selection"], SPLIT_LIMITS["holdout"]],
            },
            "category": "phase_index",
            "output_ply": "zero-based absolute ply derived from six-field FEN",
            "sampling": "uniform deterministic hash order within split",
        },
        "splits": {
            split: split_stats(split_rows)
            for split, split_rows in selected.items()
        },
        "game_overlap": {
            "tune_selection": 0,
            "tune_holdout": 0,
            "selection_holdout": 0,
        },
        "position_overlap": {
            "tune_selection": 0,
            "tune_holdout": 0,
            "selection_holdout": 0,
        },
    }
    temporary = args.output_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output_dir / "manifest.json")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
