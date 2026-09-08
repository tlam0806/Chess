from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tools.data.build_nnue_v43_all_prunes_dataset import (
    DATASET_FORMAT,
    DEFAULT_COUNTS,
    OLD_DATASET_FORMAT,
    PositionRow,
    build_dataset,
    load_rows,
    logical_source_hash,
    reconstruct_old_splits,
    row_text,
)


def create_database(path: Path, rows: list[tuple[object, ...]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE positions (
                model_key BLOB NOT NULL,
                fen TEXT NOT NULL,
                game_id TEXT NOT NULL,
                quarter INTEGER NOT NULL,
                ply INTEGER NOT NULL,
                phase INTEGER NOT NULL,
                player TEXT NOT NULL,
                player_elo INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?)", rows)
        connection.commit()
    finally:
        connection.close()


def write_prior_dataset(
    directory: Path,
    rows: list[PositionRow],
    seed: int,
    counts: dict[str, int],
) -> dict[str, list[PositionRow]]:
    selected = reconstruct_old_splits(rows, seed, counts)
    directory.mkdir()
    for split, split_rows in selected.items():
        (directory / f"{split}.tsv").write_text(
            "\n".join(row_text(row) for row in split_rows) + "\n")
    (directory / "manifest.json").write_text(json.dumps({
        "format": OLD_DATASET_FORMAT,
        "schema_version": 2,
        "seed": seed,
        "source": {"logical_sha256": logical_source_hash(rows)},
        "splits": {
            split: {"count": len(split_rows)}
            for split, split_rows in selected.items()
        },
    }))
    return selected


def test_builder_excludes_reconstructed_old_games_and_keeps_legal_check(
    tmp_path: Path,
) -> None:
    database = tmp_path / "collector.sqlite3"
    raw_rows: list[tuple[object, ...]] = []
    for index in range(200):
        # The old collector stores one-based move-to-play ply.  All ordinary
        # roots are legal and phase labels intentionally follow a natural,
        # nonuniform pattern rather than a balanced test fixture.
        phase = min(7, index // 35)
        raw_rows.append((
            index.to_bytes(16, "big"),
            "8/8/8/8/8/8/8/K6k w - - 0 1",
            f"game-{index}",
            index % 4,
            1,
            phase,
            f"bot-{index % 13}",
            2300 + index % 301,
        ))
    create_database(database, raw_rows)
    rows = load_rows(database)
    seed = 731
    old_counts = {"tune": 1, "selection": 1, "holdout": 1}
    preliminary = reconstruct_old_splits(rows, seed, old_counts)
    excluded_games = {
        row.game_id for split_rows in preliminary.values() for row in split_rows
    }

    # Change one guaranteed-fresh game to a legal checked root.  Recreate the
    # source and prior manifest after the change so its logical hash is exact.
    checked_index = next(
        index for index in range(200) if f"game-{index}" not in excluded_games)
    raw_rows[checked_index] = (
        checked_index.to_bytes(16, "big"),
        "7k/8/8/8/8/8/7R/K7 b - - 0 1",
        f"game-{checked_index}",
        checked_index % 4,
        2,
        min(7, checked_index // 35),
        f"bot-{checked_index % 13}",
        2300 + checked_index % 301,
    )
    database.unlink()
    create_database(database, raw_rows)
    rows = load_rows(database)
    assert sum(row.in_check for row in rows) == 1

    prior_dir = tmp_path / "prior"
    prior = write_prior_dataset(prior_dir, rows, seed, old_counts)
    excluded_games = {
        row.game_id for split_rows in prior.values() for row in split_rows
    }
    excluded_keys = {
        row.model_key for split_rows in prior.values() for row in split_rows
    }

    # Select every remaining game so the checked root must survive; unusual
    # phase frequencies must not be resampled into a balanced distribution.
    counts = {
        "tune": 50,
        "selection": 50,
        "aspiration_refresh": 50,
        "holdout": 47,
    }
    output = tmp_path / "fresh"
    manifest = build_dataset(database, output, [prior_dir], counts, seed=991)

    assert manifest["format"] == DATASET_FORMAT
    assert manifest["source"]["checked_root_count"] == 1
    assert sum(
        split["checked_root_count"]
        for split in manifest["splits"].values()
    ) == 1
    assert manifest["exclusions"]["unique_game_ids"] == 3
    assert manifest["exclusions"]["canonical_position_and_mirror_keys"] == 3
    assert manifest["overlap"] == {
        "games_between_new_splits": 0,
        "position_or_mirror_keys_between_new_splits": 0,
        "games_with_prior_v42_v43": 0,
        "position_or_mirror_keys_with_prior_v42_v43": 0,
    }

    source_by_key = {row.model_key: row for row in rows}
    seen_games: set[str] = set()
    seen_keys: set[str] = set()
    for split, count in counts.items():
        lines = (output / f"{split}.tsv").read_text().splitlines()
        assert len(lines) == count
        for line in lines:
            category, key, ply, fen = line.split("\t", 3)
            row = source_by_key[key]
            assert category == f"phase{row.phase}"
            assert int(ply) == row.ply
            assert fen == row.fen
            assert row.game_id not in excluded_games
            assert key not in excluded_keys
            assert row.game_id not in seen_games
            assert key not in seen_keys
            seen_games.add(row.game_id)
            seen_keys.add(key)
    assert len(seen_games) == sum(counts.values())


def test_prior_split_tamper_is_rejected_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "collector.sqlite3"
    create_database(database, [
        (
            index.to_bytes(16, "big"),
            "8/8/8/8/8/8/8/K6k w - - 0 1",
            f"game-{index}", 0, 1, 0, "bot", 2400,
        )
        for index in range(100)
    ])
    rows = load_rows(database)
    prior_dir = tmp_path / "prior"
    write_prior_dataset(
        prior_dir, rows, seed=17,
        counts={"tune": 1, "selection": 1, "holdout": 1})
    tune_path = prior_dir / "tune.tsv"
    fields = tune_path.read_text().rstrip("\n").split("\t", 3)
    fields[1] = "ff" * 16
    tune_path.write_text("\t".join(fields) + "\n")

    with pytest.raises(RuntimeError, match="does not match its seed/count"):
        build_dataset(
            database,
            tmp_path / "fresh",
            [prior_dir],
            {split: 1 for split in DEFAULT_COUNTS},
            seed=31,
        )


def test_source_duplicate_canonical_mirror_key_is_rejected(
    tmp_path: Path,
) -> None:
    database = tmp_path / "collector.sqlite3"
    duplicate_key = bytes.fromhex("12" * 16)
    create_database(database, [
        (duplicate_key, "8/8/8/8/8/8/8/K6k w - - 0 1",
         "game-a", 0, 1, 0, "bot", 2400),
        (duplicate_key, "8/8/8/8/8/8/8/K6k b - - 0 1",
         "game-b", 0, 2, 0, "bot", 2400),
    ])
    with pytest.raises(RuntimeError, match="duplicate canonical"):
        load_rows(database)
