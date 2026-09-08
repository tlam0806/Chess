from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.data.build_nnue_v42_lichess_dataset import (
    PositionRow,
    choose_one_per_game,
    fen_absolute_ply,
    load_rows,
    row_text,
    select_split,
    split_for_game,
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


class V42LichessDatasetTests(unittest.TestCase):
    def test_fen_absolute_ply_is_zero_based(self) -> None:
        self.assertEqual(fen_absolute_ply("8/8/8/8/8/8/8/K6k w - - 0 1"), 0)
        self.assertEqual(fen_absolute_ply("8/8/8/8/8/8/8/K6k b - - 0 1"), 1)
        self.assertEqual(fen_absolute_ply("8/8/8/8/8/8/8/K6k w - - 0 27"), 52)
        self.assertEqual(fen_absolute_ply("8/8/8/8/8/8/8/K6k b - - 0 27"), 53)

    def test_load_rows_normalizes_historical_one_based_collector_ply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "collector.sqlite3"
            create_database(database, [(
                bytes.fromhex("00" * 16),
                "8/8/8/8/8/8/8/K6k b - - 7 27",
                "game-1", 2, 54, 0, "bot", 2450,
            )])
            [row] = load_rows(database)
            self.assertEqual(row.source_ply, 54)
            self.assertEqual(row.ply, 53)
            self.assertEqual(row_text(row).split("\t")[2], "53")

    def test_load_rows_rejects_mixed_ply_conventions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "collector.sqlite3"
            create_database(database, [
                (bytes.fromhex("00" * 16),
                 "8/8/8/8/8/8/8/K6k w - - 0 1",
                 "game-0", 0, 0, 0, "bot", 2400),
                (bytes.fromhex("01" * 16),
                 "8/8/8/8/8/8/8/K6k b - - 0 1",
                 "game-1", 0, 2, 0, "bot", 2400),
            ])
            with self.assertRaisesRegex(RuntimeError, "mixes incompatible ply"):
                load_rows(database)

    def test_one_position_per_game_and_game_disjoint_splits(self) -> None:
        rows: list[PositionRow] = []
        for game_index in range(200):
            for sample_index in range(2):
                ply = game_index + sample_index
                rows.append(PositionRow(
                    model_key=f"{2 * game_index + sample_index:032x}",
                    fen=("8/8/8/8/8/8/8/K6k "
                         f"{'w' if ply % 2 == 0 else 'b'} - - 0 {ply // 2 + 1}"),
                    game_id=f"game-{game_index}",
                    quarter=sample_index,
                    source_ply=ply + 1,
                    ply=ply,
                    phase=0,
                    player="bot",
                    player_elo=2400,
                ))
        chosen = choose_one_per_game(rows, seed=17)
        self.assertEqual(len(chosen), 200)
        self.assertEqual(len({row.game_id for row in chosen}), 200)

        selected: dict[str, list[PositionRow]] = {}
        for split in ("tune", "selection", "holdout"):
            available = [
                row for row in chosen if split_for_game(row.game_id, 17) == split
            ]
            selected[split] = select_split(
                chosen, split, min(5, len(available)), 17)
        game_sets = {
            split: {row.game_id for row in split_rows}
            for split, split_rows in selected.items()
        }
        self.assertFalse(game_sets["tune"] & game_sets["selection"])
        self.assertFalse(game_sets["tune"] & game_sets["holdout"])
        self.assertFalse(game_sets["selection"] & game_sets["holdout"])


if __name__ == "__main__":
    unittest.main()
