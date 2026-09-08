from __future__ import annotations

import io
import unittest

import chess
import chess.pgn

from tools.data.collect_lichess_bot_population import (
    Config,
    canonical_f2m_key,
    game_candidates,
    game_is_eligible,
    pack_relative_board,
)


def config() -> Config:
    return Config(
        min_elo=2300,
        max_elo=2600,
        min_base=60,
        max_base=600,
        min_increment=0,
        max_increment=10,
        target_positions=200_000,
        min_accounts=100,
        account_position_cap=2_000,
        games_per_account=1_000,
        random_seed=20260826,
        excluded_accounts=frozenset({"trumcovuaa"}),
    )


def read_game(pgn: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    assert not game.errors
    return game


class LichessBotPopulationTests(unittest.TestCase):
    def test_relative_encoding_is_color_rank_invariant(self) -> None:
        board = chess.Board("4k3/7p/8/8/3P4/8/P7/4K3 w - - 0 1")
        mirrored = board.mirror()
        first = pack_relative_board(board)
        second = pack_relative_board(mirrored)
        self.assertEqual(first, second)

    def test_f2m_key_is_horizontal_mirror_invariant(self) -> None:
        board = chess.Board("2k5/7p/8/8/3P4/8/P7/6K1 w - - 0 1")
        mirrored = board.transform(chess.flip_horizontal)
        first = canonical_f2m_key(*pack_relative_board(board))
        second = canonical_f2m_key(*pack_relative_board(mirrored))
        self.assertEqual(first, second)

    def test_selects_only_eligible_bot_turns_and_one_per_quarter(self) -> None:
        game = read_game(
            """[Event "rated blitz game"]
[Site "https://lichess.org/testgame"]
[GameId "testgame"]
[White "EligibleBot"]
[Black "TooStrongBot"]
[WhiteTitle "BOT"]
[BlackTitle "BOT"]
[WhiteElo "2400"]
[BlackElo "2700"]
[Result "1/2-1/2"]
[TimeControl "180+2"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1/2-1/2
"""
        )
        self.assertTrue(game_is_eligible(game, config()))
        quarters = game_candidates(game, config())
        self.assertEqual([len(items) for items in quarters], [1, 1, 1, 1])
        self.assertEqual([items[0].player for items in quarters], ["eligiblebot"] * 4)
        self.assertEqual([items[0].ply for items in quarters], [1, 3, 5, 7])

    def test_rejects_non_production_time_control(self) -> None:
        game = read_game(
            """[Event "rated rapid game"]
[White "EligibleBot"]
[Black "Other"]
[WhiteTitle "BOT"]
[WhiteElo "2400"]
[BlackElo "2400"]
[Result "1-0"]
[TimeControl "900+10"]

1. e4 e5 1-0
"""
        )
        self.assertFalse(game_is_eligible(game, config()))


if __name__ == "__main__":
    unittest.main()
