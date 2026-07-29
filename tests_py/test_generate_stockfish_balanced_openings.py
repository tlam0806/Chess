from __future__ import annotations

import random
import unittest

from tools.generate_stockfish_balanced_openings import (
    AnalysisLine,
    choose_near_best,
    material_imbalance_cp,
)


class BalancedOpeningGeneratorTests(unittest.TestCase):
    def test_choose_near_best_respects_cp_window(self) -> None:
        lines = [
            AnalysisLine(1, 20, None, "e2e4"),
            AnalysisLine(2, 0, None, "d2d4"),
            AnalysisLine(3, -80, None, "a2a3"),
        ]
        observed = {
            choose_near_best(lines, 25, random.Random(seed))
            for seed in range(20)
        }
        self.assertEqual(observed, {"e2e4", "d2d4"})


    def test_choose_near_best_rejects_mate_analysis(self) -> None:
        lines = [
            AnalysisLine(1, None, 3, "h5h7"),
            AnalysisLine(2, 100, None, "h5e5"),
        ]
        self.assertIsNone(choose_near_best(lines, 50, random.Random(0)))


    def test_material_imbalance_uses_fen_board_only(self) -> None:
        equal = "8/8/8/8/8/8/4P3/4p3 w - - 0 1"
        white_up_rook = "8/8/8/8/8/8/4P3/R3p3 w - - 0 1"
        self.assertEqual(material_imbalance_cp(equal), 0)
        self.assertEqual(material_imbalance_cp(white_up_rook), 500)


if __name__ == "__main__":
    unittest.main()
