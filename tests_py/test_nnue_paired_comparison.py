from __future__ import annotations

import unittest

import numpy as np

from tools.evaluate_nnue_paired_comparison import compare_errors, paired_block_bootstrap


class NnuePairedComparisonTests(unittest.TestCase):
    def test_bootstrap_detects_consistent_candidate_improvement(self) -> None:
        delta = np.linspace(-2.0, -1.0, 4096)
        result = paired_block_bootstrap(
            delta,
            replicates=1000,
            block_size=64,
            confidence=0.95,
            rng=np.random.default_rng(7),
        )
        self.assertAlmostEqual(result["delta_mae_cp"], -1.5)
        self.assertLess(result["ci_high_cp"], 0.0)
        self.assertTrue(result["ci_excludes_zero"])
        self.assertEqual(result["probability_candidate_better"], 1.0)

    def test_compare_uses_candidate_minus_reference_and_buckets(self) -> None:
        targets = np.tile(np.array([50.0, 200.0, 450.0, 800.0]), 1024)
        reference = np.full(targets.shape, 10.0)
        candidate = np.full(targets.shape, 9.5)
        result = compare_errors(
            reference,
            candidate,
            targets,
            (0.0, 100.0, 300.0, 600.0, 1000.0),
            replicates=500,
            block_size=64,
            confidence=0.95,
            seed=11,
        )
        self.assertAlmostEqual(result["overall"]["delta_mae_cp"], -0.5)
        self.assertEqual(len(result["bins"]), 4)
        self.assertTrue(all(row["samples"] == 1024 for row in result["bins"]))
        self.assertTrue(all(row["delta_mae_cp"] == -0.5 for row in result["bins"]))


if __name__ == "__main__":
    unittest.main()
