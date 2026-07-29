from __future__ import annotations

import unittest

import numpy as np

from tools.evaluate_nnue_affine_calibration import fit_mae_affine, fit_ols_affine


class NnueAffineCalibrationTests(unittest.TestCase):
    def test_ols_recovers_exact_affine_mapping(self) -> None:
        prediction = np.linspace(-500.0, 500.0, 1001)
        target = 1.75 * prediction - 23.0
        scale, bias = fit_ols_affine(prediction, target)
        self.assertAlmostEqual(scale, 1.75, places=12)
        self.assertAlmostEqual(bias, -23.0, places=12)

    def test_mae_fit_is_robust_to_large_target_outlier(self) -> None:
        prediction = np.linspace(-100.0, 100.0, 1001)
        target = 1.4 * prediction + 17.0
        target[-1] += 100_000.0
        scale, bias = fit_mae_affine(prediction, target)
        self.assertAlmostEqual(scale, 1.4, places=5)
        self.assertAlmostEqual(bias, 17.0, places=5)


if __name__ == "__main__":
    unittest.main()
