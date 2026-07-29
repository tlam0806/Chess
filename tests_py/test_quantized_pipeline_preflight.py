from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nn.compact_board_data import HEADER, HEADER_SIZE
from tools.quantized_pipeline_preflight import validate_sweep_args


def args_for(path: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "data": path,
        "data_format": "cbin",
        "jobs": 1,
        "epochs": 1,
        "patience": 1,
        "batch_size": 8,
        "train_max_samples": 8,
        "val_max_samples": 4,
        "test_max_samples": 4,
        "calibration_max_batches": 1,
        "feature_weight_scale": 255,
        "linear_weight_scale": 64,
        "output_weight_scale": 16,
        "workers": 0,
        "eval_workers": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class QuantizedPipelinePreflightTests(unittest.TestCase):
    def test_accepts_cbin_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.cbin"
            path.write_bytes(HEADER)
            validate_sweep_args(args_for(path))

    def test_rejects_legacy_data_before_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.cbin"
            path.write_bytes(b"CHSCBIN1" + bytes(HEADER_SIZE - 8))
            with self.assertRaisesRegex(ValueError, "no ply/result metadata"):
                validate_sweep_args(args_for(path))

    def test_rejects_invalid_parallelism(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.cbin"
            path.write_bytes(HEADER)
            with self.assertRaisesRegex(ValueError, "jobs must be positive"):
                validate_sweep_args(args_for(path, jobs=0))

    def test_rejects_bad_stage_and_grid_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.cbin"
            path.write_bytes(HEADER)
            with self.assertRaisesRegex(ValueError, "stage1-train-samples.*positive"):
                validate_sweep_args(args_for(path, stage1_train_samples=0))
            with self.assertRaisesRegex(ValueError, "hidden-scales.*positive"):
                validate_sweep_args(args_for(path, hidden_scales=[64, 0]))
            with self.assertRaisesRegex(ValueError, "unknown architectures"):
                validate_sweep_args(args_for(path, architectures=["E2", "TYPO"]))
            with self.assertRaisesRegex(ValueError, "top-k.*positive"):
                validate_sweep_args(args_for(path, top_k=0))
            with self.assertRaisesRegex(ValueError, "lr-drop-factor"):
                validate_sweep_args(args_for(path, lr_drop_factor=0.0))
            with self.assertRaisesRegex(ValueError, "calibration-percentile"):
                validate_sweep_args(args_for(path, calibration_percentile=101.0))
            with self.assertRaisesRegex(ValueError, "fixed-hidden-scale"):
                validate_sweep_args(args_for(path, fixed_hidden_scale=0))
            with self.assertRaisesRegex(ValueError, "fixed-hidden-scales"):
                validate_sweep_args(args_for(path, fixed_hidden_scales=[64, 0]))
            with self.assertRaisesRegex(ValueError, "only one"):
                validate_sweep_args(
                    args_for(path, fixed_hidden_scale=64, fixed_hidden_scales=[64])
                )
            with self.assertRaisesRegex(ValueError, "patience must be greater"):
                validate_sweep_args(args_for(path, patience=2, lr_drop_patience=2))
            validate_sweep_args(args_for(path, stage1_warmup_epochs=0))


if __name__ == "__main__":
    unittest.main()
