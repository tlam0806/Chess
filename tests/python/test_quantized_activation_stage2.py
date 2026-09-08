from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chess_nnue.training_targets import TRAINING_PIPELINE_VERSION
from tools.train.run_quantized_activation_stage2 import load_stage1_candidates
from tools.train.run_quantized_scale_grid import (
    build_configs,
    build_selected_configs,
    load_completed_config_keys,
    make_config_name,
    parse_selected_scale_configs,
)


class QuantizedActivationStage2Tests(unittest.TestCase):
    def test_selects_top_validation_configs_per_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            expected = {"relu": 6, "screlu_first": 6, "screlu_all": 6}
            for activation in expected:
                rows = [
                    {
                        "arch": "E2" if index % 2 == 0 else "F2",
                        "activation": activation,
                        "hidden_scales": (
                            [index + 1]
                            if index % 2 == 0
                            else [index + 1, index + 2]
                        ),
                        "output_scale": 16,
                        "code": 0,
                        "pipeline_version": TRAINING_PIPELINE_VERSION,
                        "completed": True,
                        "best_val_loss": float(index) / 100.0,
                        "val_cp_at_best_loss": float(100 + index),
                    }
                    for index in range(6)
                ]
                path = log_dir / f"stage1_{activation}.summary.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in reversed(rows)),
                    encoding="utf-8",
                )

            selected = load_stage1_candidates(log_dir, "stage1", 4, expected)
            self.assertEqual(len(selected), 12)
            for activation in expected:
                activation_candidates = [
                    candidate for candidate in selected if candidate.activation == activation
                ]
                self.assertEqual(
                    [candidate.stage1_val_cp for candidate in activation_candidates],
                    [100.0, 101.0, 102.0, 103.0],
                )

    def test_rejects_incomplete_or_failed_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            path = log_dir / "stage1_relu.summary.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "arch": "E2",
                        "code": 1,
                        "pipeline_version": TRAINING_PIPELINE_VERSION,
                        "completed": True,
                        "best_val_loss": 1.0,
                        "val_cp_at_best_loss": 1.0,
                        "hidden_scales": [64],
                        "output_scale": 16,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_stage1_candidates(log_dir, "stage1", 1, {"relu": 1})

    def test_scale_grid_tunes_each_dense_layer_independently(self) -> None:
        configs = build_configs(
            architectures=["E2", "F2"],
            activations=["relu"],
            hidden_scale_values=[64, 128, 256],
            output_scales=[32, 64],
        )
        e2 = [config for config in configs if config[0] == "E2"]
        f2 = [config for config in configs if config[0] == "F2"]
        self.assertEqual(len(e2), 6)
        self.assertEqual(len(f2), 18)
        self.assertIn(("F2", "relu", (64, 128), 32), f2)
        self.assertIn(("F2", "relu", (128, 64), 32), f2)
        self.assertEqual(
            make_config_name("tag", "F2", "relu", (64, 128), 32),
            "tag_F2_relu_hs64x128_os32",
        )

    def test_scale_grid_can_run_only_selected_tuples(self) -> None:
        selected = parse_selected_scale_configs("32x16:16,16x64:8")
        self.assertEqual(selected, [((32, 16), 16), ((16, 64), 8)])
        self.assertEqual(
            build_selected_configs(["F2"], ["screlu_all"], selected),
            [
                ("F2", "screlu_all", (32, 16), 16),
                ("F2", "screlu_all", (16, 64), 8),
            ],
        )
        with self.assertRaises(ValueError):
            build_selected_configs(["E2"], ["screlu_all"], selected)

    def test_resume_accepts_only_successful_requested_configs(self) -> None:
        configs = build_configs(
            architectures=["E2"],
            activations=["relu"],
            hidden_scale_values=[64, 128],
            output_scales=[16],
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "grid.summary.jsonl"
            summary.write_text(
                json.dumps(
                    {
                        "arch": "E2",
                        "activation": "relu",
                        "hidden_scales": [64],
                        "output_scale": 16,
                        "code": 0,
                        "completed": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_completed_config_keys(summary, configs),
                {("E2", "relu", (64,), 16)},
            )

            summary.write_text(
                json.dumps(
                    {
                        "arch": "E2",
                        "activation": "relu",
                        "hidden_scales": [64],
                        "output_scale": 16,
                        "code": 1,
                        "completed": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_completed_config_keys(summary, configs)


if __name__ == "__main__":
    unittest.main()
