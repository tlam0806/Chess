from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chess_nnue.training_targets import TRAINING_PIPELINE_VERSION
from tools.train.quantized_training_log import parse_quantized_training_log


class QuantizedTrainingLogTests(unittest.TestCase):
    def test_accepts_preflight_rejected_config_without_epoch(self) -> None:
        events = [
            {
                "event": "start",
                "provenance": {"pipeline_version": TRAINING_PIPELINE_VERSION},
            },
            {
                "event": "hidden_saturation",
                "epoch": 0,
                "stats": [{"layer": 3, "zero_rate": 1.0, "clip_rate": 0.0}],
            },
            {
                "event": "config_rejected",
                "reasons": ["layer_3_zero_rate=1.000000"],
            },
            {
                "event": "training_complete",
                "selected_epoch": 0,
                "test_evaluated": False,
                "rejected": True,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejected.log"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            result = parse_quantized_training_log(path)

        self.assertTrue(result["completed"])
        self.assertTrue(result["rejected"])
        self.assertFalse(result["activation_health"]["healthy"])
        self.assertEqual(result["activation_health"]["dead_layers"], [3])

    def test_accepts_completed_training_without_evaluating_test(self) -> None:
        events = [
            {
                "event": "start",
                "provenance": {"pipeline_version": TRAINING_PIPELINE_VERSION},
            },
            {
                "event": "epoch", "epoch": 1, "val_loss": 0.1, "val_cp": 20.0,
                "best_epoch": 1, "best_val_loss": 0.1, "lr": 0.001,
            },
            {
                "event": "training_complete",
                "selected_epoch": 1,
                "test_evaluated": False,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            result = parse_quantized_training_log(path)

        self.assertTrue(result["completed"])
        self.assertFalse(result["test_evaluated"])
        self.assertNotIn("final_test_cp", result)

    def test_selects_by_validation_loss_and_reads_test_only_from_final_event(self) -> None:
        events = [
            {
                "event": "start",
                "provenance": {"pipeline_version": TRAINING_PIPELINE_VERSION},
            },
            {"event": "hidden_saturation", "epoch": 1, "stats": [{"clip_rate": 0.1}]},
            {
                "event": "epoch", "epoch": 1, "val_loss": 0.2, "val_cp": 10.0,
                "best_epoch": 1, "best_val_loss": 0.2, "lr": 0.001,
            },
            {"event": "hidden_saturation", "epoch": 2, "stats": [{"clip_rate": 0.2}]},
            {
                "event": "epoch", "epoch": 2, "val_loss": 0.1, "val_cp": 50.0,
                "best_epoch": 2, "best_val_loss": 0.1, "lr": 0.001,
            },
            {
                "event": "final_test",
                "selected_epoch": 2,
                "test_loss": 0.3,
                "test_val_cp": 25.0,
                "test_samples": 100,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            result = parse_quantized_training_log(path)

        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_val_loss"], 0.1)
        self.assertEqual(result["val_cp_at_best_loss"], 50.0)
        self.assertEqual(result["final_test_cp"], 25.0)
        self.assertEqual(result["best_saturation"]["stats"][0]["clip_rate"], 0.2)

    def test_rejects_legacy_log_and_mismatched_final_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.log"
            path.write_text(json.dumps({"event": "epoch", "epoch": 1, "val_loss": 0.1}) + "\n")
            self.assertIn("error", parse_quantized_training_log(path))

            events = [
                {
                    "event": "start",
                    "provenance": {"pipeline_version": TRAINING_PIPELINE_VERSION},
                },
                {
                    "event": "epoch", "epoch": 1, "val_loss": 0.1,
                    "best_epoch": 1, "best_val_loss": 0.1,
                },
                {"event": "final_test", "selected_epoch": 2},
            ]
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            self.assertIn("error", parse_quantized_training_log(path))

    def test_rejects_non_finite_epoch_or_final_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.log"
            events = [
                {
                    "event": "start",
                    "provenance": {"pipeline_version": TRAINING_PIPELINE_VERSION},
                },
                {
                    "event": "epoch",
                    "epoch": 1,
                    "val_loss": float("nan"),
                    "val_cp": 10.0,
                    "best_epoch": 1,
                    "best_val_loss": float("nan"),
                },
                {
                    "event": "final_test",
                    "selected_epoch": 1,
                    "test_loss": 0.1,
                    "test_val_cp": 20.0,
                    "test_samples": 10,
                },
            ]
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            self.assertIn("error", parse_quantized_training_log(path))

            events[1]["val_loss"] = 0.1
            events[1]["best_val_loss"] = 0.1
            events[2]["test_val_cp"] = float("inf")
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            self.assertRegex(
                str(parse_quantized_training_log(path).get("error")),
                "final_test.*non-finite",
            )

            events[2]["test_val_cp"] = 20.0
            events[2]["test_samples"] = 0
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            self.assertRegex(
                str(parse_quantized_training_log(path).get("error")),
                "no samples",
            )

    def test_rejects_skipped_test_log_that_also_contains_final_test(self) -> None:
        events = [
            {
                "event": "start",
                "provenance": {"pipeline_version": TRAINING_PIPELINE_VERSION},
            },
            {
                "event": "epoch", "epoch": 1, "val_loss": 0.1, "val_cp": 20.0,
                "best_epoch": 1, "best_val_loss": 0.1,
            },
            {"event": "training_complete", "selected_epoch": 1, "test_evaluated": False},
            {
                "event": "final_test", "selected_epoch": 1, "test_loss": 0.1,
                "test_val_cp": 20.0, "test_samples": 10,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.log"
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            self.assertRegex(
                str(parse_quantized_training_log(path).get("error")),
                "both skipped-test completion and final_test",
            )


if __name__ == "__main__":
    unittest.main()
