from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from nn.compact_board_data import HEADER, pack_record, position_split_bucket
from nn.training_targets import TRAINING_PIPELINE_VERSION


def make_board(friendly_pawn: int, enemy_pawn: int) -> bytes:
    nibbles = [0] * 64
    nibbles[4] = 6
    nibbles[60] = 12
    nibbles[friendly_pawn] = 1
    nibbles[enemy_pawn] = 7
    packed = bytearray(32)
    for square in range(0, 64, 2):
        packed[square // 2] = nibbles[square] | (nibbles[square + 1] << 4)
    return bytes(packed)


def make_split_records(count_per_split: int) -> dict[str, list[bytes]]:
    records = {"train": [], "val": [], "test": []}
    candidate = 0
    while min(len(values) for values in records.values()) < count_per_split:
        friendly = 8 + candidate % 44
        enemy = 8 + (candidate // 44) % 44
        candidate += 1
        if friendly == enemy or friendly in (4, 60) or enemy in (4, 60):
            continue
        aux = (candidate // (44 * 44)) % (1 << 13)
        board = make_board(friendly, enemy)
        bucket = position_split_bucket(board, aux, 100)
        split = "val" if bucket == 98 else "test" if bucket == 99 else "train"
        if len(records[split]) >= count_per_split:
            continue
        score = (candidate % 1200) - 600
        result = -1 if candidate % 3 == 0 else 0 if candidate % 3 == 1 else 1
        records[split].append(pack_record(board, aux, score, candidate % 300, result))
    return records


class NnueTrainingPipelineIntegrationTests(unittest.TestCase):
    def test_quantized_training_uses_v2_splits_and_tests_best_checkpoint_once(self) -> None:
        records = make_split_records(40)
        interleaved = [
            record
            for index in range(40)
            for record in (records["train"][index], records["val"][index], records["test"][index])
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tiny.cbin"
            data.write_bytes(HEADER + b"".join(interleaved))
            for architecture in ("E2", "F2"):
                with self.subTest(architecture=architecture):
                    output = root / f"model_{architecture}"
                    command = [
                        sys.executable,
                        "tools/train_quantized_nnue_architecture.py",
                        "--arch", architecture,
                        "--data", str(data),
                        "--data-format", "cbin",
                        "--output-dir", str(output),
                        "--device", "cpu",
                        "--epochs", "2",
                        "--patience", "2",
                        "--batch-size", "8",
                        "--workers", "2",
                        "--eval-workers", "2",
                        "--train-max-samples", "24",
                        "--val-max-samples", "12",
                        "--test-max-samples", "12",
                        "--shuffle-block-size", "1",
                        "--calibration-max-batches", "1",
                        "--fixed-hidden-scale", "64",
                        "--fixed-output-scale", "16",
                        "--progress-batches", "0",
                    ]
                    process = subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parents[1],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    self.assertEqual(process.returncode, 0, msg=process.stdout + process.stderr)
                    events = [
                        json.loads(line)
                        for line in process.stdout.splitlines()
                        if line.startswith("{")
                    ]
                    epochs = [event for event in events if event.get("event") == "epoch"]
                    final_tests = [event for event in events if event.get("event") == "final_test"]
                    self.assertEqual(len(epochs), 2)
                    self.assertEqual(len(final_tests), 1)
                    self.assertTrue(all(event["train_samples"] == 24 for event in epochs))
                    self.assertTrue(all(event["val_samples"] == 12 for event in epochs))
                    self.assertTrue(all("test_val_cp" not in event for event in epochs))
                    self.assertEqual(final_tests[0]["test_samples"], 12)

                    checkpoint_path = output / f"quant_nnue_arch_{architecture}_best.pt"
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    self.assertEqual(
                        checkpoint["provenance"]["pipeline_version"],
                        TRAINING_PIPELINE_VERSION,
                    )
                    self.assertEqual(
                        checkpoint["target_encoding"],
                        "stockfish_raw_score_ply_result",
                    )
                    self.assertEqual(
                        checkpoint["provenance"]["scale_calibration_split"],
                        "train",
                    )
                    self.assertEqual(final_tests[0]["selected_epoch"], checkpoint["epoch"])

    def test_vector_hidden_scales_and_sealed_test_evaluator(self) -> None:
        records = make_split_records(24)
        interleaved = [
            record
            for index in range(24)
            for record in (records["train"][index], records["val"][index], records["test"][index])
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tiny.cbin"
            data.write_bytes(HEADER + b"".join(interleaved))
            output = root / "model_F2"
            train_command = [
                sys.executable,
                "tools/train_quantized_nnue_architecture.py",
                "--arch", "F2",
                "--data", str(data),
                "--data-format", "cbin",
                "--output-dir", str(output),
                "--device", "cpu",
                "--epochs", "1",
                "--patience", "2",
                "--batch-size", "8",
                "--workers", "0",
                "--eval-workers", "0",
                "--train-max-samples", "16",
                "--val-max-samples", "8",
                "--test-max-samples", "8",
                "--calibration-max-batches", "1",
                "--fixed-hidden-scales", "64", "128",
                "--fixed-output-scale", "16",
                "--progress-batches", "0",
                "--skip-final-test",
            ]
            process = subprocess.run(
                train_command,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(process.returncode, 0, msg=process.stdout + process.stderr)
            events = [
                json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")
            ]
            self.assertEqual(sum(event.get("event") == "training_complete" for event in events), 1)
            self.assertEqual(sum(event.get("event") == "final_test" for event in events), 0)
            checkpoint_path = output / "quant_nnue_arch_F2_best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["hidden_scales"], [64, 128])

            evaluate_command = [
                sys.executable,
                "tools/evaluate_quantized_nnue_checkpoint.py",
                "--checkpoint", str(checkpoint_path),
                "--data", str(data),
                "--data-format", "cbin",
                "--device", "cpu",
                "--batch-size", "8",
                "--workers", "0",
                "--test-max-samples", "8",
            ]
            evaluation = subprocess.run(
                evaluate_command,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                evaluation.returncode,
                0,
                msg=evaluation.stdout + evaluation.stderr,
            )
            final_tests = [
                json.loads(line)
                for line in evaluation.stdout.splitlines()
                if line.startswith("{") and json.loads(line).get("event") == "final_test"
            ]
            self.assertEqual(len(final_tests), 1)
            self.assertEqual(final_tests[0]["test_samples"], 8)
            self.assertEqual(final_tests[0]["hidden_scales"], [64, 128])


if __name__ == "__main__":
    unittest.main()
