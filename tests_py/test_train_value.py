import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class TrainValueTests(unittest.TestCase):
    def test_train_one_epoch_and_save_checkpoint(self):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader

        from nn.train_value import (
            DEFAULT_TARGET_CLIP,
            DEFAULT_TARGET_SCALE,
            SparseValueDataset,
            collate_sparse_value_batch,
            denormalize_target_cp,
            normalize_target_cp,
            save_checkpoint,
            train_one_epoch,
        )
        from nn.value_net import AUX_FEATURE_COUNT, ChessValueNet

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "train.jsonl"
            checkpoint_path = Path(tmp) / "value_net.pt"

            samples = [
                {"features": [0, 1, 2], "aux": [0] * AUX_FEATURE_COUNT, "target": 0.0},
                {"features": [10, 11, 12], "aux": [1] + [0] * (AUX_FEATURE_COUNT - 1), "target": 25.0},
                {"features": [20, 21, 22], "aux": [0, 1] + [0] * (AUX_FEATURE_COUNT - 2), "target": -15.0},
            ]
            with dataset_path.open("w", encoding="utf-8") as file:
                for sample in samples:
                    file.write(json.dumps(sample) + "\n")

            dataset = SparseValueDataset(dataset_path)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda batch: batch)

            model = ChessValueNet()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            loss = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))
            self.assertGreaterEqual(loss, 0.0)

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                1,
                loss,
                None,
                DEFAULT_TARGET_SCALE,
                DEFAULT_TARGET_CLIP,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["hidden_size"], 256)
            self.assertEqual(checkpoint["target_scale"], DEFAULT_TARGET_SCALE)
            self.assertEqual(checkpoint["target_clip"], DEFAULT_TARGET_CLIP)
            self.assertIn("model_state", checkpoint)
            self.assertIn("optimizer_state", checkpoint)

            self.assertEqual(normalize_target_cp(2500.0, 1000.0, 1000.0), 1.0)
            self.assertEqual(normalize_target_cp(-1500.0, 1000.0, 1000.0), -1.0)
            self.assertEqual(normalize_target_cp(250.0, 1000.0, 1000.0), 0.25)
            self.assertEqual(denormalize_target_cp(0.25, 1000.0), 250.0)

            _, _, _, targets = collate_sparse_value_batch(
                [
                    {"features": [1], "aux": [0] * AUX_FEATURE_COUNT, "target": 2500.0},
                    {"features": [2], "aux": [0] * AUX_FEATURE_COUNT, "target": -1500.0},
                ],
                device=torch.device("cpu"),
                target_scale=1000.0,
                target_clip=1000.0,
            )
            self.assertEqual(targets.tolist(), [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
