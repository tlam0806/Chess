from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nn.compact_board_data import (
    HEADER,
    HEADER_SIZE,
    RECORD_SIZE,
    pack_aux,
    pack_board_from_raw_features,
    pack_record,
    position_split_bucket,
    validate_compact_dataset,
    validate_header,
)
from nn.training_targets import (
    STOCKFISH_VALUE_NONE,
    expected_score_from_cp,
    stockfish_score_to_cp,
)
from tools.train_nnue_architecture import (
    CompactSplitDataset,
    JsonlSplitDataset,
    cp_huber_loss,
    make_loader,
    validate_training_data,
    worker_sample_limit,
    wdl_loss,
)


class NnueDataLoaderTests(unittest.TestCase):
    def make_dataset(self, path: Path, seed: int = 0) -> CompactSplitDataset:
        return CompactSplitDataset(
            path=path,
            architecture="base768",
            split="train",
            split_mod=100,
            val_mod=98,
            test_mod=99,
            max_samples=None,
            seed=seed,
            shuffle_block_size=1,
        )

    def test_shards_are_partitioned_without_worker_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"shard_{index}.cbin" for index in range(12)]
            worker_records: list[set[int]] = []
            worker_paths: list[set[Path]] = []
            for worker_id in range(4):
                dataset = self.make_dataset(Path(directory))
                dataset._compact_paths = lambda paths=paths: paths  # type: ignore[method-assign]
                opened: set[Path] = set()
                dataset._iter_stream_blocks = (  # type: ignore[method-assign]
                    lambda path, base_index, _block_size, opened=opened: (
                        opened.add(path) or [(base_index, bytes(RECORD_SIZE))]
                    )
                )
                worker = SimpleNamespace(id=worker_id, num_workers=4)
                with patch("tools.train_nnue_architecture.get_worker_info", return_value=worker):
                    worker_records.append({index for index, _record in dataset._iter_shuffled_compact()})
                    worker_paths.append(opened)

            combined: set[int] = set()
            for records in worker_records:
                self.assertTrue(combined.isdisjoint(records))
                combined.update(records)
            self.assertEqual(combined, set(range(12)))
            combined_paths: set[Path] = set()
            for opened in worker_paths:
                self.assertEqual(len(opened), 3)
                self.assertTrue(combined_paths.isdisjoint(opened))
                combined_paths.update(opened)
            self.assertEqual(combined_paths, set(paths))

    def test_duplicate_positions_always_use_the_same_split(self) -> None:
        board = bytes(range(32))
        aux_bits = 0x123
        expected = position_split_bucket(board, aux_bits, 100)
        self.assertEqual(position_split_bucket(board, aux_bits, 100), expected)
        self.assertEqual(position_split_bucket(bytes(board), aux_bits, 100), expected)

    def test_worker_sample_quotas_sum_to_requested_limit(self) -> None:
        quotas = [worker_sample_limit(10, worker, 4) for worker in range(4)]
        self.assertEqual(quotas, [3, 3, 2, 2])
        self.assertEqual(sum(value for value in quotas if value is not None), 10)

    def test_unshuffled_shards_are_opened_by_only_one_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"shard_{index}.cbin" for index in range(12)]
            worker_paths: list[set[Path]] = []
            sample = SimpleNamespace(score=10, board=bytes(32), aux_bits=0, ply=1, result=0)
            for worker_id in range(4):
                dataset = self.make_dataset(Path(directory))
                dataset.shuffle_block_size = 0
                dataset._compact_paths = lambda paths=paths: paths  # type: ignore[method-assign]
                dataset._bucket_belongs_to_split = lambda _bucket: True  # type: ignore[method-assign]
                opened: set[Path] = set()

                def fake_samples(path: Path, opened: set[Path] = opened):
                    opened.add(path)
                    yield sample

                worker = SimpleNamespace(id=worker_id, num_workers=4)
                with (
                    patch("tools.train_nnue_architecture.get_worker_info", return_value=worker),
                    patch("tools.train_nnue_architecture.iter_compact_samples", fake_samples),
                    patch(
                        "tools.train_nnue_architecture.canonical_architecture_input",
                        return_value=(bytes(32), 0, [1], [0] * 13),
                    ),
                ):
                    list(dataset)
                worker_paths.append(opened)

            combined: set[Path] = set()
            for opened in worker_paths:
                self.assertEqual(len(opened), 3)
                self.assertTrue(combined.isdisjoint(opened))
                combined.update(opened)
            self.assertEqual(combined, set(paths))

    def test_cached_unshuffled_split_scans_shards_only_once(self) -> None:
        paths = [Path(f"shard_{index}.cbin") for index in range(3)]
        sample = SimpleNamespace(
            score=10, board=bytes(32), aux_bits=0, ply=1, result=0
        )
        dataset = CompactSplitDataset(
            path=Path("unused"),
            architecture="base768",
            split="val",
            split_mod=100,
            val_mod=98,
            test_mod=99,
            max_samples=None,
            seed=0,
            shuffle_block_size=0,
            cache_unshuffled=True,
        )
        dataset._compact_paths = lambda: paths  # type: ignore[method-assign]
        dataset._bucket_belongs_to_split = lambda _bucket: True  # type: ignore[method-assign]
        opened: list[Path] = []

        def fake_samples(path: Path):
            opened.append(path)
            yield sample

        with (
            patch("tools.train_nnue_architecture.iter_compact_samples", fake_samples),
            patch(
                "tools.train_nnue_architecture.canonical_architecture_input",
                return_value=(bytes(32), 0, [1], [0] * 13),
            ),
        ):
            self.assertEqual(len(list(dataset)), 3)
            self.assertEqual(len(list(dataset)), 3)

        self.assertEqual(opened, paths)

    def test_multiworker_loader_keeps_workers_across_epochs(self) -> None:
        loader = make_loader(
            Path("unused.cbin"),
            "base768",
            "cbin",
            "train",
            100,
            98,
            99,
            100,
            0,
            16,
            2,
            1,
        )
        self.assertTrue(loader.persistent_workers)

    def test_compact_dataset_skips_value_none_before_sample_limit(self) -> None:
        dataset = self.make_dataset(Path("unused.cbin"))
        records = [
            (0, pack_record(bytes(32), 0, STOCKFISH_VALUE_NONE, 12, 0)),
            (1, pack_record(bytes(32), 0, 25, 13, 1)),
        ]
        dataset._iter_shuffled_compact = lambda: iter(records)  # type: ignore[method-assign]
        dataset._bucket_belongs_to_split = lambda _bucket: True  # type: ignore[method-assign]
        with patch(
            "tools.train_nnue_architecture.canonical_architecture_input",
            return_value=(bytes(32), 0, [7], [0] * 13),
        ):
            samples = list(dataset)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["score"], 25)
        self.assertEqual(samples[0]["ply"], 13)
        self.assertEqual(samples[0]["result"], 1)

    def test_jsonl_dataset_skips_value_none_before_sample_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            rows = [
                {
                    "features": [1], "aux": [0] * 13,
                    "score": STOCKFISH_VALUE_NONE, "ply": 10, "result": 0,
                },
                {
                    "features": [2], "aux": [0] * 13,
                    "score": -15, "ply": 11, "result": -1,
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            board = pack_board_from_raw_features([2])
            bucket = position_split_bucket(board, pack_aux([0] * 13), 3)
            other_buckets = [value for value in range(3) if value != bucket]
            dataset = JsonlSplitDataset(
                path, "base768", "train", 3,
                other_buckets[0], other_buckets[1], 1, 0,
            )

            with patch(
                "tools.train_nnue_architecture.canonical_architecture_input",
                return_value=(board, 0, [7], [0] * 13),
            ):
                samples = list(dataset)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["score"], -15)

    def test_wdl_loss_is_invariant_to_target_normalization_scale(self) -> None:
        loss_scale_100 = wdl_loss(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([20.0]),
            torch.tensor([0.0]),
            target_scale=100.0,
        )
        loss_scale_1000 = wdl_loss(
            torch.tensor([0.1]),
            torch.tensor([0.0]),
            torch.tensor([20.0]),
            torch.tensor([0.0]),
            target_scale=1000.0,
        )
        self.assertAlmostEqual(float(loss_scale_100), float(loss_scale_1000), places=7)

    def test_wdl_loss_prioritizes_equal_cp_error_near_zero(self) -> None:
        near_equal = wdl_loss(
            torch.tensor([0.1]),
            torch.tensor([0.0]),
            torch.tensor([20.0]),
            torch.tensor([0.0]),
            target_scale=1000.0,
        )
        clearly_winning = wdl_loss(
            torch.tensor([1.6]),
            torch.tensor([3120.0]),
            torch.tensor([20.0]),
            torch.tensor([1.0]),
            target_scale=1000.0,
        )
        self.assertGreater(float(near_equal), float(clearly_winning))

    def test_cp_huber_preserves_large_cp_error_signal(self) -> None:
        prediction = torch.tensor([0.4], requires_grad=True)
        loss = cp_huber_loss(
            prediction,
            torch.tensor([4160.0]),  # 2000 CP after conversion/clamp
            target_scale=1000.0,
            delta_cp=200.0,
        )
        loss.backward()
        self.assertAlmostEqual(float(loss), 1.5, places=6)
        self.assertAlmostEqual(float(prediction.grad), -1.0, places=6)

    def test_stockfish_raw_score_converts_to_cp(self) -> None:
        actual = stockfish_score_to_cp(torch.tensor([208.0, -416.0, 50_000.0]))
        self.assertTrue(torch.equal(actual, torch.tensor([100.0, -200.0, 2000.0])))

    def test_wdl_expected_score_matches_vendored_scalar_formula(self) -> None:
        scores = [-2000.0, -137.5, 0.0, 246.25, 2000.0]
        plies = [0.0, 1.0, 64.0, 127.0, 300.0]
        actual = expected_score_from_cp(torch.tensor(scores), torch.tensor(plies))
        expected: list[float] = []
        coefficients_a = (-3.68389304, 30.07065921, -60.52878723, 149.53378557)
        coefficients_b = (-2.0181857, 15.85685038, -29.83452023, 47.59078827)
        for score, ply in zip(scores, plies):
            m = min(240.0, ply) / 64.0
            a = ((coefficients_a[0] * m + coefficients_a[1]) * m + coefficients_a[2]) * m
            a += coefficients_a[3]
            b = ((coefficients_b[0] * m + coefficients_b[1]) * m + coefficients_b[2]) * m
            b = (b + coefficients_b[3]) * 1.5
            win = 1.0 / (1.0 + math.exp((a - score) / b))
            loss = 1.0 / (1.0 + math.exp((a + score) / b))
            expected.append(0.5 * (1.0 + win - loss))
        self.assertTrue(
            torch.allclose(actual, torch.tensor(expected), rtol=1e-6, atol=1e-7)
        )

    def test_legacy_cbin_is_rejected(self) -> None:
        legacy = b"CHSCBIN1" + bytes(HEADER_SIZE - 8)
        with self.assertRaisesRegex(ValueError, "no ply/result metadata"):
            validate_header(legacy)

    def test_dataset_preflight_checks_every_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "part_000.cbin").write_bytes(HEADER)
            (root / "part_001.cbin").write_bytes(
                b"CHSCBIN1" + bytes(HEADER_SIZE - 8)
            )
            with self.assertRaisesRegex(ValueError, "part_001.*no ply/result"):
                validate_compact_dataset(root)

    def test_direct_trainer_preflight_rejects_legacy_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "part_000.cbin").write_bytes(HEADER)
            (root / "part_001.cbin").write_bytes(
                b"CHSCBIN1" + bytes(HEADER_SIZE - 8)
            )
            with self.assertRaisesRegex(ValueError, "part_001.*no ply/result"):
                validate_training_data(root, "auto")

    def test_truncated_compact_record_is_rejected_by_reader(self) -> None:
        from nn.compact_board_data import HEADER, iter_compact_samples

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.cbin"
            path.write_bytes(HEADER + bytes(RECORD_SIZE - 1))
            with self.assertRaisesRegex(ValueError, "truncated compact record"):
                list(iter_compact_samples(path))


if __name__ == "__main__":
    unittest.main()
