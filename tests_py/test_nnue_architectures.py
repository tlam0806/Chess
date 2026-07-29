import json
import unittest
from pathlib import Path

import torch

from nn.compact_board_data import (
    architecture_features_from_board,
    pack_aux,
    pack_board_from_raw_features,
    pack_record,
    raw_features_from_board,
    unpack_aux,
    unpack_record,
)
from nn.nnue_architectures import (
    ARCHITECTURES,
    BASE_FEATURE_COUNT,
    DUAL_ACCUMULATOR_FEATURE_COUNT,
    DUAL_FULL_KING_FEATURE_COUNT,
    FULL_KING_FEATURE_COUNT,
    KING_BUCKET_COUNT,
    KING_BUCKET_FEATURE_COUNT,
    SparseNnueArchitecture,
    decode_feature,
    dual_accumulator_feature,
    flip_relative_square,
    king_bucket,
    transform_features,
)


PIECES = 6
SIDES = 2
KING_CONTEXTS = 2
SQUARES = 64
RAW_FEATURE_COUNT = PIECES * SIDES * KING_CONTEXTS * SQUARES * SQUARES


def raw_feature(piece, side, king_context, king_square, piece_square):
    index = piece
    index = index * SIDES + side
    index = index * KING_CONTEXTS + king_context
    index = index * SQUARES + king_square
    index = index * SQUARES + piece_square
    return index


def expected_base_feature(piece, side, piece_square):
    return ((piece * SIDES + side) * SQUARES) + piece_square


def expected_king_bucket_feature(piece, side, king_context, king_square, piece_square):
    index = piece
    index = index * SIDES + side
    index = index * KING_CONTEXTS + king_context
    index = index * KING_BUCKET_COUNT + king_bucket(king_square)
    index = index * SQUARES + piece_square
    return index


def expected_full_king_square_feature(piece, side, king_context, king_square, piece_square):
    index = piece
    index = index * SIDES + side
    index = index * KING_CONTEXTS + king_context
    index = index * SQUARES + king_square
    index = index * SQUARES + piece_square
    return index


def expected_dual_feature(piece, side, king_context, king_square, piece_square):
    if king_context == 0:
        return dual_accumulator_feature(piece, side, king_square, piece_square)
    return DUAL_ACCUMULATOR_FEATURE_COUNT + dual_accumulator_feature(
        piece,
        1 - side,
        flip_relative_square(king_square),
        flip_relative_square(piece_square),
    )


class NnueArchitectureEncodingTests(unittest.TestCase):
    def test_architecture_shapes_are_as_requested(self):
        self.assertEqual(ARCHITECTURES["A"].feature_count, 768)
        self.assertEqual(ARCHITECTURES["A"].hidden1_size, 128)
        self.assertIsNone(ARCHITECTURES["A"].hidden2_size)

        self.assertEqual(ARCHITECTURES["B"].feature_count, 768)
        self.assertEqual(ARCHITECTURES["B"].hidden1_size, 256)
        self.assertIsNone(ARCHITECTURES["B"].hidden2_size)

        self.assertEqual(ARCHITECTURES["C"].feature_count, 768)
        self.assertEqual(ARCHITECTURES["C"].hidden1_size, 256)
        self.assertEqual(ARCHITECTURES["C"].hidden2_size, 32)

        self.assertEqual(ARCHITECTURES["D"].feature_count, KING_BUCKET_FEATURE_COUNT)
        self.assertEqual(ARCHITECTURES["D"].hidden1_size, 256)
        self.assertEqual(ARCHITECTURES["D"].hidden2_size, 32)

        self.assertEqual(ARCHITECTURES["E"].feature_count, FULL_KING_FEATURE_COUNT)
        self.assertEqual(ARCHITECTURES["E"].hidden1_size, 256)
        self.assertEqual(ARCHITECTURES["E"].hidden2_size, 32)
        self.assertEqual(ARCHITECTURES["E"].hidden_sizes, (256, 32))

        self.assertEqual(ARCHITECTURES["F"].feature_count, FULL_KING_FEATURE_COUNT)
        self.assertEqual(ARCHITECTURES["F"].hidden_sizes, (256, 32, 32))

        self.assertEqual(ARCHITECTURES["G"].feature_count, FULL_KING_FEATURE_COUNT)
        self.assertEqual(ARCHITECTURES["G"].hidden_sizes, (128, 32, 32))

        self.assertEqual(ARCHITECTURES["H"].feature_count, FULL_KING_FEATURE_COUNT)
        self.assertEqual(ARCHITECTURES["H"].hidden_sizes, (128, 32))

        self.assertEqual(ARCHITECTURES["E2"].feature_count, DUAL_FULL_KING_FEATURE_COUNT)
        self.assertEqual(ARCHITECTURES["E2"].hidden_sizes, (256, 32))

        self.assertEqual(ARCHITECTURES["F2"].feature_count, DUAL_FULL_KING_FEATURE_COUNT)
        self.assertEqual(ARCHITECTURES["F2"].hidden_sizes, (256, 32, 32))

    def test_decode_feature_roundtrip_for_entire_raw_space(self):
        for piece in range(PIECES):
            for side in range(SIDES):
                for king_context in range(KING_CONTEXTS):
                    for king_square in range(SQUARES):
                        for piece_square in range(SQUARES):
                            raw = raw_feature(piece, side, king_context, king_square, piece_square)
                            self.assertLess(raw, RAW_FEATURE_COUNT)
                            self.assertEqual(
                                decode_feature(raw),
                                (piece, side, king_context, king_square, piece_square),
                            )

    def test_transform_formulas_for_entire_raw_space(self):
        for raw in range(RAW_FEATURE_COUNT):
            piece, side, king_context, king_square, piece_square = decode_feature(raw)

            base = transform_features([raw], "base768")
            self.assertEqual(base, [expected_base_feature(piece, side, piece_square)])
            self.assertGreaterEqual(base[0], 0)
            self.assertLess(base[0], BASE_FEATURE_COUNT)

            bucket = transform_features([raw], "king_bucket")
            self.assertEqual(
                bucket,
                [expected_king_bucket_feature(piece, side, king_context, king_square, piece_square)],
            )
            self.assertGreaterEqual(bucket[0], 0)
            self.assertLess(bucket[0], KING_BUCKET_FEATURE_COUNT)

            full_king = transform_features([raw], "full_king_square")
            self.assertEqual(
                full_king,
                [expected_full_king_square_feature(piece, side, king_context, king_square, piece_square)],
            )
            self.assertEqual(full_king, [raw])
            self.assertGreaterEqual(full_king[0], 0)
            self.assertLess(full_king[0], FULL_KING_FEATURE_COUNT)

            dual = transform_features([raw], "dual_full_king_square_concat")
            self.assertEqual(
                dual,
                [expected_dual_feature(piece, side, king_context, king_square, piece_square)],
            )
            self.assertGreaterEqual(dual[0], 0)
            self.assertLess(dual[0], DUAL_FULL_KING_FEATURE_COUNT)

    def test_base768_collapses_only_king_context(self):
        raw_features = [
            raw_feature(piece, side, king_context, king_square, piece_square)
            for piece in range(PIECES)
            for side in range(SIDES)
            for king_context in range(KING_CONTEXTS)
            for king_square in [0, 7, 56, 63]
            for piece_square in [0, 9, 27, 63]
        ]
        transformed = transform_features(raw_features, "base768")
        self.assertEqual(transformed, sorted(set(transformed)))
        expected = {
            expected_base_feature(piece, side, piece_square)
            for piece in range(PIECES)
            for side in range(SIDES)
            for piece_square in [0, 9, 27, 63]
        }
        self.assertEqual(transformed, sorted(expected))

    def test_real_robotmoon_samples_have_expected_counts_and_ranges(self):
        path = Path("data/robotmoon_test80_2024_5m.jsonl")
        if not path.exists():
            self.skipTest(f"{path} is not available")

        checked = 0
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                raw_features = sample["features"]
                if not raw_features:
                    continue

                # encode_position emits two raw features per physical piece:
                # one friendly-king context and one enemy-king context.
                self.assertEqual(len(raw_features) % 2, 0)

                base = transform_features(raw_features, "base768")
                bucket = transform_features(raw_features, "king_bucket")
                full_king = transform_features(raw_features, "full_king_square")
                dual = transform_features(raw_features, "dual_full_king_square_concat")

                self.assertEqual(len(base) * 2, len(raw_features))
                self.assertEqual(len(bucket), len(raw_features))
                self.assertEqual(len(full_king), len(raw_features))
                self.assertEqual(len(dual), len(raw_features))
                self.assertEqual(base, sorted(set(base)))
                self.assertEqual(len(bucket), len(set(bucket)))
                self.assertEqual(len(full_king), len(set(full_king)))
                self.assertEqual(len(dual), len(set(dual)))

                self.assertTrue(all(0 <= feature < BASE_FEATURE_COUNT for feature in base))
                self.assertTrue(all(0 <= feature < KING_BUCKET_FEATURE_COUNT for feature in bucket))
                self.assertTrue(all(0 <= feature < FULL_KING_FEATURE_COUNT for feature in full_king))
                self.assertTrue(all(0 <= feature < DUAL_FULL_KING_FEATURE_COUNT for feature in dual))
                self.assertEqual(full_king, raw_features)

                checked += 1
                if checked >= 1000:
                    break

        self.assertEqual(checked, 1000)

    def test_compact_board_roundtrip_preserves_robotmoon_features(self):
        raw_features = [
            raw_feature(5, 0, 0, 4, 4),
            raw_feature(5, 0, 1, 60, 4),
            raw_feature(5, 1, 0, 4, 60),
            raw_feature(5, 1, 1, 60, 60),
            raw_feature(0, 0, 0, 4, 12),
            raw_feature(0, 0, 1, 60, 12),
            raw_feature(1, 1, 0, 4, 42),
            raw_feature(1, 1, 1, 60, 42),
        ]
        raw_features.sort()
        aux = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        board = pack_board_from_raw_features(raw_features)
        aux_bits = pack_aux(aux)
        record = pack_record(board, aux_bits, -123, 57, -1)
        sample = unpack_record(record)

        self.assertEqual(sample.score, -123)
        self.assertEqual(sample.ply, 57)
        self.assertEqual(sample.result, -1)
        self.assertEqual(unpack_aux(sample.aux_bits), aux)
        self.assertEqual(raw_features_from_board(sample.board), raw_features)
        self.assertEqual(
            architecture_features_from_board(sample.board, "full_king_square"),
            transform_features(raw_features, "full_king_square"),
        )
        self.assertEqual(
            architecture_features_from_board(sample.board, "king_bucket"),
            transform_features(raw_features, "king_bucket"),
        )
        self.assertEqual(
            architecture_features_from_board(sample.board, "dual_full_king_square_concat"),
            transform_features(raw_features, "dual_full_king_square_concat"),
        )

    def test_sparse_architecture_forward_matches_brute_reference(self):
        torch.manual_seed(12345)
        for arch_name in sorted(ARCHITECTURES):
            config = ARCHITECTURES[arch_name]
            model = SparseNnueArchitecture(config)
            with torch.no_grad():
                model.feature_weights.weight.uniform_(-0.2, 0.2)
                model.aux_projection.weight.uniform_(-0.2, 0.2)
                model.hidden1_bias.uniform_(-0.2, 0.2)
                for layer in model.hidden_layers:
                    layer.weight.uniform_(-0.2, 0.2)
                    layer.bias.uniform_(-0.2, 0.2)
                model.output.weight.uniform_(-0.2, 0.2)
                model.output.bias.uniform_(-0.2, 0.2)

            batch_features = [
                [0, 5, min(config.feature_count - 1, 17)],
                [min(config.feature_count - 1, 3), min(config.feature_count - 1, 71)],
                [min(config.feature_count - 1, 11)],
            ]
            feature_indices = torch.tensor(
                [feature for features in batch_features for feature in features],
                dtype=torch.long,
            )
            offsets = torch.tensor([0, 3, 5], dtype=torch.long)
            aux = torch.tensor(
                [
                    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                ],
                dtype=torch.float32,
            )

            actual = model(feature_indices, offsets, aux)
            expected_rows = []
            for row, features in enumerate(batch_features):
                if model.dual_accumulator:
                    half_features = config.feature_count // 2
                    first = torch.zeros(config.hidden1_size // 2)
                    second = torch.zeros(config.hidden1_size // 2)
                    for feature in features:
                        if feature < half_features:
                            first += model.feature_weights.weight[feature]
                        else:
                            second += model.feature_weights.weight[feature - half_features]
                    hidden = torch.cat((first, second))
                    hidden = hidden + torch.cat((model.hidden1_bias, model.hidden1_bias))
                else:
                    hidden = model.feature_weights.weight[features].sum(dim=0)
                    hidden = hidden + model.hidden1_bias
                hidden = hidden + aux[row] @ model.aux_projection.weight.T
                hidden = torch.clamp(hidden, min=0.0, max=1.0)
                for layer in model.hidden_layers:
                    hidden = hidden @ layer.weight.T + layer.bias
                    hidden = torch.clamp(hidden, min=0.0, max=1.0)
                expected_rows.append((hidden @ model.output.weight[0]) + model.output.bias[0])
            expected = torch.stack(expected_rows)
            self.assertTrue(
                torch.allclose(actual, expected, atol=1e-6),
                msg=f"forward mismatch for architecture {arch_name}",
            )

    def test_dual_accumulators_share_the_feature_transformer(self):
        for arch_name in ("E2", "F2"):
            model = SparseNnueArchitecture(ARCHITECTURES[arch_name])
            self.assertTrue(model.dual_accumulator)
            self.assertEqual(
                tuple(model.feature_weights.weight.shape),
                (DUAL_ACCUMULATOR_FEATURE_COUNT, 128),
            )

            with torch.no_grad():
                model.feature_weights.weight.zero_()
                model.feature_weights.weight[7].fill_(0.25)
                model.hidden1_bias.zero_()
                model.aux_projection.weight.zero_()
                for layer in model.hidden_layers:
                    layer.weight.zero_()
                    layer.bias.zero_()
                model.output.weight.zero_()
                model.output.bias.zero_()

            aux = torch.zeros((1, 13))
            first = model.first_hidden_accumulator(
                torch.tensor([7]), torch.tensor([0]), aux
            )[0]
            second = model.first_hidden_accumulator(
                torch.tensor([DUAL_ACCUMULATOR_FEATURE_COUNT + 7]),
                torch.tensor([0]),
                aux,
            )[0]
            self.assertTrue(torch.equal(first[:128], second[128:]))
            self.assertEqual(int(torch.count_nonzero(first[128:])), 0)
            self.assertEqual(int(torch.count_nonzero(second[:128])), 0)


if __name__ == "__main__":
    unittest.main()
