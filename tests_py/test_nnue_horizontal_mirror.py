from __future__ import annotations

import unittest

import torch

from nn.compact_board_data import (
    architecture_features_from_board,
    canonical_architecture_input,
    horizontal_mirror_aux,
    horizontal_mirror_board,
    pack_aux,
    pack_board_from_raw_features,
)
from nn.horizontal_mirror_checkpoint import (
    F2M_FEATURE_ROWS,
    F2_FEATURE_ROWS,
    convert_f2_checkpoint,
)
from nn.nnue_architectures import ARCHITECTURES


def raw_feature(
    piece: int,
    side: int,
    king_context: int,
    king_square: int,
    piece_square: int,
) -> int:
    index = piece
    index = index * 2 + side
    index = index * 2 + king_context
    index = index * 64 + king_square
    return index * 64 + piece_square


def accumulator(
    features: list[int],
    aux: list[int],
    feature_weights: torch.Tensor,
    aux_weights: torch.Tensor,
    board_feature_count: int,
) -> torch.Tensor:
    half = board_feature_count // 2
    lane_count = int(feature_weights.shape[1])
    first = torch.zeros(lane_count)
    second = torch.zeros(lane_count)
    for feature in features:
        if feature < half:
            first += feature_weights[feature]
        else:
            second += feature_weights[feature - half]
    return torch.cat((first, second)) + torch.tensor(
        aux, dtype=aux_weights.dtype
    ) @ aux_weights


class HorizontalMirrorCheckpointTests(unittest.TestCase):
    def test_symmetric_f2_and_folded_f2m_have_identical_accumulators(self) -> None:
        feature_values = torch.arange(F2_FEATURE_ROWS * 2, dtype=torch.float32)
        feature_weights = (feature_values.reshape(F2_FEATURE_ROWS, 2) % 251) / 251
        psqt_values = torch.arange(F2_FEATURE_ROWS, dtype=torch.float32).unsqueeze(1)
        aux_weights = torch.arange(13 * 4, dtype=torch.float32).reshape(13, 4) / 17
        checkpoint = {
            "architecture": "F2",
            "optimizer_state": {"discarded": True},
            "model_state": {
                "feature_weights.weight": feature_weights,
                "psqt.weight": psqt_values,
                "aux_feature_weights": aux_weights,
                "dense_sentinel": torch.tensor([123.0]),
            },
        }
        symmetric, mirror = convert_f2_checkpoint(checkpoint, "average")
        self.assertNotIn("optimizer_state", symmetric)
        self.assertNotIn("optimizer_state", mirror)
        self.assertEqual(mirror["architecture"], "F2M")
        self.assertEqual(
            tuple(mirror["model_state"]["feature_weights.weight"].shape),
            (F2M_FEATURE_ROWS, 2),
        )
        self.assertTrue(
            torch.equal(
                symmetric["model_state"]["dense_sentinel"],
                mirror["model_state"]["dense_sentinel"],
            )
        )

        raw_features = [
            raw_feature(5, 0, 0, 6, 6),
            raw_feature(5, 0, 1, 57, 6),
            raw_feature(5, 1, 0, 6, 57),
            raw_feature(5, 1, 1, 57, 57),
            raw_feature(0, 0, 0, 6, 14),
            raw_feature(0, 0, 1, 57, 14),
            raw_feature(1, 1, 0, 6, 42),
            raw_feature(1, 1, 1, 57, 42),
        ]
        board = pack_board_from_raw_features(raw_features)
        aux = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        cases = (
            (board, aux),
            (horizontal_mirror_board(board), horizontal_mirror_aux(aux)),
        )
        for case_board, case_aux in cases:
            legacy_features = architecture_features_from_board(
                case_board, ARCHITECTURES["F2"].transform
            )
            expected = accumulator(
                legacy_features,
                case_aux,
                symmetric["model_state"]["feature_weights.weight"],
                symmetric["model_state"]["aux_feature_weights"],
                ARCHITECTURES["F2"].feature_count,
            )
            _board, _aux_bits, mirror_features, mirror_aux = (
                canonical_architecture_input(
                    case_board,
                    pack_aux(case_aux),
                    ARCHITECTURES["F2M"].transform,
                )
            )
            actual = accumulator(
                mirror_features,
                mirror_aux,
                mirror["model_state"]["feature_weights.weight"],
                mirror["model_state"]["aux_feature_weights"],
                ARCHITECTURES["F2M"].feature_count,
            )
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
