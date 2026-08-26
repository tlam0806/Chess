from __future__ import annotations

import unittest

import torch
from torch import nn

from nn.quantized_nnue_architectures import (
    ACTIVATION_CHOICES,
    ACTIVATION_RELU,
    ACTIVATION_SCRELU_FIRST,
    HIDDEN_CLIP,
    PSQT_BUCKET_COUNT,
    PSQT_QUANT_MAX,
    PSQT_QUANT_MIN,
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    QuantizedNnueArchitectureConfig,
    QuantizedSparseNnueArchitecture,
    activation_kind,
    activation_numeric_clip,
    quantization_scale_audit,
    scale_from_positive_percentile,
    transform_features_with_sparse_aux,
)
from tools.train_quantized_nnue_architecture import (
    cp_regression_loss,
    initial_hidden1,
    initialize_scale_clean_screlu_biases,
    sampled_quantile,
    scheduled_learning_rate,
    validate_epoch_learning_rates,
)


def cpp_trunc_div(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


def brute_quantized_forward(
    model: QuantizedSparseNnueArchitecture,
    batch_features: list[list[int]],
    hidden_scales: list[int],
    output_scale: int,
    activation: str = ACTIVATION_RELU,
) -> list[float]:
    feature_weight = model.quantized_feature_weight().to(torch.int64)
    aux_feature_weight = (
        model.quantized_aux_feature_weight().to(torch.int64)
        if model.dual_accumulator
        else None
    )
    raw_hidden1_bias = torch.round(
        model.hidden1_bias * float(model.feature_weight_scale)
    ).to(torch.int64)
    hidden1_bias = (
        torch.cat((raw_hidden1_bias, raw_hidden1_bias), dim=0)
        if model.dual_accumulator
        else raw_hidden1_bias
    )
    layer_weights = [
        torch.clamp(torch.round(layer.weight * float(model.linear_weight_scale)), -127, 127).to(torch.int64)
        for layer in model.hidden_layers
    ]
    layer_biases = [
        torch.round(layer.bias * float(model.hidden_clip * model.linear_weight_scale)).to(torch.int64)
        for layer in model.hidden_layers
    ]
    output_weight = torch.clamp(
        torch.round(model.output.weight.squeeze(0) * float(model.output_weight_scale)),
        -127,
        127,
    ).to(torch.int64)
    output_bias = int(torch.round(model.output.bias.squeeze(0) * float(model.output_weight_scale)).item())

    def activate(value: torch.Tensor, layer_index: int) -> torch.Tensor:
        clipped = torch.clamp(
            value,
            0,
            activation_numeric_clip(activation, layer_index, model.hidden_clip),
        )
        if activation_kind(activation, layer_index) == "screlu8":
            clipped = torch.trunc((clipped * clipped) / model.screlu_divisor)
        return clipped

    results: list[float] = []
    for features in batch_features:
        hidden = hidden1_bias.clone()
        for feature in features:
            if not model.dual_accumulator:
                hidden += feature_weight[feature]
            elif feature < model.board_feature_count // 2:
                hidden[: model.hidden1_size // 2] += feature_weight[feature]
            elif feature < model.board_feature_count:
                hidden[model.hidden1_size // 2 :] += feature_weight[
                    feature - model.board_feature_count // 2
                ]
            else:
                assert aux_feature_weight is not None
                hidden += aux_feature_weight[feature - model.board_feature_count]
        hidden = activate(hidden, 0)

        for layer_index, (scale, weight, bias) in enumerate(zip(hidden_scales, layer_weights, layer_biases), 1):
            next_hidden = []
            for row, bias_value in zip(weight, bias):
                acc = int(bias_value.item())
                for value, coeff in zip(hidden, row):
                    acc += int(value.item()) * int(coeff.item())
                next_hidden.append(cpp_trunc_div(acc, scale))
            hidden = torch.tensor(next_hidden, dtype=torch.int64)
            hidden = activate(hidden, layer_index)

        raw = output_bias
        for value, coeff in zip(hidden, output_weight):
            raw += int(value.item()) * int(coeff.item())
        results.append(float(cpp_trunc_div(raw, output_scale)))
    return results


class QuantizedNnueArchitectureTests(unittest.TestCase):
    def test_psqt_uses_model_feature_rows_and_eight_phase_buckets(self) -> None:
        config = QuantizedNnueArchitectureConfig("TPsqt", "base768", 768, 768, (4,))
        model = QuantizedSparseNnueArchitecture(
            config,
            use_psqt=True,
            psqt_weight_scale=4,
        )
        self.assertEqual(tuple(model.psqt.weight.shape), (768, PSQT_BUCKET_COUNT))

        batch_features = [
            list(range(2)),
            list(range(5)),
            list(range(8)),
            list(range(9)),
            list(range(32)),
        ]
        flat = [feature for sample in batch_features for feature in sample]
        offsets = []
        cursor = 0
        for sample in batch_features:
            offsets.append(cursor)
            cursor += len(sample)
        indices = torch.tensor(flat, dtype=torch.long)
        offset_tensor = torch.tensor(offsets, dtype=torch.long)
        self.assertEqual(
            model.psqt_bucket_indices(indices, offset_tensor).tolist(),
            [0, 1, 1, 2, 7],
        )

        with torch.no_grad():
            model.feature_weights.weight.zero_()
            model.hidden1_bias.zero_()
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.psqt.weight.zero_()
            # Five active model features select bucket 1. Each contributes
            # quantized weight round(1.25 * 4) == 5.
            model.psqt.weight[:5, 1] = 1.25
            model.psqt.weight[:5, 2] = 100.0
        prediction = model(
            torch.tensor(batch_features[1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            hidden_scales=[],
            output_scale=1,
        )
        self.assertEqual(int(prediction.item()), 3)  # trunc((5 * 5 - 0) / (2 * 4))

    def test_normalized_psqt_master_weights_export_the_same_integer_cp_weight(self) -> None:
        config = QuantizedNnueArchitectureConfig(
            "TPsqtNormalized", "base768", 768, 768, (4,)
        )
        cp_master = QuantizedSparseNnueArchitecture(
            config,
            use_psqt=True,
            psqt_weight_scale=16,
            psqt_master_scale_to_cp=1.0,
        )
        normalized_master = QuantizedSparseNnueArchitecture(
            config,
            use_psqt=True,
            psqt_weight_scale=16,
            psqt_master_scale_to_cp=1000.0,
        )
        with torch.no_grad():
            cp_master.psqt.weight.zero_()
            normalized_master.psqt.weight.zero_()
            cp_master.psqt.weight[0, 0] = 40.0
            normalized_master.psqt.weight[0, 0] = 0.04

        self.assertEqual(
            int(cp_master.quantized_psqt_weight()[0, 0].item()),
            640,
        )
        self.assertTrue(
            torch.equal(
                cp_master.quantized_psqt_weight(),
                normalized_master.quantized_psqt_weight(),
            )
        )

    def test_psqt_dual_table_is_shared_and_piece_count_is_not_doubled(self) -> None:
        config = QuantizedNnueArchitectureConfig(
            "TPsqtDual",
            "dual_full_king_square_concat",
            8,
            21,
            (4,),
        )
        model = QuantizedSparseNnueArchitecture(config, use_psqt=True)
        self.assertEqual(tuple(model.psqt.weight.shape), (4, PSQT_BUCKET_COUNT))
        # Two physical pieces, represented once in each half/context.
        indices = torch.tensor([0, 1, 6, 7, 8], dtype=torch.long)  # final row is sparse aux
        offsets = torch.tensor([0], dtype=torch.long)
        self.assertEqual(model._physical_piece_count(indices, offsets).tolist(), [2])
        self.assertEqual(model.psqt_bucket_indices(indices, offsets).tolist(), [0])
        with torch.no_grad():
            model.psqt.weight.zero_()
            model.psqt.weight[0, 0] = 1.0
            model.psqt.weight[1, 0] = 2.0
            model.psqt.weight[2, 0] = 4.0
            model.psqt.weight[3, 0] = 8.0
        perspectives = model.psqt_perspective_accumulators(indices, offsets)
        self.assertEqual(perspectives[0, 0, 0].item(), 48.0)  # (1 + 2) * scale 16
        self.assertEqual(perspectives[0, 1, 0].item(), 192.0)  # (4 + 8) * scale 16

    def test_quantized_psqt_weights_and_legal_position_sum_fit_int32(self) -> None:
        config = QuantizedNnueArchitectureConfig("TPsqtRange", "base768", 64, 64, (4,))
        model = QuantizedSparseNnueArchitecture(config, use_psqt=True)
        with torch.no_grad():
            model.psqt.weight[0, 0] = 1e20
            model.psqt.weight[1, 0] = -1e20
        quantized = model.quantized_psqt_weight()
        self.assertEqual(int(quantized.max().item()), PSQT_QUANT_MAX)
        self.assertEqual(int(quantized.min().item()), PSQT_QUANT_MIN)
        self.assertLessEqual(64 * PSQT_QUANT_MAX, (1 << 31) - 1)
        self.assertGreaterEqual(64 * PSQT_QUANT_MIN, -(1 << 31))

    def test_psqt_weights_are_trained_by_the_main_loss(self) -> None:
        config = QuantizedNnueArchitectureConfig("TPsqtGrad", "base768", 768, 768, (4,))
        model = QuantizedSparseNnueArchitecture(config, use_psqt=True)
        prediction = model(
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            hidden_scales=[],
            output_scale=1,
        )
        torch.nn.functional.mse_loss(prediction, torch.tensor([100.0])).backward()
        self.assertIsNotNone(model.psqt.weight.grad)
        self.assertGreater(int(torch.count_nonzero(model.psqt.weight.grad)), 0)

    def test_psqt_changes_sign_when_side_to_move_perspective_is_flipped(self) -> None:
        config = QuantizedNnueArchitectureConfig("TPsqtPerspective", "base768", 768, 768, (4,))
        model = QuantizedSparseNnueArchitecture(config, use_psqt=True)
        with torch.no_grad():
            model.feature_weights.weight.zero_()
            model.hidden1_bias.zero_()
            model.output.weight.zero_()
            model.output.bias.zero_()
            values = torch.arange(model.psqt.weight.numel(), dtype=torch.float32)
            model.psqt.weight.copy_((torch.remainder(values, 31) - 15).view_as(model.psqt.weight) / 16)

        stm_features = [
            ((5 * 2 + 0) * 64) + 4,
            ((5 * 2 + 1) * 64) + 59,
            ((0 * 2 + 0) * 64) + 12,
            ((1 * 2 + 1) * 64) + 45,
            ((4 * 2 + 0) * 64) + 31,
        ]
        opponent_features = []
        for feature in stm_features:
            square = feature % 64
            value = feature // 64
            side = value % 2
            piece = value // 2
            opponent_features.append(((piece * 2 + (1 - side)) * 64) + (square ^ 56))

        first = model(
            torch.tensor(stm_features),
            torch.tensor([0]),
            hidden_scales=[],
            output_scale=1,
        )
        flipped = model(
            torch.tensor(opponent_features),
            torch.tensor([0]),
            hidden_scales=[],
            output_scale=1,
        )
        self.assertNotEqual(int(first.item()), 0)
        self.assertEqual(int(first.item()), -int(flipped.item()))

    def test_cp_regression_loss_supports_mse_and_huber(self) -> None:
        predictions = torch.tensor([0.0, 1.0])
        # Raw Stockfish score 208 maps to 100 CP; with target_scale=100 this
        # becomes the normalized target 1.0.
        scores = torch.tensor([0.0, 208.0])
        mse = cp_regression_loss(predictions, scores, 100.0, "cp_mse", 200.0)
        huber = cp_regression_loss(predictions, scores, 100.0, "cp_huber", 200.0)
        self.assertAlmostEqual(float(mse), 0.0, places=7)
        self.assertAlmostEqual(float(huber), 0.0, places=7)
        mixed = cp_regression_loss(
            predictions, scores, 100.0, "cp_mse_huber", 200.0, 0.75
        )
        self.assertAlmostEqual(float(mixed), 0.0, places=7)
        with self.assertRaisesRegex(ValueError, "MSE mix weight"):
            cp_regression_loss(
                predictions, scores, 100.0, "cp_mse_huber", 200.0, 1.5
            )
        with self.assertRaisesRegex(ValueError, "unknown CP loss"):
            cp_regression_loss(predictions, scores, 100.0, "bad", 200.0)

    def test_float_forward_removes_rounding_but_preserves_scale_domain(self) -> None:
        config = QuantizedNnueArchitectureConfig("TFloat", "base768", 2, 2, (1,))
        model = QuantizedSparseNnueArchitecture(
            config,
            hidden_clip=16,
            feature_weight_scale=16,
            linear_weight_scale=8,
            output_weight_scale=16,
            screlu_divisor=16,
        )
        with torch.no_grad():
            model.feature_weights.weight.zero_()
            model.feature_weights.weight[0, 0] = 0.03125  # code 0.5
            model.hidden1_bias.zero_()
            model.output.weight.fill_(0.0625)  # code 1.0
            model.output.bias.zero_()
        features = torch.tensor([0], dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)

        quantized = model(
            features,
            offsets,
            hidden_scales=[],
            output_scale=1,
            activation=ACTIVATION_RELU,
            quantization_convention=QUANTIZATION_CONVENTION_SCALE_CLEAN,
        )
        floating = model.forward_float(
            features,
            offsets,
            hidden_scales=[],
            output_scale=1,
            activation=ACTIVATION_RELU,
            quantization_convention=QUANTIZATION_CONVENTION_SCALE_CLEAN,
        )
        self.assertEqual(float(quantized.item()), 0.0)
        self.assertEqual(float(floating.item()), 0.5)

    def test_cosine_learning_rate_warms_up_and_decays(self) -> None:
        self.assertAlmostEqual(
            scheduled_learning_rate("cosine", 0, 100, 7e-4, 5e-5, 10),
            1.15e-4,
        )
        self.assertAlmostEqual(
            scheduled_learning_rate("cosine", 9, 100, 7e-4, 5e-5, 10),
            7e-4,
        )
        self.assertAlmostEqual(
            scheduled_learning_rate("cosine", 99, 100, 7e-4, 5e-5, 10),
            5e-5,
        )
        self.assertAlmostEqual(
            scheduled_learning_rate("constant", 50, 100, 7e-4, 5e-5, 10),
            7e-4,
        )

    def test_per_epoch_learning_rate_validation(self) -> None:
        validate_epoch_learning_rates([5e-4, 1e-4], [5e-5, 6e-5], 2)
        with self.assertRaisesRegex(ValueError, "used together"):
            validate_epoch_learning_rates([5e-4], None, 1)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_epoch_learning_rates([5e-4], [5e-5], 2)
        with self.assertRaisesRegex(ValueError, "epoch 1"):
            validate_epoch_learning_rates([5e-5], [5e-4], 1)

    def test_screlu_divisor_256_matches_shift_semantics(self) -> None:
        config = QuantizedNnueArchitectureConfig("TDiv", "base768", 8, 8, (4,))
        model = QuantizedSparseNnueArchitecture(config, screlu_divisor=256)
        values = torch.tensor([[53.0, 150.0, 200.0, 255.0]])
        actual = model._activate_hidden(values, "screlu_all", 0)
        expected = torch.tensor([[10.0, 87.0, 156.0, 254.0]])
        self.assertTrue(torch.equal(actual, expected))

    def test_scale_audit_reports_bias_scale_mismatches(self) -> None:
        config = QuantizedNnueArchitectureConfig("TAudit", "base768", 8, 8, (4, 2, 2))
        model = QuantizedSparseNnueArchitecture(
            config,
            hidden_clip=180,
            feature_weight_scale=255,
            linear_weight_scale=64,
            output_weight_scale=16,
            screlu_divisor=128,
        )
        audit = quantization_scale_audit(
            model,
            "screlu_all",
            hidden_scales=[32, 16],
            output_scale=8,
        )
        self.assertAlmostEqual(
            audit["layers"][0]["output_activation_scale"],
            255.0 * 255.0 / 128.0,
        )
        self.assertAlmostEqual(
            audit["layers"][1]["expected_bias_scale"],
            (255.0 * 255.0 / 128.0) * 64.0,
        )
        self.assertEqual(audit["layers"][1]["code_bias_scale"], 180.0 * 64.0)
        self.assertIn("hidden_1_bias_scale_mismatch", audit["warnings"][0])
        self.assertTrue(any("output_bias_scale_mismatch" in warning for warning in audit["warnings"]))

    def test_scale_clean_audit_uses_accumulator_scales_for_biases(self) -> None:
        config = QuantizedNnueArchitectureConfig("TClean", "base768", 8, 8, (4, 2, 2))
        model = QuantizedSparseNnueArchitecture(
            config,
            hidden_clip=180,
            feature_weight_scale=180,
            linear_weight_scale=64,
            output_weight_scale=16,
            screlu_divisor=128,
        )
        audit = quantization_scale_audit(
            model,
            "screlu_all",
            hidden_scales=[32, 16],
            output_scale=8,
            quantization_convention=QUANTIZATION_CONVENTION_SCALE_CLEAN,
        )
        self.assertEqual(audit["warnings"], [])
        for layer in audit["layers"]:
            self.assertAlmostEqual(layer["bias_scale_ratio"], 1.0)
        expected_activation_scale = 180.0 * 180.0 / 128.0
        for layer in audit["layers"][:-1]:
            self.assertAlmostEqual(
                layer["output_activation_scale"], expected_activation_scale
            )
        for layer in audit["layers"][1:-1]:
            self.assertAlmostEqual(
                layer["expected_bias_scale"], expected_activation_scale * 64.0
            )
        self.assertAlmostEqual(
            audit["layers"][-1]["expected_bias_scale"],
            expected_activation_scale * 16.0,
        )

    def test_scale_clean_forward_changes_bias_convention(self) -> None:
        config = QuantizedNnueArchitectureConfig("TCleanForward", "base768", 8, 8, (3, 2))
        model = QuantizedSparseNnueArchitecture(
            config,
            hidden_clip=180,
            feature_weight_scale=180,
            linear_weight_scale=64,
            output_weight_scale=16,
            screlu_divisor=128,
        )
        with torch.no_grad():
            model.feature_weights.weight.zero_()
            model.hidden1_bias.fill_(0.1)
            model.hidden_layers[0].weight.zero_()
            model.hidden_layers[0].bias.fill_(0.25)
            model.output.weight.zero_()
            model.output.bias.fill_(1.0)

        features = torch.tensor([0], dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        legacy = model(
            features,
            offsets,
            hidden_scales=[32],
            output_scale=8,
            activation="screlu_all",
        )
        clean = model(
            features,
            offsets,
            hidden_scales=[32],
            output_scale=8,
            activation="screlu_all",
            quantization_convention=QUANTIZATION_CONVENTION_SCALE_CLEAN,
        )
        self.assertFalse(torch.equal(legacy, clean))

    def test_scale_clean_screlu_bias_initialization_is_scale_aware(self) -> None:
        config = QuantizedNnueArchitectureConfig("TInit", "base768", 8, 8, (4, 3, 2))
        model = QuantizedSparseNnueArchitecture(
            config,
            hidden_clip=180,
            feature_weight_scale=180,
            linear_weight_scale=64,
            output_weight_scale=16,
            screlu_divisor=128,
        )
        event = initialize_scale_clean_screlu_biases(
            model,
            "screlu_all",
            [8, 32],
            QUANTIZATION_CONVENTION_SCALE_CLEAN,
            hidden_fraction=0.25,
            first_fraction=0.10,
        )
        self.assertIsNotNone(event)
        self.assertTrue(torch.allclose(model.hidden1_bias, torch.full_like(model.hidden1_bias, 0.10)))
        activation_scale = 180.0 * 180.0 / 128.0
        expected = [
            45.0 * scale / (activation_scale * 64.0)
            for scale in (8, 32)
        ]
        for layer, expected_bias in zip(model.hidden_layers, expected):
            self.assertTrue(
                torch.allclose(layer.bias, torch.full_like(layer.bias, expected_bias))
            )

    def test_architectures_a_to_h_exist(self) -> None:
        self.assertEqual(
            sorted(QUANTIZED_ARCHITECTURES),
            ["A", "B", "C", "D", "E", "E2", "F", "F2", "F2M", "F2_64", "G", "H"],
        )
        self.assertEqual(QUANTIZED_ARCHITECTURES["A"].hidden_sizes, (256, 32, 32))
        self.assertEqual(QUANTIZED_ARCHITECTURES["B"].hidden_sizes, (128, 32, 32))
        self.assertEqual(QUANTIZED_ARCHITECTURES["F"].hidden_sizes, (256, 32, 32))
        self.assertEqual(QUANTIZED_ARCHITECTURES["E2"].hidden_sizes, (256, 32))
        self.assertEqual(QUANTIZED_ARCHITECTURES["F2"].hidden_sizes, (256, 32, 32))
        self.assertEqual(QUANTIZED_ARCHITECTURES["F2M"].hidden_sizes, (256, 32, 32))
        self.assertEqual(
            QUANTIZED_ARCHITECTURES["F2M"].board_feature_count,
            QUANTIZED_ARCHITECTURES["F2"].board_feature_count // 2,
        )
        self.assertEqual(QUANTIZED_ARCHITECTURES["G"].hidden_sizes, (128, 32, 32))
        self.assertEqual(QUANTIZED_ARCHITECTURES["H"].hidden_sizes, (128, 32))

    def test_aux_is_appended_as_sparse_features(self) -> None:
        config = QUANTIZED_ARCHITECTURES["A"]
        raw_features = [0, 1]
        aux = [0] * 13
        aux[0] = 1
        aux[12] = 1
        features = transform_features_with_sparse_aux(raw_features, aux, config)
        self.assertIn(config.board_feature_count, features)
        self.assertIn(config.board_feature_count + 12, features)
        self.assertNotIn(config.board_feature_count + 1, features)

    def test_forward_matches_brute_force_for_multiple_layer_counts(self) -> None:
        torch.manual_seed(7)
        configs = [
            QuantizedNnueArchitectureConfig("T1", "base768", 8, 8, (5,)),
            QuantizedNnueArchitectureConfig("T2", "base768", 8, 8, (5, 3)),
            QuantizedNnueArchitectureConfig("T3", "base768", 8, 8, (5, 3, 2)),
            QuantizedNnueArchitectureConfig(
                "TDual", "dual_full_king_square_concat", 8, 21, (4, 3)
            ),
        ]
        for config in configs:
            for screlu_divisor in (255, 256):
                model = QuantizedSparseNnueArchitecture(
                    config,
                    hidden_clip=HIDDEN_CLIP,
                    feature_weight_scale=127,
                    linear_weight_scale=64,
                    output_weight_scale=16,
                    screlu_divisor=screlu_divisor,
                )
                with torch.no_grad():
                    model.feature_weights.weight.uniform_(-0.5, 0.5)
                    model.hidden1_bias.uniform_(-0.5, 0.5)
                    for layer in model.hidden_layers:
                        layer.weight.uniform_(-0.5, 0.5)
                        layer.bias.uniform_(-0.5, 0.5)
                    model.output.weight.uniform_(-0.5, 0.5)
                    model.output.bias.uniform_(-2, 2)

                batch_features = [[0, 3, 5], [1, 1, 7], [2, 4]]
                flat = [feature for features in batch_features for feature in features]
                offsets = []
                cursor = 0
                for features in batch_features:
                    offsets.append(cursor)
                    cursor += len(features)
                hidden_scales = [64] * len(model.hidden_layers)
                for activation in ACTIVATION_CHOICES:
                    actual = model(
                        torch.tensor(flat, dtype=torch.long),
                        torch.tensor(offsets, dtype=torch.long),
                        hidden_scales=hidden_scales,
                        output_scale=16,
                        activation=activation,
                    )
                    expected = brute_quantized_forward(model, batch_features, hidden_scales, 16, activation)
                    self.assertTrue(torch.allclose(actual, torch.tensor(expected, dtype=actual.dtype)))

    def test_feature_bias_uses_feature_weight_scale_not_hidden_clip(self) -> None:
        config = QuantizedNnueArchitectureConfig("TScale", "base768", 8, 8, (3,))
        model = QuantizedSparseNnueArchitecture(
            config,
            hidden_clip=31,
            feature_weight_scale=13,
            linear_weight_scale=7,
            output_weight_scale=5,
        )
        with torch.no_grad():
            model.feature_weights.weight.zero_()
            model.hidden1_bias.copy_(torch.tensor([0.5, -0.5, 1.0]))
            model.output.weight.copy_(torch.tensor([[1.0, 1.0, 1.0]]))
            model.output.bias.zero_()

        expected_bias = torch.tensor([6.0, -6.0, 13.0])
        self.assertTrue(torch.equal(model.quantized_hidden1_bias(), expected_bias))
        actual = model(
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            hidden_scales=[],
            output_scale=5,
        )
        expected = brute_quantized_forward(model, [[0]], [], 5)
        self.assertTrue(torch.equal(actual, torch.tensor(expected, dtype=actual.dtype)))

    def test_rejects_non_positive_quantization_scales(self) -> None:
        config = QuantizedNnueArchitectureConfig("TScale", "base768", 8, 8, (3,))
        for keyword in (
            "hidden_clip",
            "feature_weight_scale",
            "linear_weight_scale",
            "output_weight_scale",
            "psqt_weight_scale",
            "psqt_master_scale_to_cp",
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    QuantizedSparseNnueArchitecture(config, **{keyword: 0})

        model = QuantizedSparseNnueArchitecture(config)
        with self.assertRaisesRegex(ValueError, "output scale must be positive"):
            model(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                output_scale=0,
            )

    def test_dual_accumulator_uses_one_shared_feature_table(self) -> None:
        config = QUANTIZED_ARCHITECTURES["E2"]
        model = QuantizedSparseNnueArchitecture(config)
        self.assertEqual(
            tuple(model.feature_weights.weight.shape),
            (config.board_feature_count // 2, config.hidden1_size // 2),
        )
        self.assertEqual(
            tuple(model.aux_feature_weights.shape),
            (13, config.hidden1_size),
        )

    def test_dual_accumulator_diagnostics_match_masked_forward(self) -> None:
        config = QuantizedNnueArchitectureConfig(
            "TDual",
            "dual_full_king_square_concat",
            8,
            10,
            (4, 2),
        )
        model = QuantizedSparseNnueArchitecture(config)
        with torch.no_grad():
            model.feature_weights.weight.uniform_(-0.5, 0.5)
            model.hidden1_bias.uniform_(-0.5, 0.5)

        features = torch.tensor([0, 4, 8], dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        expected = torch.clamp(
            model.first_hidden_accumulator(features, offsets),
            0,
            model.hidden_clip,
        )
        actual = initial_hidden1(model, features, offsets, ACTIVATION_RELU)
        self.assertTrue(torch.equal(actual, expected))

    def test_all_parameter_groups_receive_finite_gradients(self) -> None:
        config = QuantizedNnueArchitectureConfig(
            "TGradient",
            "base768",
            32,
            32,
            (16, 8, 4),
        )
        batch_features = [
            [0, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [0, 5, 10, 15, 20, 25, 30, 31],
        ]
        flat = [feature for features in batch_features for feature in features]
        offsets = [0]
        for features in batch_features[:-1]:
            offsets.append(offsets[-1] + len(features))
        feature_indices = torch.tensor(flat, dtype=torch.long)
        offset_tensor = torch.tensor(offsets, dtype=torch.long)
        targets = torch.tensor([32.0, -24.0, 16.0, -8.0])

        for activation in ACTIVATION_CHOICES:
            torch.manual_seed(19)
            model = QuantizedSparseNnueArchitecture(config)
            with torch.no_grad():
                model.hidden1_bias.fill_(0.25)
                for layer in model.hidden_layers:
                    layer.bias.fill_(0.25)
                model.output.bias.fill_(0.25)

            prediction = model(
                feature_indices,
                offset_tensor,
                hidden_scales=[64, 64],
                output_scale=16,
                activation=activation,
            )
            torch.nn.functional.mse_loss(prediction, targets).backward()

            parameter_groups = {
                "feature_weights": model.feature_weights.weight,
                "hidden1_bias": model.hidden1_bias,
                "output_weight": model.output.weight,
                "output_bias": model.output.bias,
            }
            for index, layer in enumerate(model.hidden_layers):
                parameter_groups[f"hidden_{index}_weight"] = layer.weight
                parameter_groups[f"hidden_{index}_bias"] = layer.bias

            for name, parameter in parameter_groups.items():
                self.assertIsNotNone(parameter.grad, f"{activation}: {name} has no gradient")
                assert parameter.grad is not None
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    f"{activation}: {name} has non-finite gradient",
                )
                self.assertGreater(
                    int(torch.count_nonzero(parameter.grad)),
                    0,
                    f"{activation}: {name} has only zero gradients",
                )

    def test_scale_from_positive_percentile(self) -> None:
        values = torch.tensor([-10.0, 0.0, 1.0, 127.0, 254.0])
        self.assertGreaterEqual(scale_from_positive_percentile(values, 99.0), 1)
        self.assertEqual(scale_from_positive_percentile(torch.tensor([-1.0, 0.0]), 99.0), 1)

    def test_sampled_quantile_caps_input_without_changing_small_inputs(self) -> None:
        values = torch.arange(1000, dtype=torch.float32)
        exact = float(torch.quantile(values, 0.99))
        self.assertEqual(sampled_quantile(values, 0.99, max_values=1000), exact)
        sampled = sampled_quantile(values, 0.99, max_values=100)
        self.assertLessEqual(abs(sampled - exact), 10.0)


if __name__ == "__main__":
    unittest.main()
