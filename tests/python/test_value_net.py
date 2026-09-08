import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class ChessValueNetTests(unittest.TestCase):
    def test_forward_shape_and_backward(self):
        import torch

        from chess_nnue.value_net import AUX_FEATURE_COUNT, FEATURE_COUNT, ChessValueNet, make_batch

        model = ChessValueNet()
        self.assertEqual(model.hidden_size, 256)
        self.assertEqual(model.feature_count, FEATURE_COUNT)

        batch_features = [
            [0, 17, 1024, FEATURE_COUNT - 1],
            [7, 64, 4096],
        ]
        batch_aux = [
            [0.0] * AUX_FEATURE_COUNT,
            [1.0] + [0.0] * (AUX_FEATURE_COUNT - 1),
        ]

        feature_indices, offsets, aux = make_batch(batch_features, batch_aux)
        output = model(feature_indices, offsets, aux)

        self.assertEqual(tuple(output.shape), (2,))
        loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(model.output.weight.grad)
        self.assertIsNotNone(model.sparse_features.weight.grad)

    def test_empty_aux_batch_is_valid_when_features_present(self):
        from chess_nnue.value_net import AUX_FEATURE_COUNT, ChessValueNet, make_batch

        model = ChessValueNet()
        feature_indices, offsets, aux = make_batch([[1, 2]], [[0.0] * AUX_FEATURE_COUNT])
        output = model(feature_indices, offsets, aux)

        self.assertEqual(tuple(output.shape), (1,))


if __name__ == "__main__":
    unittest.main()
