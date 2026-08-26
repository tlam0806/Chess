from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .value_net import AUX_FEATURE_COUNT


BASE_FEATURE_COUNT = 6 * 2 * 64
KING_BUCKET_COUNT = 16
KING_BUCKET_FEATURE_COUNT = 6 * 2 * 2 * KING_BUCKET_COUNT * 64
FULL_KING_FEATURE_COUNT = 6 * 2 * 2 * 64 * 64
DUAL_ACCUMULATOR_FEATURE_COUNT = 6 * 2 * 64 * 64
DUAL_FULL_KING_FEATURE_COUNT = 2 * DUAL_ACCUMULATOR_FEATURE_COUNT
HORIZONTAL_MIRROR_KING_SQUARE_COUNT = 8 * 4
HORIZONTAL_MIRROR_DUAL_ACCUMULATOR_FEATURE_COUNT = (
    6 * 2 * HORIZONTAL_MIRROR_KING_SQUARE_COUNT * 64
)
HORIZONTAL_MIRROR_DUAL_FULL_KING_FEATURE_COUNT = (
    2 * HORIZONTAL_MIRROR_DUAL_ACCUMULATOR_FEATURE_COUNT
)
DUAL_ACCUMULATOR_TRANSFORMS = frozenset(
    {
        "dual_full_king_square_concat",
        "dual_full_king_square_concat_horizontal_mirror",
    }
)


@dataclass(frozen=True)
class NnueArchitectureConfig:
    name: str
    transform: str
    feature_count: int
    hidden_sizes: tuple[int, ...]

    @property
    def hidden1_size(self) -> int:
        return self.hidden_sizes[0]

    @property
    def hidden2_size(self) -> int | None:
        return self.hidden_sizes[1] if len(self.hidden_sizes) >= 2 else None


ARCHITECTURES: dict[str, NnueArchitectureConfig] = {
    "A": NnueArchitectureConfig("A", "base768", BASE_FEATURE_COUNT, (128,)),
    "B": NnueArchitectureConfig("B", "base768", BASE_FEATURE_COUNT, (256,)),
    "C": NnueArchitectureConfig("C", "base768", BASE_FEATURE_COUNT, (256, 32)),
    "D": NnueArchitectureConfig("D", "king_bucket", KING_BUCKET_FEATURE_COUNT, (256, 32)),
    "E": NnueArchitectureConfig("E", "full_king_square", FULL_KING_FEATURE_COUNT, (256, 32)),
    "F": NnueArchitectureConfig("F", "full_king_square", FULL_KING_FEATURE_COUNT, (256, 32, 32)),
    "G": NnueArchitectureConfig("G", "full_king_square", FULL_KING_FEATURE_COUNT, (128, 32, 32)),
    "H": NnueArchitectureConfig("H", "full_king_square", FULL_KING_FEATURE_COUNT, (128, 32)),
    "E2": NnueArchitectureConfig("E2", "dual_full_king_square_concat", DUAL_FULL_KING_FEATURE_COUNT, (256, 32)),
    "F2": NnueArchitectureConfig("F2", "dual_full_king_square_concat", DUAL_FULL_KING_FEATURE_COUNT, (256, 32, 32)),
    "F2M": NnueArchitectureConfig(
        "F2M",
        "dual_full_king_square_concat_horizontal_mirror",
        HORIZONTAL_MIRROR_DUAL_FULL_KING_FEATURE_COUNT,
        (256, 32, 32),
    ),
}


def decode_feature(feature: int) -> tuple[int, int, int, int, int]:
    piece_square = feature % 64
    feature //= 64
    king_square = feature % 64
    feature //= 64
    king_context = feature % 2
    feature //= 2
    piece_side = feature % 2
    feature //= 2
    piece = feature
    return piece, piece_side, king_context, king_square, piece_square


def king_bucket(square: int) -> int:
    file = square & 7
    rank = square >> 3
    return (rank >> 1) * 4 + (file >> 1)


def flip_relative_square(square: int) -> int:
    return square ^ 56


def horizontal_mirror_mask(king_square: int) -> int:
    """Return the branch-free square XOR mask that puts a king on files a-d."""
    return 7 if (king_square & 7) >= 4 else 0


def horizontal_mirror_king_index(king_square: int, mirror_mask: int) -> int:
    canonical = king_square ^ mirror_mask
    return (canonical >> 3) * 4 + (canonical & 7)


def is_dual_accumulator_transform(transform: str) -> bool:
    return transform in DUAL_ACCUMULATOR_TRANSFORMS


def dual_accumulator_feature(
    piece: int,
    piece_side: int,
    king_square: int,
    piece_square: int,
    king_square_count: int = 64,
) -> int:
    index = piece
    index = index * 2 + piece_side
    index = index * king_square_count + king_square
    index = index * 64 + piece_square
    return index


def transform_features(raw_features: list[int], transform: str) -> list[int]:
    if transform == "base768":
        features: set[int] = set()
        for raw in raw_features:
            piece, piece_side, _, _, piece_square = decode_feature(int(raw))
            features.add(((piece * 2 + piece_side) * 64) + piece_square)
        return sorted(features)

    if transform == "king_bucket":
        features = []
        for raw in raw_features:
            piece, piece_side, king_context, king_square, piece_square = decode_feature(int(raw))
            index = piece
            index = index * 2 + piece_side
            index = index * 2 + king_context
            index = index * KING_BUCKET_COUNT + king_bucket(king_square)
            index = index * 64 + piece_square
            features.append(index)
        return features

    if transform == "full_king_square":
        features = []
        for raw in raw_features:
            piece, piece_side, king_context, king_square, piece_square = decode_feature(int(raw))
            index = piece
            index = index * 2 + piece_side
            index = index * 2 + king_context
            index = index * 64 + king_square
            index = index * 64 + piece_square
            features.append(index)
        return features

    if transform == "dual_full_king_square_concat":
        features = []
        for raw in raw_features:
            piece, piece_side, king_context, king_square, piece_square = decode_feature(int(raw))
            if king_context == 0:
                features.append(dual_accumulator_feature(piece, piece_side, king_square, piece_square))
            else:
                features.append(
                    DUAL_ACCUMULATOR_FEATURE_COUNT
                    + dual_accumulator_feature(
                        piece,
                        1 - piece_side,
                        flip_relative_square(king_square),
                        flip_relative_square(piece_square),
                    )
                )
        return features

    if transform == "dual_full_king_square_concat_horizontal_mirror":
        features = []
        for raw in raw_features:
            piece, piece_side, king_context, king_square, piece_square = decode_feature(int(raw))
            if king_context == 0:
                mirror_mask = horizontal_mirror_mask(king_square)
                features.append(
                    dual_accumulator_feature(
                        piece,
                        piece_side,
                        horizontal_mirror_king_index(king_square, mirror_mask),
                        piece_square ^ mirror_mask,
                        HORIZONTAL_MIRROR_KING_SQUARE_COUNT,
                    )
                )
            else:
                relative_king_square = flip_relative_square(king_square)
                relative_piece_square = flip_relative_square(piece_square)
                mirror_mask = horizontal_mirror_mask(relative_king_square)
                features.append(
                    HORIZONTAL_MIRROR_DUAL_ACCUMULATOR_FEATURE_COUNT
                    + dual_accumulator_feature(
                        piece,
                        1 - piece_side,
                        horizontal_mirror_king_index(
                            relative_king_square, mirror_mask
                        ),
                        relative_piece_square ^ mirror_mask,
                        HORIZONTAL_MIRROR_KING_SQUARE_COUNT,
                    )
                )
        return features

    raise ValueError(f"unknown feature transform: {transform}")


class SparseNnueArchitecture(nn.Module):
    def __init__(
        self,
        config: NnueArchitectureConfig,
        aux_feature_count: int = AUX_FEATURE_COUNT,
    ) -> None:
        super().__init__()
        self.config = config
        self.architecture = config.name
        self.transform = config.transform
        self.feature_count = config.feature_count
        self.aux_feature_count = aux_feature_count
        self.hidden1_size = config.hidden1_size
        self.hidden2_size = config.hidden2_size
        self.hidden_sizes = config.hidden_sizes
        self.dual_accumulator = is_dual_accumulator_transform(self.transform)
        if self.dual_accumulator:
            if config.feature_count % 2 != 0 or config.hidden1_size % 2 != 0:
                raise ValueError("dual accumulator feature and hidden sizes must be even")
            embedding_count = config.feature_count // 2
            embedding_size = config.hidden1_size // 2
        else:
            embedding_count = config.feature_count
            embedding_size = config.hidden1_size

        self.feature_weights = nn.EmbeddingBag(
            num_embeddings=embedding_count,
            embedding_dim=embedding_size,
            mode="sum",
            include_last_offset=False,
        )
        self.aux_projection = nn.Linear(aux_feature_count, config.hidden1_size, bias=False)
        self.hidden1_bias = nn.Parameter(torch.zeros(embedding_size))

        self.hidden_layers = nn.ModuleList(
            nn.Linear(config.hidden_sizes[index], config.hidden_sizes[index + 1])
            for index in range(len(config.hidden_sizes) - 1)
        )
        self.output = nn.Linear(config.hidden_sizes[-1], 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_weights.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.aux_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.hidden1_bias)
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        hidden1 = self.first_hidden_accumulator(feature_indices, offsets, aux)
        hidden1 = torch.clamp(hidden1, min=0.0, max=1.0)
        hidden = hidden1
        for layer in self.hidden_layers:
            hidden = torch.clamp(layer(hidden), min=0.0, max=1.0)
        return self.output(hidden).squeeze(-1)

    def first_hidden_accumulator(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        if feature_indices.dtype != torch.long:
            feature_indices = feature_indices.long()
        if offsets.dtype != torch.long:
            offsets = offsets.long()
        aux = aux.float()

        if self.dual_accumulator:
            half_features = self.feature_count // 2
            shared_indices = torch.remainder(feature_indices, half_features)
            first_weights = (feature_indices < half_features).to(self.feature_weights.weight.dtype)
            second_weights = (feature_indices >= half_features).to(self.feature_weights.weight.dtype)
            first = self.feature_weights(
                shared_indices,
                offsets,
                per_sample_weights=first_weights,
            )
            second = self.feature_weights(
                shared_indices,
                offsets,
                per_sample_weights=second_weights,
            )
            board_hidden = torch.cat((first, second), dim=1)
            hidden1_bias = torch.cat((self.hidden1_bias, self.hidden1_bias), dim=0)
        else:
            board_hidden = self.feature_weights(feature_indices, offsets)
            hidden1_bias = self.hidden1_bias

        return board_hidden + self.aux_projection(aux) + hidden1_bias


__all__ = [
    "ARCHITECTURES",
    "DUAL_ACCUMULATOR_TRANSFORMS",
    "DUAL_ACCUMULATOR_FEATURE_COUNT",
    "DUAL_FULL_KING_FEATURE_COUNT",
    "FULL_KING_FEATURE_COUNT",
    "HORIZONTAL_MIRROR_DUAL_ACCUMULATOR_FEATURE_COUNT",
    "HORIZONTAL_MIRROR_DUAL_FULL_KING_FEATURE_COUNT",
    "HORIZONTAL_MIRROR_KING_SQUARE_COUNT",
    "NnueArchitectureConfig",
    "SparseNnueArchitecture",
    "dual_accumulator_feature",
    "flip_relative_square",
    "horizontal_mirror_king_index",
    "horizontal_mirror_mask",
    "is_dual_accumulator_transform",
    "transform_features",
]
