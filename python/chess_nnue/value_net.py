from __future__ import annotations

import torch
from torch import nn


PIECE_TYPES = 6
PIECE_SIDES = 2
KING_CONTEXTS = 2
SQUARES = 64

FEATURE_COUNT = PIECE_TYPES * PIECE_SIDES * KING_CONTEXTS * SQUARES * SQUARES
AUX_FEATURE_COUNT = 13


class ChessValueNet(nn.Module):
    """Sparse chess value network.

    Inputs:
        feature_indices: LongTensor with shape [num_active_features].
        offsets: LongTensor with shape [batch_size], EmbeddingBag offsets.
        aux: FloatTensor with shape [batch_size, 13].

    Output:
        FloatTensor with shape [batch_size], centipawn-like value from side-to-move POV.
    """

    def __init__(
        self,
        feature_count: int = FEATURE_COUNT,
        aux_feature_count: int = AUX_FEATURE_COUNT,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.aux_feature_count = aux_feature_count
        self.hidden_size = hidden_size

        self.sparse_features = nn.EmbeddingBag(
            num_embeddings=feature_count,
            embedding_dim=hidden_size,
            mode="sum",
            include_last_offset=False,
        )
        self.hidden = nn.Sequential(
            nn.Linear(hidden_size + aux_feature_count, hidden_size),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.sparse_features.weight, mean=0.0, std=0.02)
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
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

        sparse = self.sparse_features(feature_indices, offsets)
        x = torch.cat([sparse, aux], dim=1)
        x = self.hidden(x)
        return self.output(x).squeeze(-1)


def make_batch(
    batch_features: list[list[int]],
    batch_aux: list[list[float]],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build tensors for ChessValueNet from sparse feature lists."""

    flat_features: list[int] = []
    offsets: list[int] = []
    cursor = 0

    for features in batch_features:
        offsets.append(cursor)
        flat_features.extend(features)
        cursor += len(features)

    feature_tensor = torch.tensor(flat_features, dtype=torch.long, device=device)
    offset_tensor = torch.tensor(offsets, dtype=torch.long, device=device)
    aux_tensor = torch.tensor(batch_aux, dtype=torch.float32, device=device)
    return feature_tensor, offset_tensor, aux_tensor
