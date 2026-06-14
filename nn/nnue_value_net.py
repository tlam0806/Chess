from __future__ import annotations

import torch
from torch import nn

from .value_net import AUX_FEATURE_COUNT, FEATURE_COUNT, make_batch


class ChessNnueValueNet(nn.Module):
    """Minimal full-recompute NNUE-style value network.

    This keeps the current sparse JSONL feature format. Each active feature adds
    one row into an accumulator, aux features project into the same accumulator,
    then a clipped ReLU and linear output produce the normalized value.
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

        self.feature_weights = nn.EmbeddingBag(
            num_embeddings=feature_count,
            embedding_dim=hidden_size,
            mode="sum",
            include_last_offset=False,
        )
        self.aux_projection = nn.Linear(aux_feature_count, hidden_size, bias=False)
        self.accumulator_bias = nn.Parameter(torch.zeros(hidden_size))
        self.output = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_weights.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.aux_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.accumulator_bias)
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

        accumulator = (
            self.feature_weights(feature_indices, offsets)
            + self.aux_projection(aux)
            + self.accumulator_bias
        )
        x = torch.clamp(accumulator, min=0.0, max=1.0)
        return self.output(x).squeeze(-1)


__all__ = ["ChessNnueValueNet", "make_batch"]
