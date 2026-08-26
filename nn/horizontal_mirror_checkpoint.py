from __future__ import annotations

import copy
from typing import Any

import torch

from .nnue_architectures import (
    DUAL_ACCUMULATOR_FEATURE_COUNT,
    HORIZONTAL_MIRROR_DUAL_ACCUMULATOR_FEATURE_COUNT,
    HORIZONTAL_MIRROR_KING_SQUARE_COUNT,
    dual_accumulator_feature,
)


F2_FEATURE_ROWS = DUAL_ACCUMULATOR_FEATURE_COUNT
F2M_FEATURE_ROWS = HORIZONTAL_MIRROR_DUAL_ACCUMULATOR_FEATURE_COUNT


def _f2_horizontal_row_indices() -> tuple[torch.Tensor, torch.Tensor]:
    """Map every F2M row to the two horizontally symmetric F2 rows."""
    canonical: list[int] = []
    mirrored: list[int] = []
    for piece in range(6):
        for piece_side in range(2):
            for rank in range(8):
                for file in range(4):
                    king_square = rank * 8 + file
                    compact_king = rank * 4 + file
                    for piece_square in range(64):
                        canonical.append(
                            dual_accumulator_feature(
                                piece, piece_side, king_square, piece_square
                            )
                        )
                        mirrored.append(
                            dual_accumulator_feature(
                                piece,
                                piece_side,
                                king_square ^ 7,
                                piece_square ^ 7,
                            )
                        )
                        expected = dual_accumulator_feature(
                            piece,
                            piece_side,
                            compact_king,
                            piece_square,
                            HORIZONTAL_MIRROR_KING_SQUARE_COUNT,
                        )
                        if expected != len(canonical) - 1:
                            raise AssertionError("F2M row ordering mismatch")
    return torch.tensor(canonical), torch.tensor(mirrored)


F2M_TO_F2_CANONICAL, F2M_TO_F2_MIRRORED = _f2_horizontal_row_indices()


def _fold_pair(
    tensor: torch.Tensor,
    canonical: torch.Tensor,
    mirrored: torch.Tensor,
    method: str,
) -> torch.Tensor:
    left = tensor.index_select(0, canonical.to(tensor.device))
    if method == "canonical":
        return left.clone()
    if method == "average":
        right = tensor.index_select(0, mirrored.to(tensor.device))
        return (left + right) * 0.5
    raise ValueError(f"unknown horizontal fold method: {method}")


def fold_feature_rows(tensor: torch.Tensor, method: str = "average") -> torch.Tensor:
    if tensor.ndim < 1 or tensor.shape[0] != F2_FEATURE_ROWS:
        raise ValueError(
            f"expected first dimension {F2_FEATURE_ROWS}, got {tuple(tensor.shape)}"
        )
    return _fold_pair(
        tensor,
        F2M_TO_F2_CANONICAL,
        F2M_TO_F2_MIRRORED,
        method,
    )


def symmetrize_feature_rows(
    tensor: torch.Tensor,
    method: str = "average",
) -> torch.Tensor:
    folded = fold_feature_rows(tensor, method)
    result = tensor.clone()
    canonical = F2M_TO_F2_CANONICAL.to(tensor.device)
    mirrored = F2M_TO_F2_MIRRORED.to(tensor.device)
    result.index_copy_(0, canonical, folded)
    result.index_copy_(0, mirrored, folded)
    return result


def fold_aux_weights(tensor: torch.Tensor, method: str = "average") -> torch.Tensor:
    if tensor.ndim < 1 or tensor.shape[0] != 13:
        raise ValueError(f"expected 13 aux rows, got {tuple(tensor.shape)}")
    result = tensor.clone()
    for left, right in ((0, 1), (2, 3), (5, 12), (6, 11), (7, 10), (8, 9)):
        if method == "canonical":
            folded = tensor[left]
        elif method == "average":
            folded = (tensor[left] + tensor[right]) * 0.5
        else:
            raise ValueError(f"unknown horizontal fold method: {method}")
        result[left] = folded
        result[right] = folded
    return result


def convert_f2_checkpoint(
    checkpoint: dict[str, Any],
    method: str = "average",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create an exact symmetric F2 reference and its compact F2M equivalent.

    Dense layers and biases are copied unchanged. The returned checkpoints
    intentionally omit optimizer state because the sparse table shape changed.
    """
    if checkpoint.get("architecture") != "F2":
        raise ValueError("source checkpoint must use architecture F2")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("source checkpoint is missing model_state")
    required = ("feature_weights.weight", "aux_feature_weights", "psqt.weight")
    for key in required:
        if key not in state:
            raise ValueError(f"source checkpoint is missing {key}")

    symmetric = copy.deepcopy(checkpoint)
    symmetric.pop("optimizer_state", None)
    symmetric_state = copy.deepcopy(state)
    symmetric_state["feature_weights.weight"] = symmetrize_feature_rows(
        state["feature_weights.weight"], method
    )
    symmetric_state["psqt.weight"] = symmetrize_feature_rows(
        state["psqt.weight"], method
    )
    symmetric_state["aux_feature_weights"] = fold_aux_weights(
        state["aux_feature_weights"], method
    )
    symmetric["model_state"] = symmetric_state
    symmetric["horizontal_mirror_conversion"] = {
        "source_architecture": "F2",
        "target_architecture": "F2",
        "fold_method": method,
        "symmetric_reference": True,
    }

    mirror = copy.deepcopy(checkpoint)
    mirror.pop("optimizer_state", None)
    mirror_state = copy.deepcopy(state)
    mirror_state["feature_weights.weight"] = fold_feature_rows(
        state["feature_weights.weight"], method
    )
    mirror_state["psqt.weight"] = fold_feature_rows(
        state["psqt.weight"], method
    )
    mirror_state["aux_feature_weights"] = symmetric_state["aux_feature_weights"].clone()
    mirror["model_state"] = mirror_state
    mirror["architecture"] = "F2M"
    if isinstance(mirror.get("args"), dict):
        mirror["args"] = dict(mirror["args"])
        mirror["args"]["arch"] = "F2M"
        mirror["args"]["resume_checkpoint"] = None
        mirror["args"]["resume_optimizer"] = False
    mirror["horizontal_mirror_conversion"] = {
        "source_architecture": "F2",
        "target_architecture": "F2M",
        "fold_method": method,
        "symmetric_reference": False,
    }
    return symmetric, mirror


__all__ = [
    "F2M_FEATURE_ROWS",
    "F2M_TO_F2_CANONICAL",
    "F2M_TO_F2_MIRRORED",
    "F2_FEATURE_ROWS",
    "convert_f2_checkpoint",
    "fold_aux_weights",
    "fold_feature_rows",
    "symmetrize_feature_rows",
]
