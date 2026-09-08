from __future__ import annotations

import torch


STOCKFISH_VALUE_NONE = 32002
TRAINING_PIPELINE_VERSION = 3
STOCKFISH_SCORE_CP_NUMERATOR = 100.0
STOCKFISH_SCORE_CP_DENOMINATOR = 208.0
STOCKFISH_WDL_MAX_CP = 2000.0
DEFAULT_WDL_LOSS_EXPONENT = 2.6
DEFAULT_SCORE_LAMBDA = 1.0

_WDL_A = (-3.68389304, 30.07065921, -60.52878723, 149.53378557)
_WDL_B = (-2.0181857, 15.85685038, -29.83452023, 47.59078827)
_WDL_B_TRAINING_MULTIPLIER = 1.5


def is_value_none_target(target: int | float) -> bool:
    return float(target) == float(STOCKFISH_VALUE_NONE)


def stockfish_score_to_cp(score: torch.Tensor) -> torch.Tensor:
    cp = score * (STOCKFISH_SCORE_CP_NUMERATOR / STOCKFISH_SCORE_CP_DENOMINATOR)
    return torch.clamp(cp, -STOCKFISH_WDL_MAX_CP, STOCKFISH_WDL_MAX_CP)


def _wdl_parameters(ply: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = torch.clamp(ply, min=0.0, max=240.0) / 64.0
    a = ((_WDL_A[0] * m + _WDL_A[1]) * m + _WDL_A[2]) * m + _WDL_A[3]
    b = ((_WDL_B[0] * m + _WDL_B[1]) * m + _WDL_B[2]) * m + _WDL_B[3]
    return a, b * _WDL_B_TRAINING_MULTIPLIER


def expected_score_from_cp(score_cp: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
    score_cp = torch.clamp(score_cp, -STOCKFISH_WDL_MAX_CP, STOCKFISH_WDL_MAX_CP)
    a, b = _wdl_parameters(ply)
    win = torch.sigmoid((score_cp - a) / b)
    loss = torch.sigmoid((-score_cp - a) / b)
    return 0.5 * (1.0 + win - loss)


def outcome_expected_score(result: torch.Tensor) -> torch.Tensor:
    if torch.any((result < -1.0) | (result > 1.0)):
        raise ValueError("game result must be -1, 0, or 1")
    return 0.5 * (result + 1.0)


def stockfish_wdl_loss(
    prediction_cp: torch.Tensor,
    target_score: torch.Tensor,
    ply: torch.Tensor,
    result: torch.Tensor,
    score_lambda: float = DEFAULT_SCORE_LAMBDA,
    exponent: float = DEFAULT_WDL_LOSS_EXPONENT,
) -> torch.Tensor:
    if not 0.0 <= score_lambda <= 1.0:
        raise ValueError("score_lambda must be in [0, 1]")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    target_cp = stockfish_score_to_cp(target_score)
    prediction_expected = expected_score_from_cp(prediction_cp, ply)
    target_expected = expected_score_from_cp(target_cp, ply)
    if score_lambda < 1.0:
        target_expected = (
            score_lambda * target_expected
            + (1.0 - score_lambda) * outcome_expected_score(result)
        )
    return (prediction_expected - target_expected).abs().pow(exponent).mean()
