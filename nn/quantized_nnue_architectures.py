from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .nnue_architectures import (
    ARCHITECTURES as FLOAT_ARCHITECTURES,
    KING_BUCKET_COUNT,
    NnueArchitectureConfig,
    is_dual_accumulator_transform,
    transform_features,
)
from .value_net import AUX_FEATURE_COUNT


INT8_MIN = -127
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
PSQT_BUCKET_COUNT = 8
MAX_ACTIVE_BOARD_FEATURES = 64
# King-aware encodings activate two rows per physical piece. Keep the sum for
# every PSQT bucket inside int32 in all legal positions, not just each weight.
# Float32 parameters have a spacing of 4 at this magnitude; use an exactly
# representable symmetric bound so fake quantization can never round one step
# outside the declared int32-safe range.
PSQT_QUANT_MAX = ((INT32_MAX // MAX_ACTIVE_BOARD_FEATURES) // 4) * 4
PSQT_QUANT_MIN = -PSQT_QUANT_MAX
PSQT_WEIGHT_SCALE = 16
HIDDEN_CLIP = 255
FEATURE_WEIGHT_SCALE = 255
LINEAR_WEIGHT_SCALE = 64
OUTPUT_WEIGHT_SCALE = 16
SCRELU_DIVISOR = 255
ACTIVATION_RELU = "relu"
ACTIVATION_SCRELU_FIRST = "screlu_first"
ACTIVATION_SCRELU_ALL = "screlu_all"
ACTIVATION_SCRELU_RELU16_SCRELU = "screlu_relu16_screlu"
ACTIVATION_SCRELU_RELU16_ALL = "screlu_relu16_all"
ACTIVATION_CHOICES = (
    ACTIVATION_RELU,
    ACTIVATION_SCRELU_FIRST,
    ACTIVATION_SCRELU_ALL,
    ACTIVATION_SCRELU_RELU16_SCRELU,
    ACTIVATION_SCRELU_RELU16_ALL,
)
RELU16_CLIP = (1 << 16) - 1
QUANTIZATION_CONVENTION_LEGACY = "legacy"
QUANTIZATION_CONVENTION_SCALE_CLEAN = "scale_clean"
QUANTIZATION_CONVENTION_CHOICES = (
    QUANTIZATION_CONVENTION_LEGACY,
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
)


@dataclass(frozen=True)
class QuantizedNnueArchitectureConfig:
    name: str
    transform: str
    board_feature_count: int
    feature_count: int
    hidden_sizes: tuple[int, ...]

    @property
    def hidden1_size(self) -> int:
        return self.hidden_sizes[0]


_QUANTIZED_FLOAT_ARCHITECTURES = dict(FLOAT_ARCHITECTURES)
_QUANTIZED_FLOAT_ARCHITECTURES["A"] = NnueArchitectureConfig(
    "A",
    FLOAT_ARCHITECTURES["A"].transform,
    FLOAT_ARCHITECTURES["A"].feature_count,
    (256, 32, 32),
)
_QUANTIZED_FLOAT_ARCHITECTURES["B"] = NnueArchitectureConfig(
    "B",
    FLOAT_ARCHITECTURES["B"].transform,
    FLOAT_ARCHITECTURES["B"].feature_count,
    (128, 32, 32),
)
_QUANTIZED_FLOAT_ARCHITECTURES["F2_64"] = NnueArchitectureConfig(
    "F2_64",
    FLOAT_ARCHITECTURES["F2"].transform,
    FLOAT_ARCHITECTURES["F2"].feature_count,
    (256, 64, 64),
)


QUANTIZED_ARCHITECTURES: dict[str, QuantizedNnueArchitectureConfig] = {
    name: QuantizedNnueArchitectureConfig(
        name=config.name,
        transform=config.transform,
        board_feature_count=config.feature_count,
        feature_count=config.feature_count + AUX_FEATURE_COUNT,
        hidden_sizes=config.hidden_sizes,
    )
    for name, config in _QUANTIZED_FLOAT_ARCHITECTURES.items()
    if name in {
        "A", "B", "C", "D", "E", "F", "G", "H", "E2", "F2", "F2M", "F2_64"
    }
}


def transform_features_with_sparse_aux(
    raw_features: list[int],
    aux: list[int],
    config: QuantizedNnueArchitectureConfig,
) -> list[int]:
    if len(aux) != AUX_FEATURE_COUNT:
        raise ValueError(f"expected {AUX_FEATURE_COUNT} aux values, got {len(aux)}")
    features = transform_features(raw_features, config.transform)
    for index, value in enumerate(aux):
        if value:
            features.append(config.board_feature_count + index)
    features.sort()
    return features


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def trunc_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.trunc(x) - x).detach()


def fake_quantized_int(x: torch.Tensor, scale: int, qmin: int, qmax: int) -> torch.Tensor:
    scaled = x * float(scale)
    quantized = torch.clamp(round_ste(scaled), qmin, qmax)
    return quantized


def fake_int32_bias(x: torch.Tensor) -> torch.Tensor:
    return round_ste(x)


def activation_output_scale(
    input_scale: float,
    activation: str,
    layer_index: int,
    screlu_divisor: int,
    hidden_clip: int | None = None,
    quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
) -> float:
    kind = activation_kind(activation, layer_index)
    uses_screlu = kind == "screlu8"
    if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN:
        if hidden_clip is None or hidden_clip <= 0:
            raise ValueError("scale_clean activation scale requires a positive hidden_clip")
        # Every activation consumes a code in the fixed [0, hidden_clip]
        # domain. SCReLU therefore emits another fixed-scale code; its scale
        # does not inherit (and repeatedly square) the preceding accumulator
        # scale.
        if uses_screlu:
            return float(hidden_clip * hidden_clip) / float(screlu_divisor)
        if kind == "relu16":
            return input_scale
        return float(hidden_clip)
    if uses_screlu:
        return input_scale * input_scale / float(screlu_divisor)
    return input_scale


def activation_kind(activation: str, layer_index: int) -> str:
    if activation == ACTIVATION_RELU:
        return "relu8"
    if activation == ACTIVATION_SCRELU_FIRST:
        return "screlu8" if layer_index == 0 else "relu8"
    if activation == ACTIVATION_SCRELU_ALL:
        return "screlu8"
    if activation == ACTIVATION_SCRELU_RELU16_SCRELU:
        return "relu16" if layer_index == 1 else "screlu8"
    if activation == ACTIVATION_SCRELU_RELU16_ALL:
        return "screlu8" if layer_index == 0 else "relu16"
    raise ValueError(f"unknown activation: {activation}")


def activation_numeric_clip(
    activation: str,
    layer_index: int,
    hidden_clip: int,
) -> int:
    return RELU16_CLIP if activation_kind(activation, layer_index) == "relu16" else hidden_clip


def _scale_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _validate_quantization_convention(quantization_convention: str) -> None:
    if quantization_convention not in QUANTIZATION_CONVENTION_CHOICES:
        raise ValueError(f"unknown quantization convention: {quantization_convention}")


def _legacy_hidden_bias_scale(model: "QuantizedSparseNnueArchitecture") -> float:
    return float(model.hidden_clip * model.linear_weight_scale)


def _legacy_output_bias_scale(model: "QuantizedSparseNnueArchitecture") -> float:
    return float(model.output_weight_scale)


def quantization_scale_audit(
    model: "QuantizedSparseNnueArchitecture",
    activation: str,
    hidden_scales: list[int] | tuple[int, ...],
    output_scale: int,
    quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
) -> dict[str, Any]:
    model._validate_activation(activation)
    _validate_quantization_convention(quantization_convention)
    if len(hidden_scales) != len(model.hidden_layers):
        raise ValueError(
            f"expected {len(model.hidden_layers)} hidden scales, got {len(hidden_scales)}"
        )
    if output_scale <= 0:
        raise ValueError(f"output scale must be positive, got {output_scale}")

    warnings: list[str] = []
    layers: list[dict[str, Any]] = []

    first_acc_scale = float(model.feature_weight_scale)
    first_output_scale = activation_output_scale(
        first_acc_scale,
        activation,
        0,
        model.screlu_divisor,
        model.hidden_clip,
        quantization_convention,
    )
    first_input_ratio = _scale_ratio(first_acc_scale, float(model.hidden_clip))
    if (
        quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
        and first_input_ratio is not None
        and not math.isclose(first_input_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    ):
        warnings.append(
            f"feature_to_screlu_scale_mismatch ratio={first_input_ratio:.6g}"
        )
    layers.append(
        {
            "name": "feature_transformer",
            "input_activation_scale": 1.0,
            "weight_scale": float(model.feature_weight_scale),
            "accumulator_scale": first_acc_scale,
            "code_bias_scale": float(model.feature_weight_scale),
            "expected_bias_scale": first_acc_scale,
            "bias_scale_ratio": 1.0,
            "pre_activation_scale": first_acc_scale,
            "expected_activation_input_scale": float(model.hidden_clip),
            "activation_input_scale_ratio": first_input_ratio,
            "output_activation_scale": first_output_scale,
            "numeric_clip": float(activation_numeric_clip(activation, 0, model.hidden_clip)),
        }
    )

    activation_scale = first_output_scale
    for layer_index, hidden_scale in enumerate(hidden_scales, 1):
        if hidden_scale <= 0:
            raise ValueError(f"hidden scale must be positive, got {hidden_scale}")
        acc_scale = activation_scale * float(model.linear_weight_scale)
        code_hidden_bias_scale = (
            acc_scale
            if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
            else _legacy_hidden_bias_scale(model)
        )
        pre_activation_scale = acc_scale / float(hidden_scale)
        output_activation_scale = activation_output_scale(
            pre_activation_scale,
            activation,
            layer_index,
            model.screlu_divisor,
            model.hidden_clip,
            quantization_convention,
        )
        activation_input_ratio = _scale_ratio(
            pre_activation_scale, float(model.hidden_clip)
        )
        ratio = _scale_ratio(code_hidden_bias_scale, acc_scale)
        if ratio is not None and not math.isclose(ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            warnings.append(
                f"hidden_{layer_index}_bias_scale_mismatch ratio={ratio:.6g}"
            )
        layers.append(
            {
                "name": f"hidden_{layer_index}",
                "input_activation_scale": activation_scale,
                "weight_scale": float(model.linear_weight_scale),
                "accumulator_scale": acc_scale,
                "code_bias_scale": code_hidden_bias_scale,
                "expected_bias_scale": acc_scale,
                "bias_scale_ratio": ratio,
                "hidden_rescale": float(hidden_scale),
                "pre_activation_scale": pre_activation_scale,
                "expected_activation_input_scale": float(model.hidden_clip),
                "activation_input_scale_ratio": activation_input_ratio,
                "output_activation_scale": output_activation_scale,
                "numeric_clip": float(
                    activation_numeric_clip(activation, layer_index, model.hidden_clip)
                ),
            }
        )
        activation_scale = output_activation_scale

    output_acc_scale = activation_scale * float(model.output_weight_scale)
    output_bias_scale = (
        output_acc_scale
        if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
        else _legacy_output_bias_scale(model)
    )
    output_ratio = _scale_ratio(output_bias_scale, output_acc_scale)
    if output_ratio is not None and not math.isclose(output_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        warnings.append(f"output_bias_scale_mismatch ratio={output_ratio:.6g}")
    layers.append(
        {
            "name": "output",
            "input_activation_scale": activation_scale,
            "weight_scale": float(model.output_weight_scale),
            "accumulator_scale": output_acc_scale,
            "code_bias_scale": output_bias_scale,
            "expected_bias_scale": output_acc_scale,
            "bias_scale_ratio": output_ratio,
            "output_rescale": float(output_scale),
            "output_value_scale": output_acc_scale / float(output_scale),
        }
    )
    return {
        "activation": activation,
        "hidden_clip": model.hidden_clip,
        "screlu_divisor": model.screlu_divisor,
        "feature_weight_scale": model.feature_weight_scale,
        "linear_weight_scale": model.linear_weight_scale,
        "output_weight_scale": model.output_weight_scale,
        "psqt_enabled": model.use_psqt,
        "psqt_weight_scale": model.psqt_weight_scale if model.use_psqt else None,
        "psqt_master_scale_to_cp": (
            model.psqt_master_scale_to_cp if model.use_psqt else None
        ),
        "psqt_effective_quantization_scale": (
            model.psqt_weight_scale * model.psqt_master_scale_to_cp
            if model.use_psqt
            else None
        ),
        "psqt_weight_dtype": "int32" if model.use_psqt else None,
        "psqt_buckets": PSQT_BUCKET_COUNT if model.use_psqt else 0,
        "psqt_bucket_formula": "clamp((piece_count - 1) // 4, 0, 7)",
        "psqt_perspective_formula": "(stm - opponent) / 2",
        "hidden_scales": list(hidden_scales),
        "output_scale": output_scale,
        "quantization_convention": quantization_convention,
        "layers": layers,
        "warnings": warnings,
    }


class QuantizedSparseNnueArchitecture(nn.Module):
    def __init__(
        self,
        config: QuantizedNnueArchitectureConfig,
        hidden_clip: int = HIDDEN_CLIP,
        feature_weight_scale: int = FEATURE_WEIGHT_SCALE,
        linear_weight_scale: int = LINEAR_WEIGHT_SCALE,
        output_weight_scale: int = OUTPUT_WEIGHT_SCALE,
        screlu_divisor: int = SCRELU_DIVISOR,
        use_psqt: bool = False,
        psqt_weight_scale: int = PSQT_WEIGHT_SCALE,
        psqt_master_scale_to_cp: float = 1.0,
    ) -> None:
        super().__init__()
        if not config.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        if hidden_clip <= 0:
            raise ValueError("hidden_clip must be positive")
        if feature_weight_scale <= 0:
            raise ValueError("feature_weight_scale must be positive")
        if linear_weight_scale <= 0:
            raise ValueError("linear_weight_scale must be positive")
        if output_weight_scale <= 0:
            raise ValueError("output_weight_scale must be positive")
        if screlu_divisor <= 0:
            raise ValueError("screlu_divisor must be positive")
        if psqt_weight_scale <= 0:
            raise ValueError("psqt_weight_scale must be positive")
        if psqt_master_scale_to_cp <= 0.0:
            raise ValueError("psqt_master_scale_to_cp must be positive")
        self.config = config
        self.architecture = config.name
        self.transform = config.transform
        self.feature_count = config.feature_count
        self.board_feature_count = config.board_feature_count
        self.hidden1_size = config.hidden1_size
        self.hidden_sizes = config.hidden_sizes
        self.hidden_clip = hidden_clip
        self.feature_weight_scale = feature_weight_scale
        self.linear_weight_scale = linear_weight_scale
        self.output_weight_scale = output_weight_scale
        self.screlu_divisor = screlu_divisor
        self.use_psqt = bool(use_psqt)
        self.psqt_weight_scale = psqt_weight_scale
        # The trainable PSQT table is expressed in normalized target units.
        # New CP-regression models use target_scale here (normally 1000), so
        # a 40 CP coefficient is stored as 0.04.  Legacy checkpoints use 1.0.
        # Exported/C++ integer weights are unchanged because quantization
        # multiplies the master value by this conversion factor.
        self.psqt_master_scale_to_cp = float(psqt_master_scale_to_cp)
        self.default_hidden_scales = [linear_weight_scale] * max(0, len(config.hidden_sizes) - 1)
        self.default_output_scale = output_weight_scale
        self.dual_accumulator = is_dual_accumulator_transform(self.transform)
        if self.dual_accumulator:
            if self.board_feature_count % 2 != 0 or self.hidden1_size % 2 != 0:
                raise ValueError("dual accumulator feature and hidden sizes must be even")
            embedding_count = self.board_feature_count // 2
            embedding_size = self.hidden1_size // 2
            self.aux_feature_weights = nn.Parameter(
                torch.empty(AUX_FEATURE_COUNT, self.hidden1_size)
            )
        else:
            embedding_count = config.feature_count
            embedding_size = config.hidden1_size
            self.register_parameter("aux_feature_weights", None)

        self.feature_weights = nn.EmbeddingBag(
            num_embeddings=embedding_count,
            embedding_dim=embedding_size,
            mode="sum",
            include_last_offset=False,
        )
        self.hidden1_bias = nn.Parameter(torch.zeros(embedding_size))
        self.hidden_layers = nn.ModuleList(
            nn.Linear(config.hidden_sizes[index], config.hidden_sizes[index + 1])
            for index in range(len(config.hidden_sizes) - 1)
        )
        self.output = nn.Linear(config.hidden_sizes[-1], 1)
        psqt_embedding_count = (
            self.board_feature_count // 2
            if self.dual_accumulator
            else self.board_feature_count
        )
        self.psqt = (
            nn.EmbeddingBag(
                # PSQT is attached to the exact same feature rows as the NN
                # transformer. Dual accumulators share both tables.
                num_embeddings=psqt_embedding_count,
                embedding_dim=PSQT_BUCKET_COUNT,
                mode="sum",
                include_last_offset=False,
            )
            if self.use_psqt
            else None
        )
        self.reset_parameters()

    def _validate_activation(self, activation: str) -> None:
        if activation not in ACTIVATION_CHOICES:
            raise ValueError(f"unknown activation: {activation}")

    def _activate_hidden(
        self,
        value: torch.Tensor,
        activation: str,
        layer_index: int,
    ) -> torch.Tensor:
        self._validate_activation(activation)
        kind = activation_kind(activation, layer_index)
        numeric_clip = activation_numeric_clip(activation, layer_index, self.hidden_clip)
        clipped = torch.clamp(value, 0.0, float(numeric_clip))
        if kind == "screlu8":
            return trunc_ste((clipped * clipped) / float(self.screlu_divisor))
        return clipped

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_weights.weight, mean=0.0, std=0.02)
        if self.aux_feature_weights is not None:
            nn.init.normal_(self.aux_feature_weights, mean=0.0, std=0.02)
        nn.init.zeros_(self.hidden1_bias)
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.05)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.20)
        nn.init.zeros_(self.output.bias)
        if self.psqt is not None:
            # Start new PSQT models as the exact old positional model. This
            # also makes the branch safe to add before distillation training.
            nn.init.zeros_(self.psqt.weight)

    def quantized_psqt_weight(self) -> torch.Tensor:
        if self.psqt is None:
            raise RuntimeError("PSQT is disabled for this model")
        return fake_quantized_int(
            self.psqt.weight,
            self.psqt_weight_scale * self.psqt_master_scale_to_cp,
            PSQT_QUANT_MIN,
            PSQT_QUANT_MAX,
        )

    def _psqt_perspective_indices_and_masks(
        self,
        feature_indices: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one feature stream for STM and one for the opponent.

        Dual F2 inputs already contain the two normalized perspectives in
        their two halves. Other transforms are converted to the opponent's
        normalized piece side and vertically flipped square here.
        """
        indices = feature_indices.long()
        board_mask = indices < self.board_feature_count

        if self.dual_accumulator:
            half = self.board_feature_count // 2
            mapped = torch.remainder(indices, half)
            stm_valid = board_mask & (indices < half)
            opp_valid = board_mask & (indices >= half)
            stm_indices = mapped
            opp_indices = mapped
        elif self.transform == "base768":
            piece_square = torch.remainder(indices, 64)
            value = torch.div(indices, 64, rounding_mode="floor")
            piece_side = torch.remainder(value, 2)
            piece = torch.div(value, 2, rounding_mode="floor")
            stm_indices = indices
            opp_indices = ((piece * 2 + (1 - piece_side)) * 64) + (piece_square ^ 56)
            stm_valid = board_mask
            opp_valid = board_mask
        elif self.transform == "king_bucket":
            piece_square = torch.remainder(indices, 64)
            value = torch.div(indices, 64, rounding_mode="floor")
            king_bucket_value = torch.remainder(value, KING_BUCKET_COUNT)
            value = torch.div(value, KING_BUCKET_COUNT, rounding_mode="floor")
            king_context = torch.remainder(value, 2)
            value = torch.div(value, 2, rounding_mode="floor")
            piece_side = torch.remainder(value, 2)
            piece = torch.div(value, 2, rounding_mode="floor")
            stm_indices = indices
            opp_indices = piece
            opp_indices = opp_indices * 2 + (1 - piece_side)
            opp_indices = opp_indices * 2 + (1 - king_context)
            opp_indices = opp_indices * KING_BUCKET_COUNT + (king_bucket_value ^ 12)
            opp_indices = opp_indices * 64 + (piece_square ^ 56)
            stm_valid = board_mask
            opp_valid = board_mask
        elif self.transform == "full_king_square":
            piece_square = torch.remainder(indices, 64)
            value = torch.div(indices, 64, rounding_mode="floor")
            king_square = torch.remainder(value, 64)
            value = torch.div(value, 64, rounding_mode="floor")
            king_context = torch.remainder(value, 2)
            value = torch.div(value, 2, rounding_mode="floor")
            piece_side = torch.remainder(value, 2)
            piece = torch.div(value, 2, rounding_mode="floor")
            stm_indices = indices
            opp_indices = piece
            opp_indices = opp_indices * 2 + (1 - piece_side)
            opp_indices = opp_indices * 2 + (1 - king_context)
            opp_indices = opp_indices * 64 + (king_square ^ 56)
            opp_indices = opp_indices * 64 + (piece_square ^ 56)
            stm_valid = board_mask
            opp_valid = board_mask
        else:
            raise ValueError(f"unknown feature transform: {self.transform}")

        zeros = torch.zeros_like(indices)
        stm_indices = torch.where(stm_valid, stm_indices, zeros)
        opp_indices = torch.where(opp_valid, opp_indices, zeros)
        return (
            stm_indices,
            stm_valid.to(dtype),
            opp_indices,
            opp_valid.to(dtype),
        )

    def _physical_piece_count(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Count pieces once although king-aware encodings have two rows each."""
        indices = feature_indices.long()
        board_mask = indices < self.board_feature_count
        if self.transform == "base768":
            primary = board_mask
        elif self.transform == "king_bucket":
            value = torch.div(indices, 64 * KING_BUCKET_COUNT, rounding_mode="floor")
            primary = board_mask & (torch.remainder(value, 2) == 0)
        elif self.transform == "full_king_square":
            value = torch.div(indices, 64 * 64, rounding_mode="floor")
            primary = board_mask & (torch.remainder(value, 2) == 0)
        elif is_dual_accumulator_transform(self.transform):
            primary = board_mask & (indices < self.board_feature_count // 2)
        else:
            raise ValueError(f"unknown feature transform: {self.transform}")

        lengths = torch.diff(
            torch.cat(
                (
                    offsets.long(),
                    torch.tensor(
                        [indices.numel()],
                        dtype=torch.long,
                        device=indices.device,
                    ),
                )
            )
        )
        bag_indices = torch.repeat_interleave(
            torch.arange(offsets.numel(), device=indices.device),
            lengths,
        )
        counts = torch.zeros(offsets.numel(), dtype=torch.long, device=indices.device)
        counts.scatter_add_(0, bag_indices, primary.long())
        return counts

    def psqt_bucket_indices(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        pieces = self._physical_piece_count(feature_indices, offsets)
        # Legal positions always contain both kings, but clamping makes corrupt
        # or synthetic test records deterministic as well.
        return torch.clamp(
            torch.div(pieces - 1, 4, rounding_mode="floor"),
            0,
            PSQT_BUCKET_COUNT - 1,
        )

    def psqt_perspective_accumulators(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        *,
        quantized: bool = True,
    ) -> torch.Tensor:
        if self.psqt is None:
            return torch.zeros(
                (offsets.numel(), 2, PSQT_BUCKET_COUNT),
                dtype=self.feature_weights.weight.dtype,
                device=feature_indices.device,
            )
        weights = (
            self.quantized_psqt_weight()
            if quantized
            else self.psqt.weight
            * float(self.psqt_weight_scale)
            * self.psqt_master_scale_to_cp
        )
        stm_indices, stm_mask, opp_indices, opp_mask = (
            self._psqt_perspective_indices_and_masks(feature_indices, weights.dtype)
        )
        stm = nn.functional.embedding_bag(
            stm_indices,
            weights,
            offsets.long(),
            mode="sum",
            per_sample_weights=stm_mask,
            include_last_offset=False,
        )
        opp = nn.functional.embedding_bag(
            opp_indices,
            weights,
            offsets.long(),
            mode="sum",
            per_sample_weights=opp_mask,
            include_last_offset=False,
        )
        return torch.stack((stm, opp), dim=1)

    def psqt_accumulator(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        *,
        quantized: bool = True,
    ) -> torch.Tensor:
        perspectives = self.psqt_perspective_accumulators(
            feature_indices,
            offsets,
            quantized=quantized,
        )
        selected = self.psqt_bucket_indices(feature_indices, offsets)
        bucket_index = selected.view(-1, 1, 1).expand(-1, 2, 1)
        values = perspectives.gather(2, bucket_index).squeeze(2)
        return values[:, 0] - values[:, 1]

    def quantized_feature_weight(self) -> torch.Tensor:
        return fake_quantized_int(
            self.feature_weights.weight,
            self.feature_weight_scale,
            INT8_MIN,
            INT8_MAX,
        )

    def quantized_aux_feature_weight(self) -> torch.Tensor:
        if self.aux_feature_weights is None:
            raise RuntimeError("separate aux weights only exist for dual accumulators")
        return fake_quantized_int(
            self.aux_feature_weights,
            self.feature_weight_scale,
            INT8_MIN,
            INT8_MAX,
        )

    def quantized_hidden1_bias(self) -> torch.Tensor:
        # The feature-transformer bias shares the accumulator unit used by the
        # quantized feature rows, not the activation clipping range.
        bias = fake_int32_bias(self.hidden1_bias * float(self.feature_weight_scale))
        return torch.cat((bias, bias), dim=0) if self.dual_accumulator else bias

    def first_hidden_accumulator(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        indices = feature_indices.long()
        offsets = offsets.long()
        if not self.dual_accumulator:
            hidden = nn.functional.embedding_bag(
                indices,
                self.quantized_feature_weight(),
                offsets,
                mode="sum",
                include_last_offset=False,
            )
            return hidden + self.quantized_hidden1_bias()

        half_features = self.board_feature_count // 2
        shared_indices = torch.remainder(indices, half_features)
        first_mask = (indices < half_features).to(self.feature_weights.weight.dtype)
        second_mask = (
            (indices >= half_features) & (indices < self.board_feature_count)
        ).to(self.feature_weights.weight.dtype)
        shared_weight = self.quantized_feature_weight()
        first = nn.functional.embedding_bag(
            shared_indices,
            shared_weight,
            offsets,
            mode="sum",
            per_sample_weights=first_mask,
            include_last_offset=False,
        )
        second = nn.functional.embedding_bag(
            shared_indices,
            shared_weight,
            offsets,
            mode="sum",
            per_sample_weights=second_mask,
            include_last_offset=False,
        )
        aux_indices = torch.clamp(
            indices - self.board_feature_count,
            min=0,
            max=AUX_FEATURE_COUNT - 1,
        )
        aux_mask = (indices >= self.board_feature_count).to(self.feature_weights.weight.dtype)
        aux_hidden = nn.functional.embedding_bag(
            aux_indices,
            self.quantized_aux_feature_weight(),
            offsets,
            mode="sum",
            per_sample_weights=aux_mask,
            include_last_offset=False,
        )
        return torch.cat((first, second), dim=1) + aux_hidden + self.quantized_hidden1_bias()

    def float_first_hidden_accumulator(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Feature-transformer accumulator without integer rounding/clipping."""
        indices = feature_indices.long()
        offsets = offsets.long()
        feature_weight = self.feature_weights.weight * float(self.feature_weight_scale)
        bias = self.hidden1_bias * float(self.feature_weight_scale)
        if not self.dual_accumulator:
            hidden = nn.functional.embedding_bag(
                indices,
                feature_weight,
                offsets,
                mode="sum",
                include_last_offset=False,
            )
            return hidden + bias

        half_features = self.board_feature_count // 2
        shared_indices = torch.remainder(indices, half_features)
        first_mask = (indices < half_features).to(feature_weight.dtype)
        second_mask = (
            (indices >= half_features) & (indices < self.board_feature_count)
        ).to(feature_weight.dtype)
        first = nn.functional.embedding_bag(
            shared_indices,
            feature_weight,
            offsets,
            mode="sum",
            per_sample_weights=first_mask,
            include_last_offset=False,
        )
        second = nn.functional.embedding_bag(
            shared_indices,
            feature_weight,
            offsets,
            mode="sum",
            per_sample_weights=second_mask,
            include_last_offset=False,
        )
        aux_indices = torch.clamp(
            indices - self.board_feature_count,
            min=0,
            max=AUX_FEATURE_COUNT - 1,
        )
        aux_mask = (indices >= self.board_feature_count).to(feature_weight.dtype)
        aux_weight = self.aux_feature_weights * float(self.feature_weight_scale)
        aux_hidden = nn.functional.embedding_bag(
            aux_indices,
            aux_weight,
            offsets,
            mode="sum",
            per_sample_weights=aux_mask,
            include_last_offset=False,
        )
        duplicated_bias = torch.cat((bias, bias), dim=0)
        return torch.cat((first, second), dim=1) + aux_hidden + duplicated_bias

    def quantized_layer_weight(self, layer: nn.Linear) -> torch.Tensor:
        return fake_quantized_int(
            layer.weight,
            self.linear_weight_scale,
            INT8_MIN,
            INT8_MAX,
        )

    def quantized_layer_bias(self, layer: nn.Linear, bias_scale: float | None = None) -> torch.Tensor:
        if bias_scale is None:
            bias_scale = _legacy_hidden_bias_scale(self)
        return fake_int32_bias(layer.bias * float(bias_scale))

    def quantized_output_weight(self) -> torch.Tensor:
        return fake_quantized_int(
            self.output.weight.squeeze(0),
            self.output_weight_scale,
            INT8_MIN,
            INT8_MAX,
        )

    def quantized_output_bias(self, bias_scale: float | None = None) -> torch.Tensor:
        if bias_scale is None:
            bias_scale = _legacy_output_bias_scale(self)
        return fake_int32_bias(self.output.bias.squeeze(0) * float(bias_scale))

    @torch.no_grad()
    def clamp_quantized_weights(self) -> None:
        self.feature_weights.weight.clamp_(
            INT8_MIN / float(self.feature_weight_scale),
            INT8_MAX / float(self.feature_weight_scale),
        )
        if self.aux_feature_weights is not None:
            self.aux_feature_weights.clamp_(
                INT8_MIN / float(self.feature_weight_scale),
                INT8_MAX / float(self.feature_weight_scale),
            )
        for layer in self.hidden_layers:
            layer.weight.clamp_(
                INT8_MIN / float(self.linear_weight_scale),
                INT8_MAX / float(self.linear_weight_scale),
            )
        self.output.weight.clamp_(
            INT8_MIN / float(self.output_weight_scale),
            INT8_MAX / float(self.output_weight_scale),
        )
        if self.psqt is not None:
            psqt_quantization_scale = (
                float(self.psqt_weight_scale) * self.psqt_master_scale_to_cp
            )
            self.psqt.weight.clamp_(
                PSQT_QUANT_MIN / psqt_quantization_scale,
                PSQT_QUANT_MAX / psqt_quantization_scale,
            )

    def forward(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        hidden_scales: list[int] | tuple[int, ...] | None = None,
        output_scale: int | None = None,
        activation: str = ACTIVATION_RELU,
        quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
    ) -> torch.Tensor:
        _validate_quantization_convention(quantization_convention)
        if feature_indices.dtype != torch.long:
            feature_indices = feature_indices.long()
        if offsets.dtype != torch.long:
            offsets = offsets.long()
        if hidden_scales is None:
            hidden_scales = self.default_hidden_scales
        if output_scale is None:
            output_scale = self.default_output_scale
        if output_scale <= 0:
            raise ValueError(f"output scale must be positive, got {output_scale}")

        hidden = self._activate_hidden(
            self.first_hidden_accumulator(feature_indices, offsets),
            activation,
            0,
        )
        if (
            quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
            and self.feature_weight_scale != self.hidden_clip
        ):
            raise ValueError(
                "scale_clean requires feature_weight_scale == hidden_clip so "
                "the first SCReLU input uses the declared clip-domain scale"
            )
        activation_scale = activation_output_scale(
            float(self.feature_weight_scale),
            activation,
            0,
            self.screlu_divisor,
            self.hidden_clip,
            quantization_convention,
        )

        if len(hidden_scales) != len(self.hidden_layers):
            raise ValueError(f"expected {len(self.hidden_layers)} hidden scales, got {len(hidden_scales)}")
        for layer_index, (scale, layer) in enumerate(zip(hidden_scales, self.hidden_layers), 1):
            if scale <= 0:
                raise ValueError(f"hidden scale must be positive, got {scale}")
            accumulator_scale = activation_scale * float(self.linear_weight_scale)
            bias_scale = (
                accumulator_scale
                if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
                else None
            )
            acc = nn.functional.linear(
                hidden,
                self.quantized_layer_weight(layer),
                self.quantized_layer_bias(layer, bias_scale),
            )
            hidden = self._activate_hidden(trunc_ste(acc / float(scale)), activation, layer_index)
            activation_scale = activation_output_scale(
                accumulator_scale / float(scale),
                activation,
                layer_index,
                self.screlu_divisor,
                self.hidden_clip,
                quantization_convention,
            )

        output_bias_scale = (
            activation_scale * float(self.output_weight_scale)
            if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
            else None
        )
        raw = (
            hidden * self.quantized_output_weight().unsqueeze(0)
        ).sum(dim=1) + self.quantized_output_bias(output_bias_scale)
        positional = trunc_ste(raw / float(output_scale))
        if self.psqt is None:
            return positional
        psqt = trunc_ste(
            self.psqt_accumulator(feature_indices, offsets)
            / float(2 * self.psqt_weight_scale)
        )
        return positional + psqt

    def forward_float(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        hidden_scales: list[int] | tuple[int, ...] | None = None,
        output_scale: int | None = None,
        activation: str = ACTIVATION_RELU,
        quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
    ) -> torch.Tensor:
        """Continuous counterpart of ``forward`` for quantization-loss audits.

        It preserves the checkpoint's declared scale convention and hidden
        activation clipping, but removes weight/bias rounding, integer weight
        clipping, and activation/output truncation.
        """
        _validate_quantization_convention(quantization_convention)
        self._validate_activation(activation)
        if hidden_scales is None:
            hidden_scales = self.default_hidden_scales
        if output_scale is None:
            output_scale = self.default_output_scale
        if output_scale <= 0:
            raise ValueError(f"output scale must be positive, got {output_scale}")
        if len(hidden_scales) != len(self.hidden_layers):
            raise ValueError(
                f"expected {len(self.hidden_layers)} hidden scales, got {len(hidden_scales)}"
            )
        if (
            quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
            and self.feature_weight_scale != self.hidden_clip
        ):
            raise ValueError(
                "scale_clean requires feature_weight_scale == hidden_clip so "
                "the first SCReLU input uses the declared clip-domain scale"
            )

        def activate_continuous(value: torch.Tensor, layer_index: int) -> torch.Tensor:
            kind = activation_kind(activation, layer_index)
            numeric_clip = activation_numeric_clip(
                activation, layer_index, self.hidden_clip
            )
            clipped = torch.clamp(value, 0.0, float(numeric_clip))
            if kind == "screlu8":
                return (clipped * clipped) / float(self.screlu_divisor)
            return clipped

        hidden = activate_continuous(
            self.float_first_hidden_accumulator(feature_indices, offsets),
            0,
        )
        activation_scale = activation_output_scale(
            float(self.feature_weight_scale),
            activation,
            0,
            self.screlu_divisor,
            self.hidden_clip,
            quantization_convention,
        )
        for layer_index, (scale, layer) in enumerate(
            zip(hidden_scales, self.hidden_layers), 1
        ):
            if scale <= 0:
                raise ValueError(f"hidden scale must be positive, got {scale}")
            accumulator_scale = activation_scale * float(self.linear_weight_scale)
            bias_scale = (
                accumulator_scale
                if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
                else _legacy_hidden_bias_scale(self)
            )
            acc = nn.functional.linear(
                hidden,
                layer.weight * float(self.linear_weight_scale),
                layer.bias * float(bias_scale),
            )
            hidden = activate_continuous(acc / float(scale), layer_index)
            activation_scale = activation_output_scale(
                accumulator_scale / float(scale),
                activation,
                layer_index,
                self.screlu_divisor,
                self.hidden_clip,
                quantization_convention,
            )

        output_bias_scale = (
            activation_scale * float(self.output_weight_scale)
            if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
            else _legacy_output_bias_scale(self)
        )
        raw = nn.functional.linear(
            hidden,
            self.output.weight * float(self.output_weight_scale),
            self.output.bias * float(output_bias_scale),
        ).squeeze(1)
        positional = raw / float(output_scale)
        if self.psqt is None:
            return positional
        psqt = self.psqt_accumulator(
            feature_indices,
            offsets,
            quantized=False,
        ) / float(2 * self.psqt_weight_scale)
        return positional + psqt

    @torch.no_grad()
    def collect_hidden_accumulators(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        hidden_scales: list[int] | tuple[int, ...],
        activation: str = ACTIVATION_RELU,
    ) -> list[torch.Tensor]:
        if len(hidden_scales) != len(self.hidden_layers):
            raise ValueError(f"expected {len(self.hidden_layers)} hidden scales, got {len(hidden_scales)}")
        hidden = self._activate_hidden(
            self.first_hidden_accumulator(feature_indices, offsets),
            activation,
            0,
        )

        accumulators: list[torch.Tensor] = []
        for layer_index, (scale, layer) in enumerate(zip(hidden_scales, self.hidden_layers), 1):
            acc = nn.functional.linear(
                hidden,
                torch.clamp(
                    torch.round(layer.weight * float(self.linear_weight_scale)),
                    INT8_MIN,
                    INT8_MAX,
                ),
                torch.round(layer.bias * float(self.hidden_clip * self.linear_weight_scale)),
            )
            accumulators.append(acc.detach())
            hidden = self._activate_hidden(torch.trunc(acc / float(scale)), activation, layer_index)
        return accumulators


class PhaseStackQuantizedNnueArchitecture(QuantizedSparseNnueArchitecture):
    """Quantized NNUE with one shared transformer and eight dense phase stacks.

    The phase formula intentionally matches both Stockfish's layer-stack
    selection and this project's PSQT bucket selection:
    ``clamp((piece_count - 1) // 4, 0, 7)``.
    """

    def __init__(
        self,
        config: QuantizedNnueArchitectureConfig,
        phase_layout: str = "independent",
        **kwargs: Any,
    ) -> None:
        if phase_layout not in ("independent", "shared_first"):
            raise ValueError(f"unknown phase layout: {phase_layout}")
        kwargs["use_psqt"] = True
        super().__init__(config, **kwargs)
        template_layers = self.hidden_layers
        template_output = self.output
        del self.hidden_layers
        del self.output
        self.phase_layout = phase_layout
        if phase_layout == "shared_first":
            if len(template_layers) < 2:
                raise ValueError("shared_first requires at least two dense hidden layers")
            self.shared_hidden_layers = nn.ModuleList([template_layers[0]])
            phase_template = nn.ModuleList(list(template_layers[1:]))
        else:
            self.shared_hidden_layers = nn.ModuleList()
            phase_template = template_layers
        self.phase_hidden_layers = nn.ModuleList(
            [phase_template]
            + [copy.deepcopy(phase_template) for _ in range(PSQT_BUCKET_COUNT - 1)]
        )
        self.phase_outputs = nn.ModuleList(
            [template_output]
            + [copy.deepcopy(template_output) for _ in range(PSQT_BUCKET_COUNT - 1)]
        )

    @property
    def dense_layer_count(self) -> int:
        return len(self.shared_hidden_layers) + len(self.phase_hidden_layers[0])

    def layers_for_phase(self, phase: int) -> tuple[nn.Linear, ...]:
        if not 0 <= phase < len(self.phase_hidden_layers):
            raise IndexError(f"phase outside 0..7: {phase}")
        return tuple(self.shared_hidden_layers) + tuple(self.phase_hidden_layers[phase])

    def unique_layers_at_depth(self, layer_index: int) -> tuple[nn.Linear, ...]:
        if not 0 <= layer_index < self.dense_layer_count:
            raise IndexError(f"dense layer outside model: {layer_index}")
        if layer_index < len(self.shared_hidden_layers):
            return (self.shared_hidden_layers[layer_index],)
        phase_index = layer_index - len(self.shared_hidden_layers)
        return tuple(layers[phase_index] for layers in self.phase_hidden_layers)

    def _quantized_output_weight_for(self, output: nn.Linear) -> torch.Tensor:
        return fake_quantized_int(
            output.weight.squeeze(0),
            self.output_weight_scale,
            INT8_MIN,
            INT8_MAX,
        )

    def _quantized_output_bias_for(
        self,
        output: nn.Linear,
        bias_scale: float | None,
    ) -> torch.Tensor:
        if bias_scale is None:
            bias_scale = float(self.output_weight_scale)
        return fake_int32_bias(output.bias.squeeze(0) * float(bias_scale))

    def forward_positional(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        hidden_scales: list[int] | tuple[int, ...],
        output_scale: int,
        activation: str,
        quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
    ) -> torch.Tensor:
        _validate_quantization_convention(quantization_convention)
        if len(hidden_scales) != self.dense_layer_count:
            raise ValueError(
                f"expected {self.dense_layer_count} hidden scales, got {len(hidden_scales)}"
            )
        if output_scale <= 0 or any(scale <= 0 for scale in hidden_scales):
            raise ValueError("hidden and output scales must be positive")
        if (
            quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
            and self.feature_weight_scale != self.hidden_clip
        ):
            raise ValueError(
                "scale_clean requires feature_weight_scale == hidden_clip"
            )

        first_hidden = self._activate_hidden(
            self.first_hidden_accumulator(feature_indices, offsets), activation, 0
        )
        buckets = self.psqt_bucket_indices(feature_indices, offsets)
        positional = torch.zeros(
            offsets.numel(), dtype=first_hidden.dtype, device=first_hidden.device
        )

        for phase, output in enumerate(self.phase_outputs):
            selected = torch.nonzero(buckets == phase, as_tuple=False).squeeze(1)
            if selected.numel() == 0:
                continue
            hidden = first_hidden.index_select(0, selected)
            activation_scale = activation_output_scale(
                float(self.feature_weight_scale),
                activation,
                0,
                self.screlu_divisor,
                self.hidden_clip,
                quantization_convention,
            )
            for layer_index, (scale, layer) in enumerate(
                zip(hidden_scales, self.layers_for_phase(phase)), 1
            ):
                accumulator_scale = activation_scale * float(self.linear_weight_scale)
                bias_scale = (
                    accumulator_scale
                    if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
                    else None
                )
                accumulator = nn.functional.linear(
                    hidden,
                    self.quantized_layer_weight(layer),
                    self.quantized_layer_bias(layer, bias_scale),
                )
                hidden = self._activate_hidden(
                    trunc_ste(accumulator / float(scale)), activation, layer_index
                )
                activation_scale = activation_output_scale(
                    accumulator_scale / float(scale),
                    activation,
                    layer_index,
                    self.screlu_divisor,
                    self.hidden_clip,
                    quantization_convention,
                )

            output_bias_scale = (
                activation_scale * float(self.output_weight_scale)
                if quantization_convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
                else None
            )
            raw = (
                hidden * self._quantized_output_weight_for(output).unsqueeze(0)
            ).sum(dim=1) + self._quantized_output_bias_for(output, output_bias_scale)
            phase_output = trunc_ste(raw / float(output_scale))
            positional = torch.index_copy(positional, 0, selected, phase_output)
        return positional

    def forward_components(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        hidden_scales: list[int] | tuple[int, ...],
        output_scale: int,
        activation: str,
        quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positional = self.forward_positional(
            feature_indices,
            offsets,
            hidden_scales,
            output_scale,
            activation,
            quantization_convention,
        )
        psqt = trunc_ste(
            self.psqt_accumulator(feature_indices, offsets)
            / float(2 * self.psqt_weight_scale)
        )
        return positional, psqt

    def forward(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
        hidden_scales: list[int] | tuple[int, ...] | None = None,
        output_scale: int | None = None,
        activation: str = ACTIVATION_RELU,
        quantization_convention: str = QUANTIZATION_CONVENTION_LEGACY,
    ) -> torch.Tensor:
        if hidden_scales is None:
            hidden_scales = self.default_hidden_scales
        if output_scale is None:
            output_scale = self.default_output_scale
        positional, psqt = self.forward_components(
            feature_indices,
            offsets,
            hidden_scales,
            output_scale,
            activation,
            quantization_convention,
        )
        return positional + psqt

    @torch.no_grad()
    def clamp_quantized_weights(self) -> None:
        self.feature_weights.weight.clamp_(
            INT8_MIN / float(self.feature_weight_scale),
            INT8_MAX / float(self.feature_weight_scale),
        )
        if self.aux_feature_weights is not None:
            self.aux_feature_weights.clamp_(
                INT8_MIN / float(self.feature_weight_scale),
                INT8_MAX / float(self.feature_weight_scale),
            )
        for layer in self.shared_hidden_layers:
            layer.weight.clamp_(
                INT8_MIN / float(self.linear_weight_scale),
                INT8_MAX / float(self.linear_weight_scale),
            )
        for layers in self.phase_hidden_layers:
            for layer in layers:
                layer.weight.clamp_(
                    INT8_MIN / float(self.linear_weight_scale),
                    INT8_MAX / float(self.linear_weight_scale),
                )
        for output in self.phase_outputs:
            output.weight.clamp_(
                INT8_MIN / float(self.output_weight_scale),
                INT8_MAX / float(self.output_weight_scale),
            )
        assert self.psqt is not None
        psqt_quantization_scale = (
            float(self.psqt_weight_scale) * self.psqt_master_scale_to_cp
        )
        self.psqt.weight.clamp_(
            PSQT_QUANT_MIN / psqt_quantization_scale,
            PSQT_QUANT_MAX / psqt_quantization_scale,
        )


def scale_from_positive_percentile(
    values: torch.Tensor,
    percentile: float,
    clip: int = HIDDEN_CLIP,
) -> int:
    positive = values[values > 0]
    if positive.numel() == 0:
        return 1
    quantile = torch.quantile(positive.float(), percentile / 100.0).item()
    if not math.isfinite(quantile) or quantile <= 0:
        return 1
    return max(1, int(math.ceil(quantile / float(clip))))


__all__ = [
    "HIDDEN_CLIP",
    "FEATURE_WEIGHT_SCALE",
    "ACTIVATION_CHOICES",
    "ACTIVATION_RELU",
    "ACTIVATION_SCRELU_ALL",
    "ACTIVATION_SCRELU_FIRST",
    "ACTIVATION_SCRELU_RELU16_ALL",
    "ACTIVATION_SCRELU_RELU16_SCRELU",
    "INT8_MAX",
    "INT8_MIN",
    "LINEAR_WEIGHT_SCALE",
    "OUTPUT_WEIGHT_SCALE",
    "PSQT_BUCKET_COUNT",
    "PSQT_QUANT_MAX",
    "PSQT_QUANT_MIN",
    "PSQT_WEIGHT_SCALE",
    "QUANTIZATION_CONVENTION_CHOICES",
    "QUANTIZATION_CONVENTION_LEGACY",
    "QUANTIZATION_CONVENTION_SCALE_CLEAN",
    "SCRELU_DIVISOR",
    "RELU16_CLIP",
    "QUANTIZED_ARCHITECTURES",
    "QuantizedNnueArchitectureConfig",
    "PhaseStackQuantizedNnueArchitecture",
    "activation_kind",
    "activation_numeric_clip",
    "QuantizedSparseNnueArchitecture",
    "activation_output_scale",
    "quantization_scale_audit",
    "scale_from_positive_percentile",
    "transform_features_with_sparse_aux",
    "trunc_ste",
]
