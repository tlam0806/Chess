#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.quantized_nnue_architectures import (  # noqa: E402
    ACTIVATION_SCRELU_RELU16_ALL,
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    PhaseStackQuantizedNnueArchitecture,
    activation_output_scale,
)
from tools.train.train_phase_component_nnue import collate, make_total_loader  # noqa: E402


MAGIC = b"QPHNUE1\0"
LEGACY_VERSION = 1
HORIZONTAL_MIRROR_VERSION = 2
FEATURE_ROWS_BY_ARCHITECTURE = {
    "F2": 6 * 2 * 64 * 64,
    "F2M": 6 * 2 * 32 * 64,
}
PERSPECTIVE_SIZE = 128
DENSE_INPUT_SIZE = 256
HIDDEN2_SIZE = 32
HIDDEN3_SIZE = 32
PHASES = 8
PSQT_BUCKETS = 8
AUX_FEATURES = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an eight-phase F2 checkpoint for C++ search inference"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parity-data", type=Path)
    parser.add_argument("--parity-output", type=Path)
    parser.add_argument("--parity-samples", type=int, default=2048)
    return parser.parse_args()


def numpy_bytes(tensor: torch.Tensor, dtype: str) -> bytes:
    return tensor.detach().cpu().numpy().astype(dtype, copy=False).tobytes()


def require_integer_range(
    name: str,
    tensor: torch.Tensor,
    minimum: int,
    maximum: int,
) -> None:
    if tensor.numel() == 0:
        raise ValueError(f"{name} is empty")
    actual_min = int(tensor.min().item())
    actual_max = int(tensor.max().item())
    if actual_min < minimum or actual_max > maximum:
        raise OverflowError(
            f"{name} range [{actual_min}, {actual_max}] is outside "
            f"[{minimum}, {maximum}]"
        )


def compact_board_fen(board: bytes, aux: list[int], side_to_move: str) -> str:
    if len(board) != 32 or len(aux) != AUX_FEATURES:
        raise ValueError("invalid compact position for parity fixture")
    codes: list[int] = []
    for value in board:
        codes.extend((value & 0x0F, value >> 4))
    if side_to_move not in ("w", "b"):
        raise ValueError("side_to_move must be w or b")
    normalized_codes = codes
    codes = [0] * 64
    for square, code in enumerate(normalized_codes):
        actual_square = square if side_to_move == "w" else square ^ 56
        if side_to_move == "w" or code == 0:
            codes[actual_square] = code
        elif code <= 6:
            codes[actual_square] = code + 6
        else:
            codes[actual_square] = code - 6
    white = "PNBRQK"
    black = "pnbrqk"
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            code = codes[rank * 8 + file]
            if code == 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += white[code - 1] if code <= 6 else black[code - 7]
        if empty:
            row += str(empty)
        ranks.append(row)
    castling = ""
    if side_to_move == "w":
        castling += "K" if aux[0] else ""
        castling += "Q" if aux[1] else ""
        castling += "k" if aux[2] else ""
        castling += "q" if aux[3] else ""
    else:
        castling += "K" if aux[2] else ""
        castling += "Q" if aux[3] else ""
        castling += "k" if aux[0] else ""
        castling += "q" if aux[1] else ""
    castling = castling or "-"
    ep = "-"
    if aux[4]:
        files = [index for index, active in enumerate(aux[5:13]) if active]
        if len(files) != 1:
            raise ValueError("invalid en-passant aux features")
        ep_rank = "6" if side_to_move == "w" else "3"
        ep = f"{chr(ord('a') + files[0])}{ep_rank}"
    return f"{'/'.join(ranks)} {side_to_move} {castling} {ep} 0 1"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = str(checkpoint.get("architecture"))
    if architecture not in FEATURE_ROWS_BY_ARCHITECTURE:
        raise ValueError("C++ phase evaluator requires architecture F2 or F2M")
    if int(checkpoint.get("phase_stacks", -1)) != PHASES:
        raise ValueError("C++ phase evaluator requires exactly eight phase stacks")
    if tuple(checkpoint.get("hidden_sizes", ())) != (256, 32, 32):
        raise ValueError("checkpoint hidden sizes must be 256,32,32")
    if checkpoint.get("activation") != ACTIVATION_SCRELU_RELU16_ALL:
        raise ValueError("checkpoint activation must be SCReLU-ReLU16-ReLU16")
    if checkpoint.get("quantization_convention") != QUANTIZATION_CONVENTION_SCALE_CLEAN:
        raise ValueError("checkpoint must use the scale-clean convention")
    hidden_scales = [int(value) for value in checkpoint["hidden_scales"]]
    if len(hidden_scales) != 2 or any(value <= 0 for value in hidden_scales):
        raise ValueError("checkpoint must contain two positive hidden scales")
    output_scale = int(checkpoint["output_scale"])
    if output_scale <= 0:
        raise ValueError("output scale must be positive")

    config = QUANTIZED_ARCHITECTURES[architecture]
    phase_layout = str(checkpoint.get("phase_layout", "independent"))
    model = PhaseStackQuantizedNnueArchitecture(
        config,
        phase_layout=phase_layout,
        hidden_clip=int(checkpoint["hidden_clip"]),
        feature_weight_scale=int(checkpoint["feature_weight_scale"]),
        linear_weight_scale=int(checkpoint["linear_weight_scale"]),
        output_weight_scale=int(checkpoint["output_weight_scale"]),
        screlu_divisor=int(checkpoint["screlu_divisor"]),
        psqt_weight_scale=int(checkpoint["psqt_weight_scale"]),
        psqt_master_scale_to_cp=float(checkpoint["target_scale"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    feature_rows = FEATURE_ROWS_BY_ARCHITECTURE[architecture]
    if model.feature_weights.num_embeddings != feature_rows:
        raise ValueError(f"unexpected {architecture} feature-row count")
    if model.feature_weights.embedding_dim != PERSPECTIVE_SIZE:
        raise ValueError("unexpected F2 perspective accumulator size")
    if model.feature_weight_scale != model.hidden_clip:
        raise ValueError("scale-clean export requires feature scale == hidden clip")

    accumulator_bias = model.quantized_hidden1_bias()[:PERSPECTIVE_SIZE]
    feature_weights = model.quantized_feature_weight()
    aux_weights = model.quantized_aux_feature_weight()
    psqt_weights = model.quantized_psqt_weight()
    require_integer_range("accumulator_bias", accumulator_bias, -(2**31), 2**31 - 1)
    require_integer_range("feature_weights", feature_weights, -128, 127)
    require_integer_range("aux_weights", aux_weights, -128, 127)
    require_integer_range("psqt_weights", psqt_weights, -(2**31), 2**31 - 1)
    if int(psqt_weights.abs().max().item()) > (2**31 - 1) // 64:
        raise OverflowError("64 PSQT features could overflow an int32 accumulator")

    activation_scale = activation_output_scale(
        float(model.feature_weight_scale),
        str(checkpoint["activation"]),
        0,
        model.screlu_divisor,
        model.hidden_clip,
        str(checkpoint["quantization_convention"]),
    )
    dense_tensors: list[tuple[torch.Tensor, ...]] = []
    for phase, output in enumerate(model.phase_outputs):
        phase_values: list[torch.Tensor] = []
        phase_activation_scale = activation_scale
        for layer_index, (scale, layer) in enumerate(
            zip(hidden_scales, model.layers_for_phase(phase)), 1
        ):
            accumulator_scale = phase_activation_scale * model.linear_weight_scale
            bias = model.quantized_layer_bias(layer, accumulator_scale)
            weight = model.quantized_layer_weight(layer)
            require_integer_range(
                f"phase_hidden_{layer_index}_bias", bias, -(2**31), 2**31 - 1
            )
            require_integer_range(
                f"phase_hidden_{layer_index}_weight", weight, -128, 127
            )
            phase_values.extend((bias, weight))
            phase_activation_scale = activation_output_scale(
                accumulator_scale / scale,
                str(checkpoint["activation"]),
                layer_index,
                model.screlu_divisor,
                model.hidden_clip,
                str(checkpoint["quantization_convention"]),
            )
        output_bias_scale = phase_activation_scale * model.output_weight_scale
        output_bias = model._quantized_output_bias_for(output, output_bias_scale)
        output_weight = model._quantized_output_weight_for(output)
        require_integer_range("phase_output_bias", output_bias, -(2**63), 2**63 - 1)
        require_integer_range("phase_output_weight", output_weight, -128, 127)
        phase_values.extend((output_bias, output_weight))
        dense_tensors.append(tuple(phase_values))

    header = (
        HORIZONTAL_MIRROR_VERSION if architecture == "F2M" else LEGACY_VERSION,
        feature_rows,
        PERSPECTIVE_SIZE,
        DENSE_INPUT_SIZE,
        HIDDEN2_SIZE,
        HIDDEN3_SIZE,
        PHASES,
        PSQT_BUCKETS,
        AUX_FEATURES,
        int(model.hidden_clip),
        int(model.screlu_divisor),
        hidden_scales[0],
        hidden_scales[1],
        output_scale,
        int(model.feature_weight_scale),
        int(model.linear_weight_scale),
        int(model.output_weight_scale),
        int(model.psqt_weight_scale),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(struct.pack("<8s18I", MAGIC, *header))
        stream.write(numpy_bytes(accumulator_bias, "<i4"))
        stream.write(numpy_bytes(feature_weights, "i1"))
        stream.write(numpy_bytes(aux_weights, "i1"))
        stream.write(numpy_bytes(psqt_weights, "<i4"))
        for hidden2_bias, hidden2_weight, hidden3_bias, hidden3_weight, output_bias, output_weight in dense_tensors:
            stream.write(numpy_bytes(hidden2_bias, "<i4"))
            stream.write(numpy_bytes(hidden2_weight, "i1"))
            stream.write(numpy_bytes(hidden3_bias, "<i4"))
            stream.write(numpy_bytes(hidden3_weight, "i1"))
            stream.write(struct.pack("<q", int(output_bias.item())))
            stream.write(numpy_bytes(output_weight, "i1"))

    if (args.parity_data is None) != (args.parity_output is None):
        raise ValueError("--parity-data and --parity-output must be provided together")
    if args.parity_samples <= 0:
        raise ValueError("--parity-samples must be positive")
    if args.parity_data is not None and args.parity_output is not None:
        loader = make_total_loader(
            args.parity_data,
            config.transform,
            args.parity_samples,
            0,
            20260720,
            0,
            min(512, args.parity_samples),
            0,
        )
        lines: list[str] = []
        for samples in loader:
            features, offsets, _psqt, _target = collate(
                samples, config.board_feature_count, torch.device("cpu")
            )
            predictions = model(
                features,
                offsets,
                hidden_scales,
                output_scale,
                str(checkpoint["activation"]),
                str(checkpoint["quantization_convention"]),
            )
            for sample, prediction in zip(samples, predictions.tolist()):
                for side_to_move in ("w", "b"):
                    lines.append(
                        f"{compact_board_fen(sample['board'], sample['aux'], side_to_move)}"
                        f"\t{int(prediction)}"
                    )
        expected_lines = 2 * args.parity_samples
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"expected {expected_lines} parity samples, got {len(lines)}"
            )
        args.parity_output.parent.mkdir(parents=True, exist_ok=True)
        args.parity_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"wrote={args.output} bytes={args.output.stat().st_size} "
        f"architecture={architecture} phases=8 activation={checkpoint['activation']} "
        f"layout={phase_layout} hs={hidden_scales} os={output_scale} "
        f"loss={checkpoint.get('loss_type')}"
    )


if __name__ == "__main__":
    main()
