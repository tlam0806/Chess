#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.quantized_nnue_architectures import (  # noqa: E402
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    activation_output_scale,
)
from nn.train_value import choose_device  # noqa: E402
from tools.evaluate_quantized_nnue_checkpoint import load_model  # noqa: E402
from tools.train_nnue_architecture import make_loader, validate_training_data  # noqa: E402
from tools.train_quantized_nnue_architecture import collate_quantized_sparse_batch  # noqa: E402


MAGIC = b"QNNUEF1\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Python accumulators and quantized tensors for C++ inference parity"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-format", choices=("auto", "cbin", "jsonl"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def as_numpy(tensor: torch.Tensor, dtype: str) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype, copy=False)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("sample/batch counts must be positive and workers non-negative")
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    model.eval()
    architecture = str(checkpoint["architecture"])
    config = QUANTIZED_ARCHITECTURES[architecture]
    convention = str(checkpoint["quantization_convention"])
    activation = str(checkpoint["activation"])
    hidden_scales = [int(value) for value in checkpoint["hidden_scales"]]
    output_scale = int(checkpoint["output_scale"])
    if architecture != "F2" or config.hidden_sizes != (256, 32, 32):
        raise ValueError("candidate fixture currently requires F2 with hidden sizes 256,32,32")
    if activation != "screlu_all":
        raise ValueError("candidate fixture currently requires screlu_all")
    if len(hidden_scales) != 2:
        raise ValueError("candidate fixture requires exactly two dense hidden scales")
    if (
        convention == QUANTIZATION_CONVENTION_SCALE_CLEAN
        and model.feature_weight_scale != model.hidden_clip
    ):
        raise ValueError("scale_clean fixture requires feature scale equal to hidden clip")

    activation_scale = activation_output_scale(
        float(model.feature_weight_scale),
        activation,
        0,
        model.screlu_divisor,
        model.hidden_clip,
        convention,
    )
    if convention == QUANTIZATION_CONVENTION_SCALE_CLEAN:
        hidden_bias_scale = activation_scale * float(model.linear_weight_scale)
        output_bias_scale = activation_scale * float(model.output_weight_scale)
    else:
        hidden_bias_scale = float(model.hidden_clip * model.linear_weight_scale)
        output_bias_scale = float(model.output_weight_scale)
    hidden2_weight = model.quantized_layer_weight(model.hidden_layers[0])
    hidden2_bias = model.quantized_layer_bias(
        model.hidden_layers[0], hidden_bias_scale
    )
    hidden3_weight = model.quantized_layer_weight(model.hidden_layers[1])
    hidden3_bias = model.quantized_layer_bias(
        model.hidden_layers[1], hidden_bias_scale
    )
    output_weight = model.quantized_output_weight()
    output_bias = model.quantized_output_bias(output_bias_scale)
    if model.use_psqt:
        psqt_weight = model.quantized_psqt_weight()
        if psqt_weight.min() < -(2**31) or psqt_weight.max() >= 2**31:
            raise OverflowError("PSQT weight does not fit int32")

    for name, tensor in (
        ("hidden2_bias", hidden2_bias),
        ("hidden3_bias", hidden3_bias),
    ):
        if tensor.min() < -(2**31) or tensor.max() >= 2**31:
            raise OverflowError(f"{name} does not fit int32")
    output_bias_value = int(output_bias.item())
    if output_bias_value < -(2**63) or output_bias_value >= 2**63:
        raise OverflowError("output bias does not fit int64")

    loader = make_loader(
        args.data,
        config.transform,
        validate_training_data(args.data, args.data_format),
        args.split,
        100,
        98,
        99,
        args.max_samples,
        args.seed,
        args.batch_size,
        args.workers,
        0,
    )
    accumulator_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    quantized_chunks: list[torch.Tensor] = []
    float_chunks: list[torch.Tensor] = []
    psqt_chunks: list[torch.Tensor] = []
    piece_count_chunks: list[torch.Tensor] = []
    target_scale = float(checkpoint["target_scale"])
    for samples in loader:
        features, offsets, _scores, targets, _plies, _results = (
            collate_quantized_sparse_batch(
                samples,
                device,
                config.board_feature_count,
                target_scale,
            )
        )
        accumulator_chunks.append(
            model.first_hidden_accumulator(features, offsets).cpu()
        )
        psqt_chunks.append(
            model.psqt_perspective_accumulators(features, offsets).cpu()
        )
        piece_count_chunks.append(model._physical_piece_count(features, offsets).cpu())
        target_chunks.append((targets * target_scale).cpu())
        quantized_chunks.append(
            model(
                features,
                offsets,
                hidden_scales,
                output_scale,
                activation,
                convention,
            ).cpu()
        )
        float_chunks.append(
            model.forward_float(
                features,
                offsets,
                hidden_scales,
                output_scale,
                activation,
                convention,
            ).cpu()
        )

    accumulators = as_numpy(torch.cat(accumulator_chunks), "<i4")
    targets = as_numpy(torch.cat(target_chunks), "<f4")
    quantized = as_numpy(torch.cat(quantized_chunks), "<i4")
    floating = as_numpy(torch.cat(float_chunks), "<f4")
    psqt_accumulators_tensor = torch.cat(psqt_chunks)
    if (
        psqt_accumulators_tensor.min() < -(2**31)
        or psqt_accumulators_tensor.max() >= 2**31
    ):
        raise OverflowError("PSQT accumulator does not fit int32")
    psqt_accumulators = as_numpy(psqt_accumulators_tensor, "<i4")
    piece_counts = as_numpy(torch.cat(piece_count_chunks), "<u4")
    count = int(accumulators.shape[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8s12I",
                MAGIC,
                2,
                256,
                32,
                32,
                int(model.hidden_clip),
                int(model.screlu_divisor),
                hidden_scales[0],
                hidden_scales[1],
                output_scale,
                count,
                int(model.use_psqt),
                int(model.psqt_weight_scale),
            )
        )
        stream.write(as_numpy(hidden2_bias, "<i4").tobytes())
        stream.write(as_numpy(hidden2_weight, "i1").tobytes())
        stream.write(as_numpy(hidden3_bias, "<i4").tobytes())
        stream.write(as_numpy(hidden3_weight, "i1").tobytes())
        stream.write(struct.pack("<q", output_bias_value))
        stream.write(as_numpy(output_weight, "<i2").tobytes())
        for index in range(count):
            stream.write(accumulators[index].tobytes())
            stream.write(psqt_accumulators[index].tobytes())
            stream.write(struct.pack("<I", int(piece_counts[index])))
            stream.write(
                struct.pack(
                    "<fif",
                    float(targets[index]),
                    int(quantized[index]),
                    float(floating[index]),
                )
            )
    print(
        f"wrote {args.output} samples={count} clip={model.hidden_clip} "
        f"divisor={model.screlu_divisor} hs={hidden_scales} os={output_scale}"
    )


if __name__ == "__main__":
    main()
