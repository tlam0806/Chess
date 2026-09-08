#!/usr/bin/env python3
"""Convert a single-dense-stack quantized NNUE checkpoint for C++ inference.

The C++ evaluator has eight phase slots.  Replicating the single dense stack
into every slot is mathematically identical to the source model: phase choice
still happens, but every choice selects the same weights.  The shared feature
transformer and eight-bucket PSQT table are copied without modification.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch


PHASE_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def required_int(mapping: dict[str, Any], key: str) -> int:
    if key not in mapping:
        raise ValueError(f"missing metadata field: {key}")
    value = int(mapping[key])
    if value <= 0:
        raise ValueError(f"metadata field must be positive: {key}")
    return value


def main() -> None:
    args = parse_args()
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    if source.get("architecture") not in ("F2", "F2M"):
        raise ValueError("only F2/F2M checkpoints can be converted")
    if source.get("phase_stacks") not in (None, 1):
        raise ValueError("input checkpoint is not a single-stack model")
    if tuple(source.get("hidden_sizes", ())) != (256, 32, 32):
        raise ValueError("checkpoint hidden sizes must be 256,32,32")

    quantization = source.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("checkpoint is missing quantization metadata")
    psqt = source.get("psqt")
    if not isinstance(psqt, dict) or not psqt.get("enabled", False):
        raise ValueError("checkpoint must contain the eight-bucket PSQT table")
    if int(psqt.get("buckets", -1)) != PHASE_COUNT:
        raise ValueError("checkpoint must contain exactly eight PSQT buckets")

    state = source.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state")
    shared_keys = {
        "aux_feature_weights",
        "hidden1_bias",
        "feature_weights.weight",
        "psqt.weight",
    }
    dense_keys = {
        "hidden_layers.0.weight",
        "hidden_layers.0.bias",
        "hidden_layers.1.weight",
        "hidden_layers.1.bias",
        "output.weight",
        "output.bias",
    }
    missing = (shared_keys | dense_keys) - set(state)
    unexpected = set(state) - (shared_keys | dense_keys)
    if missing:
        raise ValueError(f"model_state is missing keys: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"model_state has unexpected keys: {sorted(unexpected)}")

    converted = copy.deepcopy(source)
    converted["phase_stacks"] = PHASE_COUNT
    converted["feature_weight_scale"] = required_int(
        quantization, "feature_weight_scale"
    )
    converted["linear_weight_scale"] = required_int(
        quantization, "linear_weight_scale"
    )
    converted["output_weight_scale"] = required_int(
        quantization, "output_weight_scale"
    )
    converted["psqt_weight_scale"] = required_int(psqt, "weight_scale")

    converted_state: dict[str, torch.Tensor] = {
        key: state[key].clone() for key in sorted(shared_keys)
    }
    for phase in range(PHASE_COUNT):
        for layer in range(2):
            for suffix in ("weight", "bias"):
                source_key = f"hidden_layers.{layer}.{suffix}"
                target_key = f"phase_hidden_layers.{phase}.{layer}.{suffix}"
                converted_state[target_key] = state[source_key].clone()
        for suffix in ("weight", "bias"):
            converted_state[f"phase_outputs.{phase}.{suffix}"] = state[
                f"output.{suffix}"
            ].clone()
    converted["model_state"] = converted_state
    converted["single_stack_phase_conversion"] = {
        "method": "replicate_dense_stack",
        "phase_count": PHASE_COUNT,
        "source": str(args.input),
        "logic_preserving": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(
        f"converted={args.output} phase_stacks={PHASE_COUNT} "
        "dense_stacks_identical=1"
    )


if __name__ == "__main__":
    main()
