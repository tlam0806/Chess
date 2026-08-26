#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


EXPECTED_FEATURE_ROWS = {"F2": 6 * 2 * 64 * 64, "F2M": 6 * 2 * 32 * 64}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail unless a checkpoint is an exportable eight-head phase NNUE"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--arch", required=True, choices=sorted(EXPECTED_FEATURE_ROWS))
    parser.add_argument(
        "--phase-layout",
        choices=("independent", "shared_first"),
        default="independent",
    )
    parser.add_argument("--expected-train-samples", type=int)
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--allow-missing-ranking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected_metadata = {
        "architecture": args.arch,
        "phase_stacks": 8,
        "phase_layout": args.phase_layout,
        "component_supervision": "total",
        "label_mode": "total",
        "loss_type": "huber",
        "huber_delta": 200.0,
        "hidden_sizes": (256, 32, 32),
        "hidden_scales": [2, 8],
        "output_scale": 128,
        "activation": "screlu_relu16_all",
        "quantization_convention": "scale_clean",
    }
    for key, expected in expected_metadata.items():
        actual = checkpoint.get(key, "independent" if key == "phase_layout" else None)
        if key == "hidden_sizes" and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected:
            raise SystemExit(
                f"checkpoint metadata mismatch: {key} expected={expected!r} actual={actual!r}"
            )

    if args.expected_epoch is not None and int(checkpoint.get("epoch", -1)) != args.expected_epoch:
        raise SystemExit(
            "checkpoint epoch mismatch: "
            f"expected={args.expected_epoch} actual={checkpoint.get('epoch')!r}"
        )

    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise SystemExit("checkpoint has no model_state")
    feature_weights = state.get("feature_weights.weight")
    if not isinstance(feature_weights, torch.Tensor):
        raise SystemExit("checkpoint has no sparse feature tensor")
    expected_shape = (EXPECTED_FEATURE_ROWS[args.arch], 128)
    if tuple(feature_weights.shape) != expected_shape:
        raise SystemExit(
            f"feature shape mismatch: expected={expected_shape} actual={tuple(feature_weights.shape)}"
        )
    if "hidden_layers.0.weight" in state or "output.weight" in state:
        raise SystemExit("checkpoint contains a legacy shared dense head")

    output_norms: list[float] = []
    if args.phase_layout == "shared_first":
        shared_key = "shared_hidden_layers.0.weight"
        shared = state.get(shared_key)
        if not isinstance(shared, torch.Tensor) or not torch.isfinite(shared).all():
            raise SystemExit(f"missing or non-finite shared layer: {shared_key}")
    for phase in range(8):
        phase_layer_count = 1 if args.phase_layout == "shared_first" else 2
        required = tuple(
            f"phase_hidden_layers.{phase}.{layer}.weight"
            for layer in range(phase_layer_count)
        ) + (
            f"phase_outputs.{phase}.weight",
            f"phase_outputs.{phase}.bias",
        )
        for key in required:
            tensor = state.get(key)
            if not isinstance(tensor, torch.Tensor):
                raise SystemExit(f"missing phase tensor: {key}")
            if not torch.isfinite(tensor).all():
                raise SystemExit(f"non-finite values in phase tensor: {key}")
        norm = float(state[f"phase_outputs.{phase}.weight"].abs().sum())
        if norm == 0.0:
            raise SystemExit(f"phase {phase} output stayed at its zero initialization")
        output_norms.append(norm)

    metrics = checkpoint.get("metrics", {})
    selection_phase = metrics.get("selection", {}).get("phase", [])
    ranking_phase = metrics.get("ranking", {}).get("phase", [])
    splits = [("selection", selection_phase)]
    if ranking_phase or not args.allow_missing_ranking:
        splits.append(("ranking", ranking_phase))
    for split, values in splits:
        if len(values) != 8 or any(int(value.get("samples", 0)) == 0 for value in values):
            raise SystemExit(f"{split} does not cover all eight phase heads")

    if args.expected_train_samples is not None:
        actual_samples = int(metrics.get("train", {}).get("samples", -1))
        if actual_samples != args.expected_train_samples:
            raise SystemExit(
                "training sample count mismatch: "
                f"expected={args.expected_train_samples} actual={actual_samples}"
            )

    print(
        json.dumps(
            {
                "event": "phase_checkpoint_verified",
                "checkpoint": str(args.checkpoint),
                "architecture": args.arch,
                "phase_stacks": 8,
                "phase_layout": args.phase_layout,
                "feature_rows": expected_shape[0],
                "phase_output_l1_norms": output_norms,
                "status": "pass",
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
