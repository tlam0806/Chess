#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.quantized_nnue_architectures import (  # noqa: E402
    QUANTIZATION_CONVENTION_CHOICES,
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    PSQT_WEIGHT_SCALE,
    QuantizedSparseNnueArchitecture,
)
from chess_nnue.train_value import choose_device  # noqa: E402
from tools.train.train_nnue_architecture import cp_huber_loss, make_loader, validate_training_data  # noqa: E402
from tools.train.train_quantized_nnue_architecture import collate_quantized_sparse_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Float/quantized memorization test on a fixed NNUE subset")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=("auto", "cbin", "jsonl"), default="auto")
    parser.add_argument("--arch", choices=sorted(QUANTIZED_ARCHITECTURES), default="F2")
    parser.add_argument("--activation", default="screlu_all")
    parser.add_argument("--mode", choices=("float", "quantized"), default="float")
    parser.add_argument("--quantization-convention", choices=QUANTIZATION_CONVENTION_CHOICES, default=QUANTIZATION_CONVENTION_SCALE_CLEAN)
    parser.add_argument("--hidden-clip", type=int, default=181)
    parser.add_argument("--screlu-divisor", type=int, default=128)
    parser.add_argument("--feature-weight-scale", type=int, default=181)
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--psqt-weight-scale", type=int, default=PSQT_WEIGHT_SCALE)
    psqt_group = parser.add_mutually_exclusive_group()
    psqt_group.add_argument("--psqt", dest="use_psqt", action="store_true")
    psqt_group.add_argument("--no-psqt", dest="use_psqt", action="store_false")
    parser.set_defaults(use_psqt=True)
    parser.add_argument("--hidden-scales", type=int, nargs="+", default=(32, 16))
    parser.add_argument("--output-scale", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.samples <= 0 or args.steps <= 0 or args.report_every <= 0:
        raise ValueError("samples, steps, and report interval must be positive")
    config = QUANTIZED_ARCHITECTURES[args.arch]
    if len(args.hidden_scales) != len(config.hidden_sizes) - 1:
        raise ValueError("hidden scale count does not match architecture")
    device = choose_device(args.device)
    loader = make_loader(
        args.data, config.transform, validate_training_data(args.data, args.data_format),
        "train", 100, 98, 99, args.samples, 20260717, args.samples, 0, 0,
    )
    samples = next(iter(loader))
    if len(samples) != args.samples:
        raise RuntimeError(f"requested {args.samples} fixed records, got {len(samples)}")
    features, offsets, scores, target_normalized, _plies, _results = collate_quantized_sparse_batch(
        samples, device, config.board_feature_count, 1000.0
    )
    model = QuantizedSparseNnueArchitecture(
        config,
        hidden_clip=args.hidden_clip,
        feature_weight_scale=args.feature_weight_scale,
        linear_weight_scale=args.linear_weight_scale,
        output_weight_scale=args.output_weight_scale,
        screlu_divisor=args.screlu_divisor,
        use_psqt=args.use_psqt,
        psqt_weight_scale=args.psqt_weight_scale,
        psqt_master_scale_to_cp=1000.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    for step in range(args.steps + 1):
        forward = model.forward_float if args.mode == "float" else model.forward
        prediction_cp = forward(
            features,
            offsets,
            list(args.hidden_scales),
            args.output_scale,
            args.activation,
            args.quantization_convention,
        )
        if args.loss == "mse":
            loss = torch.nn.functional.mse_loss(
                prediction_cp / 1000.0,
                target_normalized,
            )
        else:
            loss = cp_huber_loss(prediction_cp / 1000.0, scores, 1000.0, 200.0)
        if step % args.report_every == 0 or step == args.steps:
            gradient_norms = {
                name: (float(parameter.grad.norm()) if parameter.grad is not None else None)
                for name, parameter in model.named_parameters()
            }
            print(json.dumps({
                "step": step,
                "mode": args.mode,
                "objective": args.loss,
                "loss": float(loss.detach()),
                "train_cp_mae": float((prediction_cp.detach() - target_normalized * 1000.0).abs().mean()),
                "prediction_min": float(prediction_cp.detach().min()),
                "prediction_max": float(prediction_cp.detach().max()),
                "gradient_norms_before_step": gradient_norms,
            }, separators=(",", ":")), flush=True)
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if args.mode == "quantized":
            model.clamp_quantized_weights()


if __name__ == "__main__":
    main()
