from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .train_value import (
    DEFAULT_TARGET_CLIP,
    DEFAULT_TARGET_SCALE,
    choose_device,
    collate_sparse_value_batch,
    evaluate_loss,
    make_loader,
    mean_absolute_error_cp,
    save_checkpoint,
)
from .training_targets import is_value_none_target
from .value_net import AUX_FEATURE_COUNT, ChessValueNet


def validate_sample(sample: dict[str, Any], line_number: int) -> None:
    if "features" not in sample or "aux" not in sample or "target" not in sample:
        raise ValueError(f"stdin:{line_number}: expected features, aux, target")
    if len(sample["aux"]) != AUX_FEATURE_COUNT:
        raise ValueError(f"stdin:{line_number}: aux must have length {AUX_FEATURE_COUNT}")
    if not sample["features"]:
        raise ValueError(f"stdin:{line_number}: features must not be empty")


def train_batch(
    model: ChessValueNet,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
    target_scale: float,
    target_clip: float,
) -> tuple[float, float, int]:
    model.train()
    feature_indices, offsets, aux, targets = collate_sparse_value_batch(
        samples,
        device=device,
        target_scale=target_scale,
        target_clip=target_clip,
    )

    optimizer.zero_grad(set_to_none=True)
    predictions = model(feature_indices, offsets, aux)
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()

    batch_size = targets.shape[0]
    mae_cp = float(((predictions.detach() - targets).abs() * target_scale).mean().cpu())
    return float(loss.detach().cpu()), mae_cp, batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse value net from JSONL stdin")
    parser.add_argument("--val", default=None, help="Validation JSONL path")
    parser.add_argument("--test", default=None, help="Held-out baseline JSONL path")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--output", default="models/value_net_stream.pt")
    parser.add_argument("--target-scale", type=float, default=DEFAULT_TARGET_SCALE)
    parser.add_argument("--target-clip", type=float, default=DEFAULT_TARGET_CLIP)
    parser.add_argument("--log-interval", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=1000000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")

    device = choose_device(args.device)
    val_loader = make_loader(args.val, args.batch_size, shuffle=False) if args.val else None
    test_loader = make_loader(args.test, args.batch_size, shuffle=False) if args.test else None

    model = ChessValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    batch: list[dict[str, Any]] = []
    total_samples = 0
    total_loss = 0.0
    total_mae_cp = 0.0
    next_log = args.log_interval
    next_eval = args.eval_interval

    def consume_batch(samples: list[dict[str, Any]]) -> None:
        nonlocal total_samples, total_loss, total_mae_cp, next_log, next_eval
        if not samples:
            return

        loss, mae_cp, batch_size = train_batch(
            model,
            optimizer,
            loss_fn,
            samples,
            device,
            args.target_scale,
            args.target_clip,
        )
        total_samples += batch_size
        total_loss += loss * batch_size
        total_mae_cp += mae_cp * batch_size

        if total_samples >= next_log:
            print(
                f"samples={total_samples} "
                f"train_loss={total_loss / total_samples:.6f} "
                f"train_mae_cp={total_mae_cp / total_samples:.2f}",
                flush=True,
            )
            while next_log <= total_samples:
                next_log += args.log_interval

        if total_samples >= next_eval:
            report_eval("eval")
            while next_eval <= total_samples:
                next_eval += args.eval_interval

    def report_eval(prefix: str) -> None:
        fields = [
            prefix,
            f"samples={total_samples}",
            f"train_loss={total_loss / max(1, total_samples):.6f}",
            f"train_mae_cp={total_mae_cp / max(1, total_samples):.2f}",
        ]
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate_loss(
                model,
                val_loader,
                loss_fn,
                device,
                target_scale=args.target_scale,
                target_clip=args.target_clip,
            )
            val_mae_cp = mean_absolute_error_cp(
                model,
                val_loader,
                device,
                target_scale=args.target_scale,
                target_clip=args.target_clip,
            )
            fields.extend([f"val_loss={val_loss:.6f}", f"val_mae_cp={val_mae_cp:.2f}"])
        if test_loader is not None:
            test_loss = evaluate_loss(
                model,
                test_loader,
                loss_fn,
                device,
                target_scale=args.target_scale,
                target_clip=args.target_clip,
            )
            test_mae_cp = mean_absolute_error_cp(
                model,
                test_loader,
                device,
                target_scale=args.target_scale,
                target_clip=args.target_clip,
            )
            fields.extend([f"test_loss={test_loss:.6f}", f"test_mae_cp={test_mae_cp:.2f}"])
        print(" ".join(fields), flush=True)
        save_checkpoint(
            args.output,
            model,
            optimizer,
            epoch=1,
            train_loss=total_loss / max(1, total_samples),
            val_loss=val_loss,
            target_scale=args.target_scale,
            target_clip=args.target_clip,
        )

    for line_number, line in enumerate(sys.stdin, 1):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        validate_sample(sample, line_number)
        if is_value_none_target(sample["target"]):
            continue
        batch.append(sample)

        if len(batch) == args.batch_size:
            consume_batch(batch)
            batch = []

    consume_batch(batch)
    if total_samples == 0:
        raise ValueError("stdin contains no samples")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report_eval("final")


if __name__ == "__main__":
    main()
