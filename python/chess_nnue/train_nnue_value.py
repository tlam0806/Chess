from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .nnue_value_net import ChessNnueValueNet
from .train_value import (
    DEFAULT_TARGET_CLIP,
    DEFAULT_TARGET_SCALE,
    choose_device,
    evaluate_loss,
    make_loader,
    mean_absolute_error_cp,
    train_one_epoch,
)


def save_checkpoint(
    path: str | Path,
    model: ChessNnueValueNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    target_scale: float,
    target_clip: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "feature_count": model.feature_count,
            "aux_feature_count": model.aux_feature_count,
            "hidden_size": model.hidden_size,
            "target_scale": target_scale,
            "target_clip": target_clip,
            "architecture": "nnue_v1",
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train minimal NNUE-style value network")
    parser.add_argument("--train", required=True, help="Training JSONL path")
    parser.add_argument("--val", default=None, help="Validation JSONL path")
    parser.add_argument("--test", default=None, help="Held-out baseline JSONL path")
    parser.add_argument("--init-checkpoint", default=None, help="Checkpoint to initialize model weights from")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--output", default="models/nnue_value_net.pt")
    parser.add_argument("--target-scale", type=float, default=DEFAULT_TARGET_SCALE)
    parser.add_argument("--target-clip", type=float, default=DEFAULT_TARGET_CLIP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    train_loader = make_loader(args.train, args.batch_size, shuffle=True)
    val_loader = make_loader(args.val, args.batch_size, shuffle=False) if args.val else None
    test_loader = make_loader(args.test, args.batch_size, shuffle=False) if args.test else None

    model = ChessNnueValueNet().to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"initialized from {args.init_checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            target_scale=args.target_scale,
            target_clip=args.target_clip,
        )
        val_loss = (
            evaluate_loss(
                model,
                val_loader,
                loss_fn,
                device,
                target_scale=args.target_scale,
                target_clip=args.target_clip,
            )
            if val_loader
            else None
        )
        train_mae_cp = mean_absolute_error_cp(
            model,
            train_loader,
            device,
            target_scale=args.target_scale,
            target_clip=args.target_clip,
        )
        val_mae_cp = (
            mean_absolute_error_cp(
                model,
                val_loader,
                device,
                target_scale=args.target_scale,
                target_clip=args.target_clip,
            )
            if val_loader
            else None
        )
        test_loss = (
            evaluate_loss(
                model,
                test_loader,
                loss_fn,
                device,
                target_scale=args.target_scale,
                target_clip=args.target_clip,
            )
            if test_loader
            else None
        )
        test_mae_cp = (
            mean_absolute_error_cp(
                model,
                test_loader,
                device,
                target_scale=args.target_scale,
                target_clip=args.target_clip,
            )
            if test_loader
            else None
        )
        save_checkpoint(
            args.output,
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            target_scale=args.target_scale,
            target_clip=args.target_clip,
        )

        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.6f} "
            + (f"val_loss={val_loss:.6f} " if val_loss is not None else "")
            + f"train_mae_cp={train_mae_cp:.2f} "
            + (f"val_mae_cp={val_mae_cp:.2f}" if val_mae_cp is not None else "")
            + (
                f" test_loss={test_loss:.6f} test_mae_cp={test_mae_cp:.2f}"
                if test_loss is not None and test_mae_cp is not None
                else ""
            )
        )


if __name__ == "__main__":
    main()
