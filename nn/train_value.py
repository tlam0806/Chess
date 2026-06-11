from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .value_net import AUX_FEATURE_COUNT, ChessValueNet, make_batch


DEFAULT_TARGET_SCALE = 1000.0
DEFAULT_TARGET_CLIP = 1000.0


class SparseValueDataset(Dataset):
    """JSONL dataset for sparse board features.

    Expected line format:
        {"features": [1, 2, ...], "aux": [0, ...], "target": 35.0}

    target is a centipawn-like value from side-to-move POV.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.samples: list[dict[str, Any]] = []

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                self._validate_sample(sample, line_number)
                self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"{self.path} contains no samples")

    def _validate_sample(self, sample: dict[str, Any], line_number: int) -> None:
        if "features" not in sample or "aux" not in sample or "target" not in sample:
            raise ValueError(f"{self.path}:{line_number}: expected features, aux, target")
        if len(sample["aux"]) != AUX_FEATURE_COUNT:
            raise ValueError(
                f"{self.path}:{line_number}: aux must have length {AUX_FEATURE_COUNT}"
            )
        if not sample["features"]:
            raise ValueError(f"{self.path}:{line_number}: features must not be empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def collate_sparse_value_batch(
    samples: list[dict[str, Any]],
    device: torch.device | str | None = None,
    target_scale: float = DEFAULT_TARGET_SCALE,
    target_clip: float = DEFAULT_TARGET_CLIP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_features = [sample["features"] for sample in samples]
    batch_aux = [sample["aux"] for sample in samples]
    targets = torch.tensor(
        [normalize_target_cp(sample["target"], target_scale, target_clip) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    feature_indices, offsets, aux = make_batch(batch_features, batch_aux, device=device)
    return feature_indices, offsets, aux, targets


def normalize_target_cp(target_cp: float, target_scale: float, target_clip: float) -> float:
    clipped = max(-target_clip, min(target_clip, float(target_cp)))
    return clipped / target_scale


def denormalize_target_cp(target_normalized: float, target_scale: float) -> float:
    return float(target_normalized) * target_scale


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: ChessValueNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    target_scale: float = DEFAULT_TARGET_SCALE,
    target_clip: float = DEFAULT_TARGET_CLIP,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for samples in loader:
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
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def mean_absolute_error_cp(
    model: ChessValueNet,
    loader: DataLoader,
    device: torch.device,
    target_scale: float = DEFAULT_TARGET_SCALE,
    target_clip: float = DEFAULT_TARGET_CLIP,
) -> float:
    model.eval()
    total_error = 0.0
    total_samples = 0

    for samples in loader:
        feature_indices, offsets, aux, targets = collate_sparse_value_batch(
            samples,
            device=device,
            target_scale=target_scale,
            target_clip=target_clip,
        )
        predictions = model(feature_indices, offsets, aux)
        errors_cp = (predictions - targets).abs() * target_scale

        total_error += float(errors_cp.sum().detach().cpu())
        total_samples += targets.shape[0]

    return total_error / total_samples


@torch.no_grad()
def evaluate_loss(
    model: ChessValueNet,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_scale: float = DEFAULT_TARGET_SCALE,
    target_clip: float = DEFAULT_TARGET_CLIP,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for samples in loader:
        feature_indices, offsets, aux, targets = collate_sparse_value_batch(
            samples,
            device=device,
            target_scale=target_scale,
            target_clip=target_clip,
        )
        predictions = model(feature_indices, offsets, aux)
        loss = loss_fn(predictions, targets)

        batch_size = targets.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def save_checkpoint(
    path: str | Path,
    model: ChessValueNet,
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
        },
        path,
    )


def make_loader(path: str | Path, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SparseValueDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda samples: samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse chess value network")
    parser.add_argument("--train", required=True, help="Training JSONL path")
    parser.add_argument("--val", default=None, help="Validation JSONL path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--output", default="models/value_net.pt")
    parser.add_argument("--target-scale", type=float, default=DEFAULT_TARGET_SCALE)
    parser.add_argument("--target-clip", type=float, default=DEFAULT_TARGET_CLIP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    train_loader = make_loader(args.train, args.batch_size, shuffle=True)
    val_loader = make_loader(args.val, args.batch_size, shuffle=False) if args.val else None

    model = ChessValueNet().to(device)
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

        if val_loss is None:
            print(f"epoch {epoch}: train_loss={train_loss:.6f} train_mae_cp={train_mae_cp:.2f}")
        else:
            print(
                f"epoch {epoch}: "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"train_mae_cp={train_mae_cp:.2f} "
                f"val_mae_cp={val_mae_cp:.2f}"
            )


if __name__ == "__main__":
    main()
