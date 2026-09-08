from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.nnue_value_net import ChessNnueValueNet
from chess_nnue.train_value import make_loader, mean_absolute_error_cp


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ChessNnueValueNet, float, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ChessNnueValueNet(
        feature_count=int(checkpoint["feature_count"]),
        aux_feature_count=int(checkpoint["aux_feature_count"]),
        hidden_size=int(checkpoint["hidden_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, float(checkpoint["target_scale"]), float(checkpoint["target_clip"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NNUE value checkpoint on JSONL datasets")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, target_scale, target_clip = load_model(args.checkpoint, device)
    for dataset in args.datasets:
        loader = make_loader(dataset, args.batch_size, shuffle=False)
        mae = mean_absolute_error_cp(
            model,
            loader,
            device,
            target_scale=target_scale,
            target_clip=target_clip,
        )
        print(f"{dataset}: mae_cp={mae:.2f}")


if __name__ == "__main__":
    main()
