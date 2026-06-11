from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.train_value import collate_sparse_value_batch
from nn.value_net import ChessValueNet


MAGIC = b"CVN1"
VERSION = 1


def load_model(checkpoint_path: Path) -> tuple[ChessValueNet, float, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ChessValueNet(
        feature_count=int(checkpoint["feature_count"]),
        aux_feature_count=int(checkpoint["aux_feature_count"]),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, float(checkpoint["target_scale"]), float(checkpoint["target_clip"])


def write_tensor(file, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().cpu().contiguous().view(-1).float()
    file.write(contiguous.numpy().astype("<f4", copy=False).tobytes())


def export_binary(checkpoint_path: Path, output_path: Path) -> None:
    model, target_scale, target_clip = load_model(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        file.write(
            struct.pack(
                "<4sIIIIff",
                MAGIC,
                VERSION,
                model.feature_count,
                model.aux_feature_count,
                model.hidden_size,
                target_scale,
                target_clip,
            )
        )
        write_tensor(file, model.sparse_features.weight)
        write_tensor(file, model.hidden[0].weight)
        write_tensor(file, model.hidden[0].bias)
        write_tensor(file, model.output.weight)
        write_tensor(file, model.output.bias)


def validate_sample(sample: dict[str, Any], line_number: int) -> None:
    if "features" not in sample or "aux" not in sample:
        raise ValueError(f"sample {line_number}: expected features and aux")
    if not sample["features"]:
        raise ValueError(f"sample {line_number}: features must not be empty")


@torch.no_grad()
def write_compare_samples(
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    limit: int,
) -> None:
    model, _, _ = load_model(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, 1):
            if written >= limit:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            validate_sample(sample, line_number)

            feature_indices, offsets, aux, _ = collate_sparse_value_batch(
                [{"features": sample["features"], "aux": sample["aux"], "target": 0.0}]
            )
            prediction = float(model(feature_indices, offsets, aux).item())
            output_file.write(
                json.dumps(
                    {
                        "features": sample["features"],
                        "aux": sample["aux"],
                        "python_output": prediction,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1

    if written == 0:
        raise ValueError(f"{input_path} produced no comparison samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ChessValueNet weights")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare-input", type=Path, default=None)
    parser.add_argument("--compare-output", type=Path, default=None)
    parser.add_argument("--compare-limit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_binary(args.checkpoint, args.output)
    if args.compare_input or args.compare_output:
        if not args.compare_input or not args.compare_output:
            raise ValueError("--compare-input and --compare-output must be used together")
        write_compare_samples(
            args.checkpoint,
            args.compare_input,
            args.compare_output,
            args.compare_limit,
        )


if __name__ == "__main__":
    main()
