#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from tools.train.quantized_training_log import parse_quantized_training_log


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    denom = sum((index - mean_x) ** 2 for index in range(n))
    if denom == 0.0:
        return 0.0
    return sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values)) / denom


def summarize_log(path: Path, tail_epochs: int) -> dict[str, object] | None:
    epochs: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            event = json.loads(line)
            if event.get("event") == "epoch":
                epochs.append(event)
    if not epochs:
        return None
    parsed = parse_quantized_training_log(path)
    if "error" in parsed:
        return parsed
    val_losses = [float(epoch["val_loss"]) for epoch in epochs]
    train_losses = [float(epoch["train_loss"]) for epoch in epochs]
    val_values = [float(epoch["val_cp"]) for epoch in epochs]
    train_values = [float(epoch["train_val_cp"]) for epoch in epochs]
    tail = val_losses[-tail_epochs:]
    last = epochs[-1]
    return {
        "arch": last["arch"],
        "epochs": len(epochs),
        "best_epoch": parsed["best_epoch"],
        "best_val_loss": parsed["best_val_loss"],
        "val_cp_at_best_loss": parsed["val_cp_at_best_loss"],
        "final_test_loss": parsed.get("final_test_loss"),
        "final_test_cp": parsed.get("final_test_cp"),
        "last_val_loss": round(val_losses[-1], 8),
        "last_val_cp": round(val_values[-1], 4),
        "last_train_cp": round(train_values[-1], 4),
        "tail_slope_val_loss_per_epoch": round(linear_slope(tail), 8),
        "train_val_loss_gap": round(val_losses[-1] - train_losses[-1], 8),
        "train_val_cp_gap": round(val_values[-1] - train_values[-1], 4),
        "hidden_scales": last.get("hidden_scales", []),
        "output_scale": last.get("output_scale"),
        "improved_last_epoch": last.get("improved"),
        "no_improve_epochs": last.get("no_improve_epochs"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize quantized NNUE sweep logs")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--log-dir", default=Path("logs"), type=Path)
    parser.add_argument("--architectures", nargs="+", default=["A", "B", "C", "D", "E", "F", "G", "H"])
    parser.add_argument("--tail-epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for arch in args.architectures:
        path = args.log_dir / f"{args.tag}_{arch}.log"
        if not path.exists():
            rows.append({"arch": arch, "status": "missing"})
            continue
        summary = summarize_log(path, args.tail_epochs)
        if summary is None:
            rows.append({"arch": arch, "status": "no_epoch_yet"})
        else:
            rows.append(summary)
    rows.sort(key=lambda row: float(row.get("best_val_loss", 1e9)))
    print(json.dumps({"tag": args.tag, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
