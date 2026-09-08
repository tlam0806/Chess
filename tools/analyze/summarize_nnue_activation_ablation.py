#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize mixed ReLU16 NNUE ablation")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--log", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--description",
        default=(
            "Branches use the same seeded training/evaluation pipeline. Exact sample, "
            "epoch, learning-rate, activation, and scale settings are recorded in each start event."
        ),
    )
    return parser.parse_args()


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def summarize(label: str, path: Path) -> tuple[list[str], dict[str, Any]]:
    events = read_events(path)
    start = next(event for event in events if event.get("event") == "start")
    epochs = [event for event in events if event.get("event") == "epoch"]
    calibration = {
        (int(event["epoch"]), str(event["split"])): event
        for event in events
        if event.get("event") == "calibration"
    }
    saturation = {
        int(event["epoch"]): event
        for event in events
        if event.get("event") == "hidden_saturation"
    }
    rows: list[str] = []
    for event in epochs:
        epoch = int(event["epoch"])
        train = calibration[(epoch, "train_probe")]
        val = calibration[(epoch, "val")]
        stats = saturation[epoch]["stats"]
        actual_clip = "/".join(percent(float(stat["clip_rate"])) for stat in stats)
        over_u8 = "/".join(percent(float(stat.get("over_u8_rate", stat["clip_rate"]))) for stat in stats)
        rows.append(
            "| {label} | {activation} | {epoch} | {train_cp:.3f} | {train_slope:.4f} | "
            "{val_cp:.3f} | {val_slope:.4f} | {actual_clip} | {over_u8} |".format(
                label=label,
                activation=start["activation"],
                epoch=epoch,
                train_cp=float(event["train_probe_cp"]),
                train_slope=float(train["slope"]),
                val_cp=float(event["val_cp"]),
                val_slope=float(val["slope"]),
                actual_clip=actual_clip,
                over_u8=over_u8,
            )
        )
    best = min(epochs, key=lambda event: float(event["val_cp"]))
    epoch = int(best["epoch"])
    return rows, {
        "label": label,
        "activation": start["activation"],
        "epoch": epoch,
        "train_cp": float(best["train_probe_cp"]),
        "train_slope": float(calibration[(epoch, "train_probe")]["slope"]),
        "val_cp": float(best["val_cp"]),
        "val_slope": float(calibration[(epoch, "val")]["slope"]),
    }


def main() -> None:
    args = parse_args()
    rows: list[str] = []
    best: list[dict[str, Any]] = []
    for item in args.log:
        label, raw_path = item.split("=", 1)
        branch_rows, branch_best = summarize(label, Path(raw_path))
        rows.extend(branch_rows)
        best.append(branch_best)
    lines = [
        f"# Mixed ReLU16 activation ablation: {args.tag}",
        "",
        args.description,
        "",
        "Actual clip uses each activation's real maximum (255 or 65535). "
        "Over-u8 reports values that would have clipped at 255.",
        "",
        "| Model | Activation | Epoch | Train MAE | Train slope | Validation MAE | Validation slope | L1/L2/L3 actual clip | L1/L2/L3 over-u8 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Best epoch by validation CP MAE",
        "",
        "| Model | Activation | Epoch | Train MAE | Train slope | Validation MAE | Validation slope |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in best:
        lines.append(
            "| {label} | {activation} | {epoch} | {train_cp:.3f} | {train_slope:.4f} | {val_cp:.3f} | {val_slope:.4f} |".format(**item)
        )
    winner = min(best, key=lambda item: item["val_cp"])
    lines.extend(["", f"Best branch: {winner['label']} at {winner['val_cp']:.3f} CP MAE.", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(json.dumps({"event": "report", "path": str(args.output), "winner": winner}))


if __name__ == "__main__":
    main()
