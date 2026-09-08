#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the 20M NNUE capacity ablation")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--baseline-log", required=True, type=Path)
    parser.add_argument("--wide-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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


def format_rate(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def summarize(label: str, path: Path) -> tuple[list[str], dict[str, Any]]:
    events = read_events(path)
    start = next(event for event in events if event.get("event") == "start")
    epochs = [event for event in events if event.get("event") == "epoch"]
    calibrations = {
        (int(event["epoch"]), str(event["split"])): event
        for event in events
        if event.get("event") == "calibration"
    }
    saturations = {
        int(event["epoch"]): event
        for event in events
        if event.get("event") == "hidden_saturation"
    }
    rows: list[str] = []
    for epoch in epochs:
        number = int(epoch["epoch"])
        train = calibrations[(number, "train_probe")]
        val = calibrations[(number, "val")]
        stats = saturations[number]["stats"]
        clips = "/".join(format_rate(float(stat["clip_rate"])) for stat in stats)
        zeros = "/".join(format_rate(float(stat["zero_rate"])) for stat in stats)
        rows.append(
            "| {label} | {arch} | {epoch} | {train_cp:.3f} | {train_slope:.4f} | "
            "{val_cp:.3f} | {val_slope:.4f} | {gap:.3f} | {clips} | {zeros} |".format(
                label=label,
                arch=start["arch"],
                epoch=number,
                train_cp=float(epoch["train_probe_cp"]),
                train_slope=float(train["slope"]),
                val_cp=float(epoch["val_cp"]),
                val_slope=float(val["slope"]),
                gap=float(epoch["val_cp"]) - float(epoch["train_probe_cp"]),
                clips=clips,
                zeros=zeros,
            )
        )
    best = min(epochs, key=lambda event: float(event["val_cp"]))
    best_number = int(best["epoch"])
    return rows, {
        "label": label,
        "arch": start["arch"],
        "hidden_sizes": start["config"]["hidden_sizes"],
        "best_epoch": best_number,
        "train_cp": float(best["train_probe_cp"]),
        "train_slope": float(calibrations[(best_number, "train_probe")]["slope"]),
        "val_cp": float(best["val_cp"]),
        "val_slope": float(calibrations[(best_number, "val")]["slope"]),
    }


def main() -> None:
    args = parse_args()
    baseline_rows, baseline = summarize("baseline", args.baseline_log)
    wide_rows, wide = summarize("wide", args.wide_log)
    delta = wide["val_cp"] - baseline["val_cp"]
    lines = [
        f"# NNUE capacity ablation: {args.tag}",
        "",
        "Controlled comparison on the same 20,000,000 unique training records. "
        "Both models use c255/d256, hs64x64, os16, quantized forward, Huber delta=600, "
        "weight decay 0, and the same 500,000-position validation set.",
        "",
        "| Model | Architecture | Epoch | Train-probe MAE | Train slope | Validation MAE | Validation slope | MAE gap | L1/L2/L3 clip | L1/L2/L3 zero |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *baseline_rows,
        *wide_rows,
        "",
        "## Best validation checkpoint by CP MAE",
        "",
        "| Model | Hidden sizes | Epoch | Train-probe MAE | Train slope | Validation MAE | Validation slope |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| {label} | {sizes} | {best_epoch} | {train_cp:.3f} | {train_slope:.4f} | {val_cp:.3f} | {val_slope:.4f} |".format(
            sizes="-".join(str(value) for value in baseline["hidden_sizes"]), **baseline
        ),
        "| {label} | {sizes} | {best_epoch} | {train_cp:.3f} | {train_slope:.4f} | {val_cp:.3f} | {val_slope:.4f} |".format(
            sizes="-".join(str(value) for value in wide["hidden_sizes"]), **wide
        ),
        "",
        f"Wide-minus-baseline validation CP MAE: {delta:+.3f} CP (negative favors the wider model).",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(json.dumps({"event": "report", "path": str(args.output), "wide_minus_baseline_cp": delta}))


if __name__ == "__main__":
    main()
