#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def events(path: Path) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.log_dir.glob(f"{args.tag}_*.log")):
        branch = path.stem.removeprefix(f"{args.tag}_")
        parsed = events(path)
        epochs = [event for event in parsed if event.get("event") == "epoch"]
        calibrations = [
            event
            for event in parsed
            if event.get("event") == "calibration" and event.get("split") == "val"
        ]
        saturations = [event for event in parsed if event.get("event") == "hidden_saturation"]
        if not epochs:
            rows.append({"branch": branch, "status": "incomplete", "log": path})
            continue
        best = min(epochs, key=lambda event: float(event["val_cp"]))
        calibration = next(
            (event for event in calibrations if event.get("epoch") == best.get("epoch")),
            {},
        )
        saturation = next(
            (event for event in saturations if event.get("epoch") == best.get("epoch")),
            {},
        )
        rows.append(
            {
                "branch": branch,
                "status": "complete" if any(
                    event.get("event") == "training_complete" for event in parsed
                ) else "running",
                "epoch": best["epoch"],
                "train_cp": best["train_val_cp"],
                "val_cp": best["val_cp"],
                "slope": calibration.get("slope"),
                "intercept": calibration.get("intercept_cp"),
                "saturation": saturation.get("stats", []),
                "log": path,
            }
        )

    lines = [
        f"# NNUE loss/quantization ablation: {args.tag}",
        "",
        "Fixed corpus: 5,000,000 unique records. Architecture: F2 256-32-32, "
        "SCReLU c181/d128, hs32x16, os16.",
        "",
        "| Branch | Best epoch | Train rolling MAE | Validation MAE | Calibration slope | Intercept CP | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda value: float(value.get("val_cp", float("inf")))):
        if "epoch" not in row:
            lines.append(f"| {row['branch']} | - | - | - | - | - | {row['status']} |")
            continue
        lines.append(
            f"| {row['branch']} | {row['epoch']} | {row['train_cp']:.3f} | "
            f"{row['val_cp']:.3f} | {row['slope']:.4f} | {row['intercept']:.2f} | {row['status']} |"
        )
        lines.append("")
        lines.append(
            "Saturation at selected epoch: "
            + ", ".join(
                f"L{int(stat['layer'])} zero={100*stat['zero_rate']:.2f}% clip={100*stat['clip_rate']:.2f}%"
                for stat in row["saturation"]
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
