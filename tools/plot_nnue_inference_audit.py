#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot signed calibration and prediction histogram")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 7))
    all_targets: list[float] = []
    for key, label, color in (
        ("float_pytorch", "Float PyTorch", "#2878B5"),
        ("quantized_python", "Quantized Python / C++", "#D95319"),
    ):
        rows = [row for row in report[key]["signed_buckets"] if row["samples"]]
        targets = [row["mean_target"] for row in rows]
        predictions = [row["mean_prediction"] for row in rows]
        all_targets.extend(targets)
        axis.plot(targets, predictions, marker="o", linewidth=2, label=label, color=color)
    lower, upper = min(all_targets), max(all_targets)
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", label="Ideal y=x")
    quantized = report["quantized_python"]["overall"]
    slope = quantized["regression_slope"]
    intercept = quantized["regression_intercept"]
    axis.plot(
        [lower, upper],
        [intercept + slope * lower, intercept + slope * upper],
        linestyle=":",
        color="#D95319",
        label=f"Quantized fit: y={intercept:.1f}+{slope:.3f}x",
    )
    axis.set_title("NNUE signed calibration")
    axis.set_xlabel("Mean target CP")
    axis.set_ylabel("Mean prediction CP")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "signed_calibration.png", dpi=180)
    plt.close(figure)

    rows = report["quantized_python"]["prediction"]["histogram"]
    centers = [(row["min"] + row["max"]) / 2.0 for row in rows]
    widths = [row["max"] - row["min"] for row in rows]
    counts = [row["samples"] for row in rows]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(centers, counts, width=widths, align="center", edgecolor="black", color="#2878B5")
    axis.set_title("Quantized prediction histogram")
    axis.set_xlabel("Predicted CP")
    axis.set_ylabel("Samples")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "prediction_histogram.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
