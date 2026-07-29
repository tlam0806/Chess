#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize float/Python-quantized/C++ NNUE parity")
    parser.add_argument("--python-report", required=True, type=Path)
    parser.add_argument("--cpp-report", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    python_report = json.loads(args.python_report.read_text(encoding="utf-8"))
    cpp_report = json.loads(args.cpp_report.read_text(encoding="utf-8"))
    float_metrics = python_report["float_pytorch"]["overall"]
    quantized_metrics = python_report["quantized_python"]["overall"]
    cpp_metrics = cpp_report["overall"]
    rows = [
        (
            "Float PyTorch",
            float_metrics["model_mae"],
            float_metrics["regression_slope"],
            float_metrics["regression_intercept"],
            float_metrics["mean_prediction"],
            float_metrics["prediction_std"],
        ),
        (
            "Quantized Python",
            quantized_metrics["model_mae"],
            quantized_metrics["regression_slope"],
            quantized_metrics["regression_intercept"],
            quantized_metrics["mean_prediction"],
            quantized_metrics["prediction_std"],
        ),
        (
            "C++ integer",
            cpp_report["cpp_cp_mae"],
            cpp_metrics["regression_slope"],
            cpp_metrics["regression_intercept"],
            cpp_metrics["mean_prediction"],
            cpp_metrics["prediction_std"],
        ),
    ]
    lines = [
        "| Inference | CP MAE | slope | intercept | mean prediction | prediction std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {name} | {fmt(mae)} | {fmt(slope)} | {fmt(intercept)} | {fmt(mean)} | {fmt(std)} |"
        for name, mae, slope, intercept, mean, std in rows
    )
    parity = cpp_report["python_cpp_parity"]
    delta = python_report["quantization_delta_cp"]
    lines.extend(
        [
            "",
            f"Python/C++ mismatches: {parity['mismatches']} "
            f"(MAE {fmt(parity['mae'])}, max abs {fmt(parity['max_abs'])})",
            f"Float/quantized delta: MAE {fmt(delta['mae'])}, "
            f"max abs {fmt(delta['max_abs'])}",
        ]
    )
    output = "\n".join(lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
