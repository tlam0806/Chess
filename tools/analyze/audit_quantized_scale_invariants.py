#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.quantized_nnue_architectures import (  # noqa: E402
    ACTIVATION_CHOICES,
    FEATURE_WEIGHT_SCALE,
    HIDDEN_CLIP,
    LINEAR_WEIGHT_SCALE,
    OUTPUT_WEIGHT_SCALE,
    QUANTIZATION_CONVENTION_CHOICES,
    QUANTIZATION_CONVENTION_LEGACY,
    SCRELU_DIVISOR,
    QUANTIZED_ARCHITECTURES,
    QuantizedSparseNnueArchitecture,
    quantization_scale_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit NNUE quantization scale invariants")
    parser.add_argument("--arch", required=True, choices=sorted(QUANTIZED_ARCHITECTURES))
    parser.add_argument("--activation", default="relu", choices=ACTIVATION_CHOICES)
    parser.add_argument("--hidden-clip", type=int, default=HIDDEN_CLIP)
    parser.add_argument("--feature-weight-scale", type=int, default=FEATURE_WEIGHT_SCALE)
    parser.add_argument("--linear-weight-scale", type=int, default=LINEAR_WEIGHT_SCALE)
    parser.add_argument("--output-weight-scale", type=int, default=OUTPUT_WEIGHT_SCALE)
    parser.add_argument("--screlu-divisor", type=int, default=SCRELU_DIVISOR)
    parser.add_argument(
        "--quantization-convention",
        choices=QUANTIZATION_CONVENTION_CHOICES,
        default=QUANTIZATION_CONVENTION_LEGACY,
    )
    parser.add_argument("--hidden-scales", required=True, nargs="+", type=int)
    parser.add_argument("--output-scale", required=True, type=int)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    model = QuantizedSparseNnueArchitecture(
        QUANTIZED_ARCHITECTURES[args.arch],
        hidden_clip=args.hidden_clip,
        feature_weight_scale=args.feature_weight_scale,
        linear_weight_scale=args.linear_weight_scale,
        output_weight_scale=args.output_weight_scale,
        screlu_divisor=args.screlu_divisor,
    )
    audit = quantization_scale_audit(
        model,
        args.activation,
        args.hidden_scales,
        args.output_scale,
        args.quantization_convention,
    )
    payload = {"event": "scale_audit", "arch": args.arch, **audit}
    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
