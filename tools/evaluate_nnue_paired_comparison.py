#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.quantized_nnue_architectures import (  # noqa: E402
    QUANTIZATION_CONVENTION_LEGACY,
    QUANTIZED_ARCHITECTURES,
)
from nn.train_value import choose_device  # noqa: E402
from tools.evaluate_quantized_nnue_checkpoint import load_model  # noqa: E402
from tools.train_nnue_architecture import make_loader, validate_training_data  # noqa: E402
from tools.train_quantized_nnue_architecture import (  # noqa: E402
    collate_quantized_sparse_batch,
    regression_forward,
)


DEFAULT_BIN_EDGES = (0.0, 100.0, 300.0, 600.0, 1000.0, 1600.0, 2000.0)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


def parse_checkpoint_spec(raw: str) -> CheckpointSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    label, raw_path = raw.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must use non-empty LABEL=PATH")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return CheckpointSpec(label=label, path=path)


def parse_bin_edges(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(","))
    if len(values) < 2 or any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("bin edges must be strictly increasing")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired CP-MAE comparison of quantized NNUE checkpoints on identical positions"
    )
    parser.add_argument("--reference", required=True, type=parse_checkpoint_spec)
    parser.add_argument("--candidate", action="append", required=True, type=parse_checkpoint_spec)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=("auto", "cbin", "jsonl"), default="auto")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument(
        "--skip-samples",
        type=int,
        default=0,
        help="skip this many deterministic samples before collecting the paired evaluation slice",
    )
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward-mode", choices=("auto", "float", "quantized"), default="auto")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-block-size", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--target-abs-cp-bins",
        type=parse_bin_edges,
        default=DEFAULT_BIN_EDGES,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def checkpoint_runtime(checkpoint: dict[str, Any], requested_forward_mode: str) -> dict[str, Any]:
    convention = str(
        checkpoint.get(
            "quantization_convention",
            checkpoint.get("quantization", {}).get(
                "convention", QUANTIZATION_CONVENTION_LEGACY
            ),
        )
    )
    forward_mode = (
        str(checkpoint.get("forward_mode", "quantized"))
        if requested_forward_mode == "auto"
        else requested_forward_mode
    )
    return {
        "architecture": str(checkpoint["architecture"]),
        "target_scale": float(checkpoint["target_scale"]),
        "hidden_scales": [int(value) for value in checkpoint["hidden_scales"]],
        "output_scale": int(checkpoint["output_scale"]),
        "activation": str(checkpoint["activation"]),
        "quantization_convention": convention,
        "forward_mode": forward_mode,
        "epoch": int(checkpoint["epoch"]),
    }


@torch.no_grad()
def collect_absolute_errors(
    spec: CheckpointSpec,
    data: Path,
    data_format: str,
    split: str,
    skip_samples: int,
    max_samples: int,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
    requested_forward_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model, checkpoint = load_model(spec.path, device)
    runtime = checkpoint_runtime(checkpoint, requested_forward_mode)
    config = QUANTIZED_ARCHITECTURES[runtime["architecture"]]
    loader = make_loader(
        data,
        config.transform,
        data_format,
        split,
        100,
        98,
        99,
        skip_samples + max_samples,
        seed,
        batch_size,
        workers,
        0,
    )
    error_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    remaining_skip = skip_samples
    collected = 0
    model.eval()
    for batch in loader:
        features, offsets, _scores, normalized_target, _plies, _results = (
            collate_quantized_sparse_batch(
                batch,
                device=device,
                board_feature_count=config.board_feature_count,
                target_scale=runtime["target_scale"],
            )
        )
        prediction = regression_forward(
            model,
            features,
            offsets,
            runtime["hidden_scales"],
            runtime["output_scale"],
            runtime["activation"],
            runtime["quantization_convention"],
            runtime["forward_mode"],
        )
        target = normalized_target * runtime["target_scale"]
        batch_errors = (prediction - target).abs().float().cpu().numpy()
        batch_targets = target.float().cpu().numpy()
        start = min(remaining_skip, int(batch_targets.size))
        remaining_skip -= start
        take = min(int(batch_targets.size) - start, max_samples - collected)
        if take > 0:
            end = start + take
            error_chunks.append(batch_errors[start:end])
            target_chunks.append(batch_targets[start:end])
            collected += take
        if collected >= max_samples:
            break
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    if not error_chunks:
        raise RuntimeError(f"evaluation produced no samples for {spec.label}")
    errors = np.concatenate(error_chunks).astype(np.float64, copy=False)
    targets = np.concatenate(target_chunks).astype(np.float64, copy=False)
    runtime.update(
        {
            "label": spec.label,
            "checkpoint": str(spec.path),
            "samples": int(errors.size),
            "skipped_samples": skip_samples,
            "cp_mae": float(errors.mean()),
        }
    )
    return errors, targets, runtime


def paired_block_bootstrap(
    delta: np.ndarray,
    replicates: int,
    block_size: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, float | int | bool]:
    if delta.ndim != 1 or delta.size == 0:
        raise ValueError("paired delta must be a non-empty vector")
    if replicates <= 0 or block_size <= 0:
        raise ValueError("bootstrap replicates and block size must be positive")
    full_blocks = delta.size // block_size
    if full_blocks < 2:
        raise ValueError("paired comparison needs at least two complete bootstrap blocks")
    used = full_blocks * block_size
    block_means = delta[:used].reshape(full_blocks, block_size).mean(axis=1)
    draws = np.empty(replicates, dtype=np.float64)
    chunk_size = max(1, min(512, 8_000_000 // full_blocks))
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        indices = rng.integers(0, full_blocks, size=(count, full_blocks))
        draws[start : start + count] = block_means[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha))
    point = float(delta.mean())
    return {
        "delta_mae_cp": point,
        "ci_low_cp": float(lower),
        "ci_high_cp": float(upper),
        "confidence": confidence,
        "bootstrap_replicates": replicates,
        "bootstrap_block_size": block_size,
        "bootstrap_blocks": full_blocks,
        "bootstrap_samples_used": used,
        "bootstrap_standard_error_cp": float(draws.std(ddof=1)),
        "probability_candidate_better": float((draws < 0.0).mean()),
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }


def compare_errors(
    reference_errors: np.ndarray,
    candidate_errors: np.ndarray,
    targets: np.ndarray,
    bin_edges: tuple[float, ...],
    replicates: int,
    block_size: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if reference_errors.shape != candidate_errors.shape or reference_errors.shape != targets.shape:
        raise ValueError("paired arrays have different shapes")
    delta = candidate_errors - reference_errors
    overall = paired_block_bootstrap(
        delta,
        replicates,
        block_size,
        confidence,
        np.random.default_rng(seed),
    )
    overall.update(
        {
            "reference_cp_mae": float(reference_errors.mean()),
            "candidate_cp_mae": float(candidate_errors.mean()),
            "candidate_better_position_rate": float((delta < 0.0).mean()),
            "candidate_worse_position_rate": float((delta > 0.0).mean()),
            "tied_position_rate": float((delta == 0.0).mean()),
        }
    )
    bins: list[dict[str, Any]] = []
    abs_target = np.abs(targets)
    for index, (lower, upper) in enumerate(zip(bin_edges, bin_edges[1:])):
        upper_mask = abs_target <= upper if index == len(bin_edges) - 2 else abs_target < upper
        mask = (abs_target >= lower) & upper_mask
        count = int(mask.sum())
        row: dict[str, Any] = {
            "min_abs_target_cp": lower,
            "max_abs_target_cp": upper,
            "samples": count,
        }
        if count >= 2 * block_size:
            row.update(
                paired_block_bootstrap(
                    delta[mask],
                    replicates,
                    block_size,
                    confidence,
                    np.random.default_rng(seed + index + 1),
                )
            )
        else:
            row.update(
                {
                    "delta_mae_cp": float(delta[mask].mean()) if count else math.nan,
                    "ci_low_cp": math.nan,
                    "ci_high_cp": math.nan,
                    "ci_excludes_zero": False,
                }
            )
        row["reference_cp_mae"] = float(reference_errors[mask].mean()) if count else math.nan
        row["candidate_cp_mae"] = float(candidate_errors[mask].mean()) if count else math.nan
        bins.append(row)
    return {"overall": overall, "bins": bins}


def markdown_report(result: dict[str, Any]) -> str:
    reference = result["reference"]
    lines = [
        f"# Paired NNUE comparison: {result['tag']}",
        "",
        f"Reference: `{reference['label']}` (`{reference['checkpoint']}`).",
        "",
        f"Evaluation slice: skipped {result['skip_samples']:,} samples, then evaluated {result['samples']:,} samples.",
        "",
        "Delta is `candidate absolute error - reference absolute error` on the same position; negative is better. "
        "Confidence intervals use paired contiguous-block bootstrap.",
        "",
        "| Candidate | MAE | Reference MAE | Delta MAE | 95% CI | P(candidate better) | Lower-error rate | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in result["candidates"]:
        overall = candidate["comparison"]["overall"]
        decision = (
            "better"
            if overall["ci_high_cp"] < 0.0
            else "worse"
            if overall["ci_low_cp"] > 0.0
            else "tie/noise"
        )
        lines.append(
            "| {label} | {candidate_cp_mae:.3f} | {reference_cp_mae:.3f} | {delta_mae_cp:+.3f} | "
            "[{ci_low_cp:+.3f}, {ci_high_cp:+.3f}] | {probability_candidate_better:.1%} | "
            "{candidate_better_position_rate:.1%} | {decision} |".format(
                label=candidate["label"], decision=decision, **overall
            )
        )
    for candidate in result["candidates"]:
        lines.extend(
            [
                "",
                f"## {candidate['label']} by absolute-target range",
                "",
                "| Abs target CP | Samples | Candidate MAE | Reference MAE | Delta MAE | 95% CI |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in candidate["comparison"]["bins"]:
            lines.append(
                "| {min_abs_target_cp:.0f}-{max_abs_target_cp:.0f} | {samples} | "
                "{candidate_cp_mae:.3f} | {reference_cp_mae:.3f} | {delta_mae_cp:+.3f} | "
                "[{ci_low_cp:+.3f}, {ci_high_cp:+.3f}] |".format(**row)
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0 or args.workers < 0 or args.skip_samples < 0:
        raise ValueError("sample/batch counts must be positive and workers non-negative")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.bootstrap_replicates <= 0 or args.bootstrap_block_size <= 0:
        raise ValueError("bootstrap parameters must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    labels = [args.reference.label, *(candidate.label for candidate in args.candidate)]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    data_format = validate_training_data(args.data, args.data_format)
    device = choose_device(args.device)

    reference_errors, targets, reference_runtime = collect_absolute_errors(
        args.reference,
        args.data,
        data_format,
        args.split,
        args.skip_samples,
        args.max_samples,
        args.batch_size,
        args.workers,
        args.seed,
        device,
        args.forward_mode,
    )
    candidates: list[dict[str, Any]] = []
    for index, spec in enumerate(args.candidate):
        errors, candidate_targets, runtime = collect_absolute_errors(
            spec,
            args.data,
            data_format,
            args.split,
            args.skip_samples,
            args.max_samples,
            args.batch_size,
            args.workers,
            args.seed,
            device,
            args.forward_mode,
        )
        if candidate_targets.shape != targets.shape or not np.array_equal(candidate_targets, targets):
            raise RuntimeError(f"position order/targets differ for candidate {spec.label}")
        comparison = compare_errors(
            reference_errors,
            errors,
            targets,
            args.target_abs_cp_bins,
            args.bootstrap_replicates,
            args.bootstrap_block_size,
            args.confidence,
            args.bootstrap_seed + index * 100,
        )
        candidates.append({**runtime, "comparison": comparison})
        overall = comparison["overall"]
        print(
            json.dumps(
                {
                    "event": "paired_comparison",
                    "candidate": spec.label,
                    **overall,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    result = {
        "tag": args.output_json.stem,
        "data": str(args.data),
        "split": args.split,
        "skip_samples": args.skip_samples,
        "samples": int(reference_errors.size),
        "bootstrap": {
            "method": "paired_contiguous_block_bootstrap",
            "replicates": args.bootstrap_replicates,
            "block_size": args.bootstrap_block_size,
            "confidence": args.confidence,
            "seed": args.bootstrap_seed,
        },
        "reference": reference_runtime,
        "candidates": candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(result))
    print(
        json.dumps(
            {
                "event": "paired_comparison_complete",
                "output_json": str(args.output_json),
                "output_md": str(args.output_md) if args.output_md is not None else None,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
