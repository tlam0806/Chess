from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.compact_board_data import validate_compact_dataset
from chess_nnue.quantized_nnue_architectures import ACTIVATION_CHOICES, QUANTIZED_ARCHITECTURES


def validate_sweep_args(args: Namespace) -> None:
    if getattr(args, "jobs", 1) <= 0:
        raise ValueError("--jobs must be positive")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"training data does not exist: {args.data}")

    data_format = args.data_format
    if data_format == "auto":
        path = Path(args.data)
        data_format = (
            "cbin"
            if path.is_dir() or path.name.endswith(".cbin") or path.name.endswith(".cbin.zst")
            else "jsonl"
        )
    if data_format == "cbin":
        validate_compact_dataset(Path(args.data))
    elif Path(args.data).is_dir():
        raise ValueError("JSONL training data must be a file, not a directory")

    for name in (
        "epochs",
        "patience",
        "batch_size",
        "train_max_samples",
        "val_max_samples",
        "test_max_samples",
        "calibration_max_batches",
        "feature_weight_scale",
        "linear_weight_scale",
        "output_weight_scale",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("workers", "eval_workers", "torch_threads"):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("lr", "min_lr"):
        if hasattr(args, name) and getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("weight_decay", "min_delta_loss"):
        if hasattr(args, name) and getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in (
        "warmup_epochs",
        "lr_drop_patience",
        "shuffle_block_size",
        "progress_batches",
        "eval_progress_batches",
        "lr_warmup_steps",
    ):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if (
        getattr(args, "lr_schedule", "constant") == "cosine"
        and hasattr(args, "lr")
        and hasattr(args, "min_lr")
        and args.min_lr > args.lr
    ):
        raise ValueError("--min-lr cannot exceed --lr for cosine scheduling")
    if (
        hasattr(args, "lr_after_warmup")
        and args.lr_after_warmup is not None
        and args.lr_after_warmup <= 0.0
    ):
        raise ValueError("--lr-after-warmup must be positive")
    if hasattr(args, "lr_drop_factor") and not 0.0 < args.lr_drop_factor <= 1.0:
        raise ValueError("--lr-drop-factor must be in (0, 1]")
    if (
        hasattr(args, "calibration_percentile")
        and not 0.0 < args.calibration_percentile <= 100.0
    ):
        raise ValueError("--calibration-percentile must be in (0, 100]")
    for name in ("fixed_hidden_scale", "fixed_output_scale"):
        if (
            hasattr(args, name)
            and getattr(args, name) is not None
            and getattr(args, name) <= 0
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    fixed_hidden_scales = getattr(args, "fixed_hidden_scales", None)
    if fixed_hidden_scales is not None and (
        not fixed_hidden_scales or any(value <= 0 for value in fixed_hidden_scales)
    ):
        raise ValueError("--fixed-hidden-scales must contain positive values")
    if (
        getattr(args, "fixed_hidden_scale", None) is not None
        and fixed_hidden_scales is not None
    ):
        raise ValueError("use only one of --fixed-hidden-scale and --fixed-hidden-scales")
    if (
        hasattr(args, "patience")
        and getattr(args, "lr_drop_patience", 0) > 0
        and args.patience <= args.lr_drop_patience
    ):
        raise ValueError("--patience must be greater than --lr-drop-patience")
    if hasattr(args, "score_lambda") and not 0.0 <= args.score_lambda <= 1.0:
        raise ValueError("--score-lambda must be in [0, 1]")
    if hasattr(args, "wdl_loss_exponent") and args.wdl_loss_exponent <= 0.0:
        raise ValueError("--wdl-loss-exponent must be positive")

    for stage in ("stage1", "stage2"):
        for suffix in (
            "train_samples",
            "val_samples",
            "test_samples",
            "epochs",
            "patience",
            "lr",
        ):
            name = f"{stage}_{suffix}"
            if hasattr(args, name) and getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        warmup_name = f"{stage}_warmup_epochs"
        if hasattr(args, warmup_name) and getattr(args, warmup_name) < 0:
            raise ValueError(f"--{warmup_name.replace('_', '-')} must be non-negative")
        after_warmup_name = f"{stage}_lr_after_warmup"
        if (
            hasattr(args, after_warmup_name)
            and getattr(args, after_warmup_name) is not None
            and getattr(args, after_warmup_name) <= 0
        ):
            raise ValueError(f"--{after_warmup_name.replace('_', '-')} must be positive")

    for name in ("top_k", "top_k_total", "top_k_per_arch"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "max_hours") and args.max_hours < 0.0:
        raise ValueError("--max-hours must be non-negative")

    architectures = getattr(args, "architectures", None)
    if architectures is not None:
        unknown = sorted(set(architectures) - set(QUANTIZED_ARCHITECTURES))
        if unknown:
            raise ValueError(f"unknown architectures: {', '.join(unknown)}")
        if not architectures:
            raise ValueError("--architectures must not be empty")

    activations = getattr(args, "activations", None)
    if activations is not None:
        unknown = sorted(set(activations) - set(ACTIVATION_CHOICES))
        if unknown:
            raise ValueError(f"unknown activations: {', '.join(unknown)}")
        if not activations:
            raise ValueError("--activations must not be empty")

    for name in ("hidden_scales", "output_scales"):
        values = getattr(args, name, None)
        if values is not None and (not values or any(value <= 0 for value in values)):
            raise ValueError(f"--{name.replace('_', '-')} must contain positive values")
