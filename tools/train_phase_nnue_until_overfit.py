#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nn.quantized_nnue_architectures import (
    ACTIVATION_SCRELU_RELU16_ALL,
    QUANTIZED_ARCHITECTURES,
    PhaseStackQuantizedNnueArchitecture,
)
from nn.train_value import choose_device
from tools.train_nnue_architecture import make_loader, validate_training_data
from tools.train_phase_component_nnue import (
    atomic_torch_save,
    atomic_write_text,
    build_checkpoint,
    collate,
    evaluate,
    initialize_biases,
    model_forward_components,
    regression_loss,
    saturation,
    scheduled_lr,
)


DEFAULT_PEAK_LRS = (
    5e-4, 1e-4, 6e-5, 4e-5, 2.5e-5, 1.75e-5,
    1.225e-5, 8.575e-6, 6e-6, 5e-6, 5e-6, 5e-6,
)
DEFAULT_MIN_LRS = (
    5e-5, 6e-5, 4e-5, 2.5e-5, 1.75e-5, 1.225e-5,
    8.575e-6, 6e-6, 5e-6, 5e-6, 5e-6, 5e-6,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an exportable eight-head phase NNUE until validation plateaus"
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arch", choices=("F2", "F2M"), default="F2M")
    parser.add_argument(
        "--phase-layout", choices=("independent", "shared_first"),
        default="independent",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta-loss", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--eval-workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-max-samples", type=int, default=490_000_000)
    parser.add_argument("--val-max-samples", type=int, default=5_000_000)
    parser.add_argument("--test-max-samples", type=int, default=5_000_000)
    parser.add_argument("--train-probe-max-samples", type=int, default=1_000_000)
    parser.add_argument("--split-mod", type=int, default=100)
    parser.add_argument("--val-mod", type=int, default=98)
    parser.add_argument("--test-mod", type=int, default=99)
    parser.add_argument("--shuffle-block-size", type=int, default=250_000)
    parser.add_argument("--lr-steps-per-epoch", type=int, default=59_815)
    parser.add_argument("--lr-warmup-steps", type=int, default=2_000)
    parser.add_argument("--epoch-peak-lrs", type=float, nargs="+", default=DEFAULT_PEAK_LRS)
    parser.add_argument("--epoch-min-lrs", type=float, nargs="+", default=DEFAULT_MIN_LRS)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--psqt-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--checkpoint-samples", type=int, default=50_000_000)
    parser.add_argument("--progress-batches", type=int, default=500)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--saturation-batches", type=int, default=20)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--seed", type=int, default=20260824)

    # Fixed, export-compatible quantization and loss contract.
    parser.add_argument("--activation", default=ACTIVATION_SCRELU_RELU16_ALL)
    parser.add_argument("--hidden-scales", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--output-scale", type=int, default=128)
    parser.add_argument("--hidden-clip", type=int, default=181)
    parser.add_argument("--screlu-divisor", type=int, default=128)
    parser.add_argument("--feature-weight-scale", type=int, default=181)
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--psqt-weight-scale", type=int, default=16)
    parser.add_argument("--target-scale", type=float, default=1000.0)
    parser.add_argument("--loss-type", choices=("huber",), default="huber")
    parser.add_argument("--huber-delta", type=float, default=200.0)
    parser.add_argument("--wdl-exponent", type=float, default=2.5)
    parser.add_argument("--wdl-input-offset", type=float, default=270.0)
    parser.add_argument("--wdl-output-offset", type=float, default=270.0)
    parser.add_argument("--wdl-input-scaling", type=float, default=340.0)
    parser.add_argument("--wdl-output-scaling", type=float, default=380.0)
    parser.add_argument("--hidden-bias-fraction", type=float, default=0.25)
    parser.add_argument("--first-bias-fraction", type=float, default=0.1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if args.batch_size <= 0 or args.workers < 0 or args.eval_workers < 0:
        raise ValueError("invalid batch/worker count")
    if args.train_max_samples <= 0 or args.val_max_samples <= 0:
        raise ValueError("train and validation sample limits must be positive")
    if args.checkpoint_samples < 0:
        raise ValueError("checkpoint sample interval must be non-negative")
    if args.split_mod <= 2 or args.val_mod == args.test_mod:
        raise ValueError("invalid hash split")
    if not (0 <= args.val_mod < args.split_mod and 0 <= args.test_mod < args.split_mod):
        raise ValueError("validation/test bucket outside split modulus")
    if len(args.epoch_peak_lrs) != args.epochs or len(args.epoch_min_lrs) != args.epochs:
        raise ValueError("per-epoch LR lists must contain exactly --epochs values")
    if any(value <= 0 for value in (*args.epoch_peak_lrs, *args.epoch_min_lrs)):
        raise ValueError("learning rates must be positive")
    if any(low > high for high, low in zip(args.epoch_peak_lrs, args.epoch_min_lrs)):
        raise ValueError("an epoch minimum LR exceeds its peak")
    if args.hidden_scales != [2, 8] or args.output_scale != 128:
        raise ValueError("production phase runner requires hidden scales 2,8 and output 128")
    validate_training_data(args.data, "cbin")


def checkpoint_payload(
    model: PhaseStackQuantizedNnueArchitecture,
    optimizer: torch.optim.Optimizer,
    config: Any,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    best_val_loss: float,
    epochs_without_improvement: int,
    epoch_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_checkpoint(model, optimizer, config, args, metrics=metrics)
    payload.update(
        {
            "epoch": epoch,
            "history": history,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "split": {
                "policy": "crc32(canonical_board+canonical_aux)",
                "mod": args.split_mod,
                "val": args.val_mod,
                "test": args.test_mod,
            },
        }
    )
    if epoch_progress is not None:
        payload["epoch_progress"] = epoch_progress
    return payload


def validate_resume(payload: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "architecture": args.arch,
        "phase_stacks": 8,
        "phase_layout": args.phase_layout,
        "component_supervision": "total",
        "label_mode": "total",
        "loss_type": args.loss_type,
        "huber_delta": args.huber_delta,
        "hidden_scales": args.hidden_scales,
        "output_scale": args.output_scale,
        "activation": args.activation,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"resume checkpoint {key} mismatch: expected={value!r} "
                f"actual={payload.get(key)!r}"
            )
    split = payload.get("split")
    if split is not None and (
        int(split.get("mod", -1)) != args.split_mod
        or int(split.get("val", -1)) != args.val_mod
        or int(split.get("test", -1)) != args.test_mod
    ):
        raise ValueError("resume checkpoint hash split mismatch")


def make_split_loader(
    args: argparse.Namespace,
    split: str,
    max_samples: int,
    workers: int,
    seed: int,
    shuffle: int,
):
    return make_loader(
        args.data,
        QUANTIZED_ARCHITECTURES[args.arch].transform,
        "cbin",
        split,
        args.split_mod,
        args.val_mod,
        args.test_mod,
        max_samples,
        seed,
        args.batch_size,
        workers,
        shuffle,
        cache_unshuffled=shuffle == 0,
    )


def train_one_epoch(
    model: PhaseStackQuantizedNnueArchitecture,
    optimizer: torch.optim.Optimizer,
    loader: Any,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    resume: dict[str, Any] | None,
    checkpoint_callback: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    model.train()
    progress = resume or {}
    skip_batches = int(progress.get("batches_completed", 0))
    samples = int(progress.get("samples", 0))
    loss_sum = float(progress.get("loss_sum", 0.0))
    cp_abs_sum = float(progress.get("cp_abs_sum", 0.0))
    prior_elapsed = float(progress.get("elapsed_sec", 0.0))
    started = time.monotonic()
    peak_lr = float(args.epoch_peak_lrs[epoch - 1])
    min_lr = float(args.epoch_min_lrs[epoch - 1])
    warmup = args.lr_warmup_steps if epoch == 1 else 0

    for batch_index, batch in enumerate(loader, 1):
        if batch_index <= skip_batches:
            continue
        feature_indices, offsets, _psqt_target, total_target = collate(
            batch, model.board_feature_count, device
        )
        lr = scheduled_lr(
            batch_index - 1,
            args.lr_steps_per_epoch,
            peak_lr,
            min_lr,
            warmup,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_multiplier", 1.0))
        optimizer.zero_grad(set_to_none=True)
        positional, psqt = model_forward_components(
            model,
            feature_indices,
            offsets,
            args.hidden_scales,
            args.output_scale,
            args.activation,
        )
        target = torch.clamp(total_target, -2000.0, 2000.0)
        prediction = positional + psqt
        loss = regression_loss(
            prediction,
            target,
            args.target_scale,
            args.loss_type,
            args.huber_delta,
            args.wdl_exponent,
            args.wdl_input_offset,
            args.wdl_output_offset,
            args.wdl_input_scaling,
            args.wdl_output_scaling,
        )
        loss.backward()
        optimizer.step()
        model.clamp_quantized_weights()

        count = int(offsets.numel())
        samples += count
        loss_sum += float(loss.detach()) * count
        cp_abs_sum += float((prediction.detach() - target).abs().sum())
        elapsed = prior_elapsed + time.monotonic() - started
        if args.progress_batches and batch_index % args.progress_batches == 0:
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "epoch": epoch,
                        "batch": batch_index,
                        "samples": samples,
                        "loss": loss_sum / samples,
                        "cp_mae": cp_abs_sum / samples,
                        "lr": lr,
                        "samples_per_sec": samples / max(elapsed, 1e-9),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        if (
            args.checkpoint_samples
            and samples < args.train_max_samples
            and samples // args.checkpoint_samples
            > (samples - count) // args.checkpoint_samples
        ):
            checkpoint_callback(
                {
                    "epoch": epoch,
                    "samples": samples,
                    "batches_completed": batch_index,
                    "loss_sum": loss_sum,
                    "cp_abs_sum": cp_abs_sum,
                    "elapsed_sec": elapsed,
                    "resume_signature": {
                        "batch_size": args.batch_size,
                        "workers": args.workers,
                        "shuffle_block_size": args.shuffle_block_size,
                        "seed": args.seed,
                    },
                }
            )
    if samples == 0:
        raise RuntimeError("training split produced no samples")
    return {
        "loss": loss_sum / samples,
        "cp_mae": cp_abs_sum / samples,
        "samples": samples,
        "elapsed_sec": prior_elapsed + time.monotonic() - started,
    }


def main() -> None:
    args = parse_args()
    args.phase_stacks = 8
    args.objective = "total"
    args.label_mode = "total"
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_path = args.output_dir / "phase_component_current.pt"
    best_path = args.output_dir / "phase_component_best.pt"
    progress_path = args.output_dir / "phase_epoch_in_progress.pt"
    summary_path = args.output_dir / "summary.json"

    if args.resume_checkpoint is not None and args.auto_resume:
        raise ValueError("use either --resume-checkpoint or --auto-resume")
    resume_path = args.resume_checkpoint
    if args.auto_resume:
        if progress_path.is_file():
            resume_path = progress_path
        elif current_path.is_file():
            resume_path = current_path

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = QUANTIZED_ARCHITECTURES[args.arch]
    model = PhaseStackQuantizedNnueArchitecture(
        config,
        phase_layout=args.phase_layout,
        hidden_clip=args.hidden_clip,
        feature_weight_scale=args.feature_weight_scale,
        linear_weight_scale=args.linear_weight_scale,
        output_weight_scale=args.output_weight_scale,
        screlu_divisor=args.screlu_divisor,
        psqt_weight_scale=args.psqt_weight_scale,
        psqt_master_scale_to_cp=args.target_scale,
    ).to(device)
    initialize_biases(
        model,
        args.hidden_scales,
        args.activation,
        args.hidden_bias_fraction,
        args.first_bias_fraction,
    )
    assert model.psqt is not None
    psqt_ids = {id(parameter) for parameter in model.psqt.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if id(p) not in psqt_ids],
                "lr_multiplier": 1.0,
            },
            {
                "params": list(model.psqt.parameters()),
                "lr_multiplier": args.psqt_lr / args.lr,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    completed_epoch = 0
    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    epochs_without_improvement = 0
    resume_epoch_progress: dict[str, Any] | None = None
    if resume_path is not None:
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume(payload, args)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        history = list(payload.get("history", []))
        best_val_loss = float(payload.get("best_val_loss", math.inf))
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        resume_epoch_progress = payload.get("epoch_progress")
        if resume_epoch_progress is not None:
            signature = resume_epoch_progress.get("resume_signature", {})
            expected_signature = {
                "batch_size": args.batch_size,
                "workers": args.workers,
                "shuffle_block_size": args.shuffle_block_size,
                "seed": args.seed,
            }
            if signature != expected_signature:
                raise ValueError("in-epoch resume data-loader signature mismatch")
            completed_epoch = int(resume_epoch_progress["epoch"]) - 1
        else:
            completed_epoch = int(payload.get("epoch", 0))
        print(
            json.dumps(
                {
                    "event": "resume",
                    "checkpoint": str(resume_path),
                    "completed_epoch": completed_epoch,
                    "in_epoch_samples": (
                        int(resume_epoch_progress["samples"])
                        if resume_epoch_progress is not None else 0
                    ),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "start",
                "architecture": args.arch,
                "phase_stacks": 8,
                "phase_layout": args.phase_layout,
                "device": str(device),
                "workers": args.workers,
                "eval_workers": args.eval_workers,
                "batch_size": args.batch_size,
                "split": {"mod": args.split_mod, "val": args.val_mod, "test": args.test_mod},
                "train_max_samples": args.train_max_samples,
                "epochs": args.epochs,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    val_loader = make_split_loader(
        args, "val", args.val_max_samples, args.eval_workers, args.seed, 0
    )
    train_probe_loader = make_split_loader(
        args, "train", args.train_probe_max_samples,
        args.eval_workers, args.seed, 0,
    )
    for epoch in range(completed_epoch + 1, args.epochs + 1):
        train_loader = make_split_loader(
            args,
            "train",
            args.train_max_samples,
            args.workers,
            args.seed + epoch * 1_000_003,
            args.shuffle_block_size,
        )

        def save_progress(progress: dict[str, Any]) -> None:
            payload = checkpoint_payload(
                model,
                optimizer,
                config,
                args,
                epoch - 1,
                {},
                history,
                best_val_loss,
                epochs_without_improvement,
                progress,
            )
            atomic_torch_save(payload, progress_path)
            print(
                json.dumps(
                    {
                        "event": "progress_checkpoint",
                        "epoch": epoch,
                        "samples": progress["samples"],
                        "path": str(progress_path),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            args,
            epoch,
            resume_epoch_progress if resume_epoch_progress is not None else None,
            save_progress,
        )
        resume_epoch_progress = None
        progress_path.unlink(missing_ok=True)
        selection = evaluate(
            model, val_loader, device, args.hidden_scales, args.output_scale,
            args, "selection",
        )
        train_probe = evaluate(
            model, train_probe_loader, device, args.hidden_scales,
            args.output_scale, args, "train_probe",
        )
        saturation_metrics = saturation(
            # The validation workers now hold their compact split in memory,
            # so reuse that loader instead of scanning the corpus a third time.
            model, val_loader, device, args.hidden_scales,
            args, args.saturation_batches,
        )
        improved = selection["objective_loss"] + args.min_delta_loss < best_val_loss
        if improved:
            best_val_loss = float(selection["objective_loss"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "selection": selection,
            "train_probe": train_probe,
            "saturation": saturation_metrics,
            "improved": improved,
        }
        history.append(epoch_metrics)
        payload = checkpoint_payload(
            model,
            optimizer,
            config,
            args,
            epoch,
            epoch_metrics,
            history,
            best_val_loss,
            epochs_without_improvement,
        )
        atomic_torch_save(payload, current_path)
        if improved:
            atomic_torch_save(payload, best_path)
        print(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_samples": train_metrics["samples"],
                    "train_loss": train_metrics["loss"],
                    "train_cp_mae": train_metrics["cp_mae"],
                    "val_samples": selection["samples"],
                    "val_loss": selection["objective_loss"],
                    "val_cp_mae": selection["cp_mae"],
                    "train_probe_loss": train_probe["objective_loss"],
                    "improved": improved,
                    "best_val_loss": best_val_loss,
                    "no_improve_epochs": epochs_without_improvement,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(
                json.dumps(
                    {"event": "early_stop", "epoch": epoch, "best_val_loss": best_val_loss},
                    separators=(",", ":"),
                ),
                flush=True,
            )
            break

    if not best_path.is_file():
        raise RuntimeError("training finished without a best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    ranking: dict[str, Any] | None = None
    if not args.skip_final_test:
        test_loader = make_split_loader(
            args, "test", args.test_max_samples,
            args.eval_workers, args.seed, 0,
        )
        ranking = evaluate(
            model, test_loader, device, args.hidden_scales,
            args.output_scale, args, "test",
        )
        # The verifier calls the never-tuned sealed split "ranking".
        best_metrics = dict(best.get("metrics", {}))
        best_metrics["ranking"] = ranking
        best["metrics"] = best_metrics
        best["sealed_test_evaluated"] = True
        atomic_torch_save(best, best_path)
    summary = {
        "event": "training_complete",
        "selected_epoch": int(best["epoch"]),
        "best_val_loss": float(best["best_val_loss"]),
        "history": history,
        "selection": best.get("metrics", {}).get("selection"),
        "ranking": ranking,
        "checkpoint": str(best_path),
    }
    atomic_write_text(summary_path, json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
