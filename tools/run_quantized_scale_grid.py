#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from itertools import product
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.quantized_training_log import parse_quantized_training_log
from tools.quantized_pipeline_preflight import validate_sweep_args
from nn.quantized_nnue_architectures import (
    HIDDEN_CLIP,
    QUANTIZATION_CONVENTION_CHOICES,
    QUANTIZATION_CONVENTION_LEGACY,
    QUANTIZED_ARCHITECTURES,
)


TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_quantized_nnue_architecture.py"


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split()]


def parse_selected_scale_configs(value: str) -> list[tuple[tuple[int, ...], int]]:
    configs: list[tuple[tuple[int, ...], int]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for item in value.replace(",", " ").split():
        try:
            hidden_part, output_part = item.rsplit(":", 1)
            hidden_scales = tuple(int(part) for part in hidden_part.lower().split("x"))
            output_scale = int(output_part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "selected configs must use hs1xhs2:output syntax"
            ) from error
        config = (hidden_scales, output_scale)
        if not hidden_scales or any(value <= 0 for value in hidden_scales) or output_scale <= 0:
            raise argparse.ArgumentTypeError("selected scales must be positive")
        if config in seen:
            raise argparse.ArgumentTypeError(f"duplicate selected config: {item}")
        seen.add(config)
        configs.append(config)
    if not configs:
        raise argparse.ArgumentTypeError("selected configs must not be empty")
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantized NNUE hidden/output scale grid")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--eval-data", type=Path, default=None)
    parser.add_argument("--architectures", nargs="+", default=["A", "B", "C", "D", "E", "F", "G", "H"])
    parser.add_argument("--activations", nargs="+", default=["relu"])
    parser.add_argument("--hidden-scales", type=parse_int_list, default=parse_int_list("128 256 384"))
    parser.add_argument("--output-scales", type=parse_int_list, default=parse_int_list("8 16 32 64"))
    parser.add_argument(
        "--selected-scale-configs",
        type=parse_selected_scale_configs,
        default=None,
        help="run only exact hidden/output tuples, e.g. '32x16:16,16x64:16'",
    )
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--eval-data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--cp-huber-delta", type=float, default=200.0)
    parser.add_argument("--hidden-clip", type=int, default=HIDDEN_CLIP)
    parser.add_argument("--screlu-divisor", type=int, default=255)
    parser.add_argument(
        "--quantization-convention",
        choices=QUANTIZATION_CONVENTION_CHOICES,
        default=QUANTIZATION_CONVENTION_LEGACY,
        help="integer arithmetic convention passed through to the quantized trainer",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lr-after-warmup", type=float, default=None)
    parser.add_argument("--lr-drop-patience", type=int, default=2)
    parser.add_argument("--lr-drop-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--train-max-samples", type=int, default=2_000_000)
    parser.add_argument("--train-data-all-records", action="store_true")
    parser.add_argument("--val-max-samples", type=int, default=100_000)
    parser.add_argument("--test-max-samples", type=int, default=100_000)
    parser.add_argument("--calibration-max-batches", type=int, default=50)
    parser.add_argument("--calibration-percentile", type=float, default=99.5)
    parser.add_argument("--feature-weight-scale", type=int, default=255)
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--screlu-init-fraction", type=float, default=0.0)
    parser.add_argument("--screlu-first-bias-fraction", type=float, default=0.0)
    parser.add_argument("--log-initial-saturation", action="store_true")
    parser.add_argument("--reject-initial-zero-rate", type=float, default=1.0)
    parser.add_argument("--reject-initial-clip-rate", type=float, default=1.0)
    parser.add_argument("--shuffle-block-size", type=int, default=1_000_000)
    parser.add_argument("--progress-batches", type=int, default=250)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--output-dir", default=REPO_ROOT / "models" / "quantized_scale_grid", type=Path)
    parser.add_argument("--log-dir", default=REPO_ROOT / "logs", type=Path)
    parser.add_argument("--tag", default=time.strftime("quant_scale_grid_%Y%m%d_%H%M%S"))
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid rows in an existing summary and run only unfinished configs",
    )
    return parser.parse_args()


def normalize_hidden_scales(hidden_scales: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden_scales, int):
        return (hidden_scales,)
    return tuple(int(value) for value in hidden_scales)


def hidden_scales_label(hidden_scales: int | Sequence[int]) -> str:
    return "x".join(str(value) for value in normalize_hidden_scales(hidden_scales))


def make_config_name(
    tag: str,
    arch: str,
    activation: str,
    hidden_scales: int | Sequence[int],
    output_scale: int,
) -> str:
    return f"{tag}_{arch}_{activation}_hs{hidden_scales_label(hidden_scales)}_os{output_scale}"


def make_command(
    args: argparse.Namespace,
    arch: str,
    activation: str,
    hidden_scales: int | Sequence[int],
    output_scale: int,
) -> list[str]:
    normalized_hidden_scales = normalize_hidden_scales(hidden_scales)
    config_name = make_config_name(
        args.tag,
        arch,
        activation,
        normalized_hidden_scales,
        output_scale,
    )
    cmd = [
        args.python,
        str(TRAIN_SCRIPT),
        "--arch",
        arch,
        "--activation",
        activation,
        "--data",
        str(args.data),
        "--data-format",
        args.data_format,
        "--cp-huber-delta",
        str(args.cp_huber_delta),
        "--hidden-clip",
        str(args.hidden_clip),
        "--screlu-divisor",
        str(getattr(args, "screlu_divisor", 255)),
        "--quantization-convention",
        str(args.quantization_convention),
        "--output-dir",
        str(args.output_dir / args.tag / config_name),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--lr-schedule",
        args.lr_schedule,
        "--lr-warmup-steps",
        str(args.lr_warmup_steps),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--lr-drop-patience",
        str(args.lr_drop_patience),
        "--lr-drop-factor",
        str(args.lr_drop_factor),
        "--min-lr",
        str(args.min_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--eval-workers",
        str(args.eval_workers),
        "--train-max-samples",
        str(args.train_max_samples),
        "--val-max-samples",
        str(args.val_max_samples),
        "--test-max-samples",
        str(args.test_max_samples),
        "--calibration-max-batches",
        str(args.calibration_max_batches),
        "--calibration-percentile",
        str(args.calibration_percentile),
        "--feature-weight-scale",
        str(args.feature_weight_scale),
        "--linear-weight-scale",
        str(args.linear_weight_scale),
        "--output-weight-scale",
        str(args.output_weight_scale),
        "--screlu-init-fraction",
        str(args.screlu_init_fraction),
        "--screlu-first-bias-fraction",
        str(args.screlu_first_bias_fraction),
        "--reject-initial-zero-rate",
        str(args.reject_initial_zero_rate),
        "--reject-initial-clip-rate",
        str(args.reject_initial_clip_rate),
        "--shuffle-block-size",
        str(args.shuffle_block_size),
        "--progress-batches",
        str(args.progress_batches),
        "--eval-progress-batches",
        str(args.eval_progress_batches),
        "--fixed-hidden-scales",
        *(str(value) for value in normalized_hidden_scales),
        "--fixed-output-scale",
        str(output_scale),
    ]
    if args.eval_data is not None:
        cmd += [
            "--eval-data",
            str(args.eval_data),
            "--eval-data-format",
            args.eval_data_format,
            "--eval-data-all-records-for-val",
        ]
    if args.train_data_all_records:
        cmd.append("--train-data-all-records")
    if args.torch_threads > 0:
        cmd += ["--torch-threads", str(args.torch_threads)]
    if args.lr_after_warmup is not None:
        cmd += ["--lr-after-warmup", str(args.lr_after_warmup)]
    if getattr(args, "skip_final_test", False):
        cmd.append("--skip-final-test")
    if args.log_initial_saturation:
        cmd.append("--log-initial-saturation")
    return cmd


def parse_result(log_path: Path) -> dict[str, object]:
    return parse_quantized_training_log(log_path)


def build_configs(
    architectures: Sequence[str],
    activations: Sequence[str],
    hidden_scale_values: Sequence[int],
    output_scales: Sequence[int],
) -> list[tuple[str, str, tuple[int, ...], int]]:
    return [
        (arch, activation, hidden_scales, output_scale)
        for arch in architectures
        for activation in activations
        for hidden_scales in product(
            hidden_scale_values,
            repeat=len(QUANTIZED_ARCHITECTURES[arch].hidden_sizes) - 1,
        )
        for output_scale in output_scales
    ]


def build_selected_configs(
    architectures: Sequence[str],
    activations: Sequence[str],
    selected_scale_configs: Sequence[tuple[tuple[int, ...], int]],
) -> list[tuple[str, str, tuple[int, ...], int]]:
    configs: list[tuple[str, str, tuple[int, ...], int]] = []
    for arch in architectures:
        expected_hidden_scales = len(QUANTIZED_ARCHITECTURES[arch].hidden_sizes) - 1
        for hidden_scales, output_scale in selected_scale_configs:
            if len(hidden_scales) != expected_hidden_scales:
                raise ValueError(
                    f"{arch} requires {expected_hidden_scales} hidden scales, "
                    f"got {hidden_scales}"
                )
            for activation in activations:
                configs.append((arch, activation, hidden_scales, output_scale))
    return configs


def config_key(
    arch: str,
    activation: str,
    hidden_scales: int | Sequence[int],
    output_scale: int,
) -> tuple[str, str, tuple[int, ...], int]:
    return arch, activation, normalize_hidden_scales(hidden_scales), int(output_scale)


def load_completed_config_keys(
    summary_path: Path,
    expected_configs: Sequence[tuple[str, str, tuple[int, ...], int]],
) -> set[tuple[str, str, tuple[int, ...], int]]:
    if not summary_path.exists():
        return set()

    expected = set(expected_configs)
    completed: set[tuple[str, str, tuple[int, ...], int]] = set()
    for line_number, line in enumerate(
        summary_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = config_key(
                str(row["arch"]),
                str(row["activation"]),
                row["hidden_scales"],
                int(row["output_scale"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"invalid resume row {summary_path}:{line_number}: {error}"
            ) from error
        if key not in expected:
            raise RuntimeError(
                f"resume summary contains a config outside the requested grid: {key}"
            )
        if key in completed:
            raise RuntimeError(f"resume summary contains duplicate config: {key}")
        if row.get("code") != 0 or row.get("completed") is not True or "error" in row:
            raise RuntimeError(
                f"resume summary contains an unsuccessful config; remove or repair it first: {key}"
            )
        completed.add(key)
    return completed


def main() -> int:
    args = parse_args()
    validate_sweep_args(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.tag).mkdir(parents=True, exist_ok=True)

    if args.selected_scale_configs is not None:
        all_configs = build_selected_configs(
            args.architectures,
            args.activations,
            args.selected_scale_configs,
        )
    else:
        all_configs = build_configs(
            args.architectures,
            args.activations,
            args.hidden_scales,
            args.output_scales,
        )
    summary_path = args.log_dir / f"{args.tag}.summary.jsonl"
    completed_before_start = (
        load_completed_config_keys(summary_path, all_configs) if args.resume else set()
    )
    configs: deque[tuple[str, str, tuple[int, ...], int]] = deque(
        config for config in all_configs if config not in completed_before_start
    )
    running: dict[
        subprocess.Popen[bytes],
        tuple[str, str, tuple[int, ...], int, object, Path],
    ] = {}
    failures = 0
    skipped_due_deadline = 0
    total_configs = len(all_configs)
    started_at = time.monotonic()
    deadline = started_at + args.max_hours * 3600.0 if args.max_hours > 0 else None

    runner_log_path = args.log_dir / f"{args.tag}.runner.log"
    manifest_path = args.log_dir / f"{args.tag}.manifest"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write(f"tag={args.tag}\n")
        manifest.write(f"data={args.data}\n")
        manifest.write(f"eval_data={args.eval_data}\n")
        manifest.write(f"cp_huber_delta={args.cp_huber_delta}\n")
        manifest.write(f"hidden_clip={args.hidden_clip}\n")
        manifest.write(f"screlu_divisor={args.screlu_divisor}\n")
        manifest.write(f"quantization_convention={args.quantization_convention}\n")
        manifest.write(f"output_dir={args.output_dir / args.tag}\n")
        manifest.write(f"architectures={' '.join(args.architectures)}\n")
        manifest.write(f"activations={' '.join(args.activations)}\n")
        manifest.write(f"hidden_scales={' '.join(str(x) for x in args.hidden_scales)}\n")
        manifest.write("hidden_scale_grid=cartesian_per_dense_hidden_layer\n")
        manifest.write(f"output_scales={' '.join(str(x) for x in args.output_scales)}\n")
        manifest.write(f"selected_scale_configs={args.selected_scale_configs}\n")
        manifest.write(f"jobs={args.jobs}\n")
        manifest.write(f"device={args.device}\n")
        manifest.write(f"epochs={args.epochs}\n")
        manifest.write(f"train_max_samples={args.train_max_samples}\n")
        manifest.write(f"lr={args.lr}\n")
        manifest.write(f"lr_schedule={args.lr_schedule}\n")
        manifest.write(f"lr_warmup_steps={args.lr_warmup_steps}\n")
        manifest.write(f"train_data_all_records={args.train_data_all_records}\n")
        manifest.write(f"screlu_init_fraction={args.screlu_init_fraction}\n")
        manifest.write(
            f"screlu_first_bias_fraction={args.screlu_first_bias_fraction}\n"
        )
        manifest.write(f"max_hours={args.max_hours}\n")
        manifest.write(f"resume={args.resume}\n")
        manifest.write(f"completed_before_start={len(completed_before_start)}\n")

    output_mode = "a" if args.resume else "w"
    with runner_log_path.open(output_mode, encoding="utf-8") as runner_log, summary_path.open(
        output_mode, encoding="utf-8"
    ) as summary:
        print(f"tag={args.tag}", file=runner_log, flush=True)
        if args.resume:
            print(
                f"resume completed={len(completed_before_start)} "
                f"remaining={len(configs)} total={total_configs}",
                file=runner_log,
                flush=True,
            )
        while configs or running:
            while configs and len(running) < args.jobs:
                if deadline is not None and time.monotonic() >= deadline:
                    skipped_due_deadline = len(configs)
                    print(
                        f"deadline reached; {skipped_due_deadline} configs were not started",
                        file=runner_log,
                        flush=True,
                    )
                    configs.clear()
                    break
                arch, activation, hidden_scales, output_scale = configs.popleft()
                config_name = make_config_name(
                    args.tag,
                    arch,
                    activation,
                    hidden_scales,
                    output_scale,
                )
                log_path = args.log_dir / f"{config_name}.log"
                log_file = log_path.open("w", encoding="utf-8")
                cmd = make_command(args, arch, activation, hidden_scales, output_scale)
                log_file.write("# " + " ".join(cmd) + "\n")
                log_file.flush()
                process = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                running[process] = (
                    arch,
                    activation,
                    hidden_scales,
                    output_scale,
                    log_file,
                    log_path,
                )
                print(
                    f"started arch={arch} activation={activation} "
                    f"hidden_scales={hidden_scales_label(hidden_scales)} "
                    f"output_scale={output_scale} "
                    f"pid={process.pid} log={log_path}",
                    file=runner_log,
                    flush=True,
                )

            if running:
                time.sleep(5)
            for process in list(running):
                code = process.poll()
                if code is None:
                    continue
                arch, activation, hidden_scales, output_scale, log_file, log_path = running.pop(process)
                log_file.close()
                parsed = parse_result(log_path)
                result = {
                    "arch": arch,
                    "activation": activation,
                    "hidden_scales": list(hidden_scales),
                    "output_scale": output_scale,
                    "code": code,
                    "log": str(log_path),
                    "elapsed_total_sec": round(time.monotonic() - started_at, 3),
                    **parsed,
                }
                valid = code == 0 and parsed.get("completed") is True and "error" not in parsed
                if not valid:
                    failures += 1
                print(json.dumps(result, separators=(",", ":")), file=summary, flush=True)
                print(
                    f"finished arch={arch} hidden_scales={hidden_scales_label(hidden_scales)} "
                    f"output_scale={output_scale} "
                    f"code={code} valid={valid}",
                    file=runner_log,
                    flush=True,
                )

        completed_configs = total_configs - skipped_due_deadline
        print(
            f"done failures={failures} completed_configs={completed_configs} "
            f"skipped_due_deadline={skipped_due_deadline} total_configs={total_configs} tag={args.tag}",
            file=runner_log,
            flush=True,
        )
    return 1 if failures or skipped_due_deadline else 0


if __name__ == "__main__":
    raise SystemExit(main())
