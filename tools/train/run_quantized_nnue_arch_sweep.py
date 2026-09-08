#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from tools.debug.quantized_pipeline_preflight import validate_sweep_args
from tools.train.quantized_training_log import parse_quantized_training_log
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train" / "train_quantized_nnue_architecture.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantized NNUE architecture training sweep")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--architectures", nargs="+", default=["A", "B", "C", "D", "E", "F", "G", "H"])
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lr-after-warmup", type=float, default=None)
    parser.add_argument("--lr-drop-patience", type=int, default=0)
    parser.add_argument("--lr-drop-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--train-max-samples", type=int, default=2_000_000)
    parser.add_argument("--val-max-samples", type=int, default=100_000)
    parser.add_argument("--test-max-samples", type=int, default=100_000)
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--calibration-max-batches", type=int, default=50)
    parser.add_argument("--calibration-percentile", type=float, default=99.5)
    parser.add_argument("--feature-weight-scale", type=int, default=255)
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--fixed-hidden-scale", type=int, default=None)
    parser.add_argument("--fixed-output-scale", type=int, default=None)
    parser.add_argument("--shuffle-block-size", type=int, default=1_000_000)
    parser.add_argument("--progress-batches", type=int, default=500)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--output-dir", default=REPO_ROOT / "models" / "quantized_nnue_arch_sweep", type=Path)
    parser.add_argument("--log-dir", default=REPO_ROOT / "logs", type=Path)
    parser.add_argument("--tag", default=time.strftime("quant_nnue_arch_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--sleep-when-done", action="store_true")
    return parser.parse_args()


def make_command(args: argparse.Namespace, arch: str) -> list[str]:
    cmd = [
        args.python,
        str(TRAIN_SCRIPT),
        "--arch",
        arch,
        "--data",
        str(args.data),
        "--data-format",
        args.data_format,
        "--output-dir",
        str(args.output_dir / args.tag),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
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
        "--shuffle-block-size",
        str(args.shuffle_block_size),
        "--progress-batches",
        str(args.progress_batches),
        "--eval-progress-batches",
        str(args.eval_progress_batches),
    ]
    if args.torch_threads > 0:
        cmd += ["--torch-threads", str(args.torch_threads)]
    if args.lr_after_warmup is not None:
        cmd += ["--lr-after-warmup", str(args.lr_after_warmup)]
    if args.fixed_hidden_scale is not None:
        cmd += ["--fixed-hidden-scale", str(args.fixed_hidden_scale)]
    if args.fixed_output_scale is not None:
        cmd += ["--fixed-output-scale", str(args.fixed_output_scale)]
    if args.train_max_batches is not None:
        cmd += ["--train-max-batches", str(args.train_max_batches)]
    if args.eval_max_batches is not None:
        cmd += ["--eval-max-batches", str(args.eval_max_batches)]
    return cmd


def main() -> int:
    args = parse_args()
    validate_sweep_args(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.tag).mkdir(parents=True, exist_ok=True)

    pending: deque[str] = deque(args.architectures)
    running: dict[subprocess.Popen[bytes], tuple[str, object, Path]] = {}
    failures = 0

    manifest_path = args.log_dir / f"{args.tag}.manifest"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write(f"tag={args.tag}\n")
        manifest.write(f"data={args.data}\n")
        manifest.write(f"output_dir={args.output_dir / args.tag}\n")
        manifest.write(f"architectures={' '.join(args.architectures)}\n")
        manifest.write(f"jobs={args.jobs}\n")
        manifest.write(f"train_max_samples={args.train_max_samples}\n")
        manifest.write(f"val_max_samples={args.val_max_samples}\n")
        manifest.write(f"test_max_samples={args.test_max_samples}\n")
        manifest.write(f"lr={args.lr}\n")
        manifest.write(f"warmup_epochs={args.warmup_epochs}\n")
        manifest.write(f"lr_after_warmup={args.lr_after_warmup}\n")
        manifest.write(f"lr_drop_patience={args.lr_drop_patience}\n")
        manifest.write(f"lr_drop_factor={args.lr_drop_factor}\n")
        manifest.write(f"min_lr={args.min_lr}\n")
        manifest.write(f"feature_weight_scale={args.feature_weight_scale}\n")
        manifest.write(f"linear_weight_scale={args.linear_weight_scale}\n")
        manifest.write(f"output_weight_scale={args.output_weight_scale}\n")
        manifest.write(f"fixed_hidden_scale={args.fixed_hidden_scale}\n")
        manifest.write(f"fixed_output_scale={args.fixed_output_scale}\n")
        manifest.write(f"sleep_when_done={args.sleep_when_done}\n")

    while pending or running:
        while pending and len(running) < args.jobs:
            arch = pending.popleft()
            log_path = args.log_dir / f"{args.tag}_{arch}.log"
            log_file = log_path.open("w", encoding="utf-8")
            cmd = make_command(args, arch)
            log_file.write("# " + " ".join(cmd) + "\n")
            log_file.flush()
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            running[process] = (arch, log_file, log_path)
            print(f"started arch={arch} pid={process.pid} log={log_path}", flush=True)

        time.sleep(5)
        for process in list(running):
            code = process.poll()
            if code is None:
                continue
            arch, log_file, log_path = running.pop(process)
            log_file.close()
            parsed = parse_quantized_training_log(log_path)
            valid = code == 0 and parsed.get("completed") is True and "error" not in parsed
            if not valid:
                failures += 1
            print(f"finished arch={arch} code={code} valid={valid} log={log_path}", flush=True)

    print(f"done failures={failures} tag={args.tag}", flush=True)
    if args.sleep_when_done and failures == 0:
        subprocess.run(["pmset", "sleepnow"], cwd=REPO_ROOT, check=False)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
