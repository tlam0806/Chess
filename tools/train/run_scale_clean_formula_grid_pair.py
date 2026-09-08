#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_SCRIPT = REPO_ROOT / "tools" / "train" / "run_quantized_scale_grid.py"
ANALYZE_SCRIPT = REPO_ROOT / "tools" / "analyze" / "analyze_compact_cp_distribution.py"


FORMULAS = (
    ("c181_d128", 181, 128),
    ("c255_d256", 255, 256),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two requested scale-clean SCReLU formula grids in parallel"
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="cbin")
    parser.add_argument("--eval-data-format", choices=["auto", "jsonl", "cbin"], default="cbin")
    parser.add_argument("--arch", default="F2")
    parser.add_argument("--activation", default="screlu_all")
    parser.add_argument("--hidden-scales", default="8 16 32 64 128")
    parser.add_argument("--output-scales", default="4 8 16 32")
    parser.add_argument("--jobs-per-formula", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lr-drop-patience", type=int, default=2)
    parser.add_argument("--lr-drop-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--train-max-samples", type=int, default=2_000_000)
    parser.add_argument("--val-max-samples", type=int, default=500_000)
    parser.add_argument("--test-max-samples", type=int, default=100_000)
    parser.add_argument("--calibration-max-batches", type=int, default=50)
    parser.add_argument("--calibration-percentile", type=float, default=99.5)
    parser.add_argument(
        "--feature-weight-scale",
        type=int,
        default=None,
        help="defaults to hidden_clip for each formula under scale_clean",
    )
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--screlu-init-fraction", type=float, default=0.25)
    parser.add_argument("--screlu-first-bias-fraction", type=float, default=0.10)
    parser.add_argument("--reject-initial-zero-rate", type=float, default=0.995)
    parser.add_argument("--reject-initial-clip-rate", type=float, default=0.80)
    parser.add_argument("--cp-huber-delta", type=float, default=200.0)
    parser.add_argument("--progress-batches", type=int, default=100)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--shuffle-block-size", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models" / "scale_clean_formula_grid")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--tag", default=time.strftime("scale_clean_formula_grid_%Y%m%d_%H%M%S"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-skip-final-test", action="store_true")
    return parser.parse_args()


def build_formula_command(args: argparse.Namespace, label: str, hidden_clip: int, divisor: int) -> list[str]:
    feature_weight_scale = (
        hidden_clip if args.feature_weight_scale is None else args.feature_weight_scale
    )
    cmd = [
        args.python,
        str(GRID_SCRIPT),
        "--data",
        str(args.data),
        "--eval-data",
        str(args.eval_data),
        "--data-format",
        args.data_format,
        "--eval-data-format",
        args.eval_data_format,
        "--architectures",
        args.arch,
        "--activations",
        args.activation,
        "--hidden-scales",
        args.hidden_scales,
        "--output-scales",
        args.output_scales,
        "--jobs",
        str(args.jobs_per_formula),
        "--device",
        args.device,
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
        "--workers",
        str(args.workers),
        "--eval-workers",
        str(args.eval_workers),
        "--torch-threads",
        str(args.torch_threads),
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
        str(feature_weight_scale),
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
        "--log-initial-saturation",
        "--cp-huber-delta",
        str(args.cp_huber_delta),
        "--hidden-clip",
        str(hidden_clip),
        "--screlu-divisor",
        str(divisor),
        "--quantization-convention",
        "scale_clean",
        "--progress-batches",
        str(args.progress_batches),
        "--eval-progress-batches",
        str(args.eval_progress_batches),
        "--shuffle-block-size",
        str(args.shuffle_block_size),
        "--output-dir",
        str(args.output_dir),
        "--log-dir",
        str(args.log_dir),
        "--tag",
        f"{args.tag}_{label}",
        "--python",
        args.python,
        "--max-hours",
        str(args.max_hours),
    ]
    if args.resume:
        cmd.append("--resume")
    if not args.no_skip_final_test:
        cmd.append("--skip-final-test")
    return cmd


def main() -> int:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    parent_log_path = args.log_dir / f"{args.tag}.pair_runner.log"
    validation_baseline_path = args.log_dir / f"{args.tag}.validation_baseline.json"
    with validation_baseline_path.open("w", encoding="utf-8") as baseline_log:
        subprocess.run(
            [
                args.python,
                str(ANALYZE_SCRIPT),
                "--data",
                str(args.eval_data),
                "--max-samples",
                str(args.val_max_samples),
            ],
            cwd=REPO_ROOT,
            stdout=baseline_log,
            check=True,
        )
    processes: list[tuple[str, subprocess.Popen[bytes], Path]] = []

    with parent_log_path.open("w", encoding="utf-8") as parent_log:
        parent_log.write(f"tag={args.tag}\n")
        parent_log.write(f"formulas={FORMULAS}\n")
        parent_log.write(f"jobs_per_formula={args.jobs_per_formula}\n")
        parent_log.write(f"total_concurrent_jobs={args.jobs_per_formula * len(FORMULAS)}\n")
        parent_log.write(f"validation_baseline={validation_baseline_path}\n")
        parent_log.flush()
        for label, hidden_clip, divisor in FORMULAS:
            cmd = build_formula_command(args, label, hidden_clip, divisor)
            formula_log_path = args.log_dir / f"{args.tag}_{label}.stdout.log"
            formula_log = formula_log_path.open("w", encoding="utf-8")
            formula_log.write("# " + " ".join(cmd) + "\n")
            formula_log.flush()
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=formula_log,
                stderr=subprocess.STDOUT,
            )
            formula_log.close()
            processes.append((label, process, formula_log_path))
            parent_log.write(
                f"started label={label} hidden_clip={hidden_clip} divisor={divisor} "
                f"pid={process.pid} stdout={formula_log_path}\n"
            )
            parent_log.flush()

        failures = 0
        for label, process, formula_log_path in processes:
            code = process.wait()
            if code != 0:
                failures += 1
            parent_log.write(f"finished label={label} code={code} stdout={formula_log_path}\n")
            parent_log.flush()
        parent_log.write(f"done failures={failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
