#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "tools" / "train_quantized_nnue_architecture.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the Stage-2 winner with clip180/div128 ScReLU")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--python", default=".venv/bin/python")
    args = parser.parse_args()
    configs = [
        {"label": "baseline_c255_d256_hs8x8", "clip": 255, "divisor": 256, "hs": [8, 8]},
        {"label": "c180_d128_hs8x8", "clip": 180, "divisor": 128, "hs": [8, 8]},
        {"label": "c180_d128_hs8x16", "clip": 180, "divisor": 128, "hs": [8, 16]},
        {"label": "c180_d128_hs16x8", "clip": 180, "divisor": 128, "hs": [16, 8]},
        {"label": "c180_d128_hs16x16", "clip": 180, "divisor": 128, "hs": [16, 16]},
    ]
    log_dir = REPO_ROOT / "logs"
    model_root = REPO_ROOT / "models" / "quantized_scale_grid" / args.tag
    log_dir.mkdir(exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tag": args.tag,
        "comparison": "Stage-2 top-1 formula versus clip180/div128",
        "data": str(args.data),
        "eval_data": str(args.eval_data),
        "common": {"arch": "F2", "activation": "screlu_all", "output_scale": 8,
                   "epochs": 8, "train_samples_per_epoch": 5_000_000,
                   "val_samples": 500_000, "cp_huber_delta": 200.0, "lr": 0.0005},
        "configs": configs,
    }
    (log_dir / f"{args.tag}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary_path = log_dir / f"{args.tag}.summary.jsonl"
    runner_path = log_dir / f"{args.tag}.runner.log"
    failures = 0
    with summary_path.open("w") as summary, runner_path.open("w") as runner:
        for config in configs:
            label = str(config["label"])
            output = model_root / label
            log_path = log_dir / f"{args.tag}_{label}.log"
            command = [
                args.python, str(TRAINER), "--arch", "F2", "--activation", "screlu_all",
                "--hidden-clip", str(config["clip"]), "--screlu-divisor", str(config["divisor"]),
                "--data", str(args.data), "--data-format", "cbin",
                "--eval-data", str(args.eval_data), "--eval-data-format", "cbin",
                "--eval-data-all-records-for-val",
                "--output-dir", str(output), "--epochs", "8", "--patience", "4",
                "--batch-size", "8192", "--lr", "0.0005", "--warmup-epochs", "0",
                "--lr-drop-patience", "2", "--lr-drop-factor", "0.5", "--min-lr", "0.00005",
                "--weight-decay", "0.0001", "--device", args.device,
                "--workers", str(args.workers), "--eval-workers", "0",
                "--torch-threads", str(args.torch_threads),
                "--train-max-samples", "5000000", "--val-max-samples", "500000",
                "--test-max-samples", "500000", "--calibration-max-batches", "50",
                "--feature-weight-scale", "255", "--linear-weight-scale", "64",
                "--output-weight-scale", "16", "--shuffle-block-size", "1000000",
                "--progress-batches", "100", "--eval-progress-batches", "0",
                "--fixed-hidden-scales", *(str(v) for v in config["hs"]),
                "--fixed-output-scale", "8", "--cp-huber-delta", "200", "--skip-final-test",
            ]
            print(f"start {label}", file=runner, flush=True)
            started = time.monotonic()
            with log_path.open("w") as log:
                print("# " + " ".join(command), file=log, flush=True)
                code = subprocess.call(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            best = None
            complete = False
            for line in log_path.read_text(errors="ignore").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "epoch" and event.get("improved"):
                    best = event
                elif event.get("event") == "training_complete":
                    complete = True
            row = {**config, "code": code, "completed": complete,
                   "elapsed_sec": round(time.monotonic() - started, 3),
                   "best_epoch": best.get("epoch") if best else None,
                   "best_val_loss": best.get("val_loss") if best else None,
                   "best_val_cp": best.get("val_cp") if best else None,
                   "log": str(log_path)}
            print(json.dumps(row, separators=(",", ":")), file=summary, flush=True)
            valid = code == 0 and complete and best is not None
            failures += not valid
            print(f"done valid={valid} {label} cp={row['best_val_cp']}", file=runner, flush=True)
            if not valid:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
