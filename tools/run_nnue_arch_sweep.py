from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_nnue_architecture.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NNUE architecture training sweep")
    parser.add_argument("--data", default=REPO_ROOT / "data" / "robotmoon_test80_2024_5m.jsonl", type=Path)
    parser.add_argument("--architectures", nargs="+", default=["E", "F", "G", "H"])
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=50_000)
    parser.add_argument("--test-max-samples", type=int, default=50_000)
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--shuffle-block-size", type=int, default=1_000_000)
    parser.add_argument("--progress-batches", type=int, default=1000)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--output-dir", default=REPO_ROOT / "models" / "nnue_arch_sweep", type=Path)
    parser.add_argument("--log-dir", default=REPO_ROOT / "logs", type=Path)
    parser.add_argument("--tag", default=time.strftime("nnue_arch_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--python", default="/usr/bin/python3")
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
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--val-max-samples",
        str(args.val_max_samples),
        "--test-max-samples",
        str(args.test_max_samples),
        "--shuffle-block-size",
        str(args.shuffle_block_size),
        "--progress-batches",
        str(args.progress_batches),
        "--eval-progress-batches",
        str(args.eval_progress_batches),
    ]
    if args.torch_threads > 0:
        cmd += ["--torch-threads", str(args.torch_threads)]
    if args.train_max_samples is not None:
        cmd += ["--train-max-samples", str(args.train_max_samples)]
    if args.train_max_batches is not None:
        cmd += ["--train-max-batches", str(args.train_max_batches)]
    if args.eval_max_batches is not None:
        cmd += ["--eval-max-batches", str(args.eval_max_batches)]
    return cmd


def main() -> int:
    args = parse_args()
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
            if code != 0:
                failures += 1
            print(f"finished arch={arch} code={code} log={log_path}", flush=True)

    print(f"done failures={failures} tag={args.tag}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
