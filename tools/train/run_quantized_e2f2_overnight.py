#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from tools.train.quantized_training_log import parse_quantized_training_log
from tools.debug.quantized_pipeline_preflight import validate_sweep_args


TRAIN_SCRIPT = REPO_ROOT / "tools" / "train" / "train_quantized_nnue_architecture.py"


@dataclass(frozen=True)
class ScaleConfig:
    arch: str
    hidden_scale: int
    output_scale: int


def parse_int_list(value: str) -> list[int]:
    items = [part for part in value.replace(",", " ").split() if part]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    return [int(part) for part in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight E2/F2 quantized NNUE scale tune")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--architectures", nargs="+", default=["E2", "F2"])
    parser.add_argument(
        "--hidden-scales",
        type=parse_int_list,
        default=parse_int_list("64 96 128 160 192 256"),
    )
    parser.add_argument(
        "--output-scales",
        type=parse_int_list,
        default=parse_int_list("8 16 32 64"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--feature-weight-scale", type=int, default=255)
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-drop-patience", type=int, default=2)
    parser.add_argument("--lr-drop-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=6.25e-5)
    parser.add_argument("--shuffle-block-size", type=int, default=1_000_000)
    parser.add_argument("--progress-batches", type=int, default=250)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--calibration-max-batches", type=int, default=50)
    parser.add_argument("--calibration-percentile", type=float, default=99.5)
    parser.add_argument("--seed", type=int, default=20260714)

    parser.add_argument("--stage1-train-samples", type=int, default=2_000_000)
    parser.add_argument("--stage1-val-samples", type=int, default=100_000)
    parser.add_argument("--stage1-test-samples", type=int, default=100_000)
    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage1-patience", type=int, default=2)
    parser.add_argument("--stage1-lr", type=float, default=0.001)

    parser.add_argument("--stage2-train-samples", type=int, default=10_000_000)
    parser.add_argument("--stage2-val-samples", type=int, default=500_000)
    parser.add_argument("--stage2-test-samples", type=int, default=500_000)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage2-patience", type=int, default=2)
    parser.add_argument("--stage2-lr", type=float, default=0.001)

    parser.add_argument("--top-k-total", type=int, default=6)
    parser.add_argument("--top-k-per-arch", type=int, default=3)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep-when-done", action="store_true")
    parser.add_argument("--output-dir", default=REPO_ROOT / "models" / "quantized_e2f2_overnight", type=Path)
    parser.add_argument("--log-dir", default=REPO_ROOT / "logs", type=Path)
    parser.add_argument("--tag", default=time.strftime("quant_e2f2_overnight_%Y%m%d_%H%M%S"))
    parser.add_argument("--python", default=".venv/bin/python")
    return parser.parse_args()


def config_name(tag: str, stage: str, cfg: ScaleConfig) -> str:
    return f"{tag}_{stage}_{cfg.arch}_hs{cfg.hidden_scale}_os{cfg.output_scale}"


def stage1_configs(args: argparse.Namespace) -> list[ScaleConfig]:
    return [
        ScaleConfig(arch, hidden_scale, output_scale)
        for arch in args.architectures
        for hidden_scale in args.hidden_scales
        for output_scale in args.output_scales
    ]


def train_command(args: argparse.Namespace, cfg: ScaleConfig, stage: str) -> list[str]:
    if stage == "stage1":
        epochs = args.stage1_epochs
        patience = args.stage1_patience
        train_samples = args.stage1_train_samples
        val_samples = args.stage1_val_samples
        test_samples = args.stage1_test_samples
        lr = args.stage1_lr
    elif stage == "stage2":
        epochs = args.stage2_epochs
        patience = args.stage2_patience
        train_samples = args.stage2_train_samples
        val_samples = args.stage2_val_samples
        test_samples = args.stage2_test_samples
        lr = args.stage2_lr
    else:
        raise ValueError(stage)

    cmd = [
        args.python,
        str(TRAIN_SCRIPT),
        "--arch",
        cfg.arch,
        "--data",
        str(args.data),
        "--data-format",
        args.data_format,
        "--output-dir",
        str(args.output_dir / args.tag / config_name(args.tag, stage, cfg)),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(lr),
        "--warmup-epochs",
        "0",
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
        str(train_samples),
        "--val-max-samples",
        str(val_samples),
        "--test-max-samples",
        str(test_samples),
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
        "--seed",
        str(args.seed),
        "--fixed-hidden-scale",
        str(cfg.hidden_scale),
        "--fixed-output-scale",
        str(cfg.output_scale),
    ]
    if args.torch_threads > 0:
        cmd += ["--torch-threads", str(args.torch_threads)]
    return cmd


def parse_log(log_path: Path) -> dict[str, Any]:
    return parse_quantized_training_log(log_path)


def write_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as out:
        print(json.dumps(item, separators=(",", ":")), file=out, flush=True)


def run_one(args: argparse.Namespace, cfg: ScaleConfig, stage: str, runner_log: Path) -> dict[str, Any]:
    log_path = args.log_dir / f"{config_name(args.tag, stage, cfg)}.log"
    cmd = train_command(args, cfg, stage)
    with runner_log.open("a", encoding="utf-8") as log:
        print(
            f"start {stage} arch={cfg.arch} hs={cfg.hidden_scale} os={cfg.output_scale} log={log_path}",
            file=log,
            flush=True,
        )
        if args.dry_run:
            print("# " + " ".join(cmd), file=log, flush=True)
            return {
                "stage": stage,
                "arch": cfg.arch,
                "hidden_scale": cfg.hidden_scale,
                "output_scale": cfg.output_scale,
                "code": 0,
                "dry_run": True,
                "log": str(log_path),
            }

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as train_log:
        train_log.write("# " + " ".join(cmd) + "\n")
        train_log.flush()
        process = subprocess.run(cmd, cwd=REPO_ROOT, stdout=train_log, stderr=subprocess.STDOUT, check=False)

    result = {
        "stage": stage,
        "arch": cfg.arch,
        "hidden_scale": cfg.hidden_scale,
        "output_scale": cfg.output_scale,
        "code": process.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "log": str(log_path),
        **parse_log(log_path),
    }
    with runner_log.open("a", encoding="utf-8") as log:
        print(
            f"done {stage} arch={cfg.arch} hs={cfg.hidden_scale} os={cfg.output_scale} "
            f"code={process.returncode} best_val_loss={result.get('best_val_loss')} "
            f"val_cp={result.get('val_cp_at_best_loss')}",
            file=log,
            flush=True,
        )
    return result


def run_stage(
    args: argparse.Namespace,
    configs: list[ScaleConfig],
    stage: str,
    deadline: float | None,
    runner_log: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    pending: deque[ScaleConfig] = deque(configs)
    running: dict[subprocess.Popen[bytes], tuple[ScaleConfig, Any, Path, float]] = {}
    results: list[dict[str, Any]] = []

    if args.dry_run:
        for cfg in configs:
            result = run_one(args, cfg, stage, runner_log)
            write_jsonl(summary_path, result)
            results.append(result)
        return results

    while pending or running:
        while pending and len(running) < args.jobs:
            if deadline is not None and time.monotonic() >= deadline:
                with runner_log.open("a", encoding="utf-8") as log:
                    print(f"deadline reached; not starting more {stage} configs", file=log, flush=True)
                pending.clear()
                break
            cfg = pending.popleft()
            log_path = args.log_dir / f"{config_name(args.tag, stage, cfg)}.log"
            cmd = train_command(args, cfg, stage)
            train_log = log_path.open("w", encoding="utf-8")
            train_log.write("# " + " ".join(cmd) + "\n")
            train_log.flush()
            process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=train_log, stderr=subprocess.STDOUT)
            running[process] = (cfg, train_log, log_path, time.monotonic())
            with runner_log.open("a", encoding="utf-8") as log:
                print(
                    f"start {stage} arch={cfg.arch} hs={cfg.hidden_scale} os={cfg.output_scale} "
                    f"pid={process.pid} log={log_path}",
                    file=log,
                    flush=True,
                )

        if running:
            time.sleep(5)
        for process in list(running):
            code = process.poll()
            if code is None:
                continue
            cfg, train_log, log_path, started = running.pop(process)
            train_log.close()
            result = {
                "stage": stage,
                "arch": cfg.arch,
                "hidden_scale": cfg.hidden_scale,
                "output_scale": cfg.output_scale,
                "code": code,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "log": str(log_path),
                **parse_log(log_path),
            }
            write_jsonl(summary_path, result)
            results.append(result)
            with runner_log.open("a", encoding="utf-8") as log:
                print(
                    f"done {stage} arch={cfg.arch} hs={cfg.hidden_scale} os={cfg.output_scale} "
                    f"code={code} best_val_loss={result.get('best_val_loss')} "
                    f"val_cp={result.get('val_cp_at_best_loss')}",
                    file=log,
                    flush=True,
                )

    return results


def select_stage2_configs(args: argparse.Namespace, stage1_results: list[dict[str, Any]]) -> list[ScaleConfig]:
    valid = [
        item
        for item in stage1_results
        if item.get("code") == 0
        and item.get("completed")
        and "best_val_loss" in item
        and "dry_run" not in item
    ]
    selected: list[dict[str, Any]] = []
    for arch in args.architectures:
        per_arch = [item for item in valid if item.get("arch") == arch]
        selected.extend(
            sorted(per_arch, key=lambda item: float(item["best_val_loss"]))[: args.top_k_per_arch]
        )
    selected = sorted(selected, key=lambda item: float(item["best_val_loss"]))[: args.top_k_total]
    return [
        ScaleConfig(str(item["arch"]), int(item["hidden_scale"]), int(item["output_scale"]))
        for item in selected
    ]


def main() -> int:
    args = parse_args()
    validate_sweep_args(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.tag).mkdir(parents=True, exist_ok=True)

    runner_log = args.log_dir / f"{args.tag}.runner.log"
    summary_path = args.log_dir / f"{args.tag}.summary.jsonl"
    manifest_path = args.log_dir / f"{args.tag}.manifest.json"
    for path in (runner_log, summary_path):
        if path.exists():
            path.unlink()
    with manifest_path.open("w", encoding="utf-8") as manifest:
        json.dump(vars(args), manifest, default=str, indent=2)
        manifest.write("\n")

    deadline = time.monotonic() + args.max_hours * 3600.0 if args.max_hours > 0 else None
    configs = stage1_configs(args)
    if args.top_k_total > len(args.architectures) * args.top_k_per_arch:
        raise ValueError(
            "--top-k-total exceeds --top-k-per-arch times the number of architectures"
        )
    configs_per_arch = len(args.hidden_scales) * len(args.output_scales)
    if args.top_k_per_arch > configs_per_arch:
        raise ValueError("--top-k-per-arch exceeds the number of stage-1 configs per architecture")
    with runner_log.open("a", encoding="utf-8") as log:
        print(f"tag={args.tag}", file=log, flush=True)
        print(f"stage1_configs={len(configs)}", file=log, flush=True)
        print(f"jobs={args.jobs}", file=log, flush=True)

    stage1_results = run_stage(args, configs, "stage1", deadline, runner_log, summary_path)
    stage1_invalid = [
        item
        for item in stage1_results
        if not args.dry_run
        and (item.get("code") != 0 or item.get("completed") is not True or "error" in item)
    ]
    if not args.dry_run and (len(stage1_results) != len(configs) or stage1_invalid):
        with runner_log.open("a", encoding="utf-8") as log:
            print(
                f"error: incomplete stage1 results={len(stage1_results)}/{len(configs)} "
                f"invalid={len(stage1_invalid)}; stage2 will not start",
                file=log,
                flush=True,
            )
        return 1

    stage2_configs = select_stage2_configs(args, stage1_results)
    if not args.dry_run and len(stage2_configs) != args.top_k_total:
        with runner_log.open("a", encoding="utf-8") as log:
            print(
                f"error: selected {len(stage2_configs)} stage2 configs; "
                f"expected {args.top_k_total}",
                file=log,
                flush=True,
            )
        return 1
    with runner_log.open("a", encoding="utf-8") as log:
        print("stage2 selected configs:", file=log, flush=True)
        for cfg in stage2_configs:
            print(
                f"arch={cfg.arch} hs={cfg.hidden_scale} os={cfg.output_scale}",
                file=log,
                flush=True,
            )

    stage2_results: list[dict[str, Any]] = []
    if stage2_configs:
        stage2_results = run_stage(
            args, stage2_configs, "stage2", deadline, runner_log, summary_path
        )

    with runner_log.open("a", encoding="utf-8") as log:
        print("done", file=log, flush=True)
    invalid = [
        item
        for item in [*stage1_results, *stage2_results]
        if not args.dry_run
        and (item.get("code") != 0 or item.get("completed") is not True or "error" in item)
    ]
    incomplete_stage2 = not args.dry_run and len(stage2_results) != len(stage2_configs)
    if incomplete_stage2:
        with runner_log.open("a", encoding="utf-8") as log:
            print(
                f"error: incomplete stage2 results={len(stage2_results)}/{len(stage2_configs)}",
                file=log,
                flush=True,
            )
    if args.sleep_when_done and not args.dry_run and not invalid and not incomplete_stage2:
        subprocess.run(["bash", "-lc", "command -v pmset >/dev/null && pmset sleepnow || true"], check=False)
    return 1 if invalid or incomplete_stage2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
