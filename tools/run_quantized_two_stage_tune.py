#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.quantized_training_log import parse_quantized_training_log
from tools.quantized_pipeline_preflight import validate_sweep_args


TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_quantized_nnue_architecture.py"


@dataclass(frozen=True)
class ScaleConfig:
    arch: str
    hidden_scale: int
    output_scale: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage quantized NNUE scale tune")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-weight-scale", type=int, default=255)
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--progress-batches", type=int, default=250)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument("--stage1-train-samples", type=int, default=2_000_000)
    parser.add_argument("--stage1-val-samples", type=int, default=500_000)
    parser.add_argument("--stage1-test-samples", type=int, default=500_000)
    parser.add_argument("--stage1-epochs", type=int, default=6)
    parser.add_argument("--stage1-lr", type=float, default=0.001)
    parser.add_argument("--stage1-warmup-epochs", type=int, default=2)
    parser.add_argument("--stage1-lr-after-warmup", type=float, default=0.0005)
    parser.add_argument("--stage1-patience", type=int, default=3)
    parser.add_argument("--stage2-train-samples", type=int, default=5_000_000)
    parser.add_argument("--stage2-val-samples", type=int, default=1_000_000)
    parser.add_argument("--stage2-test-samples", type=int, default=1_000_000)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--stage2-lr", type=float, default=0.0005)
    parser.add_argument("--stage2-patience", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--output-dir", default=REPO_ROOT / "models" / "quantized_two_stage", type=Path)
    parser.add_argument("--log-dir", default=REPO_ROOT / "logs", type=Path)
    parser.add_argument("--tag", default=time.strftime("quant_two_stage_%Y%m%d_%H%M%S"))
    parser.add_argument("--python", default=".venv/bin/python")
    return parser.parse_args()


def stage1_configs() -> list[ScaleConfig]:
    configs: list[ScaleConfig] = []
    for arch in ("A", "B", "D", "E", "F", "G", "H"):
        configs.append(ScaleConfig(arch, 256, 16))
        configs.append(ScaleConfig(arch, 256, 32))
    return configs


def config_name(tag: str, stage: str, cfg: ScaleConfig) -> str:
    return f"{tag}_{stage}_{cfg.arch}_hs{cfg.hidden_scale}_os{cfg.output_scale}"


def train_command(args: argparse.Namespace, cfg: ScaleConfig, stage: str) -> list[str]:
    if stage == "stage1":
        epochs = args.stage1_epochs
        patience = args.stage1_patience
        train_samples = args.stage1_train_samples
        val_samples = args.stage1_val_samples
        test_samples = args.stage1_test_samples
        lr = args.stage1_lr
        warmup_epochs = args.stage1_warmup_epochs
        lr_after_warmup = args.stage1_lr_after_warmup
    elif stage == "stage2":
        epochs = args.stage2_epochs
        patience = args.stage2_patience
        train_samples = args.stage2_train_samples
        val_samples = args.stage2_val_samples
        test_samples = args.stage2_test_samples
        lr = args.stage2_lr
        warmup_epochs = 0
        lr_after_warmup = None
    else:
        raise ValueError(stage)

    out_dir = args.output_dir / args.tag / config_name(args.tag, stage, cfg)
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
        str(out_dir),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(lr),
        "--warmup-epochs",
        str(warmup_epochs),
        "--lr-drop-patience",
        "2",
        "--lr-drop-factor",
        "0.5",
        "--min-lr",
        "0.0000625",
        "--weight-decay",
        "0.00001",
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
        "--feature-weight-scale",
        str(args.feature_weight_scale),
        "--linear-weight-scale",
        str(args.linear_weight_scale),
        "--output-weight-scale",
        str(args.output_weight_scale),
        "--progress-batches",
        str(args.progress_batches),
        "--eval-progress-batches",
        str(args.eval_progress_batches),
        "--fixed-hidden-scale",
        str(cfg.hidden_scale),
        "--fixed-output-scale",
        str(cfg.output_scale),
    ]
    if lr_after_warmup is not None and warmup_epochs > 0:
        cmd += ["--lr-after-warmup", str(lr_after_warmup)]
    return cmd


def parse_log(log_path: Path) -> dict[str, object]:
    return parse_quantized_training_log(log_path)


def run_one(
    args: argparse.Namespace,
    cfg: ScaleConfig,
    stage: str,
    runner_log,
    summary,
) -> dict[str, object]:
    log_path = args.log_dir / f"{config_name(args.tag, stage, cfg)}.log"
    cmd = train_command(args, cfg, stage)
    print(
        f"start {stage} arch={cfg.arch} hs={cfg.hidden_scale} os={cfg.output_scale} log={log_path}",
        file=runner_log,
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        started = time.monotonic()
        process = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
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
    print(json.dumps(result, separators=(",", ":")), file=summary, flush=True)
    print(
        f"done {stage} arch={cfg.arch} hs={cfg.hidden_scale} os={cfg.output_scale} "
        f"code={process.returncode} best_val_loss={result.get('best_val_loss')} "
        f"val_cp={result.get('val_cp_at_best_loss')}",
        file=runner_log,
        flush=True,
    )
    return result


def main() -> int:
    args = parse_args()
    validate_sweep_args(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.tag).mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_hours * 3600.0 if args.max_hours > 0 else None

    runner_log_path = args.log_dir / f"{args.tag}.runner.log"
    summary_path = args.log_dir / f"{args.tag}.summary.jsonl"
    manifest_path = args.log_dir / f"{args.tag}.manifest"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        json.dump(vars(args), manifest, default=str, indent=2)
        manifest.write("\n")

    stage1_results: list[dict[str, object]] = []
    all_results: list[dict[str, object]] = []
    with runner_log_path.open("w", encoding="utf-8") as runner_log, summary_path.open(
        "w", encoding="utf-8"
    ) as summary:
        print(f"tag={args.tag}", file=runner_log, flush=True)
        all_stage1_configs = stage1_configs()
        for cfg in all_stage1_configs:
            if deadline is not None and time.monotonic() >= deadline:
                print("deadline reached before next stage1 config", file=runner_log, flush=True)
                break
            result = run_one(args, cfg, "stage1", runner_log, summary)
            all_results.append(result)
            if result.get("code") == 0 and result.get("completed") and "best_val_loss" in result:
                stage1_results.append(result)

        if len(all_results) != len(all_stage1_configs):
            print(
                f"error: incomplete stage1 results={len(all_results)}/{len(all_stage1_configs)}",
                file=runner_log,
                flush=True,
            )
            return 1
        if len(stage1_results) != len(all_stage1_configs):
            print(
                f"error: stage1 contains {len(all_stage1_configs) - len(stage1_results)} "
                "invalid configs; stage2 will not start",
                file=runner_log,
                flush=True,
            )
            return 1

        top = sorted(stage1_results, key=lambda item: float(item["best_val_loss"]))[: args.top_k]
        if len(top) < args.top_k:
            print(
                f"error: only {len(top)} valid stage1 configs; required {args.top_k}",
                file=runner_log,
                flush=True,
            )
            return 1
        print("stage1 top configs:", file=runner_log, flush=True)
        for item in top:
            print(json.dumps(item, separators=(",", ":")), file=runner_log, flush=True)

        for item in top:
            if deadline is not None and time.monotonic() >= deadline:
                print("deadline reached before next stage2 config", file=runner_log, flush=True)
                break
            cfg = ScaleConfig(
                str(item["arch"]),
                int(item["hidden_scale"]),
                int(item["output_scale"]),
            )
            all_results.append(run_one(args, cfg, "stage2", runner_log, summary))

        stage2_result_count = len(all_results) - len(all_stage1_configs)
        if stage2_result_count != len(top):
            print(
                f"error: incomplete stage2 results={stage2_result_count}/{len(top)}",
                file=runner_log,
                flush=True,
            )
            return 1

        print("done", file=runner_log, flush=True)
    invalid = [
        item
        for item in all_results
        if item.get("code") != 0 or item.get("completed") is not True or "error" in item
    ]
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
