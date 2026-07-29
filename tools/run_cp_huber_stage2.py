#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.run_quantized_scale_grid import make_command, make_config_name, parse_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CP-Huber Stage 2 from a Stage 1 summary")
    p.add_argument("--stage1-summary", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--eval-data", required=True, type=Path)
    p.add_argument("--python", default="python3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def config_key(row: dict[str, object]) -> tuple[str, str, tuple[int, ...], int]:
    return (str(row["arch"]), str(row["activation"]),
            tuple(int(x) for x in row["hidden_scales"]), int(row["output_scale"]))


def main() -> int:
    args = parse_args()
    stage1 = [json.loads(line) for line in args.stage1_summary.read_text().splitlines() if line.strip()]
    ranked = sorted(stage1, key=lambda r: float(r["best_val_loss"]))
    configs = [config_key(r) for r in ranked[:args.top_k]]
    if len(configs) != args.top_k or len(set(configs)) != len(configs):
        raise RuntimeError(f"invalid selected configs: {configs}")

    log_dir = REPO_ROOT / "logs"
    output_dir = REPO_ROOT / "models" / "quantized_scale_grid"
    summary_path = log_dir / f"{args.tag}.summary.jsonl"
    runner_path = log_dir / f"{args.tag}.runner.log"
    manifest_path = log_dir / f"{args.tag}.manifest.json"
    completed = set()
    if args.resume and summary_path.exists():
        completed = {config_key(json.loads(line)) for line in summary_path.read_text().splitlines() if line.strip()}
    elif summary_path.exists():
        raise RuntimeError(f"summary exists; use --resume: {summary_path}")

    settings = argparse.Namespace(
        python=args.python, data=args.data, data_format="cbin", eval_data=args.eval_data,
        eval_data_format="cbin", cp_huber_delta=200.0, output_dir=output_dir, tag=args.tag,
        screlu_divisor=256,
        epochs=8, patience=4, batch_size=8192, lr=0.0005, warmup_epochs=0,
        lr_after_warmup=None, lr_drop_patience=2, lr_drop_factor=0.5, min_lr=0.00005,
        weight_decay=0.0001, device=args.device, workers=args.workers, eval_workers=0,
        train_max_samples=5_000_000, val_max_samples=500_000, test_max_samples=500_000,
        calibration_max_batches=50, calibration_percentile=99.5, feature_weight_scale=255,
        linear_weight_scale=64, output_weight_scale=16, shuffle_block_size=1_000_000,
        progress_batches=100, eval_progress_batches=0, torch_threads=0, skip_final_test=True,
    )
    manifest = {
        "tag": args.tag, "stage1_summary": str(args.stage1_summary),
        "selection": "global top 12 by Stage 1 best_val_loss across all activations",
        "loss": "cp_huber",
        "cp_huber_delta": 200.0, "screlu_divisor": 256,
        "train_samples_per_epoch": 5_000_000,
        "val_samples": 500_000, "epochs": 8, "lr": 0.0005,
        "jobs": args.jobs, "workers_per_job": args.workers,
        "configs": [{"arch": a, "activation": x, "hidden_scales": list(h), "output_scale": o}
                    for a, x, h, o in configs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run_one(config: tuple[str, str, tuple[int, ...], int]) -> dict[str, object]:
        arch, activation, hidden_scales, output_scale = config
        name = make_config_name(args.tag, arch, activation, hidden_scales, output_scale)
        log_path = log_dir / f"{name}.log"
        command = make_command(settings, arch, activation, hidden_scales, output_scale)
        started = time.monotonic()
        with log_path.open("w") as log:
            print("# " + " ".join(command), file=log, flush=True)
            code = subprocess.call(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        parsed = parse_result(log_path)
        current = output_dir / args.tag / name / "quant_nnue_arch_F2_current.pt"
        current.unlink(missing_ok=True)
        return {"arch": arch, "activation": activation, "hidden_scales": list(hidden_scales),
                "output_scale": output_scale, "code": code, "log": str(log_path),
                "elapsed_sec": round(time.monotonic() - started, 3), **parsed}

    pending = [config for config in configs if config not in completed]
    mode = "a" if args.resume else "w"
    failures = 0
    with runner_path.open(mode) as runner, summary_path.open(mode) as summary:
        print(f"tag={args.tag} completed={len(completed)} total={len(configs)} jobs={args.jobs}",
              file=runner, flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_one, config): config for config in pending}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(json.dumps(result, separators=(",", ":")), file=summary, flush=True)
                valid = result.get("code") == 0 and result.get("completed") is True
                failures += not valid
                print(f"done valid={valid} config={futures[future]} "
                      f"loss={result.get('best_val_loss')} cp={result.get('val_cp_at_best_loss')}",
                      file=runner, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
