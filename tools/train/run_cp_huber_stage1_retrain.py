#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from tools.train.run_quantized_scale_grid import make_command, make_config_name, parse_result


def configs() -> list[tuple[str, str, tuple[int, int], int]]:
    rows: list[tuple[str, str, tuple[int, int], int]] = []
    for activation, hidden_values, output_values in (
        ("relu", (64, 128, 256), (32, 64)),
        ("screlu_first", (64, 128, 256), (8, 16, 32)),
        ("screlu_all", (8, 16, 32), (8, 16, 32)),
    ):
        rows.extend(
            ("F2", activation, hidden_scales, output_scale)
            for hidden_scales in product(hidden_values, repeat=2)
            for output_scale in output_values
        )
    rows.extend(
        ("F2", "screlu_first", hidden_scales, output_scale)
        for hidden_scales in ((64, 384), (64, 512), (96, 256), (96, 384))
        for output_scale in (32, 64)
    )
    rows.extend(
        ("F2", "screlu_all", hidden_scales, output_scale)
        for hidden_scales in ((48, 16), (64, 16), (48, 32), (64, 32))
        for output_scale in (16, 32)
    )
    assert len(rows) == 88 and len(set(rows)) == 88
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the full v2 Stage 1 with CP Huber")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-max-samples", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def key(row: dict[str, object]) -> tuple[str, str, tuple[int, ...], int]:
    return (
        str(row["arch"]), str(row["activation"]),
        tuple(int(x) for x in row["hidden_scales"]), int(row["output_scale"]),
    )


def main() -> int:
    args = parse_args()
    rows = configs()
    log_dir = REPO_ROOT / "logs"
    output_dir = REPO_ROOT / "models" / "quantized_scale_grid"
    summary_path = log_dir / f"{args.tag}.summary.jsonl"
    runner_path = log_dir / f"{args.tag}.runner.log"
    manifest_path = log_dir / f"{args.tag}.manifest.json"
    completed: set[tuple[str, str, tuple[int, ...], int]] = set()
    if args.resume and summary_path.exists():
        completed = {key(json.loads(line)) for line in summary_path.read_text().splitlines() if line.strip()}
    elif summary_path.exists():
        raise RuntimeError(f"summary exists; use --resume: {summary_path}")

    settings = argparse.Namespace(
        python=args.python, data=args.data, data_format="cbin",
        eval_data=args.eval_data, eval_data_format="cbin", cp_huber_delta=200.0,
        output_dir=output_dir, tag=args.tag, epochs=5, patience=4,
        batch_size=8192, lr=0.001, warmup_epochs=0, lr_after_warmup=None,
        lr_drop_patience=2, lr_drop_factor=0.5, min_lr=0.00005,
        weight_decay=0.0001, device=args.device, workers=args.workers, eval_workers=0,
        train_max_samples=args.train_max_samples, val_max_samples=100_000, test_max_samples=100_000,
        calibration_max_batches=50, calibration_percentile=99.5,
        feature_weight_scale=255, linear_weight_scale=64, output_weight_scale=16,
        shuffle_block_size=1_000_000, progress_batches=25, eval_progress_batches=0,
        torch_threads=0, skip_final_test=True,
    )
    manifest = {
        "tag": args.tag, "purpose": "full v2 Stage 1 retrain",
        "loss": {"type": "cp_huber", "delta_cp": 200.0},
        "train_data": str(args.data), "eval_data": str(args.eval_data),
        "settings": {"epochs": 5, "train_max_samples_per_config": args.train_max_samples,
                     "jobs": args.jobs, "workers_per_job": args.workers,
                     "val_max_samples": 100_000, "selection_metric": "best_val_loss"},
        "config_count": len(rows),
        "configs": [{"arch": a, "activation": x, "hidden_scales": list(h), "output_scale": o}
                    for a, x, h, o in rows],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    mode = "a" if args.resume else "w"
    with runner_path.open(mode) as runner, summary_path.open(mode) as summary:
        print(f"tag={args.tag} completed={len(completed)} total={len(rows)}", file=runner, flush=True)
        pending = [(index, row) for index, row in enumerate(rows, 1) if row not in completed]

        def run_one(item: tuple[int, tuple[str, str, tuple[int, int], int]]) -> dict[str, object]:
            index, (arch, activation, hidden_scales, output_scale) = item
            name = make_config_name(args.tag, arch, activation, hidden_scales, output_scale)
            log_path = log_dir / f"{name}.log"
            command = make_command(settings, arch, activation, hidden_scales, output_scale)
            started = time.monotonic()
            with log_path.open("w") as log:
                print("# " + " ".join(command), file=log, flush=True)
                code = subprocess.call(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            parsed = parse_result(log_path)
            result = {"arch": arch, "activation": activation, "hidden_scales": list(hidden_scales),
                      "output_scale": output_scale, "code": code, "log": str(log_path),
                      "elapsed_sec": round(time.monotonic() - started, 3), **parsed}
            current = output_dir / args.tag / name / "quant_nnue_arch_F2_current.pt"
            current.unlink(missing_ok=True)
            result["stage1_index"] = index
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {}
            for item in pending:
                index, config = item
                print(f"queue {index}/88 {config}", file=runner, flush=True)
                futures[executor.submit(run_one, item)] = item
            for future in concurrent.futures.as_completed(futures):
                index, config = futures[future]
                result = future.result()
                print(json.dumps(result, separators=(",", ":")), file=summary, flush=True)
                print(f"done {index}/88 code={result.get('code')} "
                      f"val={result.get('best_val_loss')} config={config}", file=runner, flush=True)
                if result.get("code") != 0 or result.get("completed") is not True:
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
