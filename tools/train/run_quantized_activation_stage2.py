#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from tools.train.run_quantized_scale_grid import make_command, make_config_name, parse_result
from tools.train.run_quantized_scale_grid import hidden_scales_label
from tools.debug.quantized_pipeline_preflight import validate_sweep_args
from chess_nnue.training_targets import TRAINING_PIPELINE_VERSION
from chess_nnue.quantized_nnue_architectures import QUANTIZED_ARCHITECTURES


EXPECTED_STAGE1_COUNTS = {
    "relu": 24,
    "screlu_first": 36,
    "screlu_all": 36,
}

EVALUATE_SCRIPT = REPO_ROOT / "tools" / "analyze" / "evaluate_quantized_nnue_checkpoint.py"


@dataclass(frozen=True)
class Candidate:
    arch: str
    activation: str
    hidden_scales: tuple[int, ...]
    output_scale: int
    stage1_val_loss: float
    stage1_val_cp: float


def finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_hidden_scales(row: dict[str, Any]) -> tuple[int, ...]:
    raw_scales = row.get("hidden_scales")
    if raw_scales is None and "hidden_scale" in row:
        raw_scales = [row["hidden_scale"]]
    if not isinstance(raw_scales, list) or not raw_scales:
        raise ValueError("stage-1 result is missing hidden_scales")
    scales = tuple(int(value) for value in raw_scales)
    if any(value <= 0 for value in scales):
        raise ValueError("stage-1 result contains a non-positive hidden scale")
    architecture = str(row["arch"])
    expected = len(QUANTIZED_ARCHITECTURES[architecture].hidden_sizes) - 1
    if len(scales) != expected:
        raise ValueError(
            f"stage-1 {architecture} expected {expected} hidden scales, got {len(scales)}"
        )
    return scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best configs per activation from a completed scale grid"
    )
    parser.add_argument("--stage1-tag", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-format", choices=["auto", "jsonl", "cbin"], default="cbin")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-drop-patience", type=int, default=2)
    parser.add_argument("--lr-drop-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--train-max-samples", type=int, default=5_000_000)
    parser.add_argument("--val-max-samples", type=int, default=500_000)
    parser.add_argument("--test-max-samples", type=int, default=500_000)
    parser.add_argument("--calibration-max-batches", type=int, default=50)
    parser.add_argument("--feature-weight-scale", type=int, default=255)
    parser.add_argument("--linear-weight-scale", type=int, default=64)
    parser.add_argument("--output-weight-scale", type=int, default=16)
    parser.add_argument("--shuffle-block-size", type=int, default=1_000_000)
    parser.add_argument("--progress-batches", type=int, default=100)
    parser.add_argument("--eval-progress-batches", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "models" / "quantized_scale_grid",
        type=Path,
    )
    parser.add_argument("--log-dir", default=REPO_ROOT / "logs", type=Path)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing tag without overwriting its manifest or completed results",
    )
    return parser.parse_args()


def load_stage1_candidates(
    log_dir: Path,
    stage1_tag: str,
    top_k: int,
    expected_counts: dict[str, int] = EXPECTED_STAGE1_COUNTS,
) -> list[Candidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    selected: list[Candidate] = []
    for activation, expected_count in expected_counts.items():
        path = log_dir / f"{stage1_tag}_{activation}.summary.jsonl"
        if not path.exists():
            raise RuntimeError(f"missing stage-1 summary: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected_count:
            raise RuntimeError(
                f"stage-1 {activation} has {len(rows)} results; expected {expected_count}"
            )
        invalid = [
            row
            for row in rows
            if row.get("code") != 0
            or not row.get("completed")
            or row.get("pipeline_version") != TRAINING_PIPELINE_VERSION
            or "best_val_loss" not in row
            or not finite_number(row.get("best_val_loss"))
            or not finite_number(row.get("val_cp_at_best_loss"))
        ]
        if invalid:
            raise RuntimeError(f"stage-1 {activation} contains {len(invalid)} invalid results")

        rows.sort(key=lambda row: float(row["best_val_loss"]))
        for row in rows[:top_k]:
            selected.append(
                Candidate(
                    arch=str(row["arch"]),
                    activation=activation,
                    hidden_scales=parse_hidden_scales(row),
                    output_scale=int(row["output_scale"]),
                    stage1_val_loss=float(row["best_val_loss"]),
                    stage1_val_cp=float(row["val_cp_at_best_loss"]),
                )
            )
    return selected


def command_args(args: argparse.Namespace, candidate: Candidate) -> argparse.Namespace:
    return argparse.Namespace(
        python=args.python,
        data=args.data,
        data_format=args.data_format,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        tag=args.tag,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=0,
        lr_after_warmup=None,
        lr_drop_patience=args.lr_drop_patience,
        lr_drop_factor=args.lr_drop_factor,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        device=args.device,
        workers=args.workers,
        eval_workers=args.eval_workers,
        train_max_samples=args.train_max_samples,
        val_max_samples=args.val_max_samples,
        test_max_samples=args.test_max_samples,
        calibration_max_batches=args.calibration_max_batches,
        calibration_percentile=99.5,
        feature_weight_scale=args.feature_weight_scale,
        linear_weight_scale=args.linear_weight_scale,
        output_weight_scale=args.output_weight_scale,
        shuffle_block_size=args.shuffle_block_size,
        progress_batches=args.progress_batches,
        eval_progress_batches=args.eval_progress_batches,
        torch_threads=0,
        skip_final_test=True,
    )


def read_final_test(log_path: Path) -> dict[str, Any]:
    final_test: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "final_test":
            final_test = event
    if final_test is None:
        raise RuntimeError("sealed test evaluator did not emit final_test")
    for metric in ("test_loss", "test_val_cp"):
        if not finite_number(final_test.get(metric)):
            raise RuntimeError(f"sealed test evaluator emitted invalid {metric}")
    if int(final_test.get("test_samples", 0)) <= 0:
        raise RuntimeError("sealed test evaluator emitted no test samples")
    return final_test


def candidate_key(candidate: Candidate | dict[str, Any]) -> tuple[str, str, tuple[int, ...], int]:
    if isinstance(candidate, Candidate):
        return (
            candidate.arch,
            candidate.activation,
            candidate.hidden_scales,
            candidate.output_scale,
        )
    return (
        str(candidate["arch"]),
        str(candidate["activation"]),
        tuple(int(value) for value in candidate["hidden_scales"]),
        int(candidate["output_scale"]),
    )


def candidates_from_manifest(manifest: dict[str, Any]) -> list[Candidate]:
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise RuntimeError("resume manifest has no selected candidates")
    return [
        Candidate(
            arch=str(row["arch"]),
            activation=str(row["activation"]),
            hidden_scales=tuple(int(value) for value in row["hidden_scales"]),
            output_scale=int(row["output_scale"]),
            stage1_val_loss=float(row["stage1_val_loss"]),
            stage1_val_cp=float(row["stage1_val_cp"]),
        )
        for row in selected
    ]


def load_completed_results(
    summary_path: Path,
    candidates: list[Candidate],
) -> dict[tuple[str, str, tuple[int, ...], int], dict[str, Any]]:
    selected_keys = {candidate_key(candidate) for candidate in candidates}
    completed: dict[tuple[str, str, tuple[int, ...], int], dict[str, Any]] = {}
    for line_number, line in enumerate(summary_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = candidate_key(row)
        if key not in selected_keys:
            raise RuntimeError(f"summary line {line_number} is not present in the manifest")
        if row.get("code") != 0 or row.get("completed") is not True or "error" in row:
            continue
        if key in completed:
            raise RuntimeError(f"summary contains duplicate successful result for {key}")
        completed[key] = row
    return completed


def main() -> int:
    args = parse_args()
    validate_sweep_args(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.tag).mkdir(parents=True, exist_ok=True)
    manifest_path = args.log_dir / f"{args.tag}.manifest.json"
    runner_path = args.log_dir / f"{args.tag}.runner.log"
    summary_path = args.log_dir / f"{args.tag}.summary.jsonl"
    winner_path = args.log_dir / f"{args.tag}.winner.json"
    final_test_log_path = args.log_dir / f"{args.tag}.final_test.log"
    if args.resume:
        if not manifest_path.exists() or not summary_path.exists():
            raise RuntimeError("--resume requires an existing manifest and summary")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("tag") != args.tag or manifest.get("stage1_tag") != args.stage1_tag:
            raise RuntimeError("resume arguments do not match the existing manifest")
        candidates = candidates_from_manifest(manifest)
        completed_by_key = load_completed_results(summary_path, candidates)
        with runner_path.open("a", encoding="utf-8") as runner:
            print(
                f"resume completed={len(completed_by_key)} pending="
                f"{len(candidates) - len(completed_by_key)}",
                file=runner,
                flush=True,
            )
    else:
        candidates = load_stage1_candidates(args.log_dir, args.stage1_tag, args.top_k)
        manifest = {
            "stage1_tag": args.stage1_tag,
            "tag": args.tag,
            "selection_metric": "best_val_loss",
            "top_k_per_activation": args.top_k,
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "selected": [candidate.__dict__ for candidate in candidates],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        runner_path.write_text(f"tag={args.tag}\nselected={len(candidates)}\n", encoding="utf-8")
        summary_path.write_text("", encoding="utf-8")
        completed_by_key = {}

    pending = deque(
        (index, candidate)
        for index, candidate in enumerate(candidates, 1)
        if candidate_key(candidate) not in completed_by_key
    )
    running: dict[
        subprocess.Popen[Any],
        tuple[int, Candidate, TextIO, Path, float],
    ] = {}
    failures = 0
    completed_results: list[dict[str, Any]] = list(completed_by_key.values())
    while pending or running:
        while pending and len(running) < args.jobs:
            index, candidate = pending.popleft()
            config_name = make_config_name(
                args.tag,
                candidate.arch,
                candidate.activation,
                candidate.hidden_scales,
                candidate.output_scale,
            )
            log_path = args.log_dir / f"{config_name}.log"
            command = make_command(
                command_args(args, candidate),
                candidate.arch,
                candidate.activation,
                candidate.hidden_scales,
                candidate.output_scale,
            )
            log = log_path.open("w", encoding="utf-8")
            print("# " + " ".join(command), file=log, flush=True)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[process] = (index, candidate, log, log_path, time.monotonic())
            with runner_path.open("a", encoding="utf-8") as runner:
                print(
                    f"start {index}/{len(candidates)} arch={candidate.arch} "
                    f"activation={candidate.activation} "
                    f"hs={hidden_scales_label(candidate.hidden_scales)} "
                    f"os={candidate.output_scale} stage1_val_loss={candidate.stage1_val_loss} "
                    f"stage1_val_cp={candidate.stage1_val_cp} pid={process.pid}",
                    file=runner,
                    flush=True,
                )

        for process in list(running):
            code = process.poll()
            if code is None:
                continue
            index, candidate, log, log_path, started = running.pop(process)
            log.close()
            parsed = parse_result(log_path)
            result: dict[str, Any] = {
                **candidate.__dict__,
                "code": code,
                "log": str(log_path),
                "elapsed_sec": round(time.monotonic() - started, 3),
                **parsed,
            }
            with summary_path.open("a", encoding="utf-8") as summary:
                print(json.dumps(result, separators=(",", ":")), file=summary, flush=True)
            with runner_path.open("a", encoding="utf-8") as runner:
                print(
                    f"done {index}/{len(candidates)} code={code} "
                    f"best_val_loss={result.get('best_val_loss')} "
                    f"val_cp={result.get('val_cp_at_best_loss')}",
                    file=runner,
                    flush=True,
                )
            failures += int(code != 0 or parsed.get("completed") is not True or "error" in parsed)
            if code == 0 and parsed.get("completed") is True and "error" not in parsed:
                completed_results.append(result)
        if running:
            time.sleep(1)

    if failures:
        with runner_path.open("a", encoding="utf-8") as runner:
            print(f"done failures={failures}; sealed test not evaluated", file=runner, flush=True)
        return 1

    if len(completed_results) != len(candidates):
        raise RuntimeError(
            f"stage 2 has {len(completed_results)}/{len(candidates)} successful candidates"
        )

    winner = min(completed_results, key=lambda result: float(result["best_val_loss"]))
    winner_hidden_scales = tuple(int(value) for value in winner["hidden_scales"])
    winner_config_name = make_config_name(
        args.tag,
        str(winner["arch"]),
        str(winner["activation"]),
        winner_hidden_scales,
        int(winner["output_scale"]),
    )
    checkpoint_path = (
        args.output_dir
        / args.tag
        / winner_config_name
        / f"quant_nnue_arch_{winner['arch']}_best.pt"
    )
    if not checkpoint_path.exists():
        raise RuntimeError(f"validation-selected checkpoint is missing: {checkpoint_path}")
    final_test_command = [
        args.python,
        str(EVALUATE_SCRIPT),
        "--checkpoint",
        str(checkpoint_path),
        "--data",
        str(args.data),
        "--data-format",
        args.data_format,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.eval_workers),
        "--test-max-samples",
        str(args.test_max_samples),
        "--eval-progress-batches",
        str(args.eval_progress_batches),
    ]
    with final_test_log_path.open("w", encoding="utf-8") as final_test_log:
        print("# " + " ".join(final_test_command), file=final_test_log, flush=True)
        final_test_process = subprocess.run(
            final_test_command,
            cwd=REPO_ROOT,
            stdout=final_test_log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if final_test_process.returncode != 0:
        raise RuntimeError(
            f"sealed test evaluator failed with code {final_test_process.returncode}: "
            f"{final_test_log_path}"
        )
    final_test = read_final_test(final_test_log_path)
    if int(final_test["selected_epoch"]) != int(winner["best_epoch"]):
        raise RuntimeError("sealed test evaluated a checkpoint other than the validation winner")
    if (
        str(final_test.get("arch")) != str(winner["arch"])
        or str(final_test.get("activation")) != str(winner["activation"])
        or tuple(int(value) for value in final_test.get("hidden_scales", []))
        != winner_hidden_scales
        or int(final_test.get("output_scale", 0)) != int(winner["output_scale"])
    ):
        raise RuntimeError("sealed test checkpoint metadata does not match the validation winner")
    if abs(float(final_test["best_val_loss"]) - float(winner["best_val_loss"])) > 1e-7:
        raise RuntimeError("sealed test checkpoint validation loss does not match the winner")
    winner_record = {
        "selection_metric": "best_val_loss",
        "winner": winner,
        "checkpoint": str(checkpoint_path),
        "final_test": final_test,
    }
    winner_path.write_text(json.dumps(winner_record, indent=2) + "\n", encoding="utf-8")
    with runner_path.open("a", encoding="utf-8") as runner:
        print(
            f"done failures=0 winner={winner_config_name} "
            f"best_val_loss={winner['best_val_loss']} test_val_cp={final_test['test_val_cp']}",
            file=runner,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
