#!/usr/bin/env python3

import argparse
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path


DEFAULTS = {
    "see_weight": 280,
    "killer1_bonus": 24_000,
    "killer2_bonus": 8_076,
    "check_bonus": 44_384,
    "counter_history_bonus": 14_000,
    "bad_capture_stage_threshold": 0,
    "qsearch_captured_value_weight": 486,
    "qsearch_capture_metric_weight": 160,
}

RANGES = {
    "see_weight": (32, 1_024),
    "killer1_bonus": (0, 100_000),
    "killer2_bonus": (0, 60_000),
    "check_bonus": (0, 160_000),
    "counter_history_bonus": (0, 80_000),
    "bad_capture_stage_threshold": (-400, 400),
    "qsearch_captured_value_weight": (0, 1_500),
    "qsearch_capture_metric_weight": (0, 600),
}

CLI_NAMES = {
    name: "--" + name.replace("_", "-")
    for name in DEFAULTS
}

SUMMARY_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def parse_summary(output):
    for line in output.splitlines():
        if line.startswith("summary "):
            return {
                key: value
                for key, value in SUMMARY_PATTERN.findall(line)
            }
    raise RuntimeError("benchmark summary not found: " + output[-2000:])


def normalize(candidate):
    result = {}
    for name, value in candidate.items():
        low, high = RANGES[name]
        result[name] = max(low, min(high, int(round(value))))
    if result["killer2_bonus"] > result["killer1_bonus"]:
        result["killer1_bonus"], result["killer2_bonus"] = (
            result["killer2_bonus"],
            result["killer1_bonus"],
        )
    return result


def random_value(rng, name):
    low, high = RANGES[name]
    if name == "bad_capture_stage_threshold" or low == 0:
        return rng.randint(low, high)
    return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))


def mutate_value(rng, name, value, progress):
    low, high = RANGES[name]
    if rng.random() < 0.08:
        return random_value(rng, name)

    radius = 0.28 * (1.0 - progress) + 0.06
    if name == "bad_capture_stage_threshold":
        sigma = max(8.0, (high - low) * radius * 0.25)
        return int(round(value + rng.gauss(0.0, sigma)))

    if value <= 0:
        return int(round(rng.uniform(low, max(low + 1, high * radius))))
    return int(round(value * math.exp(rng.gauss(0.0, radius))))


def mutate_candidate(rng, best, progress):
    base = best if best is not None and rng.random() < 0.88 else DEFAULTS
    candidate = dict(base)
    mutation_count = rng.choices([1, 2, 3, 4], weights=[35, 40, 20, 5])[0]
    for name in rng.sample(list(DEFAULTS), mutation_count):
        candidate[name] = mutate_value(
            rng, name, candidate[name], progress)
    return normalize(candidate)


def candidate_args(candidate):
    args = []
    for name in DEFAULTS:
        args.extend([CLI_NAMES[name], str(candidate[name])])
    return args


def evaluate(args, candidate, depth, offset, count, timeout):
    command = [
        str(args.benchmark),
        "--model", str(args.model),
        "--book", str(args.book),
        "--depth", str(depth),
        "--offset", str(offset),
        "--count", str(count),
        "--tt-mb", str(args.tt_mb),
        "--seed", str(args.position_seed),
        *candidate_args(candidate),
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=args.repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1.0, timeout),
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed rc={completed.returncode}: "
            + completed.stdout[-2000:])
    summary = parse_summary(completed.stdout)
    return {
        "node_ratio": float(summary["node_ratio"]),
        "baseline_nodes": int(summary["baseline_nodes"]),
        "candidate_nodes": int(summary["candidate_nodes"]),
        "time_ratio": float(summary["time_ratio"]),
        "score_mismatches": int(summary["score_mismatches"]),
        "move_mismatches": int(summary["move_mismatches"]),
        "elapsed_sec": elapsed,
        "depth": depth,
        "offset": offset,
        "count": count,
    }


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def candidate_key(candidate):
    return tuple(candidate[name] for name in DEFAULTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("build-release/benchmark_nnue_v36_ordering_weights"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/quantized_scale_grid/"
            "old_score_huber200_lr_sweep_then_5ep_20260724_142758/"
            "best/phase_quantized_nnue.bin"))
    parser.add_argument(
        "--book",
        type=Path,
        default=Path(
            "data/stockfish_balanced_openings_10ply_1k_20260723.txt"))
    parser.add_argument("--duration-sec", type=int, default=3600)
    parser.add_argument("--validation-reserve-sec", type=int, default=300)
    parser.add_argument("--tune-depth", type=int, default=6)
    parser.add_argument("--tune-count", type=int, default=48)
    parser.add_argument("--holdout-depth", type=int, default=7)
    parser.add_argument("--holdout-count", type=int, default=32)
    parser.add_argument("--holdout-top", type=int, default=5)
    parser.add_argument("--tt-mb", type=int, default=64)
    parser.add_argument("--position-seed", type=int, default=20260727)
    parser.add_argument("--mutation-seed", type=int, default=20260727)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    for attribute in ("benchmark", "model", "book", "log", "summary"):
        value = getattr(args, attribute)
        if not value.is_absolute():
            setattr(args, attribute, args.repo / value)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    if args.duration_sec <= args.validation_reserve_sec:
        raise RuntimeError("duration must exceed validation reserve")
    if args.tune_count <= 0 or args.holdout_count <= 0:
        raise RuntimeError("position counts must be positive")

    rng = random.Random(args.mutation_seed)
    started = time.monotonic()
    deadline = started + args.duration_sec
    tune_deadline = deadline - args.validation_reserve_sec
    best_candidate = None
    best_result = None
    leaderboard = {}
    elapsed_samples = []
    index = 0

    while time.monotonic() < tune_deadline:
        remaining = tune_deadline - time.monotonic()
        expected = (
            sum(elapsed_samples[-8:]) / len(elapsed_samples[-8:])
            if elapsed_samples else 1.0)
        if remaining < max(2.0, expected * 1.15):
            break

        progress = min(
            1.0,
            (time.monotonic() - started)
            / max(1.0, tune_deadline - started))
        candidate = (
            dict(DEFAULTS)
            if index == 0
            else mutate_candidate(rng, best_candidate, progress))
        record = {
            "kind": "tune",
            "index": index,
            "elapsed_total_sec": time.monotonic() - started,
            "candidate": candidate,
        }
        try:
            result = evaluate(
                args,
                candidate,
                args.tune_depth,
                0,
                args.tune_count,
                min(args.timeout_sec, remaining))
            elapsed_samples.append(result["elapsed_sec"])
            valid = result["score_mismatches"] == 0
            improved = valid and (
                best_result is None
                or result["node_ratio"] < best_result["node_ratio"])
            if valid:
                key = candidate_key(candidate)
                previous = leaderboard.get(key)
                if previous is None or result["node_ratio"] < previous["result"]["node_ratio"]:
                    leaderboard[key] = {
                        "candidate": candidate,
                        "result": result,
                    }
            if improved:
                best_candidate = candidate
                best_result = result
            record.update({
                "valid": valid,
                "improved": improved,
                "result": result,
                "best_candidate": best_candidate,
                "best_result": best_result,
            })
        except Exception as error:
            record.update({
                "valid": False,
                "improved": False,
                "error": str(error)[-2000:],
                "best_candidate": best_candidate,
                "best_result": best_result,
            })
        append_jsonl(args.log, record)
        if record["improved"]:
            print(json.dumps(record, sort_keys=True), flush=True)
        index += 1

    ranked = sorted(
        leaderboard.values(),
        key=lambda entry: entry["result"]["node_ratio"])
    holdout_results = []
    for rank, entry in enumerate(ranked[:args.holdout_top]):
        remaining = deadline - time.monotonic()
        if remaining <= 2.0:
            break
        record = {
            "kind": "holdout",
            "rank_by_tune": rank,
            "candidate": entry["candidate"],
            "tune_result": entry["result"],
            "elapsed_total_sec": time.monotonic() - started,
        }
        try:
            result = evaluate(
                args,
                entry["candidate"],
                args.holdout_depth,
                args.tune_count,
                args.holdout_count,
                min(args.timeout_sec, remaining))
            valid = result["score_mismatches"] == 0
            record.update({"valid": valid, "result": result})
            if valid:
                holdout_results.append(record)
        except Exception as error:
            record.update({
                "valid": False,
                "error": str(error)[-2000:],
            })
        append_jsonl(args.log, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    selected = (
        min(
            holdout_results,
            key=lambda entry: entry["result"]["node_ratio"])
        if holdout_results else None)
    summary = {
        "duration_requested_sec": args.duration_sec,
        "elapsed_sec": time.monotonic() - started,
        "evaluated_candidates": index,
        "tune_depth": args.tune_depth,
        "tune_count": args.tune_count,
        "holdout_depth": args.holdout_depth,
        "holdout_count": args.holdout_count,
        "best_tune_candidate": best_candidate,
        "best_tune_result": best_result,
        "selected_by_holdout": selected,
        "log": str(args.log),
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
