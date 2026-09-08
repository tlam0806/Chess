#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path


DEFAULT_WEIGHTS = {
    "history_penalty_num": 9,
    "history_penalty_den": 14,
    "tt_lower_bonus": 1_200_000_000,
    "tt_upper_bonus": 0,
    "promotion_bonus": 660_961_000,
    "good_capture_bonus": 63_613_000,
    "bad_capture_bonus": 80_000,
    "see_weight": 280,
    "killer1_bonus": 24_000,
    "killer2_bonus": 8_076,
    "check_bonus": 44_384,
    "counter_history_bonus": 21_000,
    "bad_capture_stage_threshold": 0,
    "qsearch_promotion_bonus": 1_094_272_000,
    "qsearch_good_capture_bonus": 154_962_000,
    "qsearch_bad_capture_bonus": 282_000,
    "qsearch_see_weight": 48,
    "qsearch_captured_value_weight": 486,
}

RANGES = {
    "history_penalty_num": (1, 10),
    "history_penalty_den": (1, 14),
    "tt_lower_bonus": (0, 1_200_000_000),
    "tt_upper_bonus": (0, 80_000_000),
    "promotion_bonus": (100_000_000, 1_200_000_000),
    "good_capture_bonus": (1_000_000, 160_000_000),
    "bad_capture_bonus": (-100_000, 400_000),
    "see_weight": (0, 3200),
    "killer1_bonus": (0, 180_000),
    "killer2_bonus": (0, 120_000),
    "check_bonus": (0, 800_000),
    "counter_history_bonus": (0, 80_000),
    "bad_capture_stage_threshold": (-1000, 0),
    "qsearch_promotion_bonus": (100_000_000, 1_200_000_000),
    "qsearch_good_capture_bonus": (1_000_000, 160_000_000),
    "qsearch_bad_capture_bonus": (-100_000, 400_000),
    "qsearch_see_weight": (0, 80),
    "qsearch_captured_value_weight": (0, 2000),
}

SUMMARY_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def clamp(name, value):
    low, high = RANGES[name]
    return max(low, min(high, int(value)))


def random_value(rng, name):
    low, high = RANGES[name]
    if name in {
        "history_penalty_num",
        "history_penalty_den",
        "see_weight",
        "qsearch_see_weight",
        "qsearch_captured_value_weight",
        "bad_capture_stage_threshold",
    }:
        return rng.randint(low, high)
    if low <= 0:
        return rng.randint(low, high)
    return int(round(math.exp(rng.uniform(math.log(low), math.log(high))) / 1000.0) * 1000)


def mutate_value(rng, name, value, radius_scale):
    if rng.random() < 0.12:
        return random_value(rng, name)
    low, high = RANGES[name]
    if name in {
        "history_penalty_num",
        "history_penalty_den",
        "see_weight",
        "qsearch_see_weight",
        "qsearch_captured_value_weight",
    }:
        radius = max(1, int((high - low) * rng.uniform(0.05, 0.20) * radius_scale))
        return clamp(name, value + rng.randint(-radius, radius))
    scale = rng.uniform(1.0 / (1.0 + 0.55 * radius_scale), 1.0 + 0.75 * radius_scale)
    jitter = rng.randint(-4000, 4000)
    return clamp(name, round((value * scale + jitter) / 1000.0) * 1000)


def normalize(candidate):
    for name in candidate:
        candidate[name] = clamp(name, candidate[name])
    if candidate["killer2_bonus"] > candidate["killer1_bonus"]:
        candidate["killer1_bonus"], candidate["killer2_bonus"] = (
            candidate["killer2_bonus"],
            candidate["killer1_bonus"],
        )
    return candidate


def make_candidate(rng, best, index):
    if index == 0:
        return dict(DEFAULT_WEIGHTS)
    base = best if best is not None and rng.random() < 0.82 else DEFAULT_WEIGHTS
    candidate = dict(base)
    radius_scale = 1.0 if rng.random() < 0.75 else 2.0
    names = list(DEFAULT_WEIGHTS)
    for name in rng.sample(names, rng.randint(2, 6)):
        candidate[name] = mutate_value(rng, name, candidate[name], radius_scale)
    return normalize(candidate)


def candidate_args(candidate):
    return [
        "--history-penalty-num", str(candidate["history_penalty_num"]),
        "--history-penalty-den", str(candidate["history_penalty_den"]),
        "--tt-lower-bonus", str(candidate["tt_lower_bonus"]),
        "--tt-upper-bonus", str(candidate["tt_upper_bonus"]),
        "--promotion-bonus", str(candidate["promotion_bonus"]),
        "--good-capture-bonus", str(candidate["good_capture_bonus"]),
        "--bad-capture-bonus", str(candidate["bad_capture_bonus"]),
        "--see-weight", str(candidate["see_weight"]),
        "--killer1-bonus", str(candidate["killer1_bonus"]),
        "--killer2-bonus", str(candidate["killer2_bonus"]),
        "--check-bonus", str(candidate["check_bonus"]),
        "--counter-history-bonus", str(candidate["counter_history_bonus"]),
        "--bad-capture-stage-threshold", str(candidate["bad_capture_stage_threshold"]),
        "--qsearch-promotion-bonus", str(candidate["qsearch_promotion_bonus"]),
        "--qsearch-good-capture-bonus", str(candidate["qsearch_good_capture_bonus"]),
        "--qsearch-bad-capture-bonus", str(candidate["qsearch_bad_capture_bonus"]),
        "--qsearch-see-weight", str(candidate["qsearch_see_weight"]),
        "--qsearch-captured-value-weight", str(candidate["qsearch_captured_value_weight"]),
    ]


def parse_summary(output):
    for line in output.splitlines():
        if line.startswith("summary "):
            return {key: value for key, value in SUMMARY_RE.findall(line)}
    raise RuntimeError(f"benchmark summary not found:\n{output[-2000:]}")


def evaluate(args, candidate, seed):
    cmd = [
        str(args.benchmark),
        "--input", str(args.input),
        "--samples", str(args.samples),
        "--depth", str(args.depth),
        "--iterative",
        "--seed", str(seed),
        *candidate_args(candidate),
    ]
    if args.first:
        cmd.append("--first")

    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=args.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_sec,
        check=False,
    )
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark failed rc={completed.returncode}: {' '.join(cmd)}\n{completed.stdout[-2000:]}")

    summary = parse_summary(completed.stdout)
    node_ratio = float(summary["v24_vs_v23_nodes"])
    time_ratio = float(summary["v24_vs_v23_time"])
    score_mismatches = int(summary["score_mismatches"])
    move_mismatches = int(summary["move_mismatches"])
    score = time_ratio + 0.20 * max(0.0, node_ratio - 1.0)
    score += 1000.0 * score_mismatches + 2.0 * move_mismatches
    return {
        "score": score,
        "node_ratio": node_ratio,
        "time_ratio": time_ratio,
        "score_mismatches": score_mismatches,
        "move_mismatches": move_mismatches,
        "elapsed_sec": elapsed,
        "seed": seed,
    }


def write_record(path, record):
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark", type=Path, default=Path("build/benchmark_v23_v24"))
    parser.add_argument("--input", type=Path, default=Path("data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("build/v24_move_order_tune.jsonl"))
    parser.add_argument("--duration-sec", type=int, default=7200)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--first", action="store_true")
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    if not args.benchmark.is_absolute():
        args.benchmark = args.repo / args.benchmark
    if not args.input.is_absolute():
        args.input = args.repo / args.input
    if not args.out.is_absolute():
        args.out = args.repo / args.out
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    deadline = time.time() + args.duration_sec
    best_candidate = None
    best_result = None
    index = 0

    while time.time() < deadline:
        candidate = make_candidate(rng, best_candidate, index)
        seed = rng.randint(1, 2_000_000_000)
        try:
            result = evaluate(args, candidate, seed)
            failed = False
            error = None
        except subprocess.TimeoutExpired as exc:
            result = {"score": 1e9, "elapsed_sec": args.timeout_sec, "seed": seed}
            failed = True
            error = f"timeout after {args.timeout_sec}s"
        except Exception as exc:
            result = {"score": 1e9, "elapsed_sec": 0, "seed": seed}
            failed = True
            error = str(exc)[-1000:]

        improved = False
        if not failed and (best_result is None or result["score"] < best_result["score"]):
            best_result = result
            best_candidate = candidate
            improved = True

        record = {
            "index": index,
            "elapsed_total_sec": round(time.time() - (deadline - args.duration_sec), 3),
            "failed": failed,
            "error": error,
            "improved": improved,
            "candidate": candidate,
            "result": result,
            "best_candidate": best_candidate,
            "best_result": best_result,
        }
        write_record(args.out, record)
        if improved:
            print(json.dumps(record, sort_keys=True), flush=True)
        index += 1

    if best_result is not None:
        print(json.dumps({"best_candidate": best_candidate, "best_result": best_result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
