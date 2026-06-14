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
    "tt_lower_bonus": 1_000_000_000,
    "tt_upper_bonus": 20_000_000,
    "promotion_bonus": 900_000_000,
    "good_capture_bonus": 50_000_000,
    "bad_capture_bonus": 30_000,
    "see_weight": 10,
    "killer1_bonus": 40_000,
    "killer2_bonus": 10_000,
    "check_bonus": 160_000,
    "counter_history_bonus": 30_000,
    "penalty_num": 6,
    "penalty_den": 5,
}

RANGES = {
    "tt_lower_bonus": (200_000_000, 1_500_000_000),
    "tt_upper_bonus": (0, 100_000_000),
    "promotion_bonus": (100_000_000, 1_200_000_000),
    "good_capture_bonus": (1_000_000, 150_000_000),
    "bad_capture_bonus": (-100_000, 300_000),
    "see_weight": (0, 60),
    "killer1_bonus": (0, 200_000),
    "killer2_bonus": (0, 120_000),
    "check_bonus": (0, 500_000),
    "counter_history_bonus": (0, 120_000),
    "penalty_num": (1, 12),
    "penalty_den": (1, 12),
}

SUMMARY_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def clamp(name, value):
    low, high = RANGES[name]
    return max(low, min(high, int(value)))


def random_value(rng, name):
    low, high = RANGES[name]
    if name in {"see_weight", "penalty_num", "penalty_den"}:
        return rng.randint(low, high)
    if low <= 0:
        return rng.randint(low, high)
    log_low = math.log(low)
    log_high = math.log(high)
    return int(round(math.exp(rng.uniform(log_low, log_high)) / 1000.0) * 1000)


def mutate_value(rng, name, value):
    if rng.random() < 0.18:
        return random_value(rng, name)
    low, high = RANGES[name]
    if name in {"see_weight", "penalty_num", "penalty_den"}:
        radius = max(1, int((high - low) * rng.uniform(0.08, 0.25)))
        return clamp(name, value + rng.randint(-radius, radius))
    scale = rng.uniform(0.55, 1.85)
    jitter = rng.randint(-5000, 5000)
    return clamp(name, round((value * scale + jitter) / 1000.0) * 1000)


def make_candidate(rng, best, index):
    if index == 0:
        return dict(DEFAULT_WEIGHTS)
    base = best if best is not None and rng.random() < 0.75 else DEFAULT_WEIGHTS
    candidate = dict(base)
    change_count = rng.randint(2, 5)
    names = list(DEFAULT_WEIGHTS)
    for name in rng.sample(names, change_count):
        candidate[name] = mutate_value(rng, name, candidate[name])

    if candidate["killer2_bonus"] > candidate["killer1_bonus"]:
        candidate["killer1_bonus"], candidate["killer2_bonus"] = (
            candidate["killer2_bonus"],
            candidate["killer1_bonus"],
        )
    return candidate


def parse_summary(output):
    summary = {}
    for line in output.splitlines():
        if line.startswith("summary "):
            for key, value in SUMMARY_RE.findall(line):
                summary[key] = value
    if not summary:
        raise RuntimeError(f"benchmark summary not found in output:\n{output[-1000:]}")
    return summary


def candidate_args(candidate):
    return [
        "--penalty-divisor", f"{candidate['penalty_num']}/{candidate['penalty_den']}",
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
    ]


def evaluate(args, candidate):
    node_ratios = []
    time_ratios = []
    score_mismatches = 0
    move_mismatches = 0
    runs = []
    for depth in args.depths:
        for seed in args.seeds:
            cmd = [
                str(args.benchmark),
                "--input", str(args.input),
                "--samples", str(args.samples),
                "--depth", str(depth),
                "--iterative",
                "--seed", str(seed),
                *candidate_args(candidate),
            ]
            started = time.time()
            completed = subprocess.run(
                cmd,
                cwd=args.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            elapsed = time.time() - started
            if completed.returncode != 0:
                raise RuntimeError(f"benchmark failed ({completed.returncode}): {' '.join(cmd)}\n{completed.stdout}")
            summary = parse_summary(completed.stdout)
            node_ratio = float(summary["v17_vs_v16_nodes"])
            time_ratio = float(summary["v17_vs_v16_time"])
            sm = int(summary["score_mismatches"])
            mm = int(summary["move_mismatches"])
            node_ratios.append(node_ratio)
            time_ratios.append(time_ratio)
            score_mismatches += sm
            move_mismatches += mm
            runs.append({
                "depth": depth,
                "seed": seed,
                "node_ratio": node_ratio,
                "time_ratio": time_ratio,
                "score_mismatches": sm,
                "move_mismatches": mm,
                "elapsed_sec": elapsed,
            })

    avg_node = sum(node_ratios) / len(node_ratios)
    avg_time = sum(time_ratios) / len(time_ratios)
    worst_node = max(node_ratios)
    score = avg_node + 0.15 * max(0.0, avg_time - 1.0) + 0.05 * max(0.0, worst_node - 1.0)
    score += 1000.0 * (score_mismatches + move_mismatches)
    return {
        "candidate": candidate,
        "avg_node_ratio": avg_node,
        "avg_time_ratio": avg_time,
        "worst_node_ratio": worst_node,
        "score_mismatches": score_mismatches,
        "move_mismatches": move_mismatches,
        "objective": score,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark", type=Path, default=Path("build/benchmark_v16_v17_counterhistory"))
    parser.add_argument("--input", type=Path, default=Path("data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("build/v17_move_ordering_tune.jsonl"))
    parser.add_argument("--best-out", type=Path, default=Path("build/v17_move_ordering_best.json"))
    parser.add_argument("--duration-min", type=float, default=180.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 6, 7])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404])
    parser.add_argument("--rng-seed", type=int, default=20260614)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.benchmark = (args.repo / args.benchmark).resolve()
    args.input = (args.repo / args.input).resolve()
    args.out = (args.repo / args.out).resolve()
    args.best_out = (args.repo / args.best_out).resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.best_out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.rng_seed)
    deadline = time.time() + args.duration_min * 60.0
    best_result = None

    with args.out.open("a", encoding="utf-8") as log:
        for index in range(args.max_candidates):
            if time.time() >= deadline:
                break
            candidate = make_candidate(rng, best_result["candidate"] if best_result else None, index)
            try:
                result = evaluate(args, candidate)
            except Exception as exc:
                result = {
                    "candidate": candidate,
                    "error": str(exc),
                    "objective": float("inf"),
                }

            result["index"] = index
            result["timestamp"] = time.time()
            log.write(json.dumps(result, sort_keys=True) + "\n")
            log.flush()

            if "error" not in result and (
                best_result is None or result["objective"] < best_result["objective"]
            ):
                best_result = result
                args.best_out.write_text(json.dumps(best_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(
                    "NEW_BEST",
                    f"index={index}",
                    f"objective={result['objective']:.6f}",
                    f"avg_node={result['avg_node_ratio']:.6f}",
                    f"avg_time={result['avg_time_ratio']:.6f}",
                    f"worst_node={result['worst_node_ratio']:.6f}",
                    json.dumps(result["candidate"], sort_keys=True),
                    flush=True,
                )
            elif "error" in result:
                print("ERROR", f"index={index}", result["error"], flush=True)
            else:
                print(
                    "candidate",
                    f"index={index}",
                    f"objective={result['objective']:.6f}",
                    f"avg_node={result['avg_node_ratio']:.6f}",
                    f"avg_time={result['avg_time_ratio']:.6f}",
                    flush=True,
                )

    if best_result is not None:
        print(json.dumps(best_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
