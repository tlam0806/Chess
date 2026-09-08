#!/usr/bin/env python3
"""Analyze a labeled Lichess BOT population with clustered uncertainty."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HEADER = b"CHSCBIN2" + struct.pack("<HHI", 2, 13, 1)
RECORD_SIZE = 40
EDGES = (0.0, 100.0, 300.0, 600.0, 1000.0, 1600.0, 2000.0)
LABELS = tuple(f"{int(left)}-{int(right)}" for left, right in zip(EDGES, EDGES[1:]))


def cp_bin(value: float) -> int:
    magnitude = min(abs(value), 2000.0)
    for index, (left, right) in enumerate(zip(EDGES, EDGES[1:])):
        if left <= magnitude < right or (index == len(LABELS) - 1 and magnitude <= right):
            return index
    raise AssertionError(magnitude)


def ratios(counts: np.ndarray) -> list[float]:
    total = int(counts.sum())
    return [float(value / total) for value in counts] if total else [0.0] * len(counts)


def clustered_bootstrap(
    vectors: list[np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    matrix = np.stack(vectors).astype(np.int64)
    generator = np.random.default_rng(seed)
    samples = np.empty((replicates, matrix.shape[1]), dtype=np.float64)
    cluster_count = matrix.shape[0]
    for replicate in range(replicates):
        indices = generator.integers(0, cluster_count, size=cluster_count)
        counts = matrix[indices].sum(axis=0)
        samples[replicate] = counts / counts.sum()
    lower = np.quantile(samples, 0.025, axis=0)
    upper = np.quantile(samples, 0.975, axis=0)
    halfwidth = (upper - lower) / 2.0
    return {
        "clusters": cluster_count,
        "replicates": replicates,
        "bins": [
            {
                "range": label,
                "ci95": [float(lo), float(hi)],
                "halfwidth": float(hw),
            }
            for label, lo, hi, hw in zip(LABELS, lower, upper, halfwidth)
        ],
        "max_halfwidth": float(halfwidth.max()),
    }


def vector_summary(counts: np.ndarray) -> list[dict[str, object]]:
    values = ratios(counts)
    return [
        {"range": label, "samples": int(count), "ratio": ratio}
        for label, count, ratio in zip(LABELS, counts, values)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-cbin", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--stability-threshold", type=float, default=0.0025)
    parser.add_argument("--require-stable", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0 or args.batch_size <= 0:
        raise ValueError("bootstrap replicates and batch size must be positive")

    payload = args.labeled_cbin.read_bytes()
    if payload[: len(HEADER)] != HEADER:
        raise RuntimeError("invalid CHSCBIN2 header")
    body = payload[len(HEADER) :]
    if len(body) % RECORD_SIZE:
        raise RuntimeError("truncated CHSCBIN2 payload")
    records = len(body) // RECORD_SIZE

    overall = np.zeros(len(LABELS), dtype=np.int64)
    phase = np.zeros((8, len(LABELS)), dtype=np.int64)
    quarter = np.zeros((4, len(LABELS)), dtype=np.int64)
    by_game: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(LABELS), dtype=np.int64))
    by_account: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(LABELS), dtype=np.int64))
    by_source: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(LABELS), dtype=np.int64))
    account_records: Counter[str] = Counter()
    seen_keys: set[str] = set()
    ratings: list[int] = []
    absolute_values: list[float] = []
    cumulative: list[dict[str, object]] = []
    clamped = 0
    abs_ge_1500 = 0

    with args.metadata.open(encoding="utf-8") as metadata:
        for index in range(records):
            line = metadata.readline()
            if not line:
                raise RuntimeError(f"metadata ended before record {index}")
            row = json.loads(line)
            if row.get("record") != index:
                raise RuntimeError(f"metadata record index mismatch at {index}")
            key = row["model_key"]
            if key in seen_keys:
                raise RuntimeError(f"duplicate F2M model key at record {index}")
            seen_keys.add(key)
            raw_score = struct.unpack_from("<h", body, index * RECORD_SIZE + 34)[0]
            unclamped_cp = raw_score * 100.0 / 208.0
            cp = max(-2000.0, min(2000.0, unclamped_cp))
            bucket = cp_bin(cp)
            overall[bucket] += 1
            phase[int(row["phase"])][bucket] += 1
            quarter[int(row["quarter"])][bucket] += 1
            by_game[row["game_id"]][bucket] += 1
            by_account[row["player"]][bucket] += 1
            by_source[row.get("source", row.get("source_id", "unknown"))][bucket] += 1
            account_records[row["player"]] += 1
            ratings.append(int(row["player_elo"]))
            absolute_values.append(abs(cp))
            clamped += int(abs(unclamped_cp) > 2000.0)
            abs_ge_1500 += int(abs(cp) >= 1500.0)
            if (index + 1) % args.batch_size == 0 or index + 1 == records:
                cumulative.append(
                    {
                        "samples": index + 1,
                        "ratios": ratios(overall.copy()),
                    }
                )
        if metadata.readline():
            raise RuntimeError("metadata contains more rows than the labeled CBin")

    if not records:
        raise RuntimeError("population is empty")
    if min(ratings) < 2300 or max(ratings) > 2600:
        raise RuntimeError("metadata contains a BOT rating outside 2300-2600")
    if max(account_records.values()) > 2000:
        raise RuntimeError("an account exceeds the 2000-position cap")

    changes: list[dict[str, object]] = []
    for previous, current in zip(cumulative, cumulative[1:]):
        delta = [abs(a - b) for a, b in zip(previous["ratios"], current["ratios"])]
        changes.append(
            {
                "from_samples": previous["samples"],
                "to_samples": current["samples"],
                "max_abs_ratio_change": max(delta),
            }
        )
    last_two_stable = len(changes) >= 2 and all(
        row["max_abs_ratio_change"] <= args.stability_threshold for row in changes[-2:]
    )
    game_bootstrap = clustered_bootstrap(
        list(by_game.values()), args.bootstrap_replicates, args.seed
    )
    account_bootstrap = clustered_bootstrap(
        list(by_account.values()), args.bootstrap_replicates, args.seed + 1
    )
    stable = bool(
        records >= 100_000
        and last_two_stable
        and game_bootstrap["max_halfwidth"] <= args.stability_threshold
    )

    sorted_abs = np.sort(np.asarray(absolute_values, dtype=np.float64))
    result = {
        "format": "lichess-bot-population-analysis-v1",
        "records": records,
        "games": len(by_game),
        "accounts": len(by_account),
        "rating_range": [min(ratings), max(ratings)],
        "account_record_range": [min(account_records.values()), max(account_records.values())],
        "bins": vector_summary(overall),
        "abs_cp_ge_1500_ratio": abs_ge_1500 / records,
        "clamped_at_2000_ratio": clamped / records,
        "abs_cp_quantiles": {
            "p50": float(np.quantile(sorted_abs, 0.50)),
            "p90": float(np.quantile(sorted_abs, 0.90)),
            "p95": float(np.quantile(sorted_abs, 0.95)),
        },
        "phase_bins": {
            str(index): vector_summary(values) for index, values in enumerate(phase)
        },
        "quarter_bins": {
            str(index): vector_summary(values) for index, values in enumerate(quarter)
        },
        "source_bins": {
            source: vector_summary(values) for source, values in sorted(by_source.items())
        },
        "cumulative_batches": cumulative,
        "batch_changes": changes,
        "game_cluster_bootstrap": game_bootstrap,
        "account_cluster_bootstrap": account_bootstrap,
        "stability": {
            "threshold": args.stability_threshold,
            "last_two_batches_stable": last_two_stable,
            "game_ci_stable": game_bootstrap["max_halfwidth"] <= args.stability_threshold,
            "stable": stable,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "records": records,
        "games": len(by_game),
        "accounts": len(by_account),
        "max_game_ci_halfwidth": game_bootstrap["max_halfwidth"],
        "last_two_batches_stable": last_two_stable,
        "stable": stable,
    }, sort_keys=True))
    return 0 if stable or not args.require_stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
