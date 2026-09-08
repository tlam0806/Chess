#!/usr/bin/env python3
"""Summarize a color-reversed V38 paired match."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def summarize(path: Path) -> dict:
    games = [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]
    pairs: dict[str, list[float]] = {}
    outcomes: Counter[str] = Counter()
    for game in games:
        outcome = game["candidate_outcome"]
        outcomes[outcome] += 1
        key = game["key"].rsplit(":", 1)[0]
        pairs.setdefault(key, []).append(POINTS[outcome])
    complete = [
        sum(scores) / 2.0 for scores in pairs.values() if len(scores) == 2
    ]
    if not complete:
        raise ValueError("no complete opening pairs")
    mean = sum(complete) / len(complete)
    if len(complete) > 1:
        variance = sum((score - mean) ** 2 for score in complete)
        variance /= len(complete) - 1
        half_width = 1.96 * math.sqrt(variance / len(complete))
    else:
        half_width = 0.5
    return {
        "games": len(games),
        "complete_pairs": len(complete),
        "candidate_outcomes": dict(outcomes),
        "candidate_score": mean,
        "ci95": [
            max(0.0, mean - half_width),
            min(1.0, mean + half_width),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
