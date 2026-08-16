#!/usr/bin/env python3
"""Summarize each color-reversed matchup in an NNUE round robin."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def summarize(path: Path) -> dict:
    matchups: dict[tuple[str, str], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        game = json.loads(line)
        matchup = (game["profile"], game["opponent"])
        entry = matchups.setdefault(
            matchup,
            {"games": 0, "outcomes": Counter(), "pairs": {}},
        )
        outcome = game["candidate_outcome"]
        entry["games"] += 1
        entry["outcomes"][outcome] += 1
        pair_key = game["key"].rsplit(":", 1)[0]
        entry["pairs"].setdefault(pair_key, []).append(POINTS[outcome])

    result = {"matchups": []}
    for (first, second), entry in matchups.items():
        pair_scores = [
            sum(scores) / 2.0
            for scores in entry["pairs"].values()
            if len(scores) == 2
        ]
        if not pair_scores:
            continue
        mean = sum(pair_scores) / len(pair_scores)
        if len(pair_scores) > 1:
            variance = sum((score - mean) ** 2 for score in pair_scores)
            variance /= len(pair_scores) - 1
            half_width = 1.96 * math.sqrt(variance / len(pair_scores))
        else:
            half_width = 0.5
        lower = max(0.0, mean - half_width)
        upper = min(1.0, mean + half_width)
        result["matchups"].append(
            {
                "first": first,
                "second": second,
                "games": entry["games"],
                "complete_pairs": len(pair_scores),
                "first_outcomes": dict(entry["outcomes"]),
                "first_score": mean,
                "ci95": [lower, upper],
                "ci_excludes_50": lower > 0.5 or upper < 0.5,
            }
        )
    result["matchups"].sort(key=lambda item: (item["first"], item["second"]))
    return result


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
