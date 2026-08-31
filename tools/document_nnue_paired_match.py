#!/usr/bin/env python3
"""Wait for and document a fixed-size, color-reversed NNUE match."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path


POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_games(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def paired_summary(games: list[dict], expected_games: int) -> dict:
    if len(games) != expected_games:
        raise ValueError(f"expected {expected_games} games, found {len(games)}")

    keys = [game["key"] for game in games]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate game keys in match output")

    outcomes = Counter(game["candidate_outcome"] for game in games)
    pairs: dict[str, list[float]] = {}
    for game in games:
        pair_key = game["key"].rsplit(":", 1)[0]
        pairs.setdefault(pair_key, []).append(POINTS[game["candidate_outcome"]])
    incomplete = {key: scores for key, scores in pairs.items() if len(scores) != 2}
    if incomplete:
        raise ValueError(f"incomplete opening pairs: {sorted(incomplete)}")

    pair_scores = [sum(scores) / 2.0 for scores in pairs.values()]
    mean = sum(pair_scores) / len(pair_scores)
    if len(pair_scores) > 1:
        variance = sum((score - mean) ** 2 for score in pair_scores)
        variance /= len(pair_scores) - 1
        half_width = 1.96 * math.sqrt(variance / len(pair_scores))
    else:
        half_width = 0.5
    ci = [max(0.0, mean - half_width), min(1.0, mean + half_width)]

    def score_to_elo(score: float) -> float | None:
        if score <= 0.0 or score >= 1.0:
            return None
        return 400.0 * math.log10(score / (1.0 - score))

    by_color: dict[str, Counter[str]] = {}
    for game in games:
        color = game["candidate_color"]
        by_color.setdefault(color, Counter())[game["candidate_outcome"]] += 1

    candidate_nodes = sum(int(game["candidate_nodes"]) for game in games)
    baseline_nodes = sum(int(game["control_nodes"]) for game in games)
    candidate_time_ms = sum(int(game["candidate_time_ms"]) for game in games)
    baseline_time_ms = sum(int(game["control_time_ms"]) for game in games)
    result = {
        "games": len(games),
        "complete_pairs": len(pair_scores),
        "candidate_outcomes": dict(outcomes),
        "candidate_score": mean,
        "paired_normal_ci95": ci,
        "elo_difference": score_to_elo(mean),
        "elo_ci95": [score_to_elo(ci[0]), score_to_elo(ci[1])],
        "candidate_outcomes_by_color": {
            color: dict(counts) for color, counts in sorted(by_color.items())
        },
        "termination_reasons": dict(Counter(game["reason"] for game in games)),
        "candidate_nodes": candidate_nodes,
        "baseline_nodes": baseline_nodes,
        "candidate_time_ms": candidate_time_ms,
        "baseline_time_ms": baseline_time_ms,
        "candidate_aggregate_nps": candidate_nodes / (candidate_time_ms / 1000.0),
        "baseline_aggregate_nps": baseline_nodes / (baseline_time_ms / 1000.0),
    }
    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def elo(value: float | None) -> str:
    return "unbounded" if value is None else f"{value:+.1f}"


def render_report(manifest: dict, summary: dict, evidence: dict[str, str]) -> str:
    score = summary["candidate_score"]
    lower, upper = summary["paired_normal_ci95"]
    if lower > 0.5:
        verdict = "candidate ahead; nominal 95% paired CI excludes 50%"
    elif upper < 0.5:
        verdict = "candidate behind; nominal 95% paired CI excludes 50%"
    else:
        verdict = "inconclusive; nominal 95% paired CI contains 50%"
    outcomes = summary["candidate_outcomes"]
    by_color = summary["candidate_outcomes_by_color"]
    reasons = summary["termination_reasons"]
    elo_lower, elo_upper = summary["elo_ci95"]

    def wdl(counts: dict) -> str:
        return (
            f"{counts.get('win', 0)}-{counts.get('draw', 0)}-"
            f"{counts.get('loss', 0)}"
        )

    reason_rows = "\n".join(
        f"| `{reason}` | {count} |" for reason, count in sorted(reasons.items())
    )
    return f"""# {manifest['title']}

Date: {manifest['date']}

Status: **complete — {verdict}**

## Result

All scores below are from `{manifest['candidate']['name']}`'s perspective.

| Games / pairs | W-D-L | Score | Paired 95% CI | Elo estimate | Elo 95% CI |
|---:|---:|---:|---:|---:|---:|
| {summary['games']} / {summary['complete_pairs']} | {wdl(outcomes)} | {pct(score)} | {pct(lower)}–{pct(upper)} | {elo(summary['elo_difference'])} | {elo(elo_lower)} to {elo(elo_upper)} |

| Candidate color | W-D-L |
|---|---:|
| White | {wdl(by_color.get('white', {}))} |
| Black | {wdl(by_color.get('black', {}))} |

## Protocol

- Engine: `{manifest['engine']['path']}` at git commit `{manifest['engine']['git_commit']}`.
- Candidate: `{manifest['candidate']['path']}`.
- Baseline: `{manifest['baseline']['path']}`.
- Opening book: `{manifest['book']['path']}`; {manifest['protocol']['openings']} Stockfish-balanced openings, each played with colors reversed.
- Time control: {manifest['protocol']['base_ms'] / 1000:g}s + {manifest['protocol']['increment_ms'] / 1000:g}s per move; overhead {manifest['protocol']['overhead_ms']}ms.
- Search: V41 with all dirty pruning disabled via `--clean-search`; TT {manifest['protocol']['tt_mb']} MiB; maximum {manifest['protocol']['max_plies']} plies.
- Fixed sample: exactly {manifest['expected_games']} games; no CI-based early stopping; seed `{manifest['protocol']['seed']}`.
- Both sides used the same search settings. Only the quantized NNUE artifact differed.

The harness clears TT before every game. Other move-ordering heuristic tables
remain owned by each engine for the duration of the matchup, matching this
runner's historical behavior. The reported interval is the normal 95% interval
over color-reversed opening-pair scores; it is not an SPRT boundary.

## Termination audit

| Reason | Games |
|---|---:|
{reason_rows}

Aggregate measured search throughput was
`{summary['candidate_aggregate_nps']:.0f}` NPS for the candidate and
`{summary['baseline_aggregate_nps']:.0f}` NPS for the baseline. These are
in-match wall-clock aggregates, not an isolated fixed-node benchmark.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| Engine binary | `{evidence['engine']}` |
| Candidate model | `{evidence['candidate']}` |
| Baseline model | `{evidence['baseline']}` |
| Opening book | `{evidence['book']}` |
| Game JSONL | `{evidence['games']}` |
| Runner log | `{evidence['runner_log']}` |
| Summary JSON | `{evidence['summary']}` |
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    games_path = root / manifest["games"]
    expected_games = int(manifest["expected_games"])
    while True:
        games = load_games(games_path)
        if len(games) >= expected_games:
            break
        if not args.wait:
            raise SystemExit(
                f"match incomplete: {len(games)}/{expected_games} games"
            )
        time.sleep(args.poll_seconds)

    summary = paired_summary(games, expected_games)
    summary_path = root / manifest["summary"]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    paths = {
        "engine": Path(manifest["engine"]["path"]),
        "candidate": Path(manifest["candidate"]["path"]),
        "baseline": Path(manifest["baseline"]["path"]),
        "book": Path(manifest["book"]["path"]),
        "games": games_path,
        "runner_log": root / manifest["runner_log"],
        "summary": summary_path,
    }
    evidence = {name: sha256(path) for name, path in paths.items()}
    for name in ("engine", "candidate", "baseline", "book"):
        expected = manifest[name]["sha256"]
        if evidence[name] != expected:
            raise ValueError(
                f"{name} hash changed: expected {expected}, got {evidence[name]}"
            )

    report_path = Path(manifest["report"])
    report_path.write_text(render_report(manifest, summary, evidence))
    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}))


if __name__ == "__main__":
    main()
