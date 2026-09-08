#!/usr/bin/env python3
"""Stop paired NNUE-vs-baseline workers once their bootstrap CIs separate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

import numpy as np


GAME_RE = re.compile(
    r"^game model=\S+ pair=(\d+).*candidate_color=(white|black) result=(\S+)"
)


def complete_pairs(path: Path) -> dict[int, float]:
    games: dict[int, list[float]] = {}
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        match = GAME_RE.match(line)
        if not match:
            continue
        pair_text, color, result = match.groups()
        if result == "1/2-1/2":
            points = 0.5
        else:
            points = float((result == "1-0") == (color == "white"))
        games.setdefault(int(pair_text), []).append(points)
    return {
        pair: sum(points)
        for pair, points in games.items()
        if len(points) == 2
    }


def bootstrap_ci(
    values: list[float], rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    chunks: list[np.ndarray] = []
    chunk_size = 5000
    for begin in range(0, samples, chunk_size):
        size = min(chunk_size, samples - begin)
        indices = rng.integers(0, len(data), size=(size, len(data)))
        chunks.append(data[indices].mean(axis=1) / 2.0)
    rates = np.sort(np.concatenate(chunks))
    return (
        float(rates[int(0.025 * samples)]),
        float(rates[min(samples - 1, int(0.975 * samples))]),
    )


def worker_pids(pattern: str) -> list[int]:
    result = subprocess.run(
        ["ps", "-ax", "-o", "pid=", "-o", "comm=", "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid_text, executable, command = fields
        if (
            command.startswith("./build-release/phase_nnue_strict_paired_match ")
            and pattern in command
        ):
            pids.append(int(pid_text))
    return pids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", nargs=2, metavar=("NEW", "OLD"), required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--check-step", type=int, default=25)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--process-pattern", required=True)
    args = parser.parse_args()

    pairs = [(Path(new), Path(old)) for new, old in args.pair]
    last_checked = 0
    while True:
        new_values: list[float] = []
        old_values: list[float] = []
        used: list[int] = []
        for new_path, old_path in pairs:
            new = complete_pairs(new_path)
            old = complete_pairs(old_path)
            common = min(len(new), len(old))
            used.append(common)
            new_values.extend(new[index] for index in range(common))
            old_values.extend(old[index] for index in range(common))

        total = len(new_values)
        should_check = (
            total >= args.min_pairs
            and total >= last_checked + args.check_step
        )
        if should_check:
            rng = np.random.default_rng(20260730 + total)
            new_ci = bootstrap_ci(new_values, rng, args.samples)
            old_ci = bootstrap_ci(old_values, rng, args.samples)
            new_rate = sum(new_values) / (2.0 * total)
            old_rate = sum(old_values) / (2.0 * total)
            print(
                f"ci pairs={total} games_each={2 * total} used={used} "
                f"new_rate={new_rate:.8f} new_lower={new_ci[0]:.8f} "
                f"new_upper={new_ci[1]:.8f} old_rate={old_rate:.8f} "
                f"old_lower={old_ci[0]:.8f} old_upper={old_ci[1]:.8f}",
                flush=True,
            )
            last_checked = total
            winner = None
            if new_ci[0] > old_ci[1]:
                winner = "new_200m_hs2x8_os128"
            elif old_ci[0] > new_ci[1]:
                winner = "old_huber200_50m_hs8x16_os128"
            if winner is not None:
                pids = [pid for pid in worker_pids(args.process_pattern) if pid != os.getpid()]
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                print(
                    f"decision winner={winner} pairs={total} games_each={2 * total} "
                    f"stopped_pids={pids}",
                    flush=True,
                )
                return 0

        if not worker_pids(args.process_pattern):
            print(
                f"complete_without_separation pairs={total} games_each={2 * total}",
                flush=True,
            )
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
