#!/usr/bin/env python3
"""Generate reproducible, Stockfish-filtered paired-match openings.

Candidate continuations are sampled only from Stockfish MultiPV moves close to
the best move.  The final position is accepted only when a stronger Stockfish
search considers it balanced, it is not in check, and its material imbalance
is bounded.  Output lines use the UCI move-list format consumed by
phase_nnue_strict_paired_match.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


PIECE_VALUES = {
    "p": 100,
    "n": 320,
    "b": 330,
    "r": 500,
    "q": 900,
    "k": 0,
}


@dataclass(frozen=True)
class AnalysisLine:
    multipv: int
    score_cp: int | None
    mate: int | None
    move: str


class StockfishUci:
    def __init__(self, executable: Path, threads: int, hash_mb: int) -> None:
        self.process = subprocess.Popen(
            [str(executable.resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to open Stockfish UCI pipes")
        self.stdin: TextIO = self.process.stdin
        self.stdout: TextIO = self.process.stdout
        self.send("uci")
        uci_lines = self.read_until("uciok")
        self.id_name = next(
            (
                line.removeprefix("id name ")
                for line in uci_lines
                if line.startswith("id name ")
            ),
            "unknown",
        )
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hash_mb}")
        self.send("setoption name UCI_ShowWDL value false")
        self.send("isready")
        self.read_until("readyok")

    def send(self, command: str) -> None:
        self.stdin.write(command + "\n")
        self.stdin.flush()

    def read_until(self, marker: str) -> list[str]:
        lines: list[str] = []
        while True:
            line = self.stdout.readline()
            if line == "":
                stderr = ""
                if self.process.stderr is not None:
                    stderr = self.process.stderr.read()
                raise RuntimeError(
                    f"Stockfish exited while waiting for {marker!r}: {stderr}"
                )
            clean = line.rstrip("\r\n")
            lines.append(clean)
            if clean.startswith(marker):
                return lines

    def analyze(
        self,
        moves: list[str],
        nodes: int,
        multipv: int,
    ) -> list[AnalysisLine]:
        self.send(f"setoption name MultiPV value {multipv}")
        self.send("position startpos moves " + " ".join(moves))
        self.send(f"go nodes {nodes}")
        latest: dict[int, AnalysisLine] = {}
        for line in self.read_until("bestmove"):
            words = line.split()
            if not words or words[0] != "info" or "pv" not in words or "score" not in words:
                continue
            pv_index = words.index("pv")
            if pv_index + 1 >= len(words):
                continue
            number = 1
            if "multipv" in words:
                index = words.index("multipv")
                if index + 1 >= len(words):
                    continue
                number = int(words[index + 1])
            score_index = words.index("score")
            if score_index + 2 >= len(words):
                continue
            score_type = words[score_index + 1]
            value = int(words[score_index + 2])
            latest[number] = AnalysisLine(
                multipv=number,
                score_cp=value if score_type == "cp" else None,
                mate=value if score_type == "mate" else None,
                move=words[pv_index + 1],
            )
        return [latest[key] for key in sorted(latest)]

    def describe(self, moves: list[str]) -> tuple[str, bool]:
        self.send("position startpos moves " + " ".join(moves))
        self.send("d")
        fen: str | None = None
        in_check: bool | None = None
        while True:
            line = self.stdout.readline()
            if line == "":
                raise RuntimeError("Stockfish exited while describing position")
            clean = line.rstrip("\r\n")
            if clean.startswith("Fen: "):
                fen = clean.removeprefix("Fen: ")
            if clean.startswith("Checkers:"):
                in_check = bool(clean.removeprefix("Checkers:").strip())
                break
        if fen is None or in_check is None:
            raise RuntimeError("Stockfish 'd' output omitted FEN or check state")
        return fen, in_check

    def close(self) -> None:
        if self.process.poll() is None:
            self.send("quit")
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def __enter__(self) -> "StockfishUci":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, type=Path)
    parser.add_argument("--base-book", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--extra-plies", type=int, default=4)
    parser.add_argument("--candidate-nodes", type=int, default=3000)
    parser.add_argument("--filter-nodes", type=int, default=30000)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--max-pv-loss-cp", type=int, default=35)
    parser.add_argument("--max-abs-cp", type=int, default=50)
    parser.add_argument("--max-material-imbalance-cp", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-attempts", type=int, default=100000)
    return parser.parse_args()


def read_book(path: Path) -> list[list[str]]:
    lines: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        clean = raw.split("#", 1)[0].strip()
        if clean:
            lines.append(clean.split())
    if not lines:
        raise ValueError(f"empty base opening book: {path}")
    lengths = {len(line) for line in lines}
    if len(lengths) != 1:
        raise ValueError(f"base opening lines have mixed lengths: {sorted(lengths)}")
    return lines


def material_imbalance_cp(fen: str) -> int:
    board = fen.split()[0]
    white = 0
    black = 0
    for symbol in board:
        lower = symbol.lower()
        if lower not in PIECE_VALUES:
            continue
        if symbol.isupper():
            white += PIECE_VALUES[lower]
        else:
            black += PIECE_VALUES[lower]
    return abs(white - black)


def choose_near_best(
    lines: list[AnalysisLine],
    max_loss_cp: int,
    rng: random.Random,
) -> str | None:
    if any(line.mate is not None for line in lines):
        return None
    cp_lines = [line for line in lines if line.score_cp is not None]
    if not cp_lines:
        return None
    best = max(int(line.score_cp) for line in cp_lines if line.score_cp is not None)
    eligible: list[str] = []
    seen: set[str] = set()
    for line in cp_lines:
        assert line.score_cp is not None
        if best - line.score_cp <= max_loss_cp and line.move not in seen:
            eligible.append(line.move)
            seen.add(line.move)
    return rng.choice(eligible) if eligible else None


def require_positive(args: argparse.Namespace, names: list[str]) -> None:
    for name in names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = parse_args()
    require_positive(
        args,
        [
            "count",
            "extra_plies",
            "candidate_nodes",
            "filter_nodes",
            "multipv",
            "max_abs_cp",
            "threads",
            "hash_mb",
            "max_attempts",
        ],
    )
    if args.max_pv_loss_cp < 0 or args.max_material_imbalance_cp < 0:
        raise ValueError("loss/material limits must be non-negative")
    if not args.stockfish.is_file():
        raise FileNotFoundError(args.stockfish)

    base_book = read_book(args.base_book)
    rng = random.Random(args.seed)
    accepted: list[list[str]] = []
    accepted_scores: list[int] = []
    accepted_material: list[int] = []
    unique_positions: set[str] = set()
    reject = {
        "generation_mate_or_missing": 0,
        "final_mate_or_missing": 0,
        "cp_outside_limit": 0,
        "in_check": 0,
        "material_imbalance": 0,
        "duplicate": 0,
    }
    started = time.monotonic()

    stockfish_id = "unknown"
    with StockfishUci(args.stockfish, args.threads, args.hash_mb) as engine:
        stockfish_id = engine.id_name
        attempts = 0
        while len(accepted) < args.count and attempts < args.max_attempts:
            attempts += 1
            moves = list(rng.choice(base_book))
            valid = True
            for _ in range(args.extra_plies):
                analysis = engine.analyze(
                    moves, args.candidate_nodes, args.multipv
                )
                move = choose_near_best(analysis, args.max_pv_loss_cp, rng)
                if move is None:
                    reject["generation_mate_or_missing"] += 1
                    valid = False
                    break
                moves.append(move)
            if not valid:
                continue

            final = engine.analyze(moves, args.filter_nodes, 1)
            if not final or final[0].score_cp is None:
                reject["final_mate_or_missing"] += 1
                continue
            score = int(final[0].score_cp)
            if abs(score) > args.max_abs_cp:
                reject["cp_outside_limit"] += 1
                continue
            fen, in_check = engine.describe(moves)
            if in_check:
                reject["in_check"] += 1
                continue
            imbalance = material_imbalance_cp(fen)
            if imbalance > args.max_material_imbalance_cp:
                reject["material_imbalance"] += 1
                continue
            position_key = " ".join(fen.split()[:4])
            if position_key in unique_positions:
                reject["duplicate"] += 1
                continue
            unique_positions.add(position_key)
            accepted.append(moves)
            accepted_scores.append(score)
            accepted_material.append(imbalance)
            if len(accepted) % 100 == 0 or len(accepted) == args.count:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "accepted": len(accepted),
                            "attempts": attempts,
                            "acceptance_rate": len(accepted) / attempts,
                            "elapsed_sec": time.monotonic() - started,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    if len(accepted) != args.count:
        raise RuntimeError(
            f"generated only {len(accepted)} of {args.count} openings "
            f"after {attempts} attempts"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Stockfish-filtered balanced openings; one UCI move list per line.\n"
        + "\n".join(" ".join(line) for line in accepted)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "generator": str(Path(__file__).relative_to(Path.cwd())),
        "stockfish": str(args.stockfish),
        "stockfish_id": stockfish_id,
        "base_book": str(args.base_book),
        "output": str(args.output),
        "seed": args.seed,
        "count": args.count,
        "base_plies": len(base_book[0]),
        "extra_plies": args.extra_plies,
        "total_plies": len(base_book[0]) + args.extra_plies,
        "candidate_nodes": args.candidate_nodes,
        "filter_nodes": args.filter_nodes,
        "multipv": args.multipv,
        "max_pv_loss_cp": args.max_pv_loss_cp,
        "max_abs_cp": args.max_abs_cp,
        "max_material_imbalance_cp": args.max_material_imbalance_cp,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "attempts": attempts,
        "acceptance_rate": len(accepted) / attempts,
        "rejections": reject,
        "score_cp": {
            "min": min(accepted_scores),
            "max": max(accepted_scores),
            "mean": statistics.fmean(accepted_scores),
            "mean_abs": statistics.fmean(abs(value) for value in accepted_scores),
        },
        "material_imbalance_cp": {
            "min": min(accepted_material),
            "max": max(accepted_material),
            "mean": statistics.fmean(accepted_material),
        },
        "elapsed_sec": time.monotonic() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **manifest}, separators=(",", ":")))


if __name__ == "__main__":
    main()
