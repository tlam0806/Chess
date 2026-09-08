#!/usr/bin/env python3
"""Black-box UCI ponder hit/miss regression test.

Usage: test_uci_ponder_protocol.py ENGINE MODEL CONFIG
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from tools.benchmark.nnue_v43_production_profile import (  # noqa: E402
    PRODUCTION_CONFIG_HASH,
    config_profile_hash,
)


PROMOTED_V43_HISTORY = (
    "e2e4 c7c5 d2d4 c5d4 g1f3 b8c6 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6 "
    "c3d5 f6d5 e4d5 c6b8 c2c4 f8e7 f1e2 e8g8 e1g1 a7a6 b5c3 f7f5 "
    "b2b4 a6a5 b4a5 a8a5 c3b5 b8d7 c1a3 a5a6 a1b1 f8f6 d1c2 d7c5 "
    "a3c5 d6c5 f1d1 e5e4 f2f3 f6h6 f3e4 e7d6 b5d6 d8d6 g2g3 f5f4 "
    "b1b3 d6e5 c2c3 e5g5 c3f3 a6f6 f3f2 b7b6 b3f3 c8g4 g3f4 g5h5 "
    "f3e3 f6g6 g1h1 g4e2 f2e2 h5h4 d1f1 g6g4 e3f3 h4h5 f3f2 h5h3 "
    "e4e5 g4g3 f1e1 h6g6 e5e6 g3e3 e2e3 h3e3"
)

PROMOTED_V43_OPTIONS = {
    "AspirationMinDepth": "2",
    "AspirationDeltaBaseCp": "68",
    "AspirationDeltaDivisor": "33700",
    "AspirationExpansionPermille": "2290",
    "AspirationMaxFailHighReductions": "1",
    "AspirationMeanWeightPermille": "370",
    "NullMoveReduction": "4",
    "RfpMaxDepth": "4",
    "RfpBaseMargin": "50",
    "RfpMarginPerDepth": "100",
    "LmpDepthMultiplier": "1",
    "QseeThreshold": "-25",
}


def fail(message: str) -> None:
    raise AssertionError(message)


class Engine:
    def __init__(self, executable: str, model: str) -> None:
        self.process = subprocess.Popen(
            [executable, model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_line(self, timeout: float) -> str | None:
        try:
            line = self.lines.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if line is None:
            fail(f"engine exited early with status {self.process.poll()}")
        return line

    def wait_for(self, prefix: str, timeout: float) -> tuple[str, list[str]]:
        deadline = time.monotonic() + timeout
        observed: list[str] = []
        while time.monotonic() < deadline:
            line = self.read_line(deadline - time.monotonic())
            if line is None:
                break
            observed.append(line)
            if line.startswith(prefix):
                return line, observed
        fail(f"timed out waiting for {prefix!r}; observed={observed}")

    def assert_no_bestmove(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        observed: list[str] = []
        while time.monotonic() < deadline:
            line = self.read_line(deadline - time.monotonic())
            if line is None:
                break
            observed.append(line)
            if line.startswith("bestmove "):
                fail(f"ponder returned before ponderhit/stop: {observed}")

    def close(self) -> None:
        if self.process.poll() is None:
            self.send("quit")
        try:
            self.process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3.0)
        if self.process.returncode != 0:
            assert self.process.stderr is not None
            fail(
                f"engine exited with {self.process.returncode}: "
                f"{self.process.stderr.read()}"
            )


def parse_bestmove(line: str) -> tuple[str, str]:
    parts = line.split()
    if len(parts) < 4 or parts[0] != "bestmove" or parts[2] != "ponder":
        fail(f"expected bestmove plus ponder reply, got {line!r}")
    return parts[1], parts[3]


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} ENGINE MODEL CONFIG", file=sys.stderr)
        return 2

    actual_profile_hash = config_profile_hash(Path(sys.argv[3]))
    if actual_profile_hash != PRODUCTION_CONFIG_HASH:
        fail(
            "rendered deployment config hash mismatch: "
            f"expected {PRODUCTION_CONFIG_HASH}, got {actual_profile_hash}"
        )

    engine = Engine(sys.argv[1], sys.argv[2])
    try:
        engine.send("uci")
        _, handshake = engine.wait_for("uciok", 3.0)
        if "id name ChessNNUEV43" not in handshake:
            fail(f"release engine is not promoted V43: {handshake}")
        if not any(line.startswith("option name Ponder ") for line in handshake):
            fail("engine does not advertise the UCI Ponder option")
        for name, expected in PROMOTED_V43_OPTIONS.items():
            prefix = f"option name {name} "
            line = next((value for value in handshake if value.startswith(prefix)), None)
            if line is None or f" default {expected}" not in line:
                fail(f"unexpected promoted default for {name}: {line!r}")
        engine.send("isready")
        engine.wait_for("readyok", 3.0)

        # A normal timed search must stop through the controller timer.
        engine.send("position startpos")
        started = time.monotonic()
        engine.send("go movetime 150")
        normal_line, _ = engine.wait_for("bestmove ", 2.0)
        normal_elapsed = time.monotonic() - started
        if not 0.08 <= normal_elapsed <= 1.0:
            fail(f"150 ms search took {normal_elapsed:.3f} s")
        best, ponder = parse_bestmove(normal_line)

        # A ponder search must not spend our clock before ponderhit. On a hit,
        # the same search continues and receives its normal move budget.
        engine.send(f"position startpos moves {best} {ponder}")
        engine.send("go ponder wtime 5000 btime 5000 winc 0 binc 0")
        engine.assert_no_bestmove(0.35)
        hit_started = time.monotonic()
        engine.send("ponderhit")
        engine.wait_for("bestmove ", 2.0)
        hit_elapsed = time.monotonic() - hit_started
        if not 0.08 <= hit_elapsed <= 1.0:
            fail(f"ponderhit continuation took {hit_elapsed:.3f} s")

        # On a miss python-chess sends stop and waits for bestmove before the
        # actual position/go. The adapter must unblock that exact sequence.
        engine.send("position startpos moves e2e4 e7e5")
        engine.send("go ponder wtime 5000 btime 5000 winc 0 binc 0")
        engine.assert_no_bestmove(0.20)
        miss_started = time.monotonic()
        engine.send("stop")
        engine.wait_for("bestmove ", 1.0)
        if time.monotonic() - miss_started > 0.5:
            fail("ponder miss did not stop cooperatively within 500 ms")

        # The process must remain usable after cancelling a ponder miss.
        engine.send("position startpos moves d2d4 d7d5")
        engine.send("go depth 3")
        engine.wait_for("bestmove ", 2.0)

        # Regression from Lichess game Ahm5kxQZ. V41 missed the hanging queen
        # with 41.Ref1; promoted V43 must play the immediate 41.Rxe3.
        engine.send("setoption name TwofoldSearchDraw value true")
        engine.send("setoption name ReuseStaleTtScores value false")
        engine.send("setoption name ReuseDeeperTtScores value false")
        engine.send("isready")
        engine.wait_for("readyok", 3.0)
        engine.send("ucinewgame")
        engine.send(f"position startpos moves {PROMOTED_V43_HISTORY}")
        engine.send("go depth 8")
        tactical_line, _ = engine.wait_for("bestmove ", 5.0)
        if tactical_line.split()[1] != "e1e3":
            fail(f"Ahm5kxQZ regression: expected e1e3, got {tactical_line!r}")
        print(
            "uci ponder and promoted-profile protocol passed: "
            f"normal={normal_elapsed:.3f}s hit={hit_elapsed:.3f}s"
        )
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
