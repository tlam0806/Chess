#!/usr/bin/env python3
"""Black-box UCI ponder hit/miss regression test.

Usage: test_uci_ponder_protocol.py ENGINE MODEL
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time


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
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} ENGINE MODEL", file=sys.stderr)
        return 2

    engine = Engine(sys.argv[1], sys.argv[2])
    try:
        engine.send("uci")
        _, handshake = engine.wait_for("uciok", 3.0)
        if not any(line.startswith("option name Ponder ") for line in handshake):
            fail("engine does not advertise the UCI Ponder option")
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
        print(
            "uci ponder protocol passed: "
            f"normal={normal_elapsed:.3f}s hit={hit_elapsed:.3f}s"
        )
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
