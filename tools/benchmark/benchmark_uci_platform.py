#!/usr/bin/env python3
"""Reproducible wall-clock benchmark for a UCI chess engine.

The harness deliberately starts a fresh engine process for every measured
position.  Production V43 may retain TT and search-heuristic state across
searches until ``ucinewgame``; the harness uses both a fresh process and an
explicit new-game reset so fixed-depth measurements cannot depend on a
preceding case.

Only the interval from sending ``go`` to receiving ``bestmove`` is timed.  The
engine startup, model load, UCI handshake, and host/CLI startup are excluded.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import hashlib
import json
import math
import os
import platform
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


RESULT_SENTINEL = "CHESS_BENCHMARK_RESULT="
ERROR_SENTINEL = "CHESS_BENCHMARK_ERROR="
PROGRESS_SENTINEL = "CHESS_BENCHMARK_PROGRESS="
UCI_MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)
OPTION_RE = re.compile(r"^option name (.+?) type \S+")


class BenchmarkError(RuntimeError):
    """A protocol, integrity, or configuration failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def median_absolute_deviation(values: Sequence[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def scheduled_positions(
    positions: Sequence[dict[str, str]], round_index: int
) -> list[dict[str, str]]:
    """Balance early/late thermal drift with rotate-then-reverse ordering."""
    if not positions:
        return []
    rotation = (round_index // 2) % len(positions)
    ordered = list(positions[rotation:]) + list(positions[:rotation])
    if round_index % 2:
        ordered.reverse()
    return ordered


def measurement_schedule(protocol: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Interleave limits so mode is not confounded with whole-run drift."""
    fixed_rounds = int(protocol["fixed_rounds"])
    movetime_rounds = int(protocol["movetime_rounds"])
    epochs = max(fixed_rounds, movetime_rounds)
    tasks_by_epoch: list[list[tuple[str, int, int]]] = [[] for _ in range(epochs)]
    for round_index in range(fixed_rounds):
        epoch = min(epochs - 1, (round_index * epochs) // fixed_rounds)
        tasks_by_epoch[epoch].extend(
            ("fixed_depth", int(limit), round_index)
            for limit in protocol["fixed_depths"]
        )
    for round_index in range(movetime_rounds):
        epoch = min(epochs - 1, (round_index * epochs) // movetime_rounds)
        tasks_by_epoch[epoch].extend(
            ("movetime_ms", int(limit), round_index)
            for limit in protocol["movetimes_ms"]
        )

    schedule = []
    for epoch, tasks in enumerate(tasks_by_epoch):
        if not tasks:
            continue
        rotation = epoch % len(tasks)
        ordered = tasks[rotation:] + tasks[:rotation]
        if epoch % 2:
            ordered.reverse()
        schedule.extend(ordered)
    return schedule


def option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        raise BenchmarkError("empty YAML scalar in uci_options")
    if value[0] in {'"', "'"}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise BenchmarkError(f"invalid quoted YAML scalar: {value}") from error
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_config_uci_options(path: Path) -> dict[str, Any]:
    """Read the simple scalar uci_options mapping from the deployment YAML."""
    result: dict[str, Any] = {}
    section_indent: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if section_indent is None:
            if stripped == "uci_options:":
                section_indent = indent
            continue
        if indent <= section_indent:
            break
        if ":" not in stripped:
            raise BenchmarkError(f"invalid uci_options YAML line: {raw_line}")
        name, value = stripped.split(":", 1)
        result[name.strip()] = parse_yaml_scalar(value)
    if section_indent is None or not result:
        raise BenchmarkError(f"config has no uci_options mapping: {path}")
    return result


def validate_config_options(
    config_path: Path | None,
    spec_options: dict[str, Any],
    required: bool,
) -> dict[str, Any]:
    if config_path is None:
        if required:
            raise BenchmarkError("protocol requires --config for option validation")
        return {"status": "not_checked", "options": None}
    config_options = read_config_uci_options(config_path)
    normalized_spec = {
        name: option_value(value) for name, value in spec_options.items()
    }
    normalized_config = {
        name: option_value(value) for name, value in config_options.items()
    }
    if normalized_config != normalized_spec:
        missing = sorted(set(normalized_spec) - set(normalized_config))
        extra = sorted(set(normalized_config) - set(normalized_spec))
        changed = sorted(
            name
            for name in set(normalized_spec) & set(normalized_config)
            if normalized_spec[name] != normalized_config[name]
        )
        raise BenchmarkError(
            "benchmark spec/config UCI options differ: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return {"status": "match", "options": normalized_config}


def parse_info_line(line: str) -> dict[str, Any] | None:
    tokens = line.split()
    if not tokens or tokens[0] != "info":
        return None
    parsed: dict[str, Any] = {"raw_info": line}
    index = 1
    while index < len(tokens):
        token = tokens[index]
        try:
            if token == "depth" and index + 1 < len(tokens):
                parsed["reported_depth"] = int(tokens[index + 1])
                index += 2
                continue
            if token == "nodes" and index + 1 < len(tokens):
                parsed["nodes"] = int(tokens[index + 1])
                index += 2
                continue
            if token == "score" and index + 2 < len(tokens):
                parsed["score_type"] = tokens[index + 1]
                parsed["score_value"] = int(tokens[index + 2])
                index += 3
                continue
        except ValueError as error:
            raise BenchmarkError(f"malformed UCI info line: {line}") from error
        index += 1
    required = {"reported_depth", "nodes", "score_type", "score_value"}
    return parsed if required.issubset(parsed) else None


class EngineSession:
    def __init__(
        self,
        command: Sequence[str],
        cwd: Path | None,
        timeout_seconds: float,
    ) -> None:
        self.command = list(command)
        self.timeout_seconds = timeout_seconds
        self.process = subprocess.Popen(
            self.command,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise BenchmarkError("failed to open UCI subprocess pipes")
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self._lines.put(line.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise BenchmarkError(
                f"engine exited with code {self.process.returncode} before {command!r}"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_line(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BenchmarkError("timed out waiting for UCI output")
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty as error:
            raise BenchmarkError("timed out waiting for UCI output") from error
        if line is None:
            raise BenchmarkError(f"unexpected UCI EOF (exit={self.process.poll()})")
        return line

    def initialize(
        self,
        expected_name: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        self.send("uci")
        deadline = time.monotonic() + self.timeout_seconds
        engine_name: str | None = None
        advertised: set[str] = set()
        info_strings: list[str] = []
        while True:
            line = self.read_line(deadline)
            if line.startswith("id name "):
                engine_name = line.removeprefix("id name ")
            match = OPTION_RE.match(line)
            if match:
                advertised.add(match.group(1))
            if line.startswith("info string "):
                info_strings.append(line.removeprefix("info string "))
            if line == "uciok":
                break
        if engine_name != expected_name:
            raise BenchmarkError(
                f"expected engine {expected_name!r}, received {engine_name!r}"
            )
        missing = sorted(set(options) - advertised)
        if missing:
            raise BenchmarkError(f"engine did not advertise options: {missing}")
        for name, value in options.items():
            self.send(f"setoption name {name} value {option_value(value)}")
        self.send("isready")
        deadline = time.monotonic() + self.timeout_seconds
        rejected_options = []
        while True:
            line = self.read_line(deadline)
            if "ignored invalid option" in line:
                rejected_options.append(line)
            if line == "readyok":
                break
        if rejected_options:
            raise BenchmarkError(
                "engine rejected UCI options: " + "; ".join(rejected_options)
            )
        kernel = None
        accumulator_kernel = None
        for value in info_strings:
            if value.startswith("nnue_kernel="):
                kernel = value.split("=", 1)[1]
            elif value.startswith("nnue_accumulator_kernel="):
                accumulator_kernel = value.split("=", 1)[1]
        return {
            "engine_name": engine_name,
            "advertised_options": sorted(advertised),
            "nnue_kernel": kernel,
            "nnue_accumulator_kernel": accumulator_kernel,
        }

    def search(
        self,
        fen: str,
        mode: str,
        limit: int,
    ) -> dict[str, Any]:
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        if mode == "fixed_depth":
            go_command = f"go depth {limit}"
        elif mode == "movetime_ms":
            go_command = f"go movetime {limit}"
        else:
            raise BenchmarkError(f"unsupported benchmark mode: {mode}")

        started_ns = time.perf_counter_ns()
        self.send(go_command)
        deadline = time.monotonic() + self.timeout_seconds
        latest_info: dict[str, Any] | None = None
        bestmove: str | None = None
        while bestmove is None:
            line = self.read_line(deadline)
            parsed = parse_info_line(line)
            if parsed is not None:
                latest_info = parsed
            if line.startswith("bestmove "):
                parts = line.split()
                if len(parts) < 2:
                    raise BenchmarkError(f"malformed bestmove line: {line}")
                bestmove = parts[1]
        elapsed_ns = time.perf_counter_ns() - started_ns
        if latest_info is None:
            raise BenchmarkError("bestmove arrived without a complete info line")
        if latest_info["nodes"] <= 0:
            raise BenchmarkError("engine reported a non-positive node count")
        if not UCI_MOVE_RE.fullmatch(bestmove):
            raise BenchmarkError(f"invalid benchmark bestmove: {bestmove!r}")
        return {
            **latest_info,
            "bestmove": bestmove,
            "elapsed_ns": elapsed_ns,
            "nps": latest_info["nodes"] * 1_000_000_000.0 / elapsed_ns,
        }

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=3)
            except (BenchmarkError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=3)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._reader.join(timeout=1)

    def __enter__(self) -> "EngineSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def cpuinfo_metadata() -> dict[str, Any]:
    text = read_optional(Path("/proc/cpuinfo"))
    if not text:
        return {}
    fields: dict[str, str] = {}
    processors = 0
    for line in text.splitlines():
        if line.startswith("processor"):
            processors += 1
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in {"model name", "vendor_id", "Hardware", "Features", "flags"}:
            fields.setdefault(key, value)
    return {"logical_processors": processors, **fields}


def cgroup_metadata() -> dict[str, str | None]:
    paths = {
        "membership": Path("/proc/self/cgroup"),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max"),
        "cpu_stat": Path("/sys/fs/cgroup/cpu.stat"),
        "memory_max": Path("/sys/fs/cgroup/memory.max"),
        "memory_current": Path("/sys/fs/cgroup/memory.current"),
        "v1_cpu_quota_us": Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "v1_cpu_period_us": Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        "v1_cpu_stat": Path("/sys/fs/cgroup/cpu/cpu.stat"),
        "v1_memory_limit_bytes": Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        "v1_memory_usage_bytes": Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    }
    return {name: read_optional(path) for name, path in paths.items()}


def machine_metadata() -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = None
    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = None
    allowed_environment = {
        name: os.environ[name]
        for name in (
            "DYNO",
            "HEROKU_APP_ID",
            "HEROKU_APP_NAME",
            "HEROKU_RELEASE_CREATED_AT",
            "HEROKU_RELEASE_VERSION",
            "HEROKU_SLUG_COMMIT",
            "HEROKU_SLUG_DESCRIPTION",
        )
        if name in os.environ
    }
    uname = platform.uname()
    return {
        "platform": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "os_cpu_count": os.cpu_count(),
            "cpu_affinity": affinity,
            "load_average": load_average,
        },
        "cpuinfo": cpuinfo_metadata(),
        "cgroup": cgroup_metadata(),
        "heroku": allowed_environment,
    }


def round_aggregates(
    observations: Sequence[dict[str, Any]], mode: str, limit: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        if observation["mode"] == mode and observation["limit"] == limit:
            grouped[observation["round"]].append(observation)
    result = []
    for round_index in sorted(grouped):
        rows = grouped[round_index]
        nodes = sum(row["nodes"] for row in rows)
        elapsed_ns = sum(row["elapsed_ns"] for row in rows)
        result.append(
            {
                "round": round_index,
                "positions": len(rows),
                "nodes": nodes,
                "elapsed_ns": elapsed_ns,
                "nps": nodes * 1_000_000_000.0 / elapsed_ns,
            }
        )
    return result


def bootstrap_round_nps(
    rounds: Sequence[dict[str, Any]],
    replicates: int,
    confidence: float,
    seed: int,
) -> list[float]:
    if not rounds:
        raise BenchmarkError("cannot bootstrap an empty set of rounds")
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [rng.choice(rounds) for _ in rounds]
        nodes = sum(row["nodes"] for row in sampled)
        elapsed_ns = sum(row["elapsed_ns"] for row in sampled)
        draws.append(nodes * 1_000_000_000.0 / elapsed_ns)
    alpha = (1.0 - confidence) / 2.0
    return [percentile(draws, alpha), percentile(draws, 1.0 - alpha)]


def summarize_observations(
    observations: Sequence[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    pairs = sorted({(row["mode"], row["limit"]) for row in observations})
    summaries = []
    for pair_index, (mode, limit) in enumerate(pairs):
        rows = [
            row for row in observations if row["mode"] == mode and row["limit"] == limit
        ]
        rounds = round_aggregates(observations, mode, limit)
        round_nps = [row["nps"] for row in rounds]
        total_nodes = sum(row["nodes"] for row in rows)
        total_elapsed_ns = sum(row["elapsed_ns"] for row in rows)
        mean_round = statistics.fmean(round_nps)
        position_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            position_groups[row["position_id"]].append(row)
        per_position = []
        for position_id, position_rows in sorted(position_groups.items()):
            position_nodes = sum(row["nodes"] for row in position_rows)
            position_elapsed_ns = sum(row["elapsed_ns"] for row in position_rows)
            per_position.append(
                {
                    "position_id": position_id,
                    "nodes": position_nodes,
                    "elapsed_ns": position_elapsed_ns,
                    "aggregate_nps": (
                        position_nodes * 1_000_000_000.0 / position_elapsed_ns
                    ),
                }
            )
        position_nps = [row["aggregate_nps"] for row in per_position]
        summary: dict[str, Any] = {
            "mode": mode,
            "limit": limit,
            "observations": len(rows),
            "rounds": len(rounds),
            "total_nodes": total_nodes,
            "total_elapsed_ns": total_elapsed_ns,
            "aggregate_nps": total_nodes * 1_000_000_000.0 / total_elapsed_ns,
            "median_round_nps": statistics.median(round_nps),
            "round_nps_mad": median_absolute_deviation(round_nps),
            "round_nps_min": min(round_nps),
            "round_nps_max": max(round_nps),
            "round_nps_cv": (
                statistics.stdev(round_nps) / mean_round
                if len(round_nps) > 1 and mean_round != 0.0
                else 0.0
            ),
            "bootstrap_nps_ci": bootstrap_round_nps(
                rounds,
                int(protocol["bootstrap_replicates"]),
                float(protocol["bootstrap_confidence"]),
                int(protocol["bootstrap_seed"]) + pair_index,
            ),
            "macro_median_position_nps": statistics.median(position_nps),
            "macro_geomean_position_nps": math.exp(
                statistics.fmean(math.log(value) for value in position_nps)
            ),
            "per_position": per_position,
            "round_aggregates": rounds,
        }
        if mode == "movetime_ms":
            depths = [float(row["reported_depth"]) for row in rows]
            nodes = [float(row["nodes"]) for row in rows]
            elapsed_ms = [row["elapsed_ns"] / 1_000_000.0 for row in rows]
            overshoot_ms = [value - limit for value in elapsed_ms]
            summary["completed_depth"] = {
                "median": statistics.median(depths),
                "q25": percentile(depths, 0.25),
                "q75": percentile(depths, 0.75),
            }
            summary["nodes_per_move"] = {
                "median": statistics.median(nodes),
                "q25": percentile(nodes, 0.25),
                "q75": percentile(nodes, 0.75),
            }
            summary["wall_overshoot_ms"] = {
                "median": statistics.median(overshoot_ms),
                "p95": percentile(overshoot_ms, 0.95),
                "max": max(overshoot_ms),
            }
        summaries.append(summary)
    return summaries


def fixed_depth_integrity(
    observations: Sequence[dict[str, Any]], expected_positions: int
) -> dict[str, Any]:
    fixed = [row for row in observations if row["mode"] == "fixed_depth"]
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fixed:
        grouped[(row["limit"], row["position_id"])].append(row)
    mismatches = []
    for (limit, position_id), rows in sorted(grouped.items()):
        signatures = {
            (
                row["reported_depth"],
                row["score_type"],
                row["score_value"],
                row["nodes"],
                row["bestmove"],
            )
            for row in rows
        }
        if len(signatures) != 1:
            mismatches.append(
                {
                    "limit": limit,
                    "position_id": position_id,
                    "signatures": [list(value) for value in sorted(signatures)],
                }
            )
        if any(row["reported_depth"] != limit for row in rows):
            mismatches.append(
                {
                    "limit": limit,
                    "position_id": position_id,
                    "error": "reported depth did not equal requested depth",
                }
            )
    distinct_limits = {row["limit"] for row in fixed}
    expected_groups = len(distinct_limits) * expected_positions
    return {
        "status": "pass"
        if not mismatches and len(grouped) == expected_groups
        else "fail",
        "process_isolation": "process_per_case",
        "groups_checked": len(grouped),
        "expected_groups": expected_groups,
        "mismatches": mismatches,
    }


def observation_count_integrity(
    observations: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    position_count: int,
) -> dict[str, Any]:
    actual_fixed = sum(row["mode"] == "fixed_depth" for row in observations)
    actual_movetime = sum(row["mode"] == "movetime_ms" for row in observations)
    expected_fixed = (
        position_count * len(protocol["fixed_depths"]) * int(protocol["fixed_rounds"])
    )
    expected_movetime = (
        position_count
        * len(protocol["movetimes_ms"])
        * int(protocol["movetime_rounds"])
    )
    return {
        "status": (
            "pass"
            if actual_fixed == expected_fixed and actual_movetime == expected_movetime
            else "fail"
        ),
        "fixed_depth": {"actual": actual_fixed, "expected": expected_fixed},
        "movetime_ms": {"actual": actual_movetime, "expected": expected_movetime},
        "total": {
            "actual": len(observations),
            "expected": expected_fixed + expected_movetime,
        },
    }


def timed_warnings(observation: dict[str, Any]) -> list[str]:
    if observation["mode"] != "movetime_ms":
        return []
    budget = observation["limit"]
    elapsed_ms = observation["elapsed_ns"] / 1_000_000.0
    lower = 0.8 * budget
    upper = budget + max(250.0, 0.25 * budget)
    if elapsed_ms < lower or elapsed_ms > upper:
        return [
            f"movetime wall {elapsed_ms:.1f}ms outside expected [{lower:.1f}, {upper:.1f}]ms"
        ]
    return []


def progress(payload: dict[str, Any]) -> None:
    print(
        PROGRESS_SENTINEL + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


def measure_one(
    engine_command: Sequence[str],
    engine_cwd: Path | None,
    expected_name: str,
    options: dict[str, Any],
    timeout_seconds: float,
    position: dict[str, str],
    mode: str,
    limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with EngineSession(engine_command, engine_cwd, timeout_seconds) as session:
        handshake = session.initialize(expected_name, options)
        measurement = session.search(position["fen"], mode, limit)
    return measurement, handshake


def apply_overrides(spec: dict[str, Any], args: argparse.Namespace) -> None:
    protocol = spec["protocol"]
    if args.fixed_depth:
        protocol["fixed_depths"] = args.fixed_depth
    if args.fixed_rounds is not None:
        protocol["fixed_rounds"] = args.fixed_rounds
    if args.movetime_ms:
        protocol["movetimes_ms"] = args.movetime_ms
    if args.movetime_rounds is not None:
        protocol["movetime_rounds"] = args.movetime_rounds
    if args.warmup_depth is not None:
        protocol["warmup_depth"] = args.warmup_depth
    if args.warmup_positions is not None:
        protocol["warmup_positions"] = args.warmup_positions
    if args.timeout_seconds is not None:
        protocol["case_timeout_seconds"] = args.timeout_seconds
    if args.bootstrap_replicates is not None:
        protocol["bootstrap_replicates"] = args.bootstrap_replicates


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise BenchmarkError("unsupported benchmark spec schema")
    positions = spec.get("positions")
    if not isinstance(positions, list) or not positions:
        raise BenchmarkError("benchmark spec has no positions")
    ids = [position.get("id") for position in positions]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise BenchmarkError("position ids must be non-empty and unique")
    if any(not position.get("fen") for position in positions):
        raise BenchmarkError("every benchmark position needs a FEN")
    protocol = spec.get("protocol", {})
    if protocol.get("isolation") != "process_per_case":
        raise BenchmarkError("this harness requires process_per_case isolation")
    if protocol.get("mode_schedule") != "interleaved_balanced":
        raise BenchmarkError("this harness requires interleaved_balanced mode order")
    for name in ("fixed_rounds", "movetime_rounds", "case_timeout_seconds"):
        if int(protocol.get(name, 0)) <= 0:
            raise BenchmarkError(f"protocol {name} must be positive")
    if int(protocol.get("bootstrap_replicates", 0)) <= 0:
        raise BenchmarkError("bootstrap_replicates must be positive")


def load_spec(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    if bool(args.spec) == bool(args.spec_base64):
        raise BenchmarkError("provide exactly one of --spec or --spec-base64")
    if args.spec:
        data = Path(args.spec).read_bytes()
    else:
        try:
            data = base64.b64decode(args.spec_base64, validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise BenchmarkError("invalid --spec-base64 payload") from error
    try:
        spec = json.loads(data)
    except json.JSONDecodeError as error:
        raise BenchmarkError("invalid benchmark spec JSON") from error
    apply_overrides(spec, args)
    validate_spec(spec)
    return spec, data


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec, original_spec_bytes = load_spec(args)
    engine = Path(args.engine).resolve()
    model = Path(args.model).resolve() if args.model else None
    config = Path(args.config).resolve() if args.config else None
    engine_cwd = Path(args.engine_cwd).resolve() if args.engine_cwd else None
    for label, path in (("engine", engine), ("model", model), ("config", config)):
        if path is not None and not path.is_file():
            raise BenchmarkError(f"{label} file does not exist: {path}")
    if engine_cwd is not None and not engine_cwd.is_dir():
        raise BenchmarkError(f"engine cwd does not exist: {engine_cwd}")

    command = [str(engine)]
    if model is not None:
        command.append(str(model))
    protocol = spec["protocol"]
    positions = spec["positions"]
    options = spec["uci_options"]
    config_validation = validate_config_options(
        config,
        options,
        bool(protocol.get("require_config_match", False)),
    )
    expected_name = spec["expected_engine_name"]
    timeout_seconds = float(protocol["case_timeout_seconds"])
    started_at = utc_now()
    host_before = machine_metadata()
    observations: list[dict[str, Any]] = []
    first_handshake: dict[str, Any] | None = None

    warmup_count = min(int(protocol["warmup_positions"]), len(positions))
    for index, position in enumerate(positions[:warmup_count]):
        _, handshake = measure_one(
            command,
            engine_cwd,
            expected_name,
            options,
            timeout_seconds,
            position,
            "fixed_depth",
            int(protocol["warmup_depth"]),
        )
        first_handshake = first_handshake or handshake
        progress(
            {
                "phase": "warmup",
                "completed": index + 1,
                "total": warmup_count,
                "position_id": position["id"],
            }
        )

    for mode, limit, round_index in measurement_schedule(protocol):
        rounds = int(
            protocol["fixed_rounds"]
            if mode == "fixed_depth"
            else protocol["movetime_rounds"]
        )
        ordered = scheduled_positions(positions, round_index)
        for sequence, position in enumerate(ordered):
            measured, handshake = measure_one(
                command,
                engine_cwd,
                expected_name,
                options,
                timeout_seconds,
                position,
                mode,
                limit,
            )
            first_handshake = first_handshake or handshake
            observation = {
                "type": "observation",
                "observed_at": utc_now(),
                "round": round_index,
                "sequence": sequence,
                "position_id": position["id"],
                "fen": position["fen"],
                "mode": mode,
                "limit": limit,
                **measured,
            }
            observation["warnings"] = timed_warnings(observation)
            observations.append(observation)
        round_rows = [
            row
            for row in observations
            if row["mode"] == mode
            and row["limit"] == limit
            and row["round"] == round_index
        ]
        nodes = sum(row["nodes"] for row in round_rows)
        elapsed_ns = sum(row["elapsed_ns"] for row in round_rows)
        progress(
            {
                "phase": "measure",
                "mode": mode,
                "limit": limit,
                "round": round_index + 1,
                "rounds": rounds,
                "nps": nodes * 1_000_000_000.0 / elapsed_ns,
            }
        )

    integrity = fixed_depth_integrity(observations, len(positions))
    integrity["observation_counts"] = observation_count_integrity(
        observations, protocol, len(positions)
    )
    if integrity["observation_counts"]["status"] != "pass":
        integrity["status"] = "fail"
    summaries = summarize_observations(observations, protocol)
    host_after = machine_metadata()
    run_warnings = []
    if first_handshake and first_handshake.get("nnue_kernel") is None:
        run_warnings.append(
            "engine did not report nnue_kernel; backend selection is unresolved"
        )
    result = {
        "schema_version": 1,
        "benchmark_id": spec["suite_id"],
        "status": "valid" if integrity["status"] == "pass" else "invalid",
        "started_at": started_at,
        "finished_at": utc_now(),
        "host_label": args.host_label,
        "source_label": args.source_label,
        "provenance": {
            "harness_sha256": args.harness_sha256,
            "runner_sha256": args.runner_sha256,
            "engine_path": str(engine),
            "engine_sha256": sha256_file(engine),
            "model_path": str(model) if model else None,
            "model_sha256": sha256_file(model) if model else None,
            "config_path": str(config) if config else None,
            "config_sha256": sha256_file(config) if config else None,
            "config_validation": config_validation,
            "spec_sha256": sha256_bytes(original_spec_bytes),
            "effective_spec_sha256": sha256_bytes(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
        },
        "protocol": protocol,
        "uci": {
            "expected_engine_name": expected_name,
            "options": options,
            "handshake": first_handshake,
        },
        "machine_before": host_before,
        "machine_after": host_after,
        "integrity": integrity,
        "warnings": run_warnings,
        "summaries": summaries,
        "observations": observations,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model")
    parser.add_argument("--config")
    parser.add_argument("--engine-cwd")
    spec_group = parser.add_mutually_exclusive_group(required=True)
    spec_group.add_argument("--spec")
    spec_group.add_argument("--spec-base64")
    parser.add_argument("--host-label", default=platform.node())
    parser.add_argument("--source-label", default="unknown")
    parser.add_argument("--harness-sha256")
    parser.add_argument("--runner-sha256")
    parser.add_argument("--output", default="-")
    parser.add_argument("--fixed-depth", type=int, action="append")
    parser.add_argument("--fixed-rounds", type=int)
    parser.add_argument("--movetime-ms", type=int, action="append")
    parser.add_argument("--movetime-rounds", type=int)
    parser.add_argument("--warmup-depth", type=int)
    parser.add_argument("--warmup-positions", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--bootstrap-replicates", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_benchmark(args)
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
        if args.output == "-":
            print(RESULT_SENTINEL + encoded, flush=True)
        else:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"output": str(output), "status": result["status"]}))
        return 0 if result["status"] == "valid" else 3
    except Exception as error:  # Preserve a machine-readable remote failure.
        payload = {
            "error_type": type(error).__name__,
            "error": str(error),
            "at": utc_now(),
        }
        print(
            ERROR_SENTINEL + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
