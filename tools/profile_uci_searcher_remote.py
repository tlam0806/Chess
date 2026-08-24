#!/usr/bin/env python3
"""Collect a reproducible gperftools CPU profile of a UCI engine.

This file is intended to be streamed to a Heroku one-off dyno.  It installs
the profiler into an unprivileged temporary directory, brackets only the UCI
search with SIGUSR2, and returns a compressed artifact as chunked base64.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any, Sequence


PROGRESS_SENTINEL = "CHESS_PROFILE_PROGRESS="
SUMMARY_SENTINEL = "CHESS_PROFILE_SUMMARY="
CHUNK_SENTINEL = "CHESS_PROFILE_CHUNK="
END_SENTINEL = "CHESS_PROFILE_ARTIFACT_END="
ERROR_SENTINEL = "CHESS_PROFILE_ERROR="

OPTION_RE = re.compile(r"^option name (.+?) type ")
UCI_MOVE_RE = re.compile(r"(?:[a-h][1-8]){2}[nbrq]?")


class ProfileError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        raise ProfileError("empty YAML scalar in uci_options")
    if value[0] in {'"', "'"}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ProfileError(f"invalid quoted YAML scalar: {value}") from error
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
    """Read the deployment YAML's simple scalar uci_options mapping."""
    result: dict[str, Any] = {}
    section_indent = None
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
            raise ProfileError(f"invalid uci_options YAML line: {raw_line}")
        name, value = stripped.split(":", 1)
        result[name.strip()] = parse_yaml_scalar(value)
    if section_indent is None or not result:
        raise ProfileError(f"config has no uci_options mapping: {path}")
    return result


def validate_config_options(
    config_path: Path,
    spec_options: dict[str, Any],
    required: bool,
) -> dict[str, Any]:
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
        raise ProfileError(
            "profile suite/config UCI options differ: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return {
        "status": "match" if required else "checked",
        "options": normalized_config,
    }


def run_checked(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 300,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = result.stdout[-4000:]
        raise ProfileError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{tail}"
        )
    return result


def install_profiler(temp_dir: Path) -> dict[str, Any]:
    state = temp_dir / "apt-state"
    cache = temp_dir / "apt-cache"
    root = temp_dir / "profiler-root"
    (state / "lists" / "partial").mkdir(parents=True)
    (cache / "archives" / "partial").mkdir(parents=True)
    root.mkdir()

    apt = [
        "apt-get",
        "-o",
        f"Dir::State={state}",
        "-o",
        f"Dir::Cache={cache}",
        "-o",
        "Dir::State::status=/var/lib/dpkg/status",
        "-o",
        "APT::Get::List-Cleanup=0",
        "-o",
        "Acquire::Languages=none",
    ]
    print(
        PROGRESS_SENTINEL
        + json.dumps({"phase": "profiler_setup", "status": "apt_update"}),
        flush=True,
    )
    run_checked([*apt, "update"], timeout=180)
    print(
        PROGRESS_SENTINEL
        + json.dumps({"phase": "profiler_setup", "status": "download"}),
        flush=True,
    )
    run_checked(
        [
            *apt,
            "--download-only",
            "--no-install-recommends",
            "-y",
            "install",
            "libgoogle-perftools4t64=2.16-1",
        ],
        timeout=240,
    )
    packages = sorted((cache / "archives").glob("*.deb"))
    if not packages:
        raise ProfileError("apt downloaded no profiler packages")
    for package in packages:
        run_checked(["dpkg-deb", "-x", str(package), str(root)])

    libprofiler = root / "usr/lib/x86_64-linux-gnu/libprofiler.so.0"
    if not libprofiler.exists():
        raise ProfileError("profiler package extraction is incomplete")

    package_metadata = []
    for package in packages:
        fields = run_checked(
            ["dpkg-deb", "-f", str(package), "Package", "Version"]
        ).stdout.splitlines()
        package_metadata.append(
            {
                "file": package.name,
                "sha256": sha256_file(package),
                "package": fields[0] if fields else None,
                "version": fields[1] if len(fields) > 1 else None,
            }
        )
    return {
        "root": root,
        "libprofiler": libprofiler,
        "packages": package_metadata,
    }


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
            raise ProfileError(f"malformed UCI info line: {line}") from error
        index += 1
    required = {"reported_depth", "nodes", "score_type", "score_value"}
    return parsed if required.issubset(parsed) else None


def process_cpu_seconds(pid: int) -> float:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise ProfileError(f"cannot parse /proc/{pid}/stat")
    fields = raw[close + 2 :].split()
    ticks = int(fields[11]) + int(fields[12])
    return ticks / float(os.sysconf("SC_CLK_TCK"))


class EngineSession:
    def __init__(
        self,
        command: Sequence[str],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise ProfileError("failed to open UCI subprocess pipes")
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
            raise ProfileError(
                f"engine exited with {self.process.returncode} before {command!r}"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_line(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProfileError("timed out waiting for UCI output")
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty as error:
            raise ProfileError("timed out waiting for UCI output") from error
        if line is None:
            raise ProfileError(f"unexpected UCI EOF ({self.process.poll()})")
        return line

    def initialize(self, expected_name: str, options: dict[str, Any]) -> dict[str, Any]:
        self.send("uci")
        deadline = time.monotonic() + self.timeout_seconds
        engine_name = None
        advertised: set[str] = set()
        info_strings: list[str] = []
        while True:
            line = self.read_line(deadline)
            if line.startswith("id name "):
                engine_name = line.removeprefix("id name ")
            option_match = OPTION_RE.match(line)
            if option_match:
                advertised.add(option_match.group(1))
            if line.startswith("info string "):
                info_strings.append(line.removeprefix("info string "))
            if line == "uciok":
                break
        if engine_name != expected_name:
            raise ProfileError(f"unexpected engine name: {engine_name!r}")
        missing = sorted(set(options) - advertised)
        if missing:
            raise ProfileError(f"engine is missing UCI options: {missing}")
        for name, value in options.items():
            self.send(f"setoption name {name} value {option_value(value)}")
        self.send("isready")
        deadline = time.monotonic() + self.timeout_seconds
        rejected = []
        while True:
            line = self.read_line(deadline)
            if "ignored invalid option" in line:
                rejected.append(line)
            if line == "readyok":
                break
        if rejected:
            raise ProfileError("engine rejected options: " + "; ".join(rejected))
        kernel = None
        for info in info_strings:
            if info.startswith("nnue_kernel="):
                kernel = info.split("=", 1)[1]
        return {"engine_name": engine_name, "nnue_kernel": kernel}

    def search(self, fen: str, depth: int, profile: bool) -> dict[str, Any]:
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        if profile:
            os.kill(self.process.pid, signal.SIGUSR2)
            time.sleep(0.02)
        cpu_started = process_cpu_seconds(self.process.pid)
        wall_started_ns = time.perf_counter_ns()
        self.send(f"go depth {depth}")
        deadline = time.monotonic() + self.timeout_seconds
        latest_info = None
        bestmove = None
        while bestmove is None:
            line = self.read_line(deadline)
            parsed = parse_info_line(line)
            if parsed is not None:
                latest_info = parsed
            if line.startswith("bestmove "):
                parts = line.split()
                if len(parts) < 2:
                    raise ProfileError(f"malformed bestmove line: {line}")
                bestmove = parts[1]
        wall_elapsed_ns = time.perf_counter_ns() - wall_started_ns
        cpu_elapsed_seconds = process_cpu_seconds(self.process.pid) - cpu_started
        if profile:
            os.kill(self.process.pid, signal.SIGUSR2)
            time.sleep(0.02)
        if latest_info is None:
            raise ProfileError("bestmove arrived without a complete info line")
        if latest_info["nodes"] <= 0:
            raise ProfileError("engine reported a non-positive node count")
        if not UCI_MOVE_RE.fullmatch(bestmove):
            raise ProfileError(f"invalid bestmove: {bestmove!r}")
        return {
            **latest_info,
            "bestmove": bestmove,
            "elapsed_ns": wall_elapsed_ns,
            "cpu_seconds": cpu_elapsed_seconds,
            "nps": latest_info["nodes"] * 1_000_000_000.0 / wall_elapsed_ns,
            "cpu_to_wall": cpu_elapsed_seconds / (wall_elapsed_ns / 1e9),
        }

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=5)
            except (ProfileError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._reader.join(timeout=1)

    def __enter__(self) -> "EngineSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def position_order(positions: list[dict[str, str]], round_index: int) -> list[dict[str, str]]:
    offset = (round_index // 2) % len(positions)
    rotated = positions[offset:] + positions[:offset]
    return list(reversed(rotated)) if round_index % 2 else rotated


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["reported_depth"],
        row["score_type"],
        row["score_value"],
        row["nodes"],
        row["bestmove"],
    )


def aggregate_round(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = sum(row["nodes"] for row in rows)
    elapsed_ns = sum(row["elapsed_ns"] for row in rows)
    cpu_seconds = sum(row["cpu_seconds"] for row in rows)
    return {
        "nodes": nodes,
        "elapsed_ns": elapsed_ns,
        "cpu_seconds": cpu_seconds,
        "nps": nodes * 1_000_000_000.0 / elapsed_ns,
        "cpu_to_wall": cpu_seconds / (elapsed_ns / 1e9),
    }


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def machine_metadata() -> dict[str, Any]:
    cpuinfo = read_optional(Path("/proc/cpuinfo")) or ""
    model_name = None
    flags = None
    for line in cpuinfo.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key == "model name" and model_name is None:
            model_name = value
        if key == "flags" and flags is None:
            flags = value.split()
    try:
        affinity = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = None
    return {
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "python": platform.python_version(),
        "cpu_model": model_name,
        "cpu_flags": flags,
        "affinity": affinity,
        "dyno": os.environ.get("DYNO"),
        "perf_event_paranoid": read_optional(Path("/proc/sys/kernel/perf_event_paranoid")),
        "cgroup": read_optional(Path("/proc/self/cgroup")),
        "cpu_max": read_optional(Path("/sys/fs/cgroup/cpu.max")),
        "memory_max": read_optional(Path("/sys/fs/cgroup/memory.max")),
    }


def make_archive(artifact_dir: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(artifact_dir.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(artifact_dir))
    return stream.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--engine-cwd", required=True, type=Path)
    parser.add_argument("--suite-base64", required=True)
    parser.add_argument("--remote-script-sha256", required=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--positions-limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_utc = utc_now()
    suite_bytes = base64.b64decode(args.suite_base64, validate=True)
    suite = json.loads(suite_bytes)
    positions = list(suite["positions"])
    if args.positions_limit is not None:
        positions = positions[: args.positions_limit]
    if not positions:
        raise ProfileError("suite contains no positions")
    if args.blocks <= 0 or args.frequency <= 0 or args.depth <= 0:
        raise ProfileError("depth, blocks, and frequency must be positive")
    for path in (args.engine, args.model, args.config):
        if not path.is_file():
            raise ProfileError(f"required runtime file is missing: {path}")
    config_validation = validate_config_options(
        args.config,
        suite["uci_options"],
        bool(suite.get("protocol", {}).get("require_config_match", False)),
    )

    with tempfile.TemporaryDirectory(prefix="chess-v41-profile-") as temp_name:
        temp_dir = Path(temp_name)
        artifact_dir = temp_dir / "artifact"
        raw_dir = artifact_dir / "raw"
        raw_dir.mkdir(parents=True)
        profiler = install_profiler(temp_dir)
        libprofiler = Path(profiler["libprofiler"])
        profiler_root = Path(profiler["root"])

        engine_copy = artifact_dir / "uci_nnue_v41.elf"
        shutil.copy2(args.engine, engine_copy)
        engine_copy.chmod(0o755)

        base_env = os.environ.copy()
        base_env["LD_PRELOAD"] = str(libprofiler)
        base_env["LD_LIBRARY_PATH"] = ":".join(
            [
                str(profiler_root / "usr/lib/x86_64-linux-gnu"),
                str(profiler_root / "lib/x86_64-linux-gnu"),
                base_env.get("LD_LIBRARY_PATH", ""),
            ]
        )
        base_env["CPUPROFILESIGNAL"] = str(signal.SIGUSR2.value)
        base_env["CPUPROFILE_FREQUENCY"] = str(args.frequency)

        observations: list[dict[str, Any]] = []
        round_summaries: list[dict[str, Any]] = []
        profiles: list[Path] = []
        handshake: dict[str, Any] | None = None
        # ABBA: control, profile, profile, control.  A and B both preload the
        # same library; only B toggles SIGPROF during the search.
        modes = [False, True, True, False] * args.blocks
        total_cases = len(modes) * len(positions)
        completed_cases = 0
        for round_index, profiled in enumerate(modes):
            round_rows = []
            order = position_order(positions, round_index)
            for sequence, position in enumerate(order):
                profile_base = raw_dir / (
                    f"r{round_index:02d}-{sequence:02d}-{position['id']}.prof"
                )
                env = base_env.copy()
                env["CPUPROFILE"] = str(profile_base)
                before_profiles = set(raw_dir.glob(profile_base.name + "*"))
                with EngineSession(
                    [str(args.engine), str(args.model)],
                    args.engine_cwd,
                    env,
                    args.timeout,
                ) as session:
                    current_handshake = session.initialize(
                        suite["expected_engine_name"], suite["uci_options"]
                    )
                    if handshake is None:
                        handshake = current_handshake
                    elif current_handshake != handshake:
                        raise ProfileError("UCI handshake changed between cases")
                    result = session.search(position["fen"], args.depth, profiled)
                after_profiles = set(raw_dir.glob(profile_base.name + "*"))
                new_profiles = sorted(after_profiles - before_profiles)
                if profiled:
                    nonempty = [path for path in new_profiles if path.stat().st_size > 0]
                    if len(nonempty) != 1:
                        raise ProfileError(
                            f"expected one profile for {profile_base.name}, got {new_profiles}"
                        )
                    profiles.extend(nonempty)
                elif new_profiles:
                    raise ProfileError("control case unexpectedly produced a profile")
                row = {
                    "round": round_index,
                    "sequence": sequence,
                    "mode": "profile" if profiled else "control",
                    "position_id": position["id"],
                    "fen": position["fen"],
                    **result,
                }
                observations.append(row)
                round_rows.append(row)
                completed_cases += 1
                print(
                    PROGRESS_SENTINEL
                    + json.dumps(
                        {
                            "phase": "search",
                            "completed": completed_cases,
                            "total": total_cases,
                            "round": round_index + 1,
                            "rounds": len(modes),
                            "mode": row["mode"],
                            "position": position["id"],
                        }
                    ),
                    flush=True,
                )
            round_summary = {
                "round": round_index,
                "mode": "profile" if profiled else "control",
                **aggregate_round(round_rows),
            }
            round_summaries.append(round_summary)
            print(
                PROGRESS_SENTINEL
                + json.dumps({"phase": "round", **round_summary}),
                flush=True,
            )

        signatures: dict[str, set[tuple[Any, ...]]] = {
            position["id"]: set() for position in positions
        }
        for row in observations:
            signatures[row["position_id"]].add(signature(row))
        mismatches = {
            position_id: [list(item) for item in sorted(values, key=str)]
            for position_id, values in signatures.items()
            if len(values) != 1
        }
        if mismatches:
            raise ProfileError(f"fixed-depth signature mismatch: {mismatches}")
        if any(row["reported_depth"] != args.depth for row in observations):
            raise ProfileError("engine did not complete the requested depth")

        (artifact_dir / "observations.json").write_text(
            json.dumps(observations, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        profile_rounds = [row for row in round_summaries if row["mode"] == "profile"]
        control_rounds = [row for row in round_summaries if row["mode"] == "control"]
        profile_median_nps = median([row["nps"] for row in profile_rounds])
        control_median_nps = median([row["nps"] for row in control_rounds])
        slowdown = None
        if profile_median_nps and control_median_nps:
            slowdown = 1.0 - profile_median_nps / control_median_nps
        manifest = {
            "schema_version": 1,
            "status": "valid",
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "method": {
                "profiler": "gperftools-libprofiler",
                "timer": "ITIMER_PROF",
                "frequency_hz": args.frequency,
                "signal": signal.SIGUSR2.value,
                "scope": "SIGUSR2 immediately before go through bestmove",
                "schedule": "ABBA",
                "blocks": args.blocks,
                "depth": args.depth,
                "process_isolation": "fresh process per position per round",
            },
            "identity": {
                "engine_path": str(args.engine),
                "engine_sha256": sha256_file(args.engine),
                "model_path": str(args.model),
                "model_sha256": sha256_file(args.model),
                "config_path": str(args.config),
                "config_sha256": sha256_file(args.config),
                "suite_sha256": sha256_bytes(suite_bytes),
                "remote_script_sha256": args.remote_script_sha256,
                "libprofiler_sha256": sha256_file(libprofiler),
            },
            "handshake": handshake,
            "config_validation": config_validation,
            "machine": machine_metadata(),
            "profiler_packages": profiler["packages"],
            "integrity": {
                "positions": len(positions),
                "observations": len(observations),
                "raw_profiles": len(profiles),
                "signature_mismatches": mismatches,
            },
            "rounds": round_summaries,
            "summary": {
                "profile_median_nps": profile_median_nps,
                "control_median_nps": control_median_nps,
                "sampling_slowdown_fraction": slowdown,
                "profile_median_cpu_to_wall": median(
                    [row["cpu_to_wall"] for row in profile_rounds]
                ),
                "control_median_cpu_to_wall": median(
                    [row["cpu_to_wall"] for row in control_rounds]
                ),
            },
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        archive = make_archive(artifact_dir)
        archive_sha = sha256_bytes(archive)
        encoded = base64.b64encode(archive).decode("ascii")
        chunk_size = 4096
        chunks = [encoded[index : index + chunk_size] for index in range(0, len(encoded), chunk_size)]
        print(SUMMARY_SENTINEL + json.dumps(manifest, separators=(",", ":")), flush=True)
        for index, chunk in enumerate(chunks):
            print(f"{CHUNK_SENTINEL}{index}:{chunk}", flush=True)
        print(
            END_SENTINEL
            + json.dumps(
                {
                    "chunks": len(chunks),
                    "archive_bytes": len(archive),
                    "archive_sha256": archive_sha,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # The local runner captures this structured error.
        print(
            ERROR_SENTINEL
            + json.dumps(
                {"type": type(error).__name__, "error": str(error)},
                separators=(",", ":"),
            ),
            flush=True,
        )
        raise
