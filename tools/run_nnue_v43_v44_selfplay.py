#!/usr/bin/env python3
"""Immutable, detached and resumable V43-production versus V44-clear-TT match."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_MODEL_SHA256 = (
    "a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02"
)
EXPECTED_BOOK_SHA256 = (
    "36befd4ebaee60ab9be6697dc96e2b1ac53c5797dae35a0ea79f1ee5117c51d4"
)
COMBINED_CONFIG_HASH = (
    "98b7732c9587da35554cc274a072a0a5b5f55902aaa77605e78c1ae13e88b4f2"
)
CANDIDATE_NAME = "v44_clear_each_search"
CONTROL_NAME = "v43_production"
POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}
DEFAULT_PROTOCOL = {
    "openings": 300,
    "expected_games": 600,
    "base_ms": 10_000,
    "increment_ms": 100,
    "overhead_ms": 20,
    "max_plies": 240,
    "tt_mb": 64,
    "seed": 2_026_090_155,
}
SELECTIVE_CONFIG = {
    "enable_lmr": True,
    "lmr_base": 0.45,
    "lmr_divisor": 2.9,
    "lmr_min_depth": 3,
    "lmr_min_move_index": 6,
    "enable_null_move": True,
    "null_move_min_depth": 2,
    "null_move_reduction": 4,
    "enable_reverse_futility": True,
    "reverse_futility_max_depth": 4,
    "reverse_futility_base_margin": 50,
    "reverse_futility_margin_per_depth": 100,
    "enable_late_move_pruning": True,
    "late_move_pruning_max_depth": 3,
    "late_move_pruning_base": 4,
    "late_move_pruning_depth_multiplier": 1,
    "enable_qsearch_see_pruning": True,
    "qsearch_see_threshold": -25,
    "enable_main_search_see_pruning": False,
    "main_search_see_max_depth": 5,
    "main_search_see_margin_per_depth": 100,
}
ASPIRATION_CONFIG = {
    "enabled": True,
    "min_depth": 2,
    "delta_base_cp": 68,
    "delta_divisor": 33_700,
    "expansion_factor_per_mille": 2_290,
    "max_fail_high_reductions": 1,
    "mean_score_new_weight_per_mille": 370,
    "max_researches": 6,
    "mean_score_clamp_cp": 1_500,
}


def fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True).encode() + b"\n",
    )


def copy_snapshot(source: Path, target: Path, executable: bool = False) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing source artifact: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    temporary.chmod(0o755 if executable else 0o644)
    os.replace(temporary, target)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def protocol_from_args(args: argparse.Namespace) -> dict[str, int]:
    protocol = {
        "openings": args.openings,
        "expected_games": args.openings * 2,
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "overhead_ms": args.overhead_ms,
        "max_plies": args.max_plies,
        "tt_mb": args.tt_mb,
        "seed": args.seed,
    }
    if (
        protocol["openings"] <= 0
        or protocol["base_ms"] <= 0
        or protocol["increment_ms"] < 0
        or protocol["overhead_ms"] < 0
        or protocol["max_plies"] <= 0
        or protocol["tt_mb"] <= 0
        or protocol["seed"] < 0
    ):
        raise RuntimeError("invalid match protocol")
    return protocol


def build_command(contract: dict[str, Any]) -> list[str]:
    frozen = contract["frozen_files"]
    protocol = contract["protocol"]
    return [
        frozen["binary"]["path"],
        "--book", frozen["book"]["path"],
        "--model", frozen["model"]["path"],
        "--output", contract["output"],
        "--openings", str(protocol["openings"]),
        "--base-ms", str(protocol["base_ms"]),
        "--increment-ms", str(protocol["increment_ms"]),
        "--overhead-ms", str(protocol["overhead_ms"]),
        "--max-plies", str(protocol["max_plies"]),
        "--tt-mb", str(protocol["tt_mb"]),
        "--seed", str(protocol["seed"]),
    ]


def initialize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    contract_path = run_dir / "contract.json"
    if contract_path.exists():
        return load_contract(run_dir)

    binary = args.binary.resolve()
    model = args.model.resolve()
    book = args.book.resolve()
    if fingerprint(model)["sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA256 is not the frozen production model")
    if fingerprint(book)["sha256"] != EXPECTED_BOOK_SHA256:
        raise RuntimeError("book SHA256 is not the frozen 2400-opening book")
    protocol = protocol_from_args(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = run_dir / "artifacts"
    frozen_binary = artifacts / "nnue_v43_v44_time_match"
    frozen_model = artifacts / "phase_quantized_nnue.bin"
    frozen_book = artifacts / "stockfish_balanced_openings_10ply_2400.txt"
    frozen_runner = run_dir / "bin" / Path(__file__).name
    copy_snapshot(binary, frozen_binary, executable=True)
    copy_snapshot(model, frozen_model)
    copy_snapshot(book, frozen_book)
    copy_snapshot(Path(__file__).resolve(), frozen_runner, executable=True)

    frozen_files: dict[str, dict[str, Any]] = {}
    for name, path in {
        "binary": frozen_binary,
        "model": frozen_model,
        "book": frozen_book,
        "runner": frozen_runner,
    }.items():
        frozen_files[name] = {"path": str(path), **fingerprint(path)}

    contract: dict[str, Any] = {
        "kind": "nnue_v43_v44_selfplay_contract",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "run_dir": str(run_dir),
        "output": str(run_dir / "games.jsonl"),
        "lifecycle": "experiment_only_no_automatic_promotion",
        "source_files": {
            "binary": {"path": str(binary), **fingerprint(binary)},
            "model": {"path": str(model), **fingerprint(model)},
            "book": {"path": str(book), **fingerprint(book)},
        },
        "frozen_files": frozen_files,
        "engines": {
            "candidate": {
                "profile": CANDIDATE_NAME,
                "engine": "nnue_clear_each_search_v44",
                "clear_tt_each_top_level_search": True,
                "generation_fields": False,
            },
            "control": {
                "profile": CONTROL_NAME,
                "engine": "nnue_single_bound_v43",
                "clear_tt_each_top_level_search": False,
                "generation_policy": "current_only_scores_stale_move_hint",
            },
        },
        "combined_config_hash": COMBINED_CONFIG_HASH,
        "selective_config": SELECTIVE_CONFIG,
        "aspiration_config": ASPIRATION_CONFIG,
        "common_policy": {
            "same_model": True,
            "same_selective_and_aspiration_config": True,
            "paired_color_reversed": True,
            "alternating_first_color": True,
            "serial_execution": True,
            "twofold_search_draw": True,
            "reuse_deeper_tt_scores": False,
            "clear_search_heuristics_only_between_games": True,
            "per_ply_search_telemetry": True,
            "no_early_stop": True,
        },
        "execution": {
            "detached_worker": True,
            "sleep_preventer": shutil.which("caffeinate"),
        },
        "protocol": protocol,
    }
    contract["command"] = build_command(contract)
    atomic_json(contract_path, contract)
    atomic_write(
        run_dir / "contract.sha256",
        (fingerprint(contract_path)["sha256"] + "\n").encode(),
    )
    atomic_json(run_dir / "status.json", {
        "state": "initialized",
        "completed_games": 0,
        "expected_games": protocol["expected_games"],
        "updated_at": utc_timestamp(),
    })
    return contract


def load_contract(run_dir: Path) -> dict[str, Any]:
    contract_path = run_dir / "contract.json"
    hash_path = run_dir / "contract.sha256"
    if not contract_path.is_file() or not hash_path.is_file():
        raise RuntimeError("match has not been initialized")
    expected = hash_path.read_text().strip()
    if fingerprint(contract_path)["sha256"] != expected:
        raise RuntimeError("immutable contract SHA256 mismatch")
    contract = json.loads(contract_path.read_text())
    if (
        contract.get("kind") != "nnue_v43_v44_selfplay_contract"
        or contract.get("schema_version") != SCHEMA_VERSION
    ):
        raise RuntimeError("unsupported match contract")
    return contract


def verify_contract(contract: dict[str, Any]) -> None:
    if contract.get("combined_config_hash") != COMBINED_CONFIG_HASH:
        raise RuntimeError("contract does not use the production winner config")
    if contract.get("selective_config") != SELECTIVE_CONFIG:
        raise RuntimeError("contract selective config mismatch")
    if contract.get("aspiration_config") != ASPIRATION_CONFIG:
        raise RuntimeError("contract aspiration config mismatch")
    if contract["protocol"]["expected_games"] != (
        contract["protocol"]["openings"] * 2
    ):
        raise RuntimeError("contract expected-game count mismatch")
    for name, expected in contract["frozen_files"].items():
        path = Path(expected["path"])
        if not path.is_file() or fingerprint(path) != {
            "size": expected["size"],
            "sha256": expected["sha256"],
        }:
            raise RuntimeError(f"frozen artifact changed: {name}: {path}")
    if build_command(contract) != contract["command"]:
        raise RuntimeError("contract command mismatch")


def repair_partial_jsonl_tail(path: Path) -> bool:
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    boundary = data.rfind(b"\n")
    repaired = b"" if boundary < 0 else data[:boundary + 1]
    atomic_write(path, repaired)
    return True


def read_games(path: Path, tolerate_partial_tail: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        if not tolerate_partial_tail:
            raise RuntimeError(f"non-newline JSONL tail: {path}")
        boundary = data.rfind(b"\n")
        data = b"" if boundary < 0 else data[:boundary + 1]
    games: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            game = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"malformed game JSON at {path}:{line_number}"
            ) from error
        if game.get("kind") != "game" or game.get("schema_version") != 1:
            raise RuntimeError(f"unexpected record at {path}:{line_number}")
        key = game.get("key")
        if not isinstance(key, str) or key in seen:
            raise RuntimeError(f"invalid or duplicate game key at line {line_number}")
        seen.add(key)
        if (
            game.get("candidate") != CANDIDATE_NAME
            or game.get("control") != CONTROL_NAME
            or game.get("combined_config_hash") != COMBINED_CONFIG_HASH
            or game.get("candidate_outcome") not in POINTS
        ):
            raise RuntimeError(f"game policy mismatch at line {line_number}")
        run_fingerprint = game.get("run_fingerprint")
        if (
            not isinstance(run_fingerprint, str)
            or not key.startswith(run_fingerprint + ":")
        ):
            raise RuntimeError(f"game fingerprint mismatch at line {line_number}")
        searches = game.get("searches")
        played_moves = game.get("played_moves")
        if not isinstance(searches, list) or not isinstance(played_moves, list):
            raise RuntimeError(f"missing per-ply telemetry at line {line_number}")
        if sum(search.get("applied") is True for search in searches) != len(played_moves):
            raise RuntimeError(f"move/telemetry mismatch at line {line_number}")
        games.append(game)
    return games


def paired_ci95(pair_scores: list[float]) -> list[float]:
    if not pair_scores:
        return [0.0, 1.0]
    mean = sum(pair_scores) / len(pair_scores)
    if len(pair_scores) < 2:
        return [0.0, 1.0]
    variance = sum((value - mean) ** 2 for value in pair_scores) / (
        len(pair_scores) - 1
    )
    half_width = 1.96 * math.sqrt(variance / len(pair_scores))
    return [max(0.0, mean - half_width), min(1.0, mean + half_width)]


def summarize_games(games: list[dict[str, Any]], expected_games: int) -> dict[str, Any]:
    outcomes = Counter(game["candidate_outcome"] for game in games)
    reasons = Counter(game.get("reason", "unknown") for game in games)
    pairs: dict[str, list[float]] = defaultdict(list)
    candidate_nodes = control_nodes = 0
    candidate_time = control_time = 0
    search_count = 0
    pair_colors: dict[str, set[str]] = defaultdict(set)
    for game in games:
        logical_key = game.get("logical_key")
        if not isinstance(logical_key, str) or ":" not in logical_key:
            raise RuntimeError("malformed logical game key")
        pair_key = logical_key.rsplit(":", 1)[0]
        pairs[pair_key].append(
            POINTS[game["candidate_outcome"]]
        )
        color = game.get("candidate_color")
        if color not in {"white", "black"}:
            raise RuntimeError("invalid candidate color")
        pair_colors[pair_key].add(color)
        candidate_nodes += int(game.get("candidate_nodes", 0))
        control_nodes += int(game.get("control_nodes", 0))
        candidate_time += int(game.get("candidate_time_ms", 0))
        control_time += int(game.get("control_time_ms", 0))
        search_count += len(game["searches"])
    complete_pair_scores = [
        sum(values) / 2.0 for values in pairs.values() if len(values) == 2
    ]
    if any(len(values) > 2 for values in pairs.values()):
        raise RuntimeError("opening pair contains more than two games")
    if any(
        len(values) == 2 and pair_colors[key] != {"white", "black"}
        for key, values in pairs.items()
    ):
        raise RuntimeError("complete opening pair is not color-reversed")
    if len(games) > expected_games:
        raise RuntimeError("saved game count exceeds immutable protocol")
    points = sum(POINTS[outcome] * count for outcome, count in outcomes.items())
    return {
        "kind": "nnue_v43_v44_selfplay_summary",
        "candidate": CANDIDATE_NAME,
        "control": CONTROL_NAME,
        "completed_games": len(games),
        "expected_games": expected_games,
        "complete_pairs": len(complete_pair_scores),
        "partial_pairs": sum(len(values) == 1 for values in pairs.values()),
        "candidate_wdl": {
            "wins": outcomes["win"],
            "draws": outcomes["draw"],
            "losses": outcomes["loss"],
        },
        "candidate_score": points / len(games) if games else None,
        "paired_score": (
            sum(complete_pair_scores) / len(complete_pair_scores)
            if complete_pair_scores else None
        ),
        "paired_ci95": paired_ci95(complete_pair_scores),
        "candidate_nodes": candidate_nodes,
        "control_nodes": control_nodes,
        "node_ratio": candidate_nodes / control_nodes if control_nodes else None,
        "candidate_time_ms": candidate_time,
        "control_time_ms": control_time,
        "time_ratio": candidate_time / control_time if control_time else None,
        "search_records": search_count,
        "reasons": dict(sorted(reasons.items())),
        "illegal_games": reasons["candidate_illegal"] + reasons["control_illegal"],
        "time_forfeits": reasons["candidate_time"] + reasons["control_time"],
    }


def update_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "status.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(updates)
    current["updated_at"] = utc_timestamp()
    atomic_json(path, current)
    return current


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def runner_lock_held(run_dir: Path) -> bool:
    lock_path = run_dir / ".runner.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def worker_process_matches(run_dir: Path, pid: int) -> bool:
    if not process_alive(pid):
        return False
    # The worker holds this private lock for its whole lifetime. Combining it
    # with kill(pid, 0) avoids both a sandbox-dependent `ps` call and sending a
    # stop signal to a stale/reused PID after the worker has released the lock.
    return runner_lock_held(run_dir)


def write_pid_locked(run_dir: Path, value: dict[str, Any]) -> None:
    lock_path = run_dir / ".pid.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic_json(run_dir / "pid.json", value)


def read_pid(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "pid.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_worker(contract: dict[str, Any], run_dir: Path) -> int:
    lock_path = run_dir / ".runner.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another match worker holds the lock") from error
        verify_contract(contract)
        repaired = repair_partial_jsonl_tail(Path(contract["output"]))
        games = read_games(Path(contract["output"]))
        expected = contract["protocol"]["expected_games"]
        if len(games) >= expected:
            summary = summarize_games(games, expected)
            atomic_json(run_dir / "summary.json", summary)
            update_status(
                run_dir, state="complete", completed_games=len(games),
                expected_games=expected, worker_pid=os.getpid(),
                repaired_partial_tail=repaired,
            )
            return 0

        stop = False
        child: subprocess.Popen[bytes] | None = None
        sleep_preventer: subprocess.Popen[bytes] | None = None

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop
            stop = True

        previous = {
            signum: signal.signal(signum, request_stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            update_status(
                run_dir, state="running", completed_games=len(games),
                expected_games=expected, worker_pid=os.getpid(),
                repaired_partial_tail=repaired,
            )
            caffeinate = contract.get("execution", {}).get("sleep_preventer")
            if caffeinate is not None:
                if not Path(caffeinate).is_file():
                    raise RuntimeError(
                        f"contract sleep preventer is unavailable: {caffeinate}"
                    )
                sleep_preventer = subprocess.Popen(
                    [caffeinate, "-dimsu", "-w", str(os.getpid())],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            child = subprocess.Popen(contract["command"], cwd=run_dir)
            write_pid_locked(run_dir, {
                "worker_pid": os.getpid(),
                "child_pid": child.pid,
                "sleep_preventer_pid": (
                    sleep_preventer.pid if sleep_preventer is not None else None
                ),
                "started_at": utc_timestamp(),
            })
            while child.poll() is None:
                if stop:
                    child.terminate()
                    break
                time.sleep(0.25)
            return_code = child.wait()
            games = read_games(Path(contract["output"]))
            summary = summarize_games(games, expected)
            atomic_json(run_dir / "summary.json", summary)
            if len(games) == expected and return_code == 0:
                state = "complete"
            elif stop or return_code == 0:
                state = "interrupted"
            else:
                state = "failed"
            update_status(
                run_dir, state=state, completed_games=len(games),
                expected_games=expected, worker_pid=os.getpid(),
                child_return_code=return_code,
            )
            return 0 if state in {"complete", "interrupted"} else return_code or 1
        finally:
            if sleep_preventer is not None and sleep_preventer.poll() is None:
                sleep_preventer.terminate()
                sleep_preventer.wait()
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def launch_worker(contract: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    launch_lock_path = run_dir / ".launch.lock"
    with launch_lock_path.open("a+") as launch_lock:
        try:
            fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another controller is launching the match") from error
        pid = read_pid(run_dir)
        if pid and worker_process_matches(
            run_dir, int(pid.get("worker_pid", -1))
        ):
            raise RuntimeError(f"match worker already running: {pid['worker_pid']}")
        runner = Path(contract["frozen_files"]["runner"]["path"])
        log_path = run_dir / "runner.log"
        pid_lock_path = run_dir / ".pid.lock"
        update_status(
            run_dir, state="starting", worker_pid=None,
            expected_games=contract["protocol"]["expected_games"],
        )
        with pid_lock_path.open("a+") as pid_lock:
            fcntl.flock(pid_lock, fcntl.LOCK_EX)
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [
                        sys.executable, str(runner),
                        "--run-dir", str(run_dir), "--worker",
                    ],
                    cwd=run_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pid_record = {
                "worker_pid": process.pid,
                "child_pid": None,
                "sleep_preventer_pid": None,
                "started_at": utc_timestamp(),
            }
            atomic_json(run_dir / "pid.json", pid_record)
    return pid_record


def status_snapshot(contract: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    pid = read_pid(run_dir)
    games = read_games(Path(contract["output"]), tolerate_partial_tail=True)
    result = {
        **status,
        "worker_alive": bool(
            pid and worker_process_matches(
                run_dir, int(pid.get("worker_pid", -1))
            )
        ),
        "pid": pid,
        "summary": summarize_games(
            games, contract["protocol"]["expected_games"]
        ),
    }
    return result


def stop_worker(run_dir: Path) -> dict[str, Any]:
    pid = read_pid(run_dir)
    if not pid or not worker_process_matches(
        run_dir, int(pid.get("worker_pid", -1))
    ):
        raise RuntimeError("no live match worker")
    worker_pid = int(pid["worker_pid"])
    os.kill(worker_pid, signal.SIGTERM)
    return {"state": "stop_requested", "worker_pid": worker_pid}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path,
        default=repo / "build" / "nnue_v43_v44_time_match",
    )
    parser.add_argument(
        "--model", type=Path,
        default=repo / "models" / "quantized_scale_grid"
        / "old_score_huber200_lr_sweep_then_5ep_20260724_142758"
        / "best" / "phase_quantized_nnue.bin",
    )
    parser.add_argument(
        "--book", type=Path,
        default=repo / "logs" / "nnue_v40_qsee_tune_20260817_005158"
        / "stockfish_balanced_openings_10ply_2400.txt",
    )
    parser.add_argument("--openings", type=int, default=DEFAULT_PROTOCOL["openings"])
    parser.add_argument("--base-ms", type=int, default=DEFAULT_PROTOCOL["base_ms"])
    parser.add_argument(
        "--increment-ms", type=int, default=DEFAULT_PROTOCOL["increment_ms"]
    )
    parser.add_argument(
        "--overhead-ms", type=int, default=DEFAULT_PROTOCOL["overhead_ms"]
    )
    parser.add_argument(
        "--max-plies", type=int, default=DEFAULT_PROTOCOL["max_plies"]
    )
    parser.add_argument("--tt-mb", type=int, default=DEFAULT_PROTOCOL["tt_mb"])
    parser.add_argument("--seed", type=int, default=DEFAULT_PROTOCOL["seed"])
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--init-only", action="store_true")
    actions.add_argument("--start", action="store_true")
    actions.add_argument("--resume", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--stop", action="store_true")
    actions.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    contract_path = args.run_dir / "contract.json"
    if args.worker:
        contract = load_contract(args.run_dir)
        try:
            worker_result = run_worker(contract, args.run_dir)
        except BaseException as error:
            update_status(
                args.run_dir, state="failed", worker_pid=os.getpid(),
                error=f"{type(error).__name__}: {error}",
            )
            raise
        raise SystemExit(worker_result)
    if contract_path.exists():
        contract = load_contract(args.run_dir)
        frozen_runner = Path(contract["frozen_files"]["runner"]["path"])
        if Path(__file__).resolve() != frozen_runner.resolve():
            os.execv(
                sys.executable,
                [sys.executable, str(frozen_runner), *sys.argv[1:]],
            )
    elif args.status or args.stop or args.resume:
        raise RuntimeError("match has not been initialized")
    else:
        contract = initialize(args)
    verify_contract(contract)

    if args.init_only:
        result = {
            "state": "initialized",
            "run_dir": str(args.run_dir),
            "expected_games": contract["protocol"]["expected_games"],
            "command": contract["command"],
        }
    elif args.start or args.resume:
        result = {
            "state": "started",
            "run_dir": str(args.run_dir),
            **launch_worker(contract, args.run_dir),
        }
    elif args.status:
        result = status_snapshot(contract, args.run_dir)
    else:
        result = stop_worker(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
