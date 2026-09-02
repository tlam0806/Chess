#!/usr/bin/env python3
"""Immutable, serial, resumable halving tournament for the V45 frontier."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from tools import run_nnue_v43_selfplay_race as infra
else:
    import run_nnue_v43_selfplay_race as infra  # type: ignore


SCHEMA_VERSION = 1
EXPERIMENT = "nnue-v45-pareto-halving-round-robin-v1"
PRODUCTION_V43_HASH = (
    "98b7732c9587da35554cc274a072a0a5b5f55902aaa77605e78c1ae13e88b4f2"
)
POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}
ROUNDS = (
    {"name": "r1", "entrants": 17, "survivors": 9, "openings": 4,
     "games_per_pair": 8, "base_ms": 1_000, "increment_ms": 10,
     "seed": 2026090211},
    {"name": "r2", "entrants": 9, "survivors": 5, "openings": 8,
     "games_per_pair": 16, "base_ms": 3_000, "increment_ms": 30,
     "seed": 2026090222},
    {"name": "r3", "entrants": 5, "survivors": 3, "openings": 16,
     "games_per_pair": 32, "base_ms": 5_000, "increment_ms": 50,
     "seed": 2026090233},
    {"name": "r4", "entrants": 3, "survivors": 2, "openings": 32,
     "games_per_pair": 64, "base_ms": 7_000, "increment_ms": 70,
     "seed": 2026090244},
    {"name": "r5", "entrants": 2, "survivors": 1, "openings": 64,
     "games_per_pair": 128, "base_ms": 10_000, "increment_ms": 100,
     "seed": 2026090255},
)
CONFIRMATION = {
    "name": "confirmation", "openings": 300, "games_per_pair": 600,
    "base_ms": 10_000, "increment_ms": 100, "seed": 2026090266,
}
EXPECTED_TOURNAMENT_GAMES = sum(
    math.comb(value["entrants"], 2) * value["games_per_pair"]
    for value in ROUNDS
)
assert EXPECTED_TOURNAMENT_GAMES == 2_304


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved), "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(
        json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    os.replace(temporary, path)


def write_once_json(path: Path, value: object) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"immutable artifact disagrees: {path}")
        return
    atomic_json(path, value)


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_bytes(config)).hexdigest()


def load_profiles(frontier_path: Path) -> list[dict]:
    payload = json.loads(frontier_path.read_text())
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 17:
        raise RuntimeError("V45 tournament requires exactly 17 finalists")
    profiles = []
    seen = set()
    aspirations = set()
    for item in candidates:
        config = item["config"]
        candidate_hash = item["config_hash"]
        if config_hash(config) != candidate_hash or candidate_hash in seen:
            raise RuntimeError("frontier candidate identity mismatch")
        seen.add(candidate_hash)
        aspirations.add(canonical_bytes(config["aspiration"]))
        profiles.append({
            "role": "challenger",
            "name": f"candidate_{candidate_hash[:12]}",
            "config_hash": candidate_hash,
            "config": config,
            "source_selection_metrics": {
                "mean_wdl_loss": float(item["mean_wdl_loss"]),
                "node_ratio": float(item["production_node_ratio"]),
                "critical_mistakes": int(item["critical_mistakes"]),
            },
        })
    if len(aspirations) != 1:
        raise RuntimeError("frontier does not use one frozen aspiration policy")
    profiles.sort(key=lambda item: item["config_hash"])
    return profiles


def load_production_profile(v45_run: Path) -> dict:
    path = v45_run / "evaluations" / "selection" / f"{PRODUCTION_V43_HASH}.json"
    value = json.loads(path.read_text())
    config = value["config"]
    if config_hash(config) != PRODUCTION_V43_HASH:
        raise RuntimeError("production V43 evaluation identity mismatch")
    return {
        "role": "production",
        "name": f"production_{PRODUCTION_V43_HASH[:12]}",
        "config_hash": PRODUCTION_V43_HASH,
        "config": config,
        "source_selection_metrics": {
            "mean_wdl_loss": float(value["result"]["mean_wdl_loss"]),
            "node_ratio": 1.0,
            "critical_mistakes": int(value["result"]["critical_mistakes"]),
        },
    }


def split_books(master: Path, run_dir: Path, smoke_test: bool) -> dict[str, dict]:
    lines = [
        line.strip() for line in master.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != len(set(lines)):
        raise RuntimeError("master opening book contains duplicates")
    import random
    random.Random(2026090201).shuffle(lines)
    settings = (
        [("r1", 1), ("confirmation", 1)] if smoke_test else
        [(value["name"], value["openings"]) for value in ROUNDS]
        + [(CONFIRMATION["name"], CONFIRMATION["openings"])]
    )
    required = sum(count for _, count in settings)
    if len(lines) < required:
        raise RuntimeError(f"opening book needs {required} unique positions")
    books = {}
    offset = 0
    for name, count in settings:
        path = run_dir / "openings" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "\n".join(lines[offset:offset + count]) + "\n"
        if path.exists() and path.read_text() != data:
            raise RuntimeError(f"immutable opening split changed: {name}")
        if not path.exists():
            path.write_text(data)
        books[name] = {**fingerprint(path), "count": count}
        offset += count
    return books


def initialize(args: argparse.Namespace) -> dict:
    contract_path = args.run_dir / "contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        if sha256_file(contract_path) != (args.run_dir / "contract.sha256").read_text().strip():
            raise RuntimeError("immutable tournament contract hash mismatch")
        return contract
    profiles = load_profiles(args.frontier)
    production = load_production_profile(args.v45_run)
    if production["config_hash"] not in {item["config_hash"] for item in profiles}:
        raise RuntimeError("production V43 must be one of the 17 finalists")
    books = split_books(args.book, args.run_dir, args.smoke_test)
    rounds = (
        ({**ROUNDS[0], "entrants": 2, "survivors": 1, "openings": 1,
          "games_per_pair": 2},)
        if args.smoke_test else ROUNDS
    )
    contract = {
        "kind": "nnue_v45_round_robin_contract",
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "created_at": utc_now(),
        "smoke_test": args.smoke_test,
        "frontier": fingerprint(args.frontier),
        "v45_summary": fingerprint(args.v45_run / "summary.json"),
        "binary": fingerprint(args.binary),
        "model": fingerprint(args.model),
        "master_book": fingerprint(args.book),
        "runner": fingerprint(Path(__file__)),
        "infra_runner": fingerprint(Path(infra.__file__)),
        "profiles": profiles[:2] if args.smoke_test else profiles,
        "production": production,
        "aspiration": profiles[0]["config"]["aspiration"],
        "books": books,
        "rounds": list(rounds),
        "confirmation": None if args.smoke_test else CONFIRMATION,
        "protocol": {
            "serial_execution": True,
            "round_robin": True,
            "paired_color_reversed": True,
            "fresh_disjoint_openings_per_round": True,
            "eliminate_approximately_half": True,
            "ranking": [
                "legal_desc", "score_desc", "tied_head_to_head_desc",
                "selection_wdl_loss_asc", "selection_node_ratio_asc",
                "config_hash_asc",
            ],
            "twofold_search_draw_all_profiles": True,
            "reuse_stale_tt_scores": False,
            "reuse_deeper_tt_scores": False,
            "tt_mb": 64,
            "overhead_ms": 20,
            "max_plies": 200,
            "automatic_promotion": False,
            "expected_tournament_games": (
                2 if args.smoke_test else EXPECTED_TOURNAMENT_GAMES
            ),
            "confirmation_games": 0 if args.smoke_test else 600,
        },
    }
    atomic_json(contract_path, contract)
    (args.run_dir / "contract.sha256").write_text(sha256_file(contract_path) + "\n")
    atomic_json(args.run_dir / "status.json", {
        "state": "initialized", "stage": "init", "completed_games": 0,
        "updated_at": utc_now(),
    })
    return contract


def verify_contract(contract: dict) -> None:
    for name in ("frontier", "v45_summary", "binary", "model", "master_book",
                 "runner", "infra_runner"):
        expected = contract[name]
        path = Path(expected["path"])
        if fingerprint(path) != expected:
            raise RuntimeError(f"frozen tournament artifact changed: {name}")
    seen = set()
    for entry in contract["books"].values():
        path = Path(entry["path"])
        if fingerprint(path) != {key: entry[key] for key in ("path", "size", "sha256")}:
            raise RuntimeError(f"round opening book changed: {path}")
        lines = {line for line in path.read_text().splitlines() if line.strip()}
        if len(lines) != entry["count"] or seen.intersection(lines):
            raise RuntimeError("round books are not exact and disjoint")
        seen.update(lines)


def build_command(
    contract: dict, setting: dict, profiles: Sequence[dict], output: Path
) -> list[str]:
    command = [
        contract["binary"]["path"],
        "--book", contract["books"][setting["name"]]["path"],
        "--model", contract["model"]["path"],
        "--output", str(output),
        "--openings", str(setting["openings"]),
        "--base-ms", str(setting["base_ms"]),
        "--increment-ms", str(setting["increment_ms"]),
        "--overhead-ms", str(contract["protocol"]["overhead_ms"]),
        "--max-plies", str(contract["protocol"]["max_plies"]),
        "--tt-mb", str(contract["protocol"]["tt_mb"]),
        "--seed", str(setting["seed"]),
        "--round-robin",
    ]
    for profile in profiles:
        command.extend(("--profile", infra.profile_spec(profile)))
    for profile in profiles:
        command.extend(("--aspiration-profile", infra.aspiration_spec(
            profile, contract["aspiration"]
        )))
    for profile in profiles:
        command.extend(("--twofold-search-profile", profile["name"]))
    return command


def read_complete_games(path: Path) -> list[dict]:
    return infra.read_games(path, repair_partial_tail=True)


def expected_round_games(setting: dict, profile_count: int) -> int:
    return math.comb(profile_count, 2) * setting["games_per_pair"]


def validate_round_games(
    path: Path, profiles: Sequence[dict], setting: dict
) -> tuple[list[dict], dict]:
    games = read_complete_games(path)
    expected = expected_round_games(setting, len(profiles))
    if len(games) != expected:
        raise RuntimeError(f"{setting['name']} expected {expected} games; found {len(games)}")
    manifest = infra.validate_gauntlet_manifest(path, list(profiles))
    fingerprint_value = manifest["run_fingerprint"]
    names = {profile["name"] for profile in profiles}
    keys = set()
    opening_groups: dict[tuple[str, str, str], int] = Counter()
    for game in games:
        first, second = game.get("profile"), game.get("opponent")
        if first not in names or second not in names or first == second:
            raise RuntimeError("round emitted an unknown matchup")
        key = game.get("key")
        if key in keys or game.get("run_fingerprint") != fingerprint_value:
            raise RuntimeError("round contains duplicate or foreign game identity")
        keys.add(key)
        logical = game["logical_key"].rsplit(":", 1)[0]
        opening_groups[(*sorted((first, second)), logical)] += 1
        if game.get("candidate_outcome") not in POINTS:
            raise RuntimeError("round contains invalid outcome")
    if len(opening_groups) != math.comb(len(profiles), 2) * setting["openings"]:
        raise RuntimeError("round opening-pair count mismatch")
    if any(value != 2 for value in opening_groups.values()):
        raise RuntimeError("round opening was not color reversed exactly once")
    return games, manifest


def child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def count_games(run_dir: Path) -> int:
    total = 0
    for path in sorted(run_dir.glob("rounds/*/games.jsonl")):
        total += len(read_complete_games(path))
    confirmation = run_dir / "confirmation" / "games.jsonl"
    if confirmation.exists():
        total += len(read_complete_games(confirmation))
    return total


def update_status(run_dir: Path, **values: object) -> None:
    current = {}
    path = run_dir / "status.json"
    if path.exists():
        current = json.loads(path.read_text())
    current.update(values)
    current["completed_games"] = count_games(run_dir)
    current["updated_at"] = utc_now()
    atomic_json(path, current)


def terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_round_process(
    contract: dict, run_dir: Path, setting: dict, profiles: Sequence[dict],
    output: Path,
) -> dict:
    expected = expected_round_games(setting, len(profiles))
    if output.exists() and len(read_complete_games(output)) == expected:
        games, manifest = validate_round_games(output, profiles, setting)
        resource_path = output.parent / "resource.json"
        resource = (
            json.loads(resource_path.read_text())
            if resource_path.is_file() else None
        )
        return {"games": games, "manifest": manifest, "resource": resource}
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(contract, setting, profiles, output)
    log_path = output.with_suffix(".log")
    started_wall = time.monotonic()
    started_cpu = child_cpu_seconds()
    with log_path.open("ab") as log:
        log.write(("command=" + json.dumps(command) + "\n").encode())
        log.flush()
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        )
        try:
            while True:
                try:
                    returncode = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    update_status(
                        run_dir, state="running", stage=setting["name"],
                        expected_stage_games=expected,
                        completed_stage_games=len(read_complete_games(output)),
                    )
        except BaseException:
            terminate_group(process)
            raise
    wall = time.monotonic() - started_wall
    cpu = child_cpu_seconds() - started_cpu
    if returncode:
        raise RuntimeError(f"gauntlet failed ({returncode}); see {log_path}")
    games, manifest = validate_round_games(output, profiles, setting)
    resource_payload = {
        "wall_sec": wall, "child_cpu_sec": cpu,
        "single_core_cpu_utilization": cpu / wall if wall else None,
        "measured_at": utc_now(),
    }
    write_once_json(output.parent / "resource.json", resource_payload)
    return {"games": games, "manifest": manifest, "resource": resource_payload}


def rank_round(
    games: Sequence[dict], profiles: Sequence[dict], setting: dict
) -> list[dict]:
    del setting
    standings = {
        profile["name"]: {
            "name": profile["name"], "config_hash": profile["config_hash"],
            "points": 0.0, "games": 0, "wins": 0, "draws": 0, "losses": 0,
            "nodes": 0, "time_ms": 0, "illegal_games": 0,
            "opponent_points": defaultdict(float),
            "selection_mean_wdl_loss": profile["source_selection_metrics"]["mean_wdl_loss"],
            "selection_node_ratio": profile["source_selection_metrics"]["node_ratio"],
        }
        for profile in profiles
    }
    for game in games:
        first_name, second_name = game["profile"], game["opponent"]
        first, second = standings[first_name], standings[second_name]
        outcome = game["candidate_outcome"]
        first_points = POINTS[outcome]
        second_points = 1.0 - first_points
        first["points"] += first_points
        second["points"] += second_points
        first["games"] += 1
        second["games"] += 1
        first["nodes"] += int(game.get("candidate_nodes", 0))
        second["nodes"] += int(game.get("control_nodes", 0))
        first["time_ms"] += int(game.get("candidate_time_ms", 0))
        second["time_ms"] += int(game.get("control_time_ms", 0))
        first["opponent_points"][second_name] += first_points
        second["opponent_points"][first_name] += second_points
        outcome_key = {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
        first[outcome_key] += 1
        reverse = {"win": "loss", "draw": "draw", "loss": "win"}[outcome]
        reverse_key = {"win": "wins", "draw": "draws", "loss": "losses"}[reverse]
        second[reverse_key] += 1
        reason = game.get("reason")
        first["illegal_games"] += reason == "candidate_illegal"
        second["illegal_games"] += reason == "control_illegal"
    point_groups: dict[float, list[str]] = defaultdict(list)
    for row in standings.values():
        point_groups[row["points"]].append(row["name"])
    rows = []
    for row in standings.values():
        tied_names = point_groups[row["points"]]
        row["tied_head_to_head_points"] = sum(
            row["opponent_points"].get(name, 0.0)
            for name in tied_names if name != row["name"]
        )
        row["score"] = row["points"] / row["games"]
        row["eligible"] = row["illegal_games"] == 0
        row["opponent_points"] = dict(row["opponent_points"])
        rows.append(row)
    rows.sort(key=lambda row: (
        not row["eligible"], -row["score"], -row["tied_head_to_head_points"],
        row["selection_mean_wdl_loss"], row["selection_node_ratio"],
        row["config_hash"],
    ))
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def execute(contract: dict, args: argparse.Namespace) -> dict:
    verify_contract(contract)
    if (args.run_dir / "DONE").exists():
        return json.loads((args.run_dir / "summary.json").read_text())
    profiles_by_hash = {
        profile["config_hash"]: profile for profile in contract["profiles"]
    }
    active = list(contract["profiles"])
    round_summaries = []
    for setting in contract["rounds"]:
        if len(active) != setting["entrants"]:
            raise RuntimeError(f"{setting['name']} entrant-count mismatch")
        active = sorted(active, key=lambda profile: profile["config_hash"])
        round_dir = args.run_dir / "rounds" / setting["name"]
        update_status(
            args.run_dir, state="running", stage=setting["name"],
            entrants=len(active), expected_stage_games=expected_round_games(
                setting, len(active)
            ),
        )
        executed = run_round_process(
            contract, args.run_dir, setting, active, round_dir / "games.jsonl"
        )
        ranking = rank_round(executed["games"], active, setting)
        survivors = ranking[:setting["survivors"]]
        if any(not row["eligible"] for row in survivors):
            raise RuntimeError(f"{setting['name']} has too few legal survivors")
        selected_hashes = [row["config_hash"] for row in survivors]
        summary = {
            "kind": "nnue_v45_round_robin_round",
            "schema_version": SCHEMA_VERSION,
            "stage": setting["name"], "setting": setting,
            "ranking": ranking, "selected_hashes": selected_hashes,
            "resource": executed["resource"],
            "artifacts": {
                "games": fingerprint(round_dir / "games.jsonl"),
                "manifest": fingerprint(Path(str(round_dir / "games.jsonl") + ".manifest.json")),
                "run_fingerprint": executed["manifest"]["run_fingerprint"],
            },
        }
        write_once_json(round_dir / "summary.json", summary)
        round_summaries.append(summary)
        active = [profiles_by_hash[value] for value in selected_hashes]
        print(
            f"{setting['name']} complete games={len(executed['games'])} "
            f"survivors={len(active)} leader={selected_hashes[0][:12]}",
            flush=True,
        )
    winner = active[0]
    confirmation_summary = None
    if contract["confirmation"] is not None and winner["config_hash"] != PRODUCTION_V43_HASH:
        setting = contract["confirmation"]
        output = args.run_dir / "confirmation" / "games.jsonl"
        update_status(
            args.run_dir, state="running", stage="confirmation",
            entrants=2, expected_stage_games=setting["games_per_pair"],
        )
        profiles = [winner, contract["production"]]
        executed = run_round_process(
            contract, args.run_dir, setting, profiles, output
        )
        ranking = rank_round(executed["games"], profiles, setting)
        candidate_row = next(
            row for row in ranking if row["config_hash"] == winner["config_hash"]
        )
        confirmation_summary = {
            "kind": "nnue_v45_winner_confirmation",
            "winner_hash": winner["config_hash"],
            "production_hash": PRODUCTION_V43_HASH,
            "candidate_score": candidate_row["score"],
            "candidate_points": candidate_row["points"],
            "games": candidate_row["games"],
            "ranking": ranking, "resource": executed["resource"],
            "artifacts": {
                "games": fingerprint(output),
                "manifest": fingerprint(Path(str(output) + ".manifest.json")),
                "run_fingerprint": executed["manifest"]["run_fingerprint"],
            },
            "automatic_promotion": False,
        }
        write_once_json(args.run_dir / "confirmation" / "summary.json", confirmation_summary)
    result = {
        "kind": "nnue_v45_round_robin_summary",
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "completed_at": utc_now(),
        "winner_hash": winner["config_hash"],
        "winner_config": winner["config"],
        "production_won_tournament": winner["config_hash"] == PRODUCTION_V43_HASH,
        "rounds": [
            {"stage": item["stage"], "selected_hashes": item["selected_hashes"]}
            for item in round_summaries
        ],
        "tournament_games": sum(
            len(read_complete_games(args.run_dir / "rounds" / item["stage"] / "games.jsonl"))
            for item in round_summaries
        ),
        "confirmation": confirmation_summary,
        "automatic_promotion": False,
        "decision": "complete_for_human_review",
    }
    write_once_json(args.run_dir / "summary.json", result)
    (args.run_dir / "DONE").write_text("\n")
    update_status(
        args.run_dir, state="complete", stage="done",
        winner_hash=winner["config_hash"],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--v45-run", type=Path, required=True)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = args.run_dir.resolve()
    args.v45_run = args.v45_run.resolve(strict=True)
    args.frontier = (
        args.frontier.resolve(strict=True) if args.frontier is not None
        else (args.v45_run / "stage2_frontier.json").resolve(strict=True)
    )
    args.binary = args.binary.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    args.book = args.book.resolve(strict=True)
    return args


def main() -> None:
    args = parse_args()
    with (args.run_dir / ".runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another tournament runner owns this directory") from error
        contract = initialize(args)
        (args.run_dir / "RUNNING").write_text(f"{os.getpid()}\n")
        (args.run_dir / "FAILED").unlink(missing_ok=True)
        try:
            result = execute(contract, args)
            print(json.dumps(result, sort_keys=True), flush=True)
        except BaseException as error:
            atomic_json(args.run_dir / "FAILED", {
                "state": "failed", "error_type": type(error).__name__,
                "error": str(error), "updated_at": utc_now(),
            })
            update_status(
                args.run_dir, state="failed", error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            (args.run_dir / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
