#!/usr/bin/env python3
"""Immutable, serial and resumable V43 common-control self-play race."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}
SELECTIVE_FIELDS = (
    "enable_lmr", "lmr_base", "lmr_divisor", "lmr_min_depth",
    "lmr_min_move_index", "enable_null_move", "null_move_min_depth",
    "null_move_reduction", "enable_reverse_futility",
    "reverse_futility_max_depth", "reverse_futility_base_margin",
    "reverse_futility_margin_per_depth", "enable_late_move_pruning",
    "late_move_pruning_max_depth", "late_move_pruning_base",
    "late_move_pruning_depth_multiplier", "enable_qsearch_see_pruning",
    "qsearch_see_threshold", "enable_main_search_see_pruning",
    "main_search_see_max_depth", "main_search_see_margin_per_depth",
)
ASPIRATION_FIELDS = (
    "enabled", "min_depth", "delta_base_cp", "delta_divisor",
    "expansion_factor_per_mille", "max_fail_high_reductions",
    "mean_score_new_weight_per_mille", "max_researches",
    "mean_score_clamp_cp",
)
ROUNDS = {
    "r1": {"openings": 16, "games_per_match": 32, "base_ms": 1000,
           "increment_ms": 10, "seed": 2026090111, "advance": 8},
    "r2": {"openings": 48, "games_per_match": 96, "base_ms": 3000,
           "increment_ms": 30, "seed": 2026090122, "advance": 3},
    "r3": {"openings": 50, "games_per_match": 100, "base_ms": 5000,
           "increment_ms": 50, "seed": 2026090133, "advance": 1},
    "final": {"openings": 300, "games_per_match": 600,
              "base_ms": 10000, "increment_ms": 100,
              "seed": 2026090144, "advance": 0},
}
EXPECTED_TOTAL_GAMES = 23 * 32 + 8 * 96 + 3 * 100 + 600


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def write_once_json(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"immutable artifact disagrees: {path}")
        return
    atomic_json(path, value)


def copy_snapshot(source: Path, target: Path, executable: bool = False) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing source artifact: {source}")
    mode = 0o755 if executable else 0o644
    atomic_write(target, source.read_bytes(), mode)


def candidate_hash(config: dict) -> str:
    return sha256_bytes(canonical_bytes(config))


def load_selection_records(results_path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for number, line in enumerate(results_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"malformed source results line {number}") from error
        if record.get("kind") != "result" or record.get("stage") != "selection":
            continue
        key = record.get("config_hash")
        if key in records:
            raise RuntimeError(f"duplicate selection record: {key}")
        records[key] = record
    return records


def profile_name(role: str, config_hash: str) -> str:
    stem = "candidate" if role == "selection" else role
    return f"{stem}_{config_hash[:12]}"


def profile_spec(profile: dict) -> str:
    config = profile["config"]["selective"]
    required = set(SELECTIVE_FIELDS)
    if set(config) != required:
        raise RuntimeError("non-canonical selective profile")
    def b(value: Any) -> str:
        return "1" if value is True else "0" if value is False else str(value)
    order = (
        "lmr_base", "lmr_divisor", "lmr_min_depth", "lmr_min_move_index",
        "null_move_min_depth", "null_move_reduction",
        "enable_reverse_futility", "reverse_futility_max_depth",
        "reverse_futility_base_margin", "reverse_futility_margin_per_depth",
        "enable_late_move_pruning", "late_move_pruning_max_depth",
        "late_move_pruning_base", "late_move_pruning_depth_multiplier",
        "enable_qsearch_see_pruning", "qsearch_see_threshold",
        "enable_main_search_see_pruning", "main_search_see_max_depth",
        "main_search_see_margin_per_depth", "enable_lmr", "enable_null_move",
    )
    return ",".join([profile["name"], *(b(config[field]) for field in order)])


def aspiration_spec(profile: dict, aspiration: dict) -> str:
    if set(aspiration) != set(ASPIRATION_FIELDS):
        raise RuntimeError("non-canonical aspiration profile")
    def b(value: Any) -> str:
        return "1" if value is True else "0" if value is False else str(value)
    return ",".join([profile["name"], *(b(aspiration[field])
                     for field in ASPIRATION_FIELDS)])


def _profile_from_record(record: dict, role: str, aspiration: dict) -> dict:
    config = record.get("config")
    if not isinstance(config, dict) or candidate_hash(config) != record["config_hash"]:
        raise RuntimeError(f"source config hash mismatch: {record.get('config_hash')}")
    if config.get("aspiration") != aspiration:
        raise RuntimeError(f"profile does not use frozen Balanced aspiration: {record['config_hash']}")
    result = record.get("result", {})
    return {
        "role": role,
        "name": profile_name(role, record["config_hash"]),
        "config_hash": record["config_hash"],
        "config": config,
        "source_selection_metrics": {
            key: result.get(key) for key in (
                "mean_wdl_loss", "node_ratio", "critical_mistakes",
                "aspiration_accepted_depth_ratio",
            )
        },
    }


def split_openings(source: Path) -> dict[str, list[str]]:
    lines = [line.strip() for line in source.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != len(set(lines)):
        raise RuntimeError("source opening book contains duplicate lines")
    needed = sum(value["openings"] for value in ROUNDS.values())
    if len(lines) < needed:
        raise RuntimeError(f"opening book has {len(lines)} lines; need {needed}")
    random.Random(2026090101).shuffle(lines)
    result: dict[str, list[str]] = {}
    offset = 0
    for stage, settings in ROUNDS.items():
        count = settings["openings"]
        result[stage] = lines[offset:offset + count]
        offset += count
    assert len({line for values in result.values() for line in values}) == needed
    return result


def initialize(args: argparse.Namespace) -> dict:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "contract.json"
    if contract_path.exists():
        return load_contract(run_dir)
    source_run = args.source_run.resolve()
    results_path = source_run / "results.jsonl"
    summary_path = source_run / "summary.json"
    pool_spec = json.loads(args.pool_spec.read_text())
    sources_to_pin = {
        "binary": args.binary.resolve(), "model": args.model.resolve(),
        "master_book": args.book.resolve(), "source_results": results_path,
        "source_summary": summary_path,
    }
    for name, source in sources_to_pin.items():
        actual = fingerprint(source)["sha256"]
        if actual != pool_spec["expected_sha256"][name]:
            raise RuntimeError(f"{name} SHA256 disagrees with frozen pool spec")
    hashes = pool_spec["selection_candidate_hashes"]
    if len(hashes) != 22 or len(set(hashes)) != 22:
        raise RuntimeError("pool must contain exactly 22 distinct selection candidates")
    records = load_selection_records(results_path)
    all_hashes = [*hashes, pool_spec["pre_refresh_fast_hash"],
                  pool_spec["production_control_hash"]]
    missing = [value for value in all_hashes if value not in records]
    if missing:
        raise RuntimeError(f"pool hashes missing from selection records: {missing}")
    aspiration = pool_spec["balanced_aspiration"]
    entrants = [_profile_from_record(records[value], "selection", aspiration)
                for value in hashes]
    entrants.append(_profile_from_record(
        records[pool_spec["pre_refresh_fast_hash"]], "fast", aspiration))
    control = _profile_from_record(
        records[pool_spec["production_control_hash"]], "production", aspiration)
    control_nodes = control["source_selection_metrics"]["node_ratio"]
    if any(profile["source_selection_metrics"]["node_ratio"] > 2 * control_nodes + 1e-15
           for profile in entrants[:-1]):
        raise RuntimeError("a 22-candidate pool member exceeds 2x production nodes")

    artifacts = run_dir / "artifacts"
    copy_snapshot(args.binary.resolve(), artifacts / "nnue_v43_time_gauntlet", True)
    copy_snapshot(args.model.resolve(), artifacts / "phase_quantized_nnue.bin")
    copy_snapshot(Path(__file__).resolve(), run_dir / "bin" / Path(__file__).name, True)
    copy_snapshot(args.pool_spec.resolve(), artifacts / "pool.json")
    copy_snapshot(summary_path, artifacts / "source_summary.json")
    books = split_openings(args.book.resolve())
    book_entries = {}
    for stage, lines in books.items():
        path = run_dir / "openings" / f"{stage}.txt"
        atomic_write(path, ("\n".join(lines) + "\n").encode())
        book_entries[stage] = {"path": str(path), **fingerprint(path),
                               "count": len(lines)}

    frozen_files = {}
    for name, path in {
        "binary": artifacts / "nnue_v43_time_gauntlet",
        "model": artifacts / "phase_quantized_nnue.bin",
        "runner": run_dir / "bin" / Path(__file__).name,
        "pool": artifacts / "pool.json",
        "source_summary": artifacts / "source_summary.json",
    }.items():
        frozen_files[name] = {"path": str(path), **fingerprint(path)}
    contract = {
        "kind": "nnue_v43_selfplay_race_contract",
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "source": {
            "run_dir": str(source_run),
            "results": {"path": str(results_path), **fingerprint(results_path)},
            "summary": {"path": str(summary_path), **fingerprint(summary_path)},
            "master_book": {"path": str(args.book.resolve()),
                            **fingerprint(args.book.resolve())},
        },
        "frozen_files": frozen_files,
        "round_books": book_entries,
        "balanced_aspiration": aspiration,
        "entrants": entrants,
        "production_control": control,
        "rounds": ROUNDS,
        "protocol": {
            "serial_execution": True,
            "paired_color_reversed": True,
            "no_early_stop": True,
            "twofold_search_draw_all_profiles": True,
            "reuse_stale_tt_scores": False,
            "reuse_deeper_tt_scores": False,
            "tt_mb": 64,
            "overhead_ms": 20,
            "max_plies": 200,
            "expected_total_games": EXPECTED_TOTAL_GAMES,
            "r1_r2_ranking": ["score_desc", "paired_ci95_lower_desc",
                              "selection_wdl_loss_asc", "node_ratio_asc",
                              "config_hash_asc"],
            "r3_ranking": ["score_desc", "two_way_head_to_head_desc",
                           "r2_control_score_desc", "config_hash_asc"],
            "automatic_promotion": False,
        },
    }
    atomic_json(contract_path, contract)
    atomic_write(run_dir / "contract.sha256",
                 (fingerprint(contract_path)["sha256"] + "\n").encode())
    atomic_json(run_dir / "status.json", {
        "state": "initialized", "phase": "init", "completed_games": 0,
        "expected_total_games": EXPECTED_TOTAL_GAMES,
    })
    return contract


def load_contract(run_dir: Path) -> dict:
    path = run_dir / "contract.json"
    expected = (run_dir / "contract.sha256").read_text().strip()
    actual = fingerprint(path)["sha256"]
    if actual != expected:
        raise RuntimeError("immutable contract SHA256 mismatch")
    contract = json.loads(path.read_text())
    if (contract.get("kind") != "nnue_v43_selfplay_race_contract"
            or contract.get("schema_version") != SCHEMA_VERSION):
        raise RuntimeError("unsupported race contract")
    return contract


def verify_contract(contract: dict) -> None:
    for entry in contract["frozen_files"].values():
        path = Path(entry["path"])
        if not path.is_file() or fingerprint(path) != {
                "size": entry["size"], "sha256": entry["sha256"]}:
            raise RuntimeError(f"frozen artifact changed: {path}")
    seen: set[str] = set()
    for entry in contract["round_books"].values():
        path = Path(entry["path"])
        if fingerprint(path) != {"size": entry["size"], "sha256": entry["sha256"]}:
            raise RuntimeError(f"frozen opening book changed: {path}")
        lines = {line for line in path.read_text().splitlines() if line.strip()}
        if len(lines) != entry["count"] or seen.intersection(lines):
            raise RuntimeError("round opening books are not exact and disjoint")
        seen.update(lines)
    aspiration = contract["balanced_aspiration"]
    for profile in [*contract["entrants"], contract["production_control"]]:
        if profile["config"]["aspiration"] != aspiration:
            raise RuntimeError("contract contains non-Balanced aspiration")


def build_command(contract: dict, stage: str, profiles: list[dict],
                  output: Path) -> list[str]:
    settings = contract["rounds"][stage]
    command = [
        contract["frozen_files"]["binary"]["path"],
        "--book", contract["round_books"][stage]["path"],
        "--model", contract["frozen_files"]["model"]["path"],
        "--output", str(output), "--openings", str(settings["openings"]),
        "--base-ms", str(settings["base_ms"]),
        "--increment-ms", str(settings["increment_ms"]),
        "--overhead-ms", str(contract["protocol"]["overhead_ms"]),
        "--max-plies", str(contract["protocol"]["max_plies"]),
        "--tt-mb", str(contract["protocol"]["tt_mb"]),
        "--seed", str(settings["seed"]), "--round-robin",
    ]
    for profile in profiles:
        command += ["--profile", profile_spec(profile)]
    for profile in profiles:
        command += ["--aspiration-profile", aspiration_spec(
            profile, contract["balanced_aspiration"])]
    for profile in profiles:
        command += ["--twofold-search-profile", profile["name"]]
    return command


def read_games(path: Path, repair_partial_tail: bool = False) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        if not repair_partial_tail:
            raise RuntimeError(f"non-newline JSONL tail: {path}")
        boundary = data.rfind(b"\n")
        if boundary < 0:
            atomic_write(path, b"")
            data = b""
        else:
            atomic_write(path, data[:boundary + 1])
            data = data[:boundary + 1]
    games = []
    for number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"malformed JSONL record {path}:{number}") from error
        if value.get("kind") == "game":
            games.append(value)
    return games


def validate_gauntlet_manifest(path: Path, profiles: list[dict]) -> dict:
    manifest_path = Path(str(path) + ".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"missing gauntlet manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"malformed gauntlet manifest: {manifest_path}") from error
    if (manifest.get("kind") != "gauntlet_manifest"
            or not isinstance(manifest.get("run_fingerprint"), str)):
        raise RuntimeError(f"invalid gauntlet manifest: {manifest_path}")
    emitted = manifest.get("identity", {}).get("profiles")
    if not isinstance(emitted, list) or len(emitted) != len(profiles):
        raise RuntimeError(f"gauntlet manifest profile-count mismatch: {manifest_path}")
    for actual, expected in zip(emitted, profiles):
        if (actual.get("name") != expected["name"]
                or actual.get("selective_config") != expected["config"]["selective"]
                or actual.get("aspiration_config") != expected["config"]["aspiration"]
                or actual.get("twofold_search_draw") is not True
                or actual.get("reuse_stale_tt_scores") is not False
                or actual.get("reuse_deeper_tt_scores") is not False):
            raise RuntimeError(f"gauntlet manifest policy mismatch: {manifest_path}")
    return manifest


def paired_summary(path: Path, first: dict, second: dict,
                   expected_games: int | None = None,
                   require_manifest: bool = True) -> dict:
    games = read_games(path)
    manifest = validate_gauntlet_manifest(path, [first, second]) if require_manifest else None
    run_fingerprint = manifest["run_fingerprint"] if manifest else None
    seen_keys: set[str] = set()
    pairs: dict[str, list[float]] = defaultdict(list)
    outcomes = Counter()
    reasons = Counter()
    first_nodes = second_nodes = first_time = second_time = 0
    aspiration = first["config"]["aspiration"]
    for game in games:
        if game.get("profile") != first["name"] or game.get("opponent") != second["name"]:
            raise RuntimeError(f"unexpected matchup in {path}")
        key = game.get("key")
        if key in seen_keys:
            raise RuntimeError(f"duplicate game key in {path}: {key}")
        seen_keys.add(key)
        if run_fingerprint is not None and (
                game.get("run_fingerprint") != run_fingerprint
                or not key.startswith(run_fingerprint + ":")):
            raise RuntimeError(f"game/manifest fingerprint mismatch in {path}")
        outcome = game.get("candidate_outcome")
        if outcome not in POINTS:
            raise RuntimeError(f"invalid outcome in {path}")
        if (game.get("candidate_aspiration_config") != aspiration
                or game.get("opponent_aspiration_config") != aspiration
                or game.get("candidate_twofold_search_draw") is not True
                or game.get("candidate_reuse_stale_tt_scores") is not False
                or game.get("candidate_reuse_deeper_tt_scores") is not False
                or game.get("opponent_reuse_stale_tt_scores") is not False
                or game.get("opponent_reuse_deeper_tt_scores") is not False):
            raise RuntimeError(f"emitted game policy mismatch in {path}")
        pair_key = game["logical_key"].rsplit(":", 1)[0]
        pairs[pair_key].append(POINTS[outcome])
        outcomes[outcome] += 1
        reasons[game.get("reason", "unknown")] += 1
        first_nodes += int(game.get("candidate_nodes", 0))
        second_nodes += int(game.get("control_nodes", 0))
        first_time += int(game.get("candidate_time_ms", 0))
        second_time += int(game.get("control_time_ms", 0))
    if any(len(values) != 2 for values in pairs.values()):
        raise RuntimeError(f"incomplete opening pair in {path}")
    if expected_games is not None and len(games) != expected_games:
        raise RuntimeError(f"expected {expected_games} games in {path}; found {len(games)}")
    pair_scores = [sum(values) / 2 for values in pairs.values()]
    mean = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
    if len(pair_scores) > 1:
        variance = sum((value - mean) ** 2 for value in pair_scores) / (len(pair_scores) - 1)
        half = 1.96 * math.sqrt(variance / len(pair_scores))
    else:
        half = 0.5
    summary = {
        "first": first["name"], "first_hash": first["config_hash"],
        "second": second["name"], "second_hash": second["config_hash"],
        "games": len(games), "complete_pairs": len(pair_scores),
        "first_outcomes": dict(outcomes), "termination_reasons": dict(reasons),
        "first_points": sum(POINTS[key] * value for key, value in outcomes.items()),
        "first_score": mean,
        "ci95": [max(0.0, mean - half), min(1.0, mean + half)],
        "first_nodes": first_nodes, "second_nodes": second_nodes,
        "node_ratio": first_nodes / second_nodes if second_nodes else None,
        "first_time_ms": first_time, "second_time_ms": second_time,
        "first_illegal_games": reasons.get("candidate_illegal", 0),
        "second_illegal_games": reasons.get("control_illegal", 0),
    }
    if require_manifest:
        manifest_path = Path(str(path) + ".manifest.json")
        summary["artifacts"] = {
            "games": {"path": str(path), **fingerprint(path)},
            "manifest": {"path": str(manifest_path), **fingerprint(manifest_path)},
            "run_fingerprint": run_fingerprint,
        }
    return summary


def rank_control(summaries: list[dict], profiles: dict[str, dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        profile = profiles[summary["first_hash"]]
        outcomes = summary["first_outcomes"]
        rows.append({
            **summary,
            "selection_mean_wdl_loss": profile["source_selection_metrics"]["mean_wdl_loss"],
            "selection_node_ratio": profile["source_selection_metrics"]["node_ratio"],
            "wins": outcomes.get("win", 0), "draws": outcomes.get("draw", 0),
            "losses": outcomes.get("loss", 0),
            "eligible": summary["first_illegal_games"] == 0,
        })
    rows.sort(key=lambda row: (not row["eligible"], -row["first_score"], -row["ci95"][0],
                               row["selection_mean_wdl_loss"],
                               row["selection_node_ratio"], row["first_hash"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def require_legal_match(summary: dict, context: str) -> None:
    if summary["first_illegal_games"] or summary["second_illegal_games"]:
        raise RuntimeError(f"illegal engine move in {context}")


def count_all_games(run_dir: Path) -> int:
    paths = [*sorted((run_dir / "r1" / "matches").glob("*.jsonl")),
             *sorted((run_dir / "r2" / "matches").glob("*.jsonl")),
             run_dir / "r3" / "round_robin.jsonl",
             run_dir / "final" / "winner_vs_production.jsonl"]
    total = 0
    for path in paths:
        if not path.exists():
            continue
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            data = data[:data.rfind(b"\n") + 1]
        for line in data.splitlines():
            if not line.strip():
                continue
            try:
                total += json.loads(line).get("kind") == "game"
            except json.JSONDecodeError:
                continue
    return total


def update_status(run_dir: Path, **values: Any) -> None:
    current = json.loads((run_dir / "status.json").read_text())
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    current["completed_games"] = count_all_games(run_dir)
    atomic_json(run_dir / "status.json", current)


def run_match(contract: dict, run_dir: Path, stage: str,
              profiles: list[dict], output: Path) -> dict:
    expected = contract["rounds"][stage]["games_per_match"]
    if output.exists():
        partial_games = read_games(output, repair_partial_tail=True)
        if len(partial_games) > expected:
            raise RuntimeError(f"too many games in {output}")
        if len(partial_games) == expected:
            return paired_summary(output, profiles[0], profiles[1], expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(contract, stage, profiles, output)
    log_path = output.with_suffix(".log")
    update_status(run_dir, state="running", phase=stage,
                  current_match=[profile["config_hash"] for profile in profiles])
    with log_path.open("ab") as log:
        log.write(("command=" + json.dumps(command) + "\n").encode())
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=log,
                                   start_new_session=True)
        try:
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    update_status(run_dir, state="running", phase=stage,
                                  current_match=[profile["config_hash"] for profile in profiles])
        except BaseException:
            terminate_process_group(process)
            raise
        if code != 0:
            raise RuntimeError(f"gauntlet failed ({code}); see {log_path}")
    return paired_summary(output, profiles[0], profiles[1], expected)


def terminate_process_group(process: subprocess.Popen) -> None:
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


def control_round(contract: dict, run_dir: Path, stage: str,
                  entrants: list[dict], advance: int) -> list[dict]:
    control = contract["production_control"]
    summaries = []
    for entrant in entrants:
        output = run_dir / stage / "matches" / f"{entrant['config_hash']}.jsonl"
        result = run_match(contract, run_dir, stage, [entrant, control], output)
        if result["second_illegal_games"]:
            raise RuntimeError(f"production made an illegal move in {stage}")
        summaries.append(result)
    by_hash = {profile["config_hash"]: profile for profile in entrants}
    ranking = rank_control(summaries, by_hash)
    summary = {"kind": "common_control_round", "stage": stage,
               "ranking": ranking, "advance": advance}
    write_once_json(run_dir / stage / "summary.json", summary)
    selected = [row["first_hash"] for row in ranking[:advance]]
    if any(not row["eligible"] for row in ranking[:advance]):
        raise RuntimeError(f"fewer than {advance} legal candidates remain in {stage}")
    write_once_json(run_dir / stage / "selection.json", {
        "stage": stage, "selected_hashes": selected,
        "ranking_rule": contract["protocol"]["r1_r2_ranking"],
    })
    return [by_hash[value] for value in selected]


def round_robin(contract: dict, run_dir: Path, entrants: list[dict],
                r2_ranking: dict[str, float]) -> dict:
    output = run_dir / "r3" / "round_robin.jsonl"
    expected = contract["rounds"]["r3"]["games_per_match"]
    if output.exists():
        read_games(output, repair_partial_tail=True)
    total_expected = 3 * expected
    if len(read_games(output)) != total_expected:
        command = build_command(contract, "r3", entrants, output)
        log_path = output.with_suffix(".log")
        output.parent.mkdir(parents=True, exist_ok=True)
        update_status(run_dir, state="running", phase="r3",
                      current_match=[p["config_hash"] for p in entrants])
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log,
                                       start_new_session=True)
            try:
                while True:
                    try:
                        returncode = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        update_status(run_dir, state="running", phase="r3",
                                      current_match=[p["config_hash"] for p in entrants])
            except BaseException:
                terminate_process_group(process)
                raise
        if returncode:
            raise RuntimeError(f"round robin failed; see {log_path}")
    games = read_games(output)
    if len(games) != total_expected:
        raise RuntimeError(f"R3 expected {total_expected} games; found {len(games)}")
    manifest = validate_gauntlet_manifest(output, entrants)
    run_fingerprint = manifest["run_fingerprint"]
    if any(game.get("run_fingerprint") != run_fingerprint
           or not game.get("key", "").startswith(run_fingerprint + ":")
           for game in games):
        raise RuntimeError("R3 game/manifest fingerprint mismatch")
    by_name = {profile["name"]: profile for profile in entrants}
    match_games: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for game in games:
        match_games[(game["profile"], game["opponent"])].append(game)
    if len(match_games) != 3:
        raise RuntimeError("R3 did not emit exactly three matchups")
    standings = {p["config_hash"]: {"config_hash": p["config_hash"],
                 "name": p["name"], "points": 0.0, "games": 0,
                 "wins": 0, "draws": 0, "losses": 0,
                 "opponent_points": {}} for p in entrants}
    matchup_summaries = []
    for (first_name, second_name), values in match_games.items():
        temporary = run_dir / "r3" / "matches" / f"{first_name}_vs_{second_name}.jsonl"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(canonical_bytes(game) + b"\n" for game in values)
        atomic_write(temporary, payload)
        first, second = by_name[first_name], by_name[second_name]
        summary = paired_summary(temporary, first, second, expected,
                                 require_manifest=False)
        require_legal_match(
            summary, f"R3 {first['config_hash']} vs {second['config_hash']}")
        matchup_summaries.append(summary)
        first_row, second_row = standings[first["config_hash"]], standings[second["config_hash"]]
        first_points = summary["first_points"]
        second_points = expected - first_points
        first_row["points"] += first_points
        first_row["games"] += expected
        second_row["points"] += second_points
        second_row["games"] += expected
        first_row["opponent_points"][second["config_hash"]] = first_points
        second_row["opponent_points"][first["config_hash"]] = second_points
        outcomes = summary["first_outcomes"]
        first_row["wins"] += outcomes.get("win", 0)
        first_row["draws"] += outcomes.get("draw", 0)
        first_row["losses"] += outcomes.get("loss", 0)
        second_row["wins"] += outcomes.get("loss", 0)
        second_row["draws"] += outcomes.get("draw", 0)
        second_row["losses"] += outcomes.get("win", 0)
    rows = list(standings.values())
    for row in rows:
        row["score"] = row["points"] / row["games"]
        row["r2_control_score"] = r2_ranking[row["config_hash"]]
    point_groups = Counter(row["points"] for row in rows)
    for row in rows:
        tied = [other["config_hash"] for other in rows
                if other["points"] == row["points"]]
        row["two_way_head_to_head_points"] = (
            row["opponent_points"].get(tied[1] if tied[0] == row["config_hash"] else tied[0], 0.0)
            if point_groups[row["points"]] == 2 else 0.0)
    rows.sort(key=lambda row: (-row["score"], -row["two_way_head_to_head_points"],
                               -row["r2_control_score"], row["config_hash"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    result = {"kind": "round_robin", "stage": "r3", "ranking": rows,
              "matchups": matchup_summaries, "winner_hash": rows[0]["config_hash"],
              "artifacts": {
                  "games": {"path": str(output), **fingerprint(output)},
                  "manifest": {"path": str(Path(str(output) + '.manifest.json')),
                               **fingerprint(Path(str(output) + '.manifest.json'))},
                  "run_fingerprint": run_fingerprint,
              }}
    write_once_json(run_dir / "r3" / "summary.json", result)
    return result


def execute(contract: dict, run_dir: Path) -> dict:
    verify_contract(contract)
    if (run_dir / "DONE").exists():
        return json.loads((run_dir / "summary.json").read_text())
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    def interrupted(signum, _frame):
        raise InterruptedError(f"runner interrupted by signal {signum}")
    for signum in previous_handlers:
        signal.signal(signum, interrupted)
    atomic_write(run_dir / "RUNNING", (str(os.getpid()) + "\n").encode())
    (run_dir / "FAILED").unlink(missing_ok=True)
    (run_dir / "INTERRUPTED").unlink(missing_ok=True)
    try:
        r1 = control_round(contract, run_dir, "r1", contract["entrants"], 8)
        r2 = control_round(contract, run_dir, "r2", r1, 3)
        r2_summary = json.loads((run_dir / "r2" / "summary.json").read_text())
        r2_scores = {row["first_hash"]: row["first_score"]
                     for row in r2_summary["ranking"]}
        r3 = round_robin(contract, run_dir, r2, r2_scores)
        by_hash = {p["config_hash"]: p for p in contract["entrants"]}
        winner = by_hash[r3["winner_hash"]]
        final_path = run_dir / "final" / "winner_vs_production.jsonl"
        confirmation = run_match(contract, run_dir, "final",
                                 [winner, contract["production_control"]], final_path)
        require_legal_match(confirmation, "final confirmation")
        result = {
            "kind": "nnue_v43_selfplay_race_summary", "schema_version": 1,
            "winner_hash": winner["config_hash"], "winner_name": winner["name"],
            "confirmation_vs_production": confirmation,
            "completed_games": count_all_games(run_dir),
            "expected_total_games": EXPECTED_TOTAL_GAMES,
            "automatic_promotion": False,
            "decision": "completed_for_human_promotion_review",
        }
        if result["completed_games"] != EXPECTED_TOTAL_GAMES:
            raise RuntimeError("completed game count disagrees with frozen protocol")
        write_once_json(run_dir / "summary.json", result)
        update_status(run_dir, state="complete", phase="done", current_match=None)
        atomic_write(run_dir / "DONE", b"\n")
        return result
    except BaseException as error:
        interrupted_run = isinstance(error, (InterruptedError, KeyboardInterrupt))
        state = "interrupted" if interrupted_run else "failed"
        update_status(run_dir, state=state, error=str(error))
        marker = "INTERRUPTED" if interrupted_run else "FAILED"
        atomic_write(run_dir / marker, (str(error) + "\n").encode())
        raise
    finally:
        (run_dir / "RUNNING").unlink(missing_ok=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def planned_commands(contract: dict, run_dir: Path) -> dict:
    control = contract["production_control"]
    return {
        "r1": [build_command(contract, "r1", [profile, control],
                 run_dir / "r1" / "matches" / f"{profile['config_hash']}.jsonl")
               for profile in contract["entrants"]],
        "later_rounds": "entrants are frozen from completed prior-round ranking",
        "expected_total_games": EXPECTED_TOTAL_GAMES,
    }


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=repo / "logs" /
                        "nnue_v43_all_prunes_balanced_baseline_12h_20260831_030546")
    parser.add_argument("--binary", type=Path, default=repo / "logs" /
                        "nnue_v43_all_prunes_balanced_baseline_12h_20260831_030546" /
                        "bin" / "nnue_v43_time_gauntlet")
    parser.add_argument("--model", type=Path, default=repo / "models" /
                        "quantized_scale_grid" /
                        "old_score_huber200_lr_sweep_then_5ep_20260724_142758" /
                        "best" / "phase_quantized_nnue.bin")
    parser.add_argument("--book", type=Path, default=repo / "logs" /
                        "nnue_v40_qsee_tune_20260817_005158" /
                        "stockfish_balanced_openings_10ply_2400.txt")
    parser.add_argument("--pool-spec", type=Path,
                        default=repo / "tools" / "nnue_v43_selfplay_race_pool.json")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    contract_path = args.run_dir / "contract.json"
    if contract_path.exists():
        contract = load_contract(args.run_dir)
        frozen_runner = Path(contract["frozen_files"]["runner"]["path"])
        if Path(__file__).resolve() != frozen_runner.resolve():
            os.execv(sys.executable, [sys.executable, str(frozen_runner), *sys.argv[1:]])
    if args.status:
        status_path = args.run_dir / "status.json"
        if not status_path.exists():
            raise RuntimeError("race has not been initialized")
        status = json.loads(status_path.read_text())
        status["live_completed_games"] = count_all_games(args.run_dir)
        print(json.dumps(status, indent=2, sort_keys=True) + "\n", end="")
        return
    lock_path = args.run_dir / ".runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another self-play race runner holds the lock") from error
        contract = initialize(args)
        verify_contract(contract)
        write_once_json(args.run_dir / "plan.json", planned_commands(contract, args.run_dir))
        if args.init_only:
            print(json.dumps({"state": "initialized", "run_dir": str(args.run_dir),
                              "expected_total_games": EXPECTED_TOTAL_GAMES}))
            return
        print(json.dumps(execute(contract, args.run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
