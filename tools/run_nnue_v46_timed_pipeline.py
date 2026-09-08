#!/usr/bin/env python3
"""Resumable teacher-filtered, timed-production pruning pipeline.

The strict teacher is used only to generate and safety-filter candidates.
Every promotion decision is then made by serial, color-reversed, equal-time
matches against the deployed production profile on fresh common openings.
"""

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
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from tools import run_nnue_v43_selfplay_race as race
    from tools import tune_nnue_v45_pareto as teacher
else:
    import run_nnue_v43_selfplay_race as race  # type: ignore
    import tune_nnue_v45_pareto as teacher  # type: ignore


SCHEMA_VERSION = 1
EXPERIMENT = "v46-teacher-filtered-timed-production-v1"
PRODUCTION_HASH = teacher.DEPLOYED_V45_HASH
SETTINGS = (
    {"name": "timed_screen", "pool_limit": 24, "advance": 8,
     "openings": 8, "games_per_candidate": 16,
     "base_ms": 1_000, "increment_ms": 10, "seed": 2026090511},
    {"name": "timed_selection", "entrants": 8, "advance": 3,
     "openings": 32, "games_per_candidate": 64,
     "base_ms": 3_000, "increment_ms": 30, "seed": 2026090522},
    {"name": "timed_gauntlet", "entrants": 3, "advance": 1,
     "openings": 64, "games_per_candidate": 128,
     "base_ms": 7_000, "increment_ms": 70, "seed": 2026090533},
)
CONFIRMATION = {
    "name": "confirmation", "entrants": 1, "advance": 0,
    "openings": 300, "games_per_candidate": 600,
    "base_ms": 10_000, "increment_ms": 100, "seed": 2026090544,
}


class PipelinePaused(RuntimeError):
    pass


class PipelineInterrupted(KeyboardInterrupt):
    pass


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


def schedule(smoke_test: bool) -> tuple[tuple[dict, ...], dict]:
    if not smoke_test:
        return SETTINGS, CONFIRMATION
    tiny = (
        {**SETTINGS[0], "pool_limit": 1, "advance": 1,
         "openings": 1, "games_per_candidate": 2},
        *({**setting, "entrants": 1, "advance": 1,
           "openings": 1, "games_per_candidate": 2}
          for setting in SETTINGS[1:]),
    )
    return tiny, {
        **CONFIRMATION, "openings": 1, "games_per_candidate": 2,
    }


def excluded_openings(path: Path | None) -> tuple[list[str], list[dict]]:
    if path is None or not path.exists():
        return [], []
    files = sorted(path.glob("*.txt"))
    lines: list[str] = []
    for item in files:
        lines.extend(
            line.strip() for line in item.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return lines, [fingerprint(item) for item in files]


def split_openings(
    master: Path, run_dir: Path, settings: Sequence[dict],
    confirmation: dict, excluded: Sequence[str], seed: int,
) -> dict[str, dict]:
    lines = [
        line.strip() for line in master.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != len(set(lines)):
        raise RuntimeError("master opening book contains duplicates")
    excluded_set = set(excluded)
    lines = [line for line in lines if line not in excluded_set]
    random.Random(seed).shuffle(lines)
    requested = [
        *( (item["name"], item["openings"]) for item in settings ),
        (confirmation["name"], confirmation["openings"]),
    ]
    required = sum(count for _, count in requested)
    if len(lines) < required:
        raise RuntimeError(
            f"opening book has {len(lines)} unused positions; needs {required}"
        )
    result = {}
    offset = 0
    for name, count in requested:
        path = run_dir / "openings" / f"{name}.txt"
        data = "\n".join(lines[offset:offset + count]) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != data:
            raise RuntimeError(f"immutable opening split changed: {name}")
        if not path.exists():
            path.write_text(data)
        result[name] = {**fingerprint(path), "count": count}
        offset += count
    return result


def initialize_manifest(args: argparse.Namespace) -> dict:
    path = args.run_dir / "manifest.json"
    settings, confirmation = schedule(args.smoke_test)
    excluded, excluded_files = excluded_openings(args.exclude_openings_dir)
    books = split_openings(
        args.book, args.run_dir, settings, confirmation, excluded,
        args.seed ^ 0x4F50454E,
    )
    expected = {
        "kind": "nnue_v46_timed_pipeline_contract",
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "runner": fingerprint(Path(__file__)),
        "teacher_runner": fingerprint(Path(teacher.__file__)),
        "legacy_teacher_runner": fingerprint(Path(teacher.legacy.__file__)),
        "teacher_infra_runner": fingerprint(Path(teacher.legacy.infra.__file__)),
        "race_runner": fingerprint(Path(race.__file__)),
        "source_results": fingerprint(args.source_run / "results.jsonl"),
        "source_summary": fingerprint(args.source_run / "summary.json"),
        "evaluation_cache_runs": [
            {
                "path": str(path),
                "summary": fingerprint(path / "summary.json"),
            }
            for path in args.evaluation_cache_runs
        ],
        "binary": fingerprint(args.binary),
        "model": fingerprint(args.model),
        "master_book": fingerprint(args.book),
        "excluded_opening_files": excluded_files,
        "books": books,
        "production_hash": PRODUCTION_HASH,
        "teacher": {
            "mutation_count": args.mutation_count,
            "workers": args.workers,
            "duration_sec": args.teacher_duration_sec,
            "hard_node_ratio": args.hard_node_ratio,
            "local_mutation_probability": 0.60,
            "cross_mutation_probability": 0.20,
            "multi_block_restart_probability": 0.20,
            "strict_teacher_is_filter_only": True,
        },
        "settings": list(settings),
        "confirmation": confirmation,
        "protocol": {
            "serial_timed_matches": True,
            "common_control": "deployed_v45",
            "paired_color_reversed": True,
            "fresh_disjoint_openings": True,
            "excluded_prior_tournament_openings": bool(excluded_files),
            "tt_mb": 64, "overhead_ms": 20, "max_plies": 200,
            "automatic_promotion": False,
            "candidate_ranking": [
                "legal_desc", "timed_score_desc", "paired_ci_lower_desc",
                "selection_node_ratio_asc", "selection_wdl_loss_asc",
                "config_hash_asc",
            ],
        },
    }
    if path.exists():
        actual = json.loads(path.read_text())
        if actual != expected:
            raise RuntimeError("immutable pipeline manifest disagrees")
        return actual
    atomic_json(path, expected)
    return expected


def verify_manifest(manifest: dict) -> None:
    for name in (
        "runner", "teacher_runner", "legacy_teacher_runner",
        "teacher_infra_runner", "race_runner", "source_results",
        "source_summary", "binary", "model", "master_book",
    ):
        if fingerprint(Path(manifest[name]["path"])) != manifest[name]:
            raise RuntimeError(f"frozen pipeline artifact changed: {name}")
    for entry in [
        *manifest["excluded_opening_files"], *manifest["books"].values()
    ]:
        expected = {key: entry[key] for key in ("path", "size", "sha256")}
        if fingerprint(Path(entry["path"])) != expected:
            raise RuntimeError(f"frozen opening artifact changed: {entry['path']}")
    for cache_run in manifest["evaluation_cache_runs"]:
        if fingerprint(Path(cache_run["summary"]["path"])) != cache_run["summary"]:
            raise RuntimeError(
                f"frozen evaluation cache changed: {cache_run['path']}"
            )


def teacher_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=args.run_dir / "teacher", source_run=args.source_run,
        mutation_count=args.mutation_count, workers=args.workers,
        duration_sec=args.teacher_duration_sec, seed=args.seed,
        baseline_profile="v45-deployed", hard_node_ratio=args.hard_node_ratio,
        minimum_disk_free_bytes=args.minimum_disk_free_bytes,
        local_mutation_probability=0.60,
        cross_mutation_probability=0.20,
        evaluation_cache_runs=args.evaluation_cache_runs,
        smoke_test=args.smoke_test,
    )


def run_teacher(args: argparse.Namespace) -> None:
    configured = teacher_args(args)
    configured.run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = configured.run_dir / ".tuner.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        teacher._main_locked(configured)
    if not (configured.run_dir / "DONE").exists():
        marker = configured.run_dir / "PAUSED"
        reason = (
            json.loads(marker.read_text()).get("reason", "teacher paused")
            if marker.exists() else "teacher did not complete"
        )
        raise PipelinePaused(reason)


def normalized_quality(entries: Sequence[dict]) -> dict[str, float]:
    losses = [entry["mean_wdl_loss"] for entry in entries]
    nodes = [entry["production_node_ratio"] for entry in entries]
    loss_span = max(losses) - min(losses)
    node_span = max(nodes) - min(nodes)
    return {
        entry["config_hash"]: 0.5 * (
            (entry["mean_wdl_loss"] - min(losses)) / (loss_span or 1.0)
            + (entry["production_node_ratio"] - min(nodes)) /
              (node_span or 1.0)
        )
        for entry in entries
    }


def config_distance(left: dict, right: dict) -> float:
    first = left["config"]["selective"]
    second = right["config"]["selective"]
    keys = sorted(first)
    return sum(first[key] != second[key] for key in keys) / len(keys)


def choose_diverse_pool(entries: Sequence[dict], limit: int) -> list[dict]:
    if len(entries) <= limit:
        return sorted(entries, key=lambda item: item["config_hash"])
    quality = normalized_quality(entries)
    raw_entries = [entry["evaluation"] for entry in entries]
    pareto_hashes = {
        entry["config_hash"] for entry in teacher.legacy.frontier(raw_entries)
    }
    candidates = list(entries)
    selected: list[dict] = []
    for key in ("mean_wdl_loss", "production_node_ratio"):
        choice = min(candidates, key=lambda item: (item[key], item["config_hash"]))
        if choice not in selected:
            selected.append(choice)
        if len(selected) == limit:
            break
    while len(selected) < limit:
        remaining = [item for item in candidates if item not in selected]
        choice = max(remaining, key=lambda item: (
            min(config_distance(item, prior) for prior in selected)
            + (0.20 if item["config_hash"] in pareto_hashes else 0.0)
            + 0.15 * (1.0 - quality[item["config_hash"]]),
            -quality[item["config_hash"]], item["config_hash"],
        ))
        selected.append(choice)
    return sorted(selected, key=lambda item: item["config_hash"])


def build_profiles(args: argparse.Namespace, manifest: dict) -> tuple[list[dict], dict]:
    path = args.run_dir / "candidate_pool.json"
    selection_dir = args.run_dir / "teacher" / "evaluations" / "selection"
    baseline_path = selection_dir / f"{PRODUCTION_HASH}.json"
    baseline = json.loads(baseline_path.read_text())
    baseline_nodes = float(baseline["result"]["candidate_nodes"])
    aspiration = baseline["config"]["aspiration"]
    entries = []
    for item in sorted(selection_dir.glob("*.json")):
        entry = json.loads(item.read_text())
        candidate = teacher.legacy.Candidate.from_dict(entry["config"])
        if candidate.hash != entry["config_hash"]:
            raise RuntimeError(f"selection identity mismatch: {item}")
        if candidate.hash == PRODUCTION_HASH:
            continue
        node_ratio = float(entry["result"]["candidate_nodes"]) / baseline_nodes
        if node_ratio > manifest["teacher"]["hard_node_ratio"]:
            continue
        entries.append({
            "config_hash": candidate.hash, "config": entry["config"],
            "mean_wdl_loss": float(entry["result"]["mean_wdl_loss"]),
            "production_node_ratio": node_ratio,
            "critical_mistakes": int(entry["result"]["critical_mistakes"]),
            "evaluation": entry,
        })
    limit = manifest["settings"][0]["pool_limit"]
    if len(entries) < min(3, limit):
        raise RuntimeError(
            f"teacher produced only {len(entries)} eligible challengers"
        )
    selected = choose_diverse_pool(entries, min(limit, len(entries)))
    profiles = [{
        "role": "challenger", "name": f"candidate_{entry['config_hash'][:12]}",
        "config_hash": entry["config_hash"], "config": entry["config"],
        "source_selection_metrics": {
            "mean_wdl_loss": entry["mean_wdl_loss"],
            "node_ratio": entry["production_node_ratio"],
            "critical_mistakes": entry["critical_mistakes"],
        },
    } for entry in selected]
    production = {
        "role": "production", "name": f"production_{PRODUCTION_HASH[:12]}",
        "config_hash": PRODUCTION_HASH, "config": baseline["config"],
        "source_selection_metrics": {
            "mean_wdl_loss": float(baseline["result"]["mean_wdl_loss"]),
            "node_ratio": 1.0,
            "critical_mistakes": int(baseline["result"]["critical_mistakes"]),
        },
    }
    payload = {
        "kind": "nnue_v46_timed_candidate_pool",
        "schema_version": SCHEMA_VERSION,
        "eligible_count": len(entries), "selected_count": len(profiles),
        "selection_policy": "teacher_pareto_bonus_quality_weighted_maximin",
        "profiles": profiles, "production": production,
        "aspiration": aspiration,
    }
    write_once_json(path, payload)
    return profiles, production


def count_games(run_dir: Path) -> int:
    total = 0
    for path in run_dir.glob("timed/**/*.jsonl"):
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            boundary = data.rfind(b"\n")
            data = data[:boundary + 1] if boundary >= 0 else b""
        for line in data.splitlines():
            try:
                total += json.loads(line).get("kind") == "game"
            except json.JSONDecodeError:
                continue
    return total


def update_status(run_dir: Path, **values: object) -> None:
    current = {}
    path = run_dir / "status.json"
    if path.exists():
        current = json.loads(path.read_text())
    current.update(values)
    current["timed_games"] = count_games(run_dir)
    current["updated_at"] = utc_now()
    atomic_json(path, current)


def build_command(
    manifest: dict, setting: dict, candidate: dict, production: dict,
    output: Path,
) -> list[str]:
    profiles = [candidate, production]
    command = [
        manifest["binary"]["path"],
        "--book", manifest["books"][setting["name"]]["path"],
        "--model", manifest["model"]["path"],
        "--output", str(output),
        "--openings", str(setting["openings"]),
        "--base-ms", str(setting["base_ms"]),
        "--increment-ms", str(setting["increment_ms"]),
        "--overhead-ms", str(manifest["protocol"]["overhead_ms"]),
        "--max-plies", str(manifest["protocol"]["max_plies"]),
        "--tt-mb", str(manifest["protocol"]["tt_mb"]),
        "--seed", str(setting["seed"]), "--round-robin",
    ]
    for profile in profiles:
        command.extend(("--profile", race.profile_spec(profile)))
    aspiration = candidate["config"]["aspiration"]
    for profile in profiles:
        command.extend((
            "--aspiration-profile", race.aspiration_spec(profile, aspiration)
        ))
        command.extend(("--twofold-search-profile", profile["name"]))
    return command


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


def run_match(
    args: argparse.Namespace, manifest: dict, setting: dict,
    candidate: dict, production: dict,
) -> dict:
    output = (
        args.run_dir / "timed" / setting["name"] / "matches" /
        f"{candidate['config_hash']}.jsonl"
    )
    expected = setting["games_per_candidate"]
    if output.exists():
        games = race.read_games(output, repair_partial_tail=True)
        if len(games) > expected:
            raise RuntimeError(f"too many games in {output}")
        if len(games) == expected:
            result = race.paired_summary(
                output, candidate, production, expected
            )
            resource_path = Path(str(output) + ".resource.json")
            result["resource"] = (
                json.loads(resource_path.read_text())
                if resource_path.exists() else None
            )
            return result
    teacher.verify_disk_headroom(args.run_dir, args.minimum_disk_free_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(manifest, setting, candidate, production, output)
    log_path = output.with_suffix(".log")
    update_status(
        args.run_dir, state="RUNNING", stage=setting["name"],
        current_candidate=candidate["config_hash"],
        completed_candidate_games=len(race.read_games(output)),
        expected_candidate_games=expected,
    )
    started = time.monotonic()
    with log_path.open("ab") as log:
        log.write(("command=" + json.dumps(command) + "\n").encode())
        log.flush()
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        )
        try:
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    update_status(
                        args.run_dir, state="RUNNING", stage=setting["name"],
                        current_candidate=candidate["config_hash"],
                        completed_candidate_games=len(race.read_games(output)),
                        expected_candidate_games=expected,
                    )
        except BaseException:
            terminate_group(process)
            raise
    if code:
        raise RuntimeError(f"timed gauntlet failed ({code}); see {log_path}")
    result = race.paired_summary(output, candidate, production, expected)
    resource = {"wall_sec": time.monotonic() - started, "measured_at": utc_now()}
    write_once_json(Path(str(output) + ".resource.json"), resource)
    result["resource"] = resource
    return result


def ranked_rows(rows: Sequence[dict], profiles: dict[str, dict]) -> list[dict]:
    result = []
    for row in rows:
        profile = profiles[row["first_hash"]]
        result.append({
            **row,
            "eligible": row["first_illegal_games"] == 0,
            "selection_mean_wdl_loss": profile["source_selection_metrics"][
                "mean_wdl_loss"
            ],
            "selection_node_ratio": profile["source_selection_metrics"][
                "node_ratio"
            ],
        })
        if row["second_illegal_games"]:
            raise RuntimeError("production made an illegal move")
    result.sort(key=lambda row: (
        not row["eligible"], -row["first_score"], -row["ci95"][0],
        row["selection_node_ratio"], row["selection_mean_wdl_loss"],
        row["first_hash"],
    ))
    for index, row in enumerate(result, 1):
        row["rank"] = index
    return result


def run_timed_pipeline(
    args: argparse.Namespace, manifest: dict,
    profiles: list[dict], production: dict,
) -> dict:
    by_hash = {profile["config_hash"]: profile for profile in profiles}
    active = profiles
    stage_summaries = []
    for setting in manifest["settings"]:
        if not active:
            raise RuntimeError(f"{setting['name']} has no entrants")
        rows = [
            run_match(args, manifest, setting, candidate, production)
            for candidate in active
        ]
        ranking = ranked_rows(rows, by_hash)
        advance = min(setting["advance"], len(ranking))
        selected = ranking[:advance]
        if any(not row["eligible"] for row in selected):
            raise RuntimeError(f"{setting['name']} has too few legal candidates")
        selected_hashes = [row["first_hash"] for row in selected]
        summary = {
            "kind": "nnue_v46_timed_common_control_stage",
            "schema_version": SCHEMA_VERSION,
            "stage": setting["name"], "setting": setting,
            "production_hash": production["config_hash"],
            "ranking": ranking, "selected_hashes": selected_hashes,
        }
        write_once_json(
            args.run_dir / "timed" / setting["name"] / "summary.json",
            summary,
        )
        stage_summaries.append(summary)
        active = [by_hash[value] for value in selected_hashes]
        print(
            f"{setting['name']} complete entrants={len(ranking)} "
            f"advance={advance} leader={selected_hashes[0][:12]}",
            flush=True,
        )
    winner = active[0]
    setting = manifest["confirmation"]
    confirmation = run_match(args, manifest, setting, winner, production)
    confirmation_summary = {
        "kind": "nnue_v46_timed_confirmation",
        "schema_version": SCHEMA_VERSION,
        "winner_hash": winner["config_hash"],
        "production_hash": production["config_hash"],
        "candidate_score": confirmation["first_score"],
        "candidate_points": confirmation["first_points"],
        "games": confirmation["games"], "ci95": confirmation["ci95"],
        "node_ratio": confirmation["node_ratio"],
        "result": confirmation, "automatic_promotion": False,
    }
    write_once_json(
        args.run_dir / "timed" / "confirmation" / "summary.json",
        confirmation_summary,
    )
    summary_path = args.run_dir / "summary.json"
    completed_at = (
        json.loads(summary_path.read_text())["completed_at"]
        if summary_path.exists() else utc_now()
    )
    result = {
        "kind": "nnue_v46_timed_pipeline_summary",
        "schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT,
        "completed_at": completed_at,
        "winner_hash": winner["config_hash"],
        "winner_config": winner["config"],
        "production_hash": production["config_hash"],
        "stages": [
            {"stage": item["stage"], "selected_hashes": item["selected_hashes"]}
            for item in stage_summaries
        ],
        "timed_games": count_games(args.run_dir),
        "confirmation": confirmation_summary,
        "decision": "complete_for_human_review",
        "automatic_promotion": False,
    }
    write_once_json(summary_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--evaluation-cache-run", dest="evaluation_cache_runs",
        action="append", type=Path, default=[],
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--exclude-openings-dir", type=Path)
    parser.add_argument("--mutation-count", type=int, default=480)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--teacher-duration-sec", type=int, default=43_200)
    parser.add_argument("--hard-node-ratio", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--minimum-disk-free-bytes", type=int, default=5 * 1024**3
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if min(
        args.mutation_count, args.workers, args.teacher_duration_sec,
        args.hard_node_ratio,
    ) <= 0:
        parser.error("positive mutation, worker, duration, and node settings required")
    if args.hard_node_ratio <= 1.0:
        parser.error("hard node ratio must exceed 1.0")
    if args.smoke_test:
        args.mutation_count = min(args.mutation_count, 3)
        args.workers = min(args.workers, 2)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = args.run_dir.resolve()
    args.source_run = args.source_run.resolve(strict=True)
    args.evaluation_cache_runs = [
        path.resolve(strict=True) for path in args.evaluation_cache_runs
    ]
    args.binary = args.binary.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    args.book = args.book.resolve(strict=True)
    if args.exclude_openings_dir is not None:
        args.exclude_openings_dir = args.exclude_openings_dir.resolve(strict=True)
    return args


def handle_signal(_signum: int, _frame: object) -> None:
    raise PipelineInterrupted("pipeline interrupted")


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, handle_signal)
    lock_path = args.run_dir / ".pipeline.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another pipeline runner owns this directory") from error
        if (args.run_dir / "DONE").exists():
            print((args.run_dir / "summary.json").read_text(), flush=True)
            return
        manifest = initialize_manifest(args)
        verify_manifest(manifest)
        (args.run_dir / "RUNNING").write_text(f"{os.getpid()}\n")
        for marker in ("PAUSED", "FAILED"):
            (args.run_dir / marker).unlink(missing_ok=True)
        update_status(args.run_dir, state="RUNNING", stage="teacher")
        try:
            run_teacher(args)
            profiles, production = build_profiles(args, manifest)
            result = run_timed_pipeline(
                args, manifest, profiles, production
            )
            (args.run_dir / "DONE").write_text("\n")
            update_status(
                args.run_dir, state="DONE", stage="complete",
                winner_hash=result["winner_hash"],
            )
            print(json.dumps(result, sort_keys=True), flush=True)
        except (PipelinePaused, PipelineInterrupted, KeyboardInterrupt) as error:
            payload = {
                "state": "PAUSED", "reason": str(error),
                "updated_at": utc_now(),
            }
            atomic_json(args.run_dir / "PAUSED", payload)
            update_status(args.run_dir, **payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
        except BaseException as error:
            payload = {
                "state": "FAILED", "error_type": type(error).__name__,
                "error": str(error), "updated_at": utc_now(),
            }
            atomic_json(args.run_dir / "FAILED", payload)
            update_status(args.run_dir, **payload)
            raise
        finally:
            (args.run_dir / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
