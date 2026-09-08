#!/usr/bin/env python3
"""Resumable two-stage steady-state Pareto tuner for V45 pruning.

Stage 1 mutates uniformly sampled parents from the active tune frontier.  A
cheap screen rejects offspring whose node count exceeds the production V43
baseline by the configured hard ratio.  Surviving offspring are evaluated on
the full tune sample and update the frontier in both directions: dominated
offspring are archived, while an admitted offspring removes every frontier
member it dominates.

Stage 2 evaluates the frozen stage-1 frontier once on a fresh selection split
and constructs a new Pareto frontier.  Selection is never fed back into
mutation, so it remains an overfit check rather than becoming tune data.

The evaluator's immutable V36 strict-root caches are reused for every
candidate.  Aggregate point-Pareto decisions avoid retaining detail sidecars
or bootstrap matrices in memory.  Every batch is planned and decided through
write-once JSON artifacts, making interruption and per-candidate failures
resumable without losing the active frontier.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__:
    from tools.tune import tune_nnue_v43_all_prunes as legacy
else:
    import tune_nnue_v43_all_prunes as legacy  # type: ignore


SCHEMA_VERSION = 1
EXPERIMENT = "v45-steady-state-pareto-prunes-v1"
PRODUCTION_V43_HASH = (
    "98b7732c9587da35554cc274a072a0a5b5f55902aaa77605e78c1ae13e88b4f2"
)
DEPLOYED_V45_HASH = (
    "28c848b51bd93c402e873a0154e1fc61c953efa683898dc4a67d1fecd5a76aa8"
)
DEFAULT_SOURCE_RUN = Path(
    "logs/nnue_v43_all_prunes_balanced_baseline_12h_20260831_030546"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "path": str(resolved),
        "size": resolved.stat().st_size,
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


def production_v43_candidate() -> legacy.Candidate:
    selective = legacy.SelectiveConfig(
        enable_lmr=True,
        lmr_base=0.45,
        lmr_divisor=2.9,
        lmr_min_depth=3,
        lmr_min_move_index=6,
        enable_null_move=True,
        null_move_min_depth=2,
        null_move_reduction=4,
        enable_reverse_futility=True,
        reverse_futility_max_depth=4,
        reverse_futility_base_margin=50,
        reverse_futility_margin_per_depth=100,
        enable_late_move_pruning=True,
        late_move_pruning_max_depth=3,
        late_move_pruning_base=4,
        late_move_pruning_depth_multiplier=1,
        enable_qsearch_see_pruning=True,
        qsearch_see_threshold=-25,
        enable_main_search_see_pruning=False,
        main_search_see_max_depth=5,
        main_search_see_margin_per_depth=100,
    )
    candidate = legacy.Candidate(
        selective, legacy.PRODUCTION_BASELINE_ASPIRATION
    )
    if candidate.hash != PRODUCTION_V43_HASH:
        raise RuntimeError("hard-coded production V43 identity drift")
    return candidate


def deployed_v45_candidate() -> legacy.Candidate:
    """Return the pruning profile provisionally deployed in Heroku v20."""
    selective = replace(
        production_v43_candidate().selective,
        lmr_base=0.6,
        lmr_min_move_index=5,
        late_move_pruning_depth_multiplier=0,
        qsearch_see_threshold=25,
        enable_main_search_see_pruning=True,
    )
    candidate = legacy.Candidate(
        selective, legacy.PRODUCTION_BASELINE_ASPIRATION
    )
    if candidate.hash != DEPLOYED_V45_HASH:
        raise RuntimeError("hard-coded deployed V45 identity drift")
    return candidate


def baseline_candidate(profile: str) -> legacy.Candidate:
    if profile == "v43-original":
        return production_v43_candidate()
    if profile == "v45-deployed":
        return deployed_v45_candidate()
    raise ValueError(f"unknown baseline profile: {profile}")


def point_dominates(left: dict, right: dict) -> bool:
    """Return exact two-objective dominance on one fixed evaluation rung."""
    return legacy.dominates(left, right)


def pareto_insert(
    frontier_entries: Sequence[dict], candidate_entry: dict
) -> tuple[list[dict], str, list[str]]:
    """Insert bidirectionally: reject dominated, remove what child dominates."""
    dominators = [
        entry for entry in frontier_entries
        if point_dominates(entry, candidate_entry)
    ]
    if dominators:
        return list(frontier_entries), "dominated", []
    removed = [
        entry["config_hash"] for entry in frontier_entries
        if point_dominates(candidate_entry, entry)
    ]
    kept = [
        entry for entry in frontier_entries
        if entry["config_hash"] not in set(removed)
    ]
    kept.append(candidate_entry)
    kept.sort(key=lambda entry: entry["config_hash"])
    return kept, "inserted", sorted(removed)


def _neighbor_values(name: str, value: Any) -> list[Any]:
    grid = legacy.GRIDS[name]
    index = grid.index(value)
    return [
        grid[index + delta]
        for delta in (-1, 1)
        if 0 <= index + delta < len(grid)
    ]


def block_mutations(
    config: legacy.SelectiveConfig, block: str
) -> list[tuple[legacy.SelectiveConfig, str]]:
    """Enumerate one-grid-step children in one logical pruning block."""
    parent = config.normalized()
    values = asdict(parent)
    enabled_name = legacy.ENABLE_FIELD[block]
    mutations: list[tuple[legacy.SelectiveConfig, str]] = []
    toggled = replace(parent, **{enabled_name: not values[enabled_name]}).normalized()
    mutations.append((toggled, f"{enabled_name}:toggle"))
    if not values[enabled_name]:
        enabled = replace(parent, **{enabled_name: True}).normalized()
        values = asdict(enabled)
        parent = enabled
    for name in legacy.BLOCKS[block]:
        if name == enabled_name:
            continue
        for neighbor in _neighbor_values(name, values[name]):
            mutations.append((
                replace(parent, **{name: neighbor}).normalized(),
                f"{name}:{values[name]}->{neighbor}",
            ))
    unique: dict[tuple, tuple[legacy.SelectiveConfig, str]] = {}
    for child, description in mutations:
        unique.setdefault(legacy.behavior_signature(child), (child, description))
    return list(unique.values())


def mutate_parent(
    parent: legacy.SelectiveConfig,
    rng: random.Random,
    local_probability: float = 0.85,
    cross_probability: float = 0.15,
) -> tuple[legacy.SelectiveConfig, str]:
    """Mix local steps, cross-block steps, and multi-block restarts."""
    blocks = tuple(legacy.BLOCKS)
    draw = rng.random()
    if draw < local_probability:
        block = rng.choice(blocks)
        child, detail = rng.choice(block_mutations(parent, block))
        return child, f"single:{block}:{detail}"
    if draw < local_probability + cross_probability:
        first, second = rng.sample(blocks, 2)
        child, first_detail = rng.choice(block_mutations(parent, first))
        child, second_detail = rng.choice(block_mutations(child, second))
        return child, f"cross:{first}:{first_detail}|{second}:{second_detail}"
    restart_blocks = rng.sample(blocks, rng.randint(2, 4))
    child = legacy.random_selective_config(
        rng, mutable_blocks=restart_blocks, base=parent.normalized()
    )
    return child, f"restart:{'+'.join(sorted(restart_blocks))}"


def generate_batch(
    frontier: Sequence[legacy.Candidate],
    seen_behaviors: set[tuple],
    count: int,
    seed: int,
    local_probability: float = 0.85,
    cross_probability: float = 0.15,
) -> list[dict]:
    rng = random.Random(seed)
    proposals: list[dict] = []
    local_seen = set(seen_behaviors)
    for _ in range(count):
        for _attempt in range(20_000):
            parent = rng.choice(frontier)
            selective, mutation = mutate_parent(
                parent.selective, rng, local_probability, cross_probability
            )
            child = legacy.Candidate(
                selective, legacy.PRODUCTION_BASELINE_ASPIRATION
            )
            signature = legacy.candidate_behavior_signature(child)
            if signature in local_seen:
                continue
            local_seen.add(signature)
            proposals.append({
                "parent_hash": parent.hash,
                "mutation": mutation,
                "config_hash": child.hash,
                "config": child.canonical(),
            })
            break
        else:
            raise RuntimeError("unable to generate a new frontier offspring")
    return proposals


def source_plans(source_run: Path) -> list[dict]:
    path = source_run / "results.jsonl"
    result: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"malformed source results at line {line_number}"
            ) from error
        if value.get("kind") == "plan":
            result.append(value)
    return result


def source_result_index(source_run: Path) -> dict[tuple[str, int, str], dict]:
    path = source_run / "results.jsonl"
    result: dict[tuple[str, int, str], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("kind") != "result":
            continue
        key = (
            str(value.get("dataset_sha256")),
            int(value.get("depth", -1)),
            str(value.get("config_hash")),
        )
        previous = result.get(key)
        if previous is not None and (
            legacy.result_metrics(previous) != legacy.result_metrics(value)
        ):
            raise RuntimeError(f"conflicting source result cache for {key}")
        result[key] = value
    return result


def extend_result_index_from_v45_runs(
    result: dict[tuple[str, int, str], dict],
    cache_runs: Sequence[Path],
) -> dict[tuple[str, int, str], dict]:
    """Add immutable candidate-level evaluations from earlier V45 runs."""
    for cache_run in cache_runs:
        for path in sorted((cache_run / "evaluations").glob("*/*.json")):
            value = json.loads(path.read_text())
            if value.get("kind") != "v45_candidate_evaluation":
                continue
            cached = {
                "config": value["config"],
                "config_hash": value["config_hash"],
                "dataset_sha256": value["dataset_sha256"],
                "depth": int(value["depth"]),
                "result": value["result"],
                "wall_sec": float(value.get("wall_sec", 0.0)),
                "stage": f"external:{cache_run.name}:{value.get('rung')}",
            }
            key = (
                cached["dataset_sha256"], cached["depth"],
                cached["config_hash"],
            )
            previous = result.get(key)
            if previous is not None and (
                previous.get("config") != cached["config"]
                or legacy.result_metrics(previous)
                    != legacy.result_metrics(cached)
            ):
                raise RuntimeError(f"conflicting external result cache for {key}")
            result[key] = cached
    return result


def find_plan(plans: Sequence[dict], rung: str) -> dict:
    matches = []
    for plan in plans:
        stage = str(plan.get("stage", ""))
        if rung == "screen":
            matches.append(plan) if stage.startswith("screen_") else None
        elif stage == rung:
            matches.append(plan)
    if not matches:
        raise RuntimeError(f"source run has no plan for rung {rung}")
    first = matches[0]
    identity = (
        first.get("dataset", {}).get("sha256"),
        first.get("depth"),
        first.get("control_cache", {}).get("identity_sha256"),
    )
    if any((
        plan.get("dataset", {}).get("sha256"),
        plan.get("depth"),
        plan.get("control_cache", {}).get("identity_sha256"),
    ) != identity for plan in matches):
        raise RuntimeError(f"source plans disagree for rung {rung}")
    return first


def prepare_rung(source_plan: dict) -> dict:
    dataset = Path(source_plan["dataset"]["path"]).resolve(strict=True)
    cache = dict(source_plan["control_cache"])
    cache_path = Path(cache["path"]).resolve(strict=True)
    if fingerprint(dataset) != source_plan["dataset"]:
        raise RuntimeError(f"source dataset changed: {dataset}")
    if fingerprint(cache_path) != cache["artifact"]:
        raise RuntimeError(f"source strict-teacher cache changed: {cache_path}")
    legacy.validate_control_cache(cache_path, cache, dataset)
    return {
        "dataset": dataset,
        "dataset_fingerprint": fingerprint(dataset),
        "depth": int(source_plan["depth"]),
        "control_cache": cache,
        "control_cache_fingerprint": fingerprint(cache_path),
    }


def memory_free_percent() -> int | None:
    if sys.platform != "darwin" or shutil.which("memory_pressure") is None:
        return None
    try:
        completed = subprocess.run(
            ["memory_pressure", "-Q"], text=True, capture_output=True,
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"free percentage:\s*(\d+)%", completed.stdout)
    return int(match.group(1)) if match else None


def memory_bounded_workers(requested: int) -> tuple[int, int | None]:
    free = memory_free_percent()
    if free is None or free >= 20:
        return requested, free
    if free >= 10:
        return min(requested, 2), free
    return 1, free


def verify_disk_headroom(run_dir: Path, minimum_bytes: int) -> None:
    available = shutil.disk_usage(run_dir).free
    if available < minimum_bytes:
        raise legacy.EvaluationBudgetExhausted(
            f"disk headroom {available} is below required {minimum_bytes}"
        )


def evaluation_path(run_dir: Path, rung_name: str, candidate_hash: str) -> Path:
    return run_dir / "evaluations" / rung_name / f"{candidate_hash}.json"


def validate_evaluation(
    entry: dict, candidate: legacy.Candidate, rung: dict
) -> None:
    if (
        entry.get("kind") != "v45_candidate_evaluation"
        or entry.get("config_hash") != candidate.hash
        or entry.get("config") != candidate.canonical()
        or entry.get("dataset_sha256")
            != rung["dataset_fingerprint"]["sha256"]
        or int(entry.get("depth", -1)) != rung["depth"]
        or entry.get("control_cache_identity_sha256")
            != rung["control_cache"]["identity_sha256"]
    ):
        raise RuntimeError(f"cached evaluation identity mismatch: {candidate.hash}")
    legacy.validate_result(
        entry["result"], candidate, rung["depth"], rung["control_cache"]
    )


def cached_source_evaluation(
    source_index: dict[tuple[str, int, str], dict],
    candidate: legacy.Candidate,
    rung: dict,
) -> dict | None:
    key = (
        rung["dataset_fingerprint"]["sha256"], rung["depth"], candidate.hash
    )
    source = source_index.get(key)
    if source is None:
        return None
    if source.get("config") != candidate.canonical():
        raise RuntimeError(f"source result config mismatch: {candidate.hash}")
    legacy.validate_result(
        source["result"], candidate, rung["depth"], rung["control_cache"]
    )
    return {
        "result": source["result"],
        "wall_sec": float(source.get("wall_sec", 0.0)),
        "reused_source_result": True,
        "source_stage": source.get("stage"),
    }


def evaluate_candidate(
    *,
    candidate: legacy.Candidate,
    rung_name: str,
    rung: dict,
    binary: Path,
    model: Path,
    run_dir: Path,
    source_index: dict[tuple[str, int, str], dict],
    deadline_monotonic: float,
) -> dict:
    path = evaluation_path(run_dir, rung_name, candidate.hash)
    if path.exists():
        entry = json.loads(path.read_text())
        validate_evaluation(entry, candidate, rung)
        return entry
    reused = cached_source_evaluation(source_index, candidate, rung)
    if reused is None:
        started = time.monotonic()
        result, detail = legacy.evaluate(
            binary, rung["dataset"], model, rung["depth"], candidate,
            rung["control_cache"], None, deadline_monotonic,
        )
        if detail is not None:
            raise RuntimeError("aggregate V45 evaluation unexpectedly wrote details")
        wall_sec = time.monotonic() - started
        reused = {
            "result": result,
            "wall_sec": wall_sec,
            "reused_source_result": False,
            "source_stage": None,
        }
    entry = {
        "kind": "v45_candidate_evaluation",
        "schema_version": SCHEMA_VERSION,
        "rung": rung_name,
        "config": candidate.canonical(),
        "config_hash": candidate.hash,
        "dataset_sha256": rung["dataset_fingerprint"]["sha256"],
        "depth": rung["depth"],
        "control_cache_identity_sha256": (
            rung["control_cache"]["identity_sha256"]
        ),
        "completed_at": utc_now(),
        **reused,
    }
    validate_evaluation(entry, candidate, rung)
    write_once_json(path, entry)
    return entry


def evaluate_batch(
    candidates: Sequence[legacy.Candidate],
    *,
    rung_name: str,
    rung: dict,
    binary: Path,
    model: Path,
    run_dir: Path,
    source_index: dict[tuple[str, int, str], dict],
    workers: int,
    deadline_monotonic: float,
) -> tuple[dict[str, dict], dict[str, str], int, int | None]:
    effective_workers, free_percent = memory_bounded_workers(workers)
    effective_workers = min(effective_workers, max(1, len(candidates)))
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    def execute(candidate: legacy.Candidate) -> dict:
        return evaluate_candidate(
            candidate=candidate, rung_name=rung_name, rung=rung,
            binary=binary, model=model, run_dir=run_dir,
            source_index=source_index, deadline_monotonic=deadline_monotonic,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=effective_workers
    ) as pool:
        futures = {pool.submit(execute, candidate): candidate for candidate in candidates}
        for future in concurrent.futures.as_completed(futures):
            candidate = futures[future]
            try:
                results[candidate.hash] = future.result()
            except legacy.EvaluationBudgetExhausted:
                raise
            except BaseException as error:
                failures[candidate.hash] = f"{type(error).__name__}: {error}"

    # A concurrent SIGKILL or transient launch failure must not stop the whole
    # night. Retry once with only one evaluator alive; persistent candidate-
    # specific failures are archived and excluded from the frontier.
    for candidate in candidates:
        if candidate.hash not in failures:
            continue
        if time.monotonic() >= deadline_monotonic:
            raise legacy.EvaluationBudgetExhausted("V45 deadline reached")
        try:
            results[candidate.hash] = execute(candidate)
            failures.pop(candidate.hash, None)
        except legacy.EvaluationBudgetExhausted:
            raise
        except BaseException as error:
            failures[candidate.hash] += (
                f"; retry={type(error).__name__}: {error}"
            )
    return results, failures, effective_workers, free_percent


def candidate_from_payload(payload: dict) -> legacy.Candidate:
    candidate = legacy.Candidate.from_dict(payload["config"])
    if candidate.hash != payload["config_hash"]:
        raise RuntimeError("proposal candidate hash mismatch")
    return candidate


def seen_behavior_signatures(
    run_dir: Path, baseline: legacy.Candidate
) -> set[tuple]:
    seen = {legacy.candidate_behavior_signature(baseline)}
    for path in sorted((run_dir / "plans").glob("batch-*.json")):
        plan = json.loads(path.read_text())
        for proposal in plan["proposals"]:
            seen.add(legacy.candidate_behavior_signature(
                candidate_from_payload(proposal)
            ))
    return seen


def config_catalog(
    run_dir: Path, baseline: legacy.Candidate
) -> dict[str, legacy.Candidate]:
    result = {baseline.hash: baseline}
    for path in sorted((run_dir / "plans").glob("batch-*.json")):
        for proposal in json.loads(path.read_text())["proposals"]:
            candidate = candidate_from_payload(proposal)
            result[candidate.hash] = candidate
    return result


def load_frontier_entries(
    state: dict, catalog: dict[str, legacy.Candidate], run_dir: Path,
    rung_name: str, rung: dict,
) -> list[dict]:
    entries = []
    for candidate_hash in state["frontier_hashes"]:
        path = evaluation_path(run_dir, rung_name, candidate_hash)
        if not path.is_file():
            raise RuntimeError(f"frontier evaluation is missing: {candidate_hash}")
        entry = json.loads(path.read_text())
        validate_evaluation(entry, catalog[candidate_hash], rung)
        entries.append(entry)
    return entries


def result_ratio(entry: dict, baseline_entry: dict, name: str) -> float:
    denominator = float(baseline_entry["result"][name])
    if denominator <= 0:
        raise RuntimeError(f"baseline {name} is not positive")
    return float(entry["result"][name]) / denominator


def wall_ratio(entry: dict, baseline_entry: dict) -> float | None:
    denominator = float(baseline_entry.get("wall_sec", 0.0))
    return float(entry.get("wall_sec", 0.0)) / denominator if denominator > 0 else None


def update_status(run_dir: Path, state: str, **extra: object) -> None:
    atomic_json(run_dir / "status.json", {
        "state": state,
        "updated_at": utc_now(),
        **extra,
    })


def initialize_manifest(
    args: argparse.Namespace,
    binary: Path,
    model: Path,
    rungs: dict[str, dict],
) -> dict:
    path = args.run_dir / "manifest.json"
    baseline = baseline_candidate(args.baseline_profile)
    expected = {
        "kind": "nnue_v45_pareto_tune_contract",
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "source_run": fingerprint(args.source_run / "results.jsonl"),
        "source_summary": fingerprint(args.source_run / "summary.json"),
        "evaluation_cache_runs": [
            {
                "path": str(path),
                "summary": fingerprint(path / "summary.json"),
            }
            for path in args.evaluation_cache_runs
        ],
        "binary": fingerprint(binary),
        "model": fingerprint(model),
        "tuner": fingerprint(Path(__file__)),
        "legacy_tuner": fingerprint(Path(legacy.__file__)),
        "baseline_profile": args.baseline_profile,
        "baseline_hash": baseline.hash,
        "seed": args.seed,
        "mutation_count": args.mutation_count,
        "workers": args.workers,
        "duration_sec": args.duration_sec,
        "hard_node_ratio": args.hard_node_ratio,
        "minimum_disk_free_bytes": args.minimum_disk_free_bytes,
        "mutation_policy": {
            "parent": "uniform_active_pareto_frontier",
            "single_grid_step_probability": args.local_mutation_probability,
            "two_block_grid_step_probability": args.cross_mutation_probability,
            "multi_block_restart_probability": (
                1.0 - args.local_mutation_probability
                - args.cross_mutation_probability
            ),
            "safe_profile_anchor": False,
        },
        "pareto_policy": {
            "objectives": ["mean_wdl_loss", "candidate_nodes"],
            "direction": ["minimize", "minimize"],
            "reject_if_dominated_by_active_frontier": True,
            "remove_active_members_dominated_by_offspring": True,
            "archive_every_evaluated_candidate": True,
        },
        "selection_policy": {
            "fresh_split": True,
            "fed_back_into_mutation": False,
            "point_pareto_reselection": True,
        },
        "memory_policy": {
            "detail_sidecars": False,
            "bootstrap_matrices": False,
            "requested_workers": args.workers,
            "macos_free_below_20_percent_workers": 2,
            "macos_free_below_10_percent_workers": 1,
            "failed_evaluation_single_worker_retries": 1,
        },
        "strict_teacher_cache_policy": "immutable_reuse_once_per_rung",
        "rungs": {
            name: {
                "dataset": value["dataset_fingerprint"],
                "depth": value["depth"],
                "control_cache": value["control_cache_fingerprint"],
                "control_cache_identity_sha256": (
                    value["control_cache"]["identity_sha256"]
                ),
            }
            for name, value in rungs.items()
        },
    }
    if path.exists():
        actual = json.loads(path.read_text())
        if actual != expected:
            raise RuntimeError("immutable V45 manifest disagrees")
        return actual
    atomic_json(path, expected)
    return expected


def initial_state(args: argparse.Namespace) -> dict:
    baseline = baseline_candidate(args.baseline_profile)
    return {
        "kind": "nnue_v45_pareto_state",
        "schema_version": SCHEMA_VERSION,
        "stage": "stage1",
        "next_batch": 0,
        "proposed_count": 0,
        "frontier_hashes": [baseline.hash],
        "mutation_target": args.mutation_count,
        "updated_at": utc_now(),
    }


def recover_decided_batches(run_dir: Path, state: dict) -> dict:
    while True:
        path = run_dir / "decisions" / f"batch-{state['next_batch']:06d}.json"
        if not path.exists():
            return state
        decision = json.loads(path.read_text())
        if decision["frontier_before"] != state["frontier_hashes"]:
            raise RuntimeError("saved decision frontier does not match state")
        state = {
            **state,
            "next_batch": state["next_batch"] + 1,
            "proposed_count": decision["proposed_count_after"],
            "frontier_hashes": decision["frontier_after"],
            "updated_at": utc_now(),
        }
        atomic_json(run_dir / "state.json", state)


def run_stage1(
    args: argparse.Namespace,
    rungs: dict[str, dict],
    binary: Path,
    model: Path,
    source_index: dict[tuple[str, int, str], dict],
    deadline_monotonic: float,
) -> dict:
    state_path = args.run_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = initial_state(args)
        atomic_json(state_path, state)
    state = recover_decided_batches(args.run_dir, state)
    baseline = baseline_candidate(args.baseline_profile)
    baseline_screen = evaluate_candidate(
        candidate=baseline, rung_name="screen", rung=rungs["screen"],
        binary=binary, model=model, run_dir=args.run_dir,
        source_index=source_index, deadline_monotonic=deadline_monotonic,
    )
    baseline_tune = evaluate_candidate(
        candidate=baseline, rung_name="tune", rung=rungs["tune"],
        binary=binary, model=model, run_dir=args.run_dir,
        source_index=source_index, deadline_monotonic=deadline_monotonic,
    )

    while state["proposed_count"] < args.mutation_count:
        if time.monotonic() >= deadline_monotonic:
            raise legacy.EvaluationBudgetExhausted("V45 stage-1 deadline reached")
        verify_disk_headroom(args.run_dir, args.minimum_disk_free_bytes)
        catalog = config_catalog(args.run_dir, baseline)
        frontier_candidates = [
            catalog[value] for value in state["frontier_hashes"]
        ]
        remaining = args.mutation_count - state["proposed_count"]
        batch_count = min(args.workers, remaining)
        batch_number = state["next_batch"]
        plan_path = args.run_dir / "plans" / f"batch-{batch_number:06d}.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan["frontier_before"] != state["frontier_hashes"]:
                raise RuntimeError("saved batch plan frontier mismatch")
        else:
            proposals = generate_batch(
                frontier_candidates,
                seen_behavior_signatures(args.run_dir, baseline),
                batch_count,
                args.seed ^ (batch_number * 0x9E3779B1),
                args.local_mutation_probability,
                args.cross_mutation_probability,
            )
            plan = {
                "kind": "nnue_v45_mutation_batch_plan",
                "schema_version": SCHEMA_VERSION,
                "batch": batch_number,
                "frontier_before": state["frontier_hashes"],
                "proposed_count_before": state["proposed_count"],
                "proposals": proposals,
            }
            write_once_json(plan_path, plan)
        candidates = [candidate_from_payload(item) for item in plan["proposals"]]
        screen_results, screen_failures, used_workers, free_percent = evaluate_batch(
            candidates, rung_name="screen", rung=rungs["screen"],
            binary=binary, model=model, run_dir=args.run_dir,
            source_index=source_index, workers=args.workers,
            deadline_monotonic=deadline_monotonic,
        )
        full_candidates = []
        screen_metrics: dict[str, dict] = {}
        for candidate in candidates:
            entry = screen_results.get(candidate.hash)
            if entry is None:
                continue
            node_ratio = result_ratio(entry, baseline_screen, "candidate_nodes")
            screen_metrics[candidate.hash] = {
                "production_node_ratio": node_ratio,
                "production_wall_ratio": wall_ratio(entry, baseline_screen),
                "mean_wdl_loss": float(entry["result"]["mean_wdl_loss"]),
                "critical_mistakes": int(entry["result"]["critical_mistakes"]),
            }
            if node_ratio <= args.hard_node_ratio:
                full_candidates.append(candidate)
        tune_results: dict[str, dict] = {}
        tune_failures: dict[str, str] = {}
        if full_candidates:
            tune_results, tune_failures, used_workers, free_percent = evaluate_batch(
                full_candidates, rung_name="tune", rung=rungs["tune"],
                binary=binary, model=model, run_dir=args.run_dir,
                source_index=source_index, workers=args.workers,
                deadline_monotonic=deadline_monotonic,
            )
        catalog = config_catalog(args.run_dir, baseline)
        frontier = load_frontier_entries(
            state, catalog, args.run_dir, "tune", rungs["tune"]
        )
        items = []
        for proposal, candidate in zip(plan["proposals"], candidates):
            screen = screen_results.get(candidate.hash)
            if screen is None:
                status = "evaluation_failed"
                reason = screen_failures[candidate.hash]
                removed: list[str] = []
                tune_metrics = None
            elif screen_metrics[candidate.hash]["production_node_ratio"] \
                    > args.hard_node_ratio:
                status = "hard_node_reject"
                reason = (
                    f"screen node ratio "
                    f"{screen_metrics[candidate.hash]['production_node_ratio']:.6f} "
                    f"> {args.hard_node_ratio:.6f}"
                )
                removed = []
                tune_metrics = None
            elif candidate.hash not in tune_results:
                status = "evaluation_failed"
                reason = tune_failures[candidate.hash]
                removed = []
                tune_metrics = None
            else:
                tune_entry = tune_results[candidate.hash]
                frontier, status, removed = pareto_insert(frontier, tune_entry)
                reason = None
                tune_metrics = {
                    "production_node_ratio": result_ratio(
                        tune_entry, baseline_tune, "candidate_nodes"
                    ),
                    "production_wall_ratio": wall_ratio(tune_entry, baseline_tune),
                    "mean_wdl_loss": float(
                        tune_entry["result"]["mean_wdl_loss"]
                    ),
                    "critical_mistakes": int(
                        tune_entry["result"]["critical_mistakes"]
                    ),
                }
            items.append({
                **proposal,
                "status": status,
                "reason": reason,
                "removed_frontier_hashes": removed,
                "screen": screen_metrics.get(candidate.hash),
                "tune": tune_metrics,
            })
        frontier_after = sorted(entry["config_hash"] for entry in frontier)
        decision = {
            "kind": "nnue_v45_mutation_batch_decision",
            "schema_version": SCHEMA_VERSION,
            "batch": batch_number,
            "frontier_before": state["frontier_hashes"],
            "frontier_after": frontier_after,
            "proposed_count_after": (
                state["proposed_count"] + len(plan["proposals"])
            ),
            "effective_workers": used_workers,
            "memory_free_percent": free_percent,
            "items": items,
            "completed_at": utc_now(),
        }
        decision_path = (
            args.run_dir / "decisions" / f"batch-{batch_number:06d}.json"
        )
        write_once_json(decision_path, decision)
        state = {
            **state,
            "next_batch": batch_number + 1,
            "proposed_count": decision["proposed_count_after"],
            "frontier_hashes": frontier_after,
            "updated_at": utc_now(),
        }
        atomic_json(state_path, state)
        counts: dict[str, int] = {}
        for item in items:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        update_status(
            args.run_dir, "RUNNING", stage="stage1",
            proposed=state["proposed_count"], target=args.mutation_count,
            frontier_count=len(frontier_after), batch_status_counts=counts,
            effective_workers=used_workers, memory_free_percent=free_percent,
        )
        print(
            f"stage1 proposed={state['proposed_count']}/{args.mutation_count} "
            f"frontier={len(frontier_after)} decisions={counts} "
            f"workers={used_workers} memory_free_pct={free_percent}",
            flush=True,
        )
    state = {**state, "stage": "stage2", "updated_at": utc_now()}
    atomic_json(state_path, state)
    return state


def summarize_stage1(run_dir: Path) -> dict:
    counts: dict[str, int] = {}
    removed_events = 0
    for path in sorted((run_dir / "decisions").glob("batch-*.json")):
        for item in json.loads(path.read_text())["items"]:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
            removed_events += len(item["removed_frontier_hashes"])
    return {"decision_counts": counts, "dominated_frontier_removals": removed_events}


def run_stage2(
    args: argparse.Namespace,
    state: dict,
    rungs: dict[str, dict],
    binary: Path,
    model: Path,
    source_index: dict[tuple[str, int, str], dict],
    deadline_monotonic: float,
) -> dict:
    baseline = baseline_candidate(args.baseline_profile)
    hashes = sorted(set([*state["frontier_hashes"], baseline.hash]))
    catalog = config_catalog(args.run_dir, baseline)
    candidates = [catalog[value] for value in hashes]
    plan = {
        "kind": "nnue_v45_selection_plan",
        "schema_version": SCHEMA_VERSION,
        "source_stage1_frontier_hashes": state["frontier_hashes"],
        "candidate_hashes": hashes,
        "selection_is_not_fed_back_into_mutation": True,
    }
    write_once_json(args.run_dir / "stage2_plan.json", plan)
    baseline_selection = evaluate_candidate(
        candidate=baseline, rung_name="selection", rung=rungs["selection"],
        binary=binary, model=model, run_dir=args.run_dir,
        source_index=source_index, deadline_monotonic=deadline_monotonic,
    )
    pending = [
        candidate for candidate in candidates
        if not evaluation_path(args.run_dir, "selection", candidate.hash).exists()
    ]
    for offset in range(0, len(pending), args.workers):
        if time.monotonic() >= deadline_monotonic:
            raise legacy.EvaluationBudgetExhausted("V45 stage-2 deadline reached")
        verify_disk_headroom(args.run_dir, args.minimum_disk_free_bytes)
        batch = pending[offset:offset + args.workers]
        results, failures, used_workers, free_percent = evaluate_batch(
            batch, rung_name="selection", rung=rungs["selection"],
            binary=binary, model=model, run_dir=args.run_dir,
            source_index=source_index, workers=args.workers,
            deadline_monotonic=deadline_monotonic,
        )
        failure_path = args.run_dir / "stage2_failures.json"
        existing = (
            json.loads(failure_path.read_text()) if failure_path.exists() else {}
        )
        for candidate_hash in results:
            existing.pop(candidate_hash, None)
        existing.update(failures)
        atomic_json(failure_path, existing)
        completed = len(list((args.run_dir / "evaluations" / "selection").glob("*.json")))
        update_status(
            args.run_dir, "RUNNING", stage="stage2",
            completed=completed, target=len(candidates),
            effective_workers=used_workers, memory_free_percent=free_percent,
        )
        print(
            f"stage2 completed={completed}/{len(candidates)} "
            f"workers={used_workers} memory_free_pct={free_percent}", flush=True,
        )
    failure_hashes = set()
    failure_path = args.run_dir / "stage2_failures.json"
    if failure_path.exists():
        failure_hashes = set(json.loads(failure_path.read_text()))
    entries = []
    hard_rejected = []
    for candidate in candidates:
        if candidate.hash in failure_hashes:
            continue
        path = evaluation_path(args.run_dir, "selection", candidate.hash)
        if not path.exists():
            continue
        entry = json.loads(path.read_text())
        validate_evaluation(entry, candidate, rungs["selection"])
        production_node_ratio = result_ratio(
            entry, baseline_selection, "candidate_nodes"
        )
        if production_node_ratio > args.hard_node_ratio:
            hard_rejected.append(candidate.hash)
        else:
            entries.append(entry)
    selected = legacy.frontier(entries)
    selected.sort(key=lambda entry: (
        float(entry["result"]["mean_wdl_loss"]),
        float(entry["result"]["candidate_nodes"]),
        entry["config_hash"],
    ))
    frontier_payload = []
    for entry in selected:
        frontier_payload.append({
            "config_hash": entry["config_hash"],
            "config": entry["config"],
            "mean_wdl_loss": float(entry["result"]["mean_wdl_loss"]),
            "production_node_ratio": result_ratio(
                entry, baseline_selection, "candidate_nodes"
            ),
            "production_wall_ratio": wall_ratio(entry, baseline_selection),
            "critical_mistakes": int(entry["result"]["critical_mistakes"]),
            "source_result_cache_hit": bool(entry["reused_source_result"]),
        })
    summary = {
        "kind": "nnue_v45_stage1_stage2_summary",
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "completed_at": utc_now(),
        "baseline_profile": args.baseline_profile,
        "baseline_hash": baseline.hash,
        "mutation_count": args.mutation_count,
        "stage1": {
            **summarize_stage1(args.run_dir),
            "frontier_count": len(state["frontier_hashes"]),
            "frontier_hashes": state["frontier_hashes"],
        },
        "stage2": {
            "evaluated_count": len(entries),
            "evaluation_failure_hashes": sorted(failure_hashes),
            "hard_node_rejected_hashes": hard_rejected,
            "frontier_count": len(frontier_payload),
            "frontier": frontier_payload,
        },
        "ready_for_round_robin": bool(frontier_payload),
        "automatic_promotion": False,
    }
    write_once_json(args.run_dir / "stage2_frontier.json", {
        "kind": "nnue_v45_stage2_frontier",
        "schema_version": SCHEMA_VERSION,
        "candidates": frontier_payload,
    })
    write_once_json(args.run_dir / "summary.json", summary)
    atomic_json(args.run_dir / "state.json", {
        **state, "stage": "complete", "updated_at": utc_now()
    })
    (args.run_dir / "DONE").write_text("\n")
    update_status(
        args.run_dir, "DONE", stage="complete",
        stage1_frontier_count=len(state["frontier_hashes"]),
        stage2_frontier_count=len(frontier_payload),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--evaluation-cache-run", dest="evaluation_cache_runs",
        action="append", type=Path, default=[],
    )
    parser.add_argument("--mutation-count", type=int, default=360)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--duration-sec", type=int, default=28_800)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--baseline-profile",
        choices=("v43-original", "v45-deployed"),
        default="v43-original",
    )
    parser.add_argument("--hard-node-ratio", type=float, default=2.0)
    parser.add_argument(
        "--local-mutation-probability", type=float, default=0.85
    )
    parser.add_argument(
        "--cross-mutation-probability", type=float, default=0.15
    )
    parser.add_argument(
        "--minimum-disk-free-bytes", type=int, default=5 * 1024**3
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if min(args.mutation_count, args.workers, args.duration_sec) <= 0:
        parser.error("mutation count, workers, and duration must be positive")
    if args.hard_node_ratio <= 1.0:
        parser.error("hard node ratio must exceed 1.0")
    if (
        args.local_mutation_probability < 0
        or args.cross_mutation_probability < 0
        or args.local_mutation_probability + args.cross_mutation_probability > 1
    ):
        parser.error("mutation probabilities must be nonnegative and sum to <= 1")
    if args.minimum_disk_free_bytes < 0:
        parser.error("minimum disk headroom cannot be negative")
    if args.smoke_test:
        args.mutation_count = min(args.mutation_count, 3)
        args.workers = min(args.workers, 2)
    return args


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = args.run_dir.resolve()
    args.source_run = args.source_run.resolve(strict=True)
    args.evaluation_cache_runs = [
        path.resolve(strict=True) for path in args.evaluation_cache_runs
    ]
    lock_path = args.run_dir / ".tuner.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another V45 tuner owns this run directory") from error
        _main_locked(args)


def _main_locked(args: argparse.Namespace) -> None:
    if (args.run_dir / "DONE").exists():
        print((args.run_dir / "summary.json").read_text(), flush=True)
        return
    binary = (args.source_run / "bin" / "evaluate_nnue_v43_selective").resolve(
        strict=True
    )
    model = (
        args.source_run / "inputs" / "phase_quantized_nnue.bin"
    ).resolve(strict=True)
    plans = source_plans(args.source_run)
    rung_names = (
        {"screen": "preflight", "tune": "screen", "selection": "selection"}
        if args.smoke_test
        else {"screen": "screen", "tune": "halving", "selection": "selection"}
    )
    rungs = {
        name: prepare_rung(find_plan(plans, source_name))
        for name, source_name in rung_names.items()
    }
    initialize_manifest(args, binary, model, rungs)
    verify_disk_headroom(args.run_dir, args.minimum_disk_free_bytes)
    source_index = source_result_index(args.source_run)
    extend_result_index_from_v45_runs(
        source_index, args.evaluation_cache_runs
    )
    deadline_monotonic = time.monotonic() + args.duration_sec
    (args.run_dir / "RUNNING").write_text(f"{os.getpid()}\n")
    for marker in ("PAUSED", "FAILED"):
        (args.run_dir / marker).unlink(missing_ok=True)
    update_status(args.run_dir, "RUNNING", stage="initializing")
    try:
        state = run_stage1(
            args, rungs, binary, model, source_index, deadline_monotonic
        )
        run_stage2(
            args, state, rungs, binary, model, source_index, deadline_monotonic
        )
    except legacy.EvaluationBudgetExhausted as error:
        payload = {
            "state": "PAUSED", "stage": json.loads(
                (args.run_dir / "state.json").read_text()
            )["stage"],
            "reason": str(error), "duration_sec": args.duration_sec,
            "updated_at": utc_now(),
        }
        atomic_json(args.run_dir / "PAUSED", payload)
        update_status(args.run_dir, "PAUSED", **{
            key: value for key, value in payload.items() if key != "state"
        })
        print(json.dumps(payload, sort_keys=True), flush=True)
    except (KeyboardInterrupt, InterruptedError) as error:
        payload = {
            "state": "PAUSED",
            "stage": json.loads(
                (args.run_dir / "state.json").read_text()
            )["stage"],
            "reason": str(error) or "tuner interrupted",
            "duration_sec": args.duration_sec,
            "updated_at": utc_now(),
        }
        atomic_json(args.run_dir / "PAUSED", payload)
        update_status(args.run_dir, "PAUSED", **{
            key: value for key, value in payload.items() if key != "state"
        })
        print(json.dumps(payload, sort_keys=True), flush=True)
        raise
    except BaseException as error:
        payload = {
            "state": "FAILED", "error_type": type(error).__name__,
            "error": str(error), "updated_at": utc_now(),
        }
        atomic_json(args.run_dir / "FAILED", payload)
        update_status(args.run_dir, "FAILED", **{
            key: value for key, value in payload.items() if key != "state"
        })
        raise
    finally:
        (args.run_dir / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
