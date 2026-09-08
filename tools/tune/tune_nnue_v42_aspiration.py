#!/usr/bin/env python3
"""Resumable multi-fidelity tuner for V42 adaptive aspiration windows.

The tuner deliberately keeps the two safety guards (retry count and mean-score
clamp) fixed.  It explores the six behavioural parameters globally, refines the
whole WDL/node Pareto frontier, and only promotes frontier configurations to
the more expensive rungs.  Every candidate in a rung sees the exact same
dataset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = 3
DATASET_SCHEMA = "category_hash_ply_payload_v1"
CONTROL_CACHE_SCHEMA = "nnue_v42_v36_root_cache_v1"
RANKING_TARGET_ABS_CP = 1_500
WDL_FORMULA = "symmetric_phase_ply"
WDL_CALIBRATION_RUN = "nnue_wdl_calibration_20260729_120718/fit_v2"
CONTROL_CACHE_COLUMNS = (
    "index", "sample_hash", "calibration_ply", "zobrist_key",
    "move_value", "score", "nodes", "result_depth", "stopped",
    "full_window_fallbacks", "range_conflicts", "unresolved_ranges",
)
CONTROL_TEACHER_PROFILE = {
    "searcher": "NnueSearcherV36",
    "root_semantics": "strict-exact-fixed-depth",
    "tt_mb": 64,
    "tt_bucket_size": 4,
    "clear_tt_per_sample": True,
    "clear_search_heuristics_per_sample": True,
    "neon_dotprod_requested": True,
    "cached_values": [
        "best_move", "score", "nodes", "result_depth", "stopped",
        "exactness_counters",
    ],
    "cached_timing": False,
    "candidate_child_rescore": "fresh-strict-v36-depth-minus-one",
}
BUDGET_POLICY_VERSION = 1
BUDGET_POLICY = {
    "version": BUDGET_POLICY_VERSION,
    "semantics": "cumulative_elapsed_ceiling",
}
CONFIG_FIELDS = (
    "enabled",
    "min_depth",
    "delta_base_cp",
    "delta_divisor",
    "expansion_factor_per_mille",
    "max_fail_high_reductions",
    "mean_score_new_weight_per_mille",
    "max_researches",
    "mean_score_clamp_cp",
)
TELEMETRY_FIELDS = (
    "aspiration_completed_iterations",
    "aspiration_narrow_iterations",
    "aspiration_narrow_attempts",
    "aspiration_initial_window_successes",
    "aspiration_fail_lows",
    "aspiration_fail_highs",
    "aspiration_reduced_depth_attempts",
    "aspiration_accepted_reduced_depth_iterations",
    "aspiration_accepted_narrow_nominal_depth_sum",
    "aspiration_accepted_narrow_search_depth_sum",
    "aspiration_max_accepted_depth_reduction",
    "aspiration_full_window_fallbacks",
    "aspiration_range_conflict_fallbacks",
    "aspiration_unresolved_ranges",
    "aspiration_retry_limit_fallbacks",
    "aspiration_initial_delta_sum_cp",
    "aspiration_initial_delta_min_cp",
    "aspiration_initial_delta_max_cp",
    "aspiration_final_mean_score_count",
    "aspiration_final_mean_score_sum_cp",
)
SELECTIVE_CONFIG_FIELDS = (
    "enable_lmr",
    "lmr_base",
    "lmr_divisor",
    "lmr_min_depth",
    "lmr_min_move_index",
    "enable_null_move",
    "null_move_min_depth",
    "null_move_reduction",
    "enable_reverse_futility",
    "reverse_futility_max_depth",
    "reverse_futility_base_margin",
    "reverse_futility_margin_per_depth",
    "enable_late_move_pruning",
    "late_move_pruning_max_depth",
    "late_move_pruning_base",
    "late_move_pruning_depth_multiplier",
    "enable_qsearch_see_pruning",
    "qsearch_see_threshold",
    "enable_main_search_see_pruning",
    "main_search_see_max_depth",
    "main_search_see_margin_per_depth",
)

# The aspiration experiment must vary only aspiration.  Spell out the full
# promoted V41/Fast+QSEE profile instead of inheriting mutable C++ defaults.
PRODUCTION_SELECTIVE_CONFIG = {
    "enable_lmr": True,
    "lmr_base": 0.45,
    "lmr_divisor": 2.9,
    "lmr_min_depth": 3,
    "lmr_min_move_index": 6,
    "enable_null_move": True,
    "null_move_min_depth": 2,
    "null_move_reduction": 3,
    "enable_reverse_futility": True,
    "reverse_futility_max_depth": 2,
    "reverse_futility_base_margin": 175,
    "reverse_futility_margin_per_depth": 275,
    "enable_late_move_pruning": True,
    "late_move_pruning_max_depth": 3,
    "late_move_pruning_base": 4,
    "late_move_pruning_depth_multiplier": 2,
    "enable_qsearch_see_pruning": True,
    "qsearch_see_threshold": -75,
    "enable_main_search_see_pruning": False,
    "main_search_see_max_depth": 5,
    "main_search_see_margin_per_depth": 100,
}
TWOFOLD_SEARCH_DRAW_ENABLED = True
PRODUCTION_GAUNTLET_PROFILE_FIELDS = (
    "0.45", "2.9", "3", "6", "2", "3",
    "1", "2", "175", "275",
    "1", "3", "4", "2",
    "1", "-75",
)


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    min_depth: int = 3
    delta_base_cp: int = 30
    delta_divisor: int = 10_000
    expansion_factor_per_mille: int = 1_750
    max_fail_high_reductions: int = 2
    mean_score_new_weight_per_mille: int = 500
    max_researches: int = 6
    mean_score_clamp_cp: int = 1_500

    def __post_init__(self) -> None:
        if not 2 <= self.min_depth <= 5:
            raise ValueError("min_depth must be in [2, 5]")
        if not 10 <= self.delta_base_cp <= 90:
            raise ValueError("delta_base_cp must be in [10, 90]")
        if not 4_000 <= self.delta_divisor <= 40_000:
            raise ValueError("delta_divisor must be in [4000, 40000]")
        if not 1_300 <= self.expansion_factor_per_mille <= 3_000:
            raise ValueError(
                "expansion_factor_per_mille must be in [1300, 3000]")
        if not 0 <= self.max_fail_high_reductions <= 3:
            raise ValueError("max_fail_high_reductions must be in [0, 3]")
        if not 0 <= self.mean_score_new_weight_per_mille <= 1_000:
            raise ValueError(
                "mean_score_new_weight_per_mille must be in [0, 1000]")
        if self.max_researches < 1:
            raise ValueError("max_researches must be positive")
        if self.mean_score_clamp_cp < 1:
            raise ValueError("mean_score_clamp_cp must be positive")

    def canonical(self) -> dict[str, bool | int]:
        raw = asdict(self)
        return {name: raw[name] for name in CONFIG_FIELDS}

    @property
    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.canonical())).hexdigest()

    @property
    def label(self) -> str:
        prefix = "legacy" if not self.enabled else "v42"
        return f"{prefix}-{self.hash[:12]}"

    def evaluator_args(self) -> list[str]:
        return [
            (
                "--enable-adaptive-aspiration"
                if self.enabled
                else "--disable-adaptive-aspiration"
            ),
            "--aspiration-min-depth", str(self.min_depth),
            "--aspiration-delta-base-cp", str(self.delta_base_cp),
            "--aspiration-delta-divisor", str(self.delta_divisor),
            "--aspiration-expansion-factor-per-mille",
            str(self.expansion_factor_per_mille),
            "--aspiration-max-fail-high-reductions",
            str(self.max_fail_high_reductions),
            "--aspiration-mean-score-new-weight-per-mille",
            str(self.mean_score_new_weight_per_mille),
            "--aspiration-max-researches", str(self.max_researches),
            "--aspiration-mean-score-clamp-cp",
            str(self.mean_score_clamp_cp),
        ]

    @classmethod
    def from_dict(cls, value: dict) -> "Config":
        if set(value) != set(CONFIG_FIELDS):
            missing = sorted(set(CONFIG_FIELDS) - set(value))
            extra = sorted(set(value) - set(CONFIG_FIELDS))
            raise ValueError(
                f"non-canonical config fields: missing={missing} extra={extra}")
        return cls(**{name: value[name] for name in CONFIG_FIELDS})


@dataclass(frozen=True)
class Proposal:
    config: Config
    origin: str


@dataclass
class Budget:
    limit_sec: float
    previously_used_sec: float
    started: float

    @property
    def elapsed_sec(self) -> float:
        return self.previously_used_sec + (time.monotonic() - self.started)

    @property
    def exhausted(self) -> bool:
        return self.elapsed_sec >= self.limit_sec

    @property
    def remaining_sec(self) -> float:
        return max(0.0, self.limit_sec - self.elapsed_sec)


class EvaluationFailure(RuntimeError):
    """A single evaluator invocation/configuration failed cleanly."""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str] = (),
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.command = list(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class CacheBudgetExhausted(RuntimeError):
    """The cumulative run budget expired during root-cache precomputation."""


class EvaluationBudgetExhausted(RuntimeError):
    """A candidate evaluator reached the shared wall-clock batch deadline."""


def selective_evaluator_args() -> list[str]:
    config = PRODUCTION_SELECTIVE_CONFIG
    return [
        "--enable-lmr" if config["enable_lmr"] else "--disable-lmr",
        "--lmr-base", str(config["lmr_base"]),
        "--lmr-divisor", str(config["lmr_divisor"]),
        "--lmr-min-depth", str(config["lmr_min_depth"]),
        "--lmr-min-move-index", str(config["lmr_min_move_index"]),
        (
            "--enable-null-move"
            if config["enable_null_move"] else "--disable-null-move"
        ),
        "--null-min-depth", str(config["null_move_min_depth"]),
        "--null-reduction", str(config["null_move_reduction"]),
        (
            "--enable-reverse-futility"
            if config["enable_reverse_futility"]
            else "--disable-reverse-futility"
        ),
        "--reverse-futility-max-depth",
        str(config["reverse_futility_max_depth"]),
        "--reverse-futility-base-margin",
        str(config["reverse_futility_base_margin"]),
        "--reverse-futility-margin-per-depth",
        str(config["reverse_futility_margin_per_depth"]),
        (
            "--enable-late-move-pruning"
            if config["enable_late_move_pruning"]
            else "--disable-late-move-pruning"
        ),
        "--late-move-pruning-max-depth",
        str(config["late_move_pruning_max_depth"]),
        "--late-move-pruning-base", str(config["late_move_pruning_base"]),
        "--late-move-pruning-depth-multiplier",
        str(config["late_move_pruning_depth_multiplier"]),
        (
            "--enable-qsearch-see-pruning"
            if config["enable_qsearch_see_pruning"]
            else "--disable-qsearch-see-pruning"
        ),
        "--qsearch-see-threshold", str(config["qsearch_see_threshold"]),
        (
            "--enable-main-search-see-pruning"
            if config["enable_main_search_see_pruning"]
            else "--disable-main-search-see-pruning"
        ),
        "--main-search-see-max-depth",
        str(config["main_search_see_max_depth"]),
        "--main-search-see-margin-per-depth",
        str(config["main_search_see_margin_per_depth"]),
        (
            "--enable-twofold-search-draw"
            if TWOFOLD_SEARCH_DRAW_ENABLED
            else "--disable-twofold-search-draw"
        ),
    ]


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, str | int]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict) -> None:
    payload = canonical_json(record) + b"\n"
    with path.open("ab") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _truncate_torn_jsonl(path: Path, raw_lines: list[bytes], index: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.repair.tmp")
    temporary.write_bytes(b"".join(raw_lines[:index]))
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL while repairing one torn, trailing append.

    Invalid complete lines and invalid non-final lines are data corruption and
    remain fatal.  A final unterminated or malformed line is ignored so a
    killed evaluator can be resumed without hand-editing results.jsonl.
    """
    if not path.exists():
        return []
    data = path.read_bytes()
    raw_lines = data.splitlines(keepends=True)
    records: list[dict] = []
    for index, raw in enumerate(raw_lines):
        final = index == len(raw_lines) - 1
        terminated = raw.endswith(b"\n") or raw.endswith(b"\r")
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if final and not terminated:
                _truncate_torn_jsonl(path, raw_lines, index)
                break
            raise RuntimeError(
                f"corrupt JSONL record {index + 1} in {path}") from error
        if not isinstance(value, dict):
            if final and not terminated:
                _truncate_torn_jsonl(path, raw_lines, index)
                break
            raise RuntimeError(
                f"JSONL record {index + 1} in {path} is not an object")
        records.append(value)
    return records


def load_by_category(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if line:
            groups.setdefault(line.split("\t", 1)[0], []).append(line)
    return groups


def parse_dataset_row(line: str, path: Path, line_number: int) -> tuple[str, str, int, str]:
    parts = line.split("\t", 3)
    if len(parts) != 4:
        raise RuntimeError(
            f"{path}:{line_number}: V42 requires four-column {DATASET_SCHEMA}")
    category, position_hash, ply_text, payload = parts
    if not category or not position_hash or not payload:
        raise RuntimeError(f"{path}:{line_number}: empty dataset field")
    try:
        ply = int(ply_text)
    except ValueError as error:
        raise RuntimeError(
            f"{path}:{line_number}: invalid explicit ply {ply_text!r}") from error
    if not 0 <= ply <= 0x3FFF:
        raise RuntimeError(f"{path}:{line_number}: ply outside [0, 16383]")
    if payload.startswith("book:"):
        move_count = len(payload[5:].split())
        if ply != move_count:
            raise RuntimeError(
                f"{path}:{line_number}: book ply {ply} != {move_count} moves")
    return category, position_hash, ply, payload


def validate_dataset_schema(path: Path) -> dict[str, int | str]:
    count = 0
    minimum = 0x3FFF
    maximum = 0
    hashes: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            raise RuntimeError(f"{path}:{line_number}: empty dataset row")
        _, position_hash, ply, _ = parse_dataset_row(line, path, line_number)
        previous_line = hashes.get(position_hash)
        if previous_line is not None:
            raise RuntimeError(
                f"{path}:{line_number}: duplicate position hash "
                f"{position_hash!r} (first at line {previous_line})")
        hashes[position_hash] = line_number
        minimum = min(minimum, ply)
        maximum = max(maximum, ply)
        count += 1
    if count == 0:
        raise RuntimeError(f"empty dataset: {path}")
    return {
        "schema": DATASET_SCHEMA,
        "count": count,
        "ply_min": minimum,
        "ply_max": maximum,
    }


def _content_identity(path: Path) -> dict[str, int | str]:
    fingerprint = file_fingerprint(path)
    return {
        "size": fingerprint["size"],
        "sha256": fingerprint["sha256"],
    }


def control_cache_spec(
    run_dir: Path,
    cache_key: str,
    binary: Path,
    model: Path,
    dataset: Path,
    depth: int,
) -> dict:
    dataset_schema = validate_dataset_schema(dataset)
    identity = {
        "schema": CONTROL_CACHE_SCHEMA,
        "evaluator": _content_identity(binary),
        "model": _content_identity(model),
        "ordered_dataset": _content_identity(dataset),
        "dataset_schema": DATASET_SCHEMA,
        "depth": depth,
        "offset": 0,
        "count": dataset_schema["count"],
        "teacher_profile": CONTROL_TEACHER_PROFILE,
    }
    return {
        "cache_key": cache_key,
        "path": str(
            (run_dir / f"control-root-{cache_key}-d{depth}.tsv").resolve()),
        "identity": identity,
        "identity_sha256": hashlib.sha256(canonical_json(identity)).hexdigest(),
    }


def control_cache_command_args(cache: dict, write: bool = False) -> list[str]:
    identity = cache["identity"]
    return [
        "--write-control-cache" if write else "--control-cache",
        str(cache["path"]),
        "--control-cache-identity-sha256", cache["identity_sha256"],
        "--control-cache-evaluator-sha256",
        identity["evaluator"]["sha256"],
        "--control-cache-model-sha256", identity["model"]["sha256"],
        "--control-cache-dataset-sha256",
        identity["ordered_dataset"]["sha256"],
    ]


def validate_control_cache(path: Path, spec: dict, dataset: Path) -> dict:
    lines = path.read_text().splitlines()
    if len(lines) < 9:
        raise RuntimeError(f"truncated control cache: {path}")
    identity = spec["identity"]
    recomputed_identity = hashlib.sha256(
        canonical_json(identity)).hexdigest()
    if spec.get("identity_sha256") != recomputed_identity:
        raise RuntimeError("control cache spec identity digest mismatch")
    if _content_identity(dataset) != identity.get("ordered_dataset"):
        raise RuntimeError("control cache dataset identity changed")
    expected_headers = [
        CONTROL_CACHE_SCHEMA,
        f"identity_sha256\t{spec['identity_sha256']}",
        f"evaluator_sha256\t{identity['evaluator']['sha256']}",
        f"model_sha256\t{identity['model']['sha256']}",
        f"dataset_sha256\t{identity['ordered_dataset']['sha256']}",
        f"depth\t{identity['depth']}",
        f"offset\t{identity['offset']}",
        f"count\t{identity['count']}",
        "\t".join(CONTROL_CACHE_COLUMNS),
    ]
    if lines[:9] != expected_headers:
        differing = next(
            (index for index, (actual, expected) in enumerate(
                zip(lines[:9], expected_headers)) if actual != expected),
            0,
        )
        raise RuntimeError(
            f"control cache header mismatch at line {differing + 1}: {path}")
    if any(
        "time" in field.lower() or "wall" in field.lower()
        for field in CONTROL_CACHE_COLUMNS
    ):
        raise RuntimeError("control cache schema must not contain timing")

    dataset_lines = dataset.read_text().splitlines()
    expected_count = int(identity["count"])
    if len(dataset_lines) != expected_count:
        raise RuntimeError("control cache identity count disagrees with dataset")
    rows = lines[9:]
    if len(rows) != expected_count:
        raise RuntimeError(
            f"control cache row count mismatch: {len(rows)} != {expected_count}")

    control_nodes = 0
    full_window_fallbacks = 0
    range_conflicts = 0
    unresolved_ranges = 0
    # Length equality was checked above. Avoid zip(strict=True), because the
    # repository's pinned .venv currently runs Python 3.9.
    for index, (cache_line, dataset_line) in enumerate(zip(rows, dataset_lines)):
        fields = cache_line.split("\t")
        if len(fields) != len(CONTROL_CACHE_COLUMNS):
            raise RuntimeError(
                f"control cache column count mismatch at row {index}")
        _, sample_hash, ply, _ = parse_dataset_row(
            dataset_line, dataset, index + 1)
        try:
            numeric = [int(value) for value in fields[2:]]
            row_index = int(fields[0])
        except ValueError as error:
            raise RuntimeError(
                f"non-integer control cache value at row {index}") from error
        (
            cached_ply, zobrist_key, move_value, score, nodes,
            result_depth, stopped, fallbacks, conflicts, unresolved,
        ) = numeric
        if row_index != index or fields[1] != sample_hash or cached_ply != ply:
            raise RuntimeError(
                f"control cache row identity/order mismatch at row {index}")
        if (
            not 0 <= zobrist_key <= (1 << 64) - 1
            or not 0 <= move_value <= 0xFFFF
            or not -1_000_000 <= score <= 1_000_000
            or nodes <= 0
            or result_depth != int(identity["depth"])
            or stopped != 0
            or min(fallbacks, conflicts, unresolved) < 0
            or unresolved != 0
        ):
            raise RuntimeError(f"invalid control cache result at row {index}")
        control_nodes += nodes
        full_window_fallbacks += fallbacks
        range_conflicts += conflicts
        unresolved_ranges += unresolved
    return {
        "count": expected_count,
        "depth": int(identity["depth"]),
        "control_nodes": control_nodes,
        "control_exact_searches": expected_count,
        "control_exact_full_window_fallbacks": full_window_fallbacks,
        "control_exact_range_conflicts": range_conflicts,
        "control_exact_unresolved_ranges": unresolved_ranges,
    }


def _new_cache_temporaries(path: Path, before: set[Path]) -> list[Path]:
    return [
        item for item in path.parent.glob(f"{path.name}.tmp-*")
        if item not in before and item.is_file()
    ]


def ensure_control_cache(
    records: list[dict],
    log_path: Path,
    spec: dict,
    binary: Path,
    model: Path,
    dataset: Path,
    budget: Budget,
) -> dict:
    cache_key = spec["cache_key"]
    path = Path(spec["path"])
    matches = [
        record for record in records
        if record.get("kind") == "control_cache_ready"
        and record.get("cache_key") == cache_key
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple cache-ready records for {cache_key}")
    if matches:
        descriptor = matches[0].get("control_cache")
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"invalid cache-ready record for {cache_key}")
        expected_semantics = {
            key: spec[key]
            for key in ("cache_key", "path", "identity", "identity_sha256")
        }
        actual_semantics = {
            key: descriptor.get(key) for key in expected_semantics
        }
        if canonical_json(actual_semantics) != canonical_json(expected_semantics):
            raise RuntimeError(
                f"saved control cache identity mismatch for {cache_key}")
        artifact = descriptor.get("artifact")
        if not isinstance(artifact, dict) or file_fingerprint(path) != artifact:
            raise RuntimeError(
                f"immutable control cache artifact changed for {cache_key}")
        summary = validate_control_cache(path, spec, dataset)
        if descriptor.get("summary") != summary:
            raise RuntimeError(
                f"saved control cache summary mismatch for {cache_key}")
        return descriptor

    if budget.exhausted:
        raise CacheBudgetExhausted(
            f"budget exhausted before building control cache {cache_key}")

    reported: dict | None = None
    if not path.exists():
        before_temporaries = set(path.parent.glob(f"{path.name}.tmp-*"))
        command = [
            str(binary),
            "--dataset", str(dataset),
            "--model", str(model),
            "--depth", str(spec["identity"]["depth"]),
            *control_cache_command_args(spec, write=True),
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=max(0.001, budget.remaining_sec),
            )
        except subprocess.TimeoutExpired as error:
            for temporary in _new_cache_temporaries(path, before_temporaries):
                temporary.unlink(missing_ok=True)
            raise CacheBudgetExhausted(
                f"budget exhausted while building control cache {cache_key}"
            ) from error
        except OSError as error:
            for temporary in _new_cache_temporaries(path, before_temporaries):
                temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"failed to launch control-cache builder: {error}") from error
        if completed.returncode:
            for temporary in _new_cache_temporaries(path, before_temporaries):
                temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "control-cache build failed fatally: "
                f"returncode={completed.returncode} "
                f"stdout={completed.stdout[-2000:]!r} "
                f"stderr={completed.stderr[-2000:]!r}")
        try:
            reported = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "control-cache builder produced invalid JSON") from error
        expected_report = {
            "control_cache_written": True,
            "control_cache_schema": CONTROL_CACHE_SCHEMA,
            "control_cache_identity_sha256": spec["identity_sha256"],
            "count": int(spec["identity"]["count"]),
            "depth": int(spec["identity"]["depth"]),
        }
        if any(reported.get(key) != value for key, value in expected_report.items()):
            raise RuntimeError("control-cache builder summary identity mismatch")

    summary = validate_control_cache(path, spec, dataset)
    if reported is not None:
        for key, value in summary.items():
            if reported.get(key) != value:
                raise RuntimeError(
                    f"control-cache builder summary mismatch for {key}")
    descriptor = {
        **spec,
        "artifact": file_fingerprint(path),
        "summary": summary,
    }
    record = {
        "kind": "control_cache_ready",
        "schema_version": SCHEMA_VERSION,
        "cache_key": cache_key,
        "control_cache": descriptor,
        "budget_elapsed_sec": budget.elapsed_sec,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(log_path, record)
    records.append(record)
    return descriptor


def validate_dataset_hash_disjointness(datasets: dict[str, Path]) -> None:
    owner: dict[str, tuple[str, Path, int]] = {}
    for split, path in datasets.items():
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            _, position_hash, _, _ = parse_dataset_row(line, path, line_number)
            previous = owner.get(position_hash)
            if previous is not None:
                raise RuntimeError(
                    "position hash overlaps dataset splits: "
                    f"{previous[0]} {previous[1]}:{previous[2]} vs "
                    f"{split} {path}:{line_number} ({position_hash})")
            owner[position_hash] = (split, path, line_number)


def validate_no_balanced_lineage_overlap(
    datasets: dict[str, Path],
) -> None:
    """Reject exact or ancestor/descendant book lines across data splits."""
    full_owner: dict[tuple[str, ...], tuple[str, Path, int]] = {}
    prefix_owner: dict[tuple[str, ...], tuple[str, Path, int]] = {}
    for split, path in datasets.items():
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            category, _, _, payload = parse_dataset_row(line, path, line_number)
            if category != "balanced" or not payload.startswith("book:"):
                continue
            moves = tuple(payload[5:].split())
            related = prefix_owner.get(moves)
            if related is None:
                for length in range(1, len(moves) + 1):
                    candidate = full_owner.get(moves[:length])
                    if candidate is not None and candidate[0] != split:
                        related = candidate
                        break
            if related is not None and related[0] != split:
                raise RuntimeError(
                    "balanced opening lineage overlaps splits: "
                    f"{related[0]} {related[1]}:{related[2]} vs "
                    f"{split} {path}:{line_number}")
            full_owner.setdefault(moves, (split, path, line_number))
            for length in range(1, len(moves) + 1):
                prefix_owner.setdefault(
                    moves[:length], (split, path, line_number))


def fen_absolute_ply(fen: str) -> int:
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ("w", "b"):
        raise ValueError("expected a six-field FEN")
    fullmove = int(fields[5])
    if fullmove < 1:
        raise ValueError("FEN fullmove number must be positive")
    return 2 * (fullmove - 1) + (fields[1] == "b")


def fen_piece_count(fen: str) -> int:
    """Return the piece count while validating the FEN board geometry."""
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("expected a six-field FEN")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError("FEN board must contain eight ranks")
    piece_count = 0
    for rank in ranks:
        width = 0
        for symbol in rank:
            if symbol in "12345678":
                width += int(symbol)
            elif symbol in "pnbrqkPNBRQK":
                width += 1
                piece_count += 1
            else:
                raise ValueError("invalid FEN board symbol")
        if width != 8:
            raise ValueError("FEN rank does not contain eight squares")
    if piece_count == 0:
        raise ValueError("FEN board contains no pieces")
    return piece_count


def fen_phase_category(fen: str) -> str:
    phase = max(0, min(7, (fen_piece_count(fen) - 1) // 4))
    return f"phase{phase}"


def validate_sealed_dataset_manifest(
    dataset_dir: Path,
    datasets: dict[str, Path],
    allow_unsealed: bool = False,
) -> dict:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        if allow_unsealed:
            return {
                "sealed": False,
                "allow_unsealed_dataset": True,
            }
        raise RuntimeError(
            f"missing sealed V42 dataset manifest: {path}; "
            "use --allow-unsealed-dataset only for tests")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid dataset manifest JSON: {path}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("dataset manifest must be a JSON object")
    if manifest.get("schema_version") != 2:
        raise RuntimeError("dataset manifest schema_version must be 2")
    if manifest.get("format") != "nnue-v42-lichess-game-disjoint-v2":
        raise RuntimeError("dataset manifest has the wrong sealed format")
    policy = manifest.get("selection_policy")
    if not isinstance(policy, dict):
        raise RuntimeError("dataset manifest lacks selection_policy")
    expected_policy = {
        "split_unit": "game_id",
        "positions_per_game": 1,
        "output_ply": "zero-based absolute ply derived from six-field FEN",
    }
    for name, expected in expected_policy.items():
        if policy.get(name) != expected:
            raise RuntimeError(
                f"dataset manifest selection_policy.{name} is not {expected!r}")

    overlap_keys = {
        "tune_selection", "tune_holdout", "selection_holdout"}
    for field in ("game_overlap", "position_overlap"):
        overlaps = manifest.get(field)
        if (
            not isinstance(overlaps, dict)
            or set(overlaps) != overlap_keys
            or any(value != 0 for value in overlaps.values())
        ):
            raise RuntimeError(f"dataset manifest {field} is not sealed at zero")

    split_manifests = manifest.get("splits")
    if not isinstance(split_manifests, dict):
        raise RuntimeError("dataset manifest lacks split statistics")
    for split, dataset in datasets.items():
        split_manifest = split_manifests.get(split)
        if not isinstance(split_manifest, dict):
            raise RuntimeError(f"dataset manifest lacks split {split}")
        actual_count = 0
        actual_phase_counts: Counter[str] = Counter()
        for line_number, line in enumerate(dataset.read_text().splitlines(), 1):
            category, position_hash, ply, payload = parse_dataset_row(
                line, dataset, line_number)
            if category not in {f"phase{index}" for index in range(8)}:
                raise RuntimeError(
                    f"{dataset}:{line_number}: sealed V42 category must be phase0..7")
            if payload.startswith("book:"):
                raise RuntimeError(
                    f"{dataset}:{line_number}: sealed V42 row must contain FEN")
            try:
                canonical_hash = len(position_hash) == 32 and int(
                    position_hash, 16) >= 0
            except ValueError:
                canonical_hash = False
            if not canonical_hash or position_hash != position_hash.lower():
                raise RuntimeError(
                    f"{dataset}:{line_number}: invalid canonical position hash")
            try:
                payload_ply = fen_absolute_ply(payload)
                expected_category = fen_phase_category(payload)
            except (ValueError, IndexError) as error:
                raise RuntimeError(
                    f"{dataset}:{line_number}: invalid sealed V42 FEN") from error
            if category != expected_category:
                raise RuntimeError(
                    f"{dataset}:{line_number}: phase label {category!r} != "
                    f"piece-count phase {expected_category!r}")
            if payload_ply != ply:
                raise RuntimeError(
                    f"{dataset}:{line_number}: explicit ply {ply} != FEN ply "
                    f"{payload_ply}")
            actual_count += 1
            actual_phase_counts[category.removeprefix("phase")] += 1
        if split_manifest.get("count") != actual_count:
            raise RuntimeError(
                f"dataset manifest count mismatch for {split}: "
                f"{split_manifest.get('count')} != {actual_count}")
        if split_manifest.get("games") != actual_count:
            raise RuntimeError(
                f"dataset manifest violates one-position-per-game for {split}")
        declared_phase_counts = split_manifest.get("phase_counts")
        if declared_phase_counts is not None:
            if (
                not isinstance(declared_phase_counts, dict)
                or any(
                    key not in {str(index) for index in range(8)}
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for key, value in declared_phase_counts.items()
                )
            ):
                raise RuntimeError(
                    f"dataset manifest has invalid phase_counts for {split}")
            if dict(sorted(actual_phase_counts.items())) != dict(
                sorted(declared_phase_counts.items())
            ):
                raise RuntimeError(
                    f"dataset manifest phase_counts mismatch for {split}")
    return {
        "sealed": True,
        "manifest": file_fingerprint(path),
        "format": manifest["format"],
        "schema_version": manifest["schema_version"],
    }


def fixed_subset(source: Path, total: int, seed: int) -> list[str]:
    if total <= 0:
        raise ValueError("subset size must be positive")
    groups = load_by_category(source)
    available = sum(len(rows) for rows in groups.values())
    if total > available:
        raise RuntimeError(
            f"not enough rows in {source}: need {total}, got {available}")
    # Preserve the source's empirical mixture.  The old hard-coded category
    # weights distorted natural Lichess distributions and made the tune target
    # a synthetic workload.  Largest remainder gives an exact, deterministic
    # total without rotating categories between runs.
    quotas = {
        category: total * len(rows) / available
        for category, rows in groups.items()
    }
    counts = {category: int(math.floor(quota)) for category, quota in quotas.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        groups,
        key=lambda category: (-(quotas[category] - counts[category]), category),
    )
    for category in order[:remainder]:
        counts[category] += 1
    selected: list[str] = []
    for category in sorted(counts):
        count = counts[category]
        candidates = groups.get(category, []).copy()
        random.Random(seed ^ sum(map(ord, category))).shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(
                f"not enough {category} rows in {source}: "
                f"need {count}, got {len(candidates)}")
        selected.extend(candidates[:count])
    random.Random(seed).shuffle(selected)
    return selected


def category_counts(path: Path) -> dict[str, int]:
    return {
        category: len(rows)
        for category, rows in sorted(load_by_category(path).items())
    }


def ensure_subset(path: Path, source: Path, total: int, seed: int) -> None:
    expected = ("\n".join(fixed_subset(source, total, seed)) + "\n").encode()
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(
                f"existing rung dataset does not match seed/size/source: {path}")
        return
    path.write_bytes(expected)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def quantize(value: float, quantum: int, lower: int, upper: int) -> int:
    return clamp(int(round(value / quantum)) * quantum, lower, upper)


def anchors(max_researches: int, mean_score_clamp_cp: int) -> list[Proposal]:
    guard = {
        "max_researches": max_researches,
        "mean_score_clamp_cp": mean_score_clamp_cp,
    }
    seed = Config(**guard)
    return [
        Proposal(replace(seed, enabled=False), "anchor:legacy-v41"),
        Proposal(seed, "anchor:v42-seed"),
        Proposal(replace(
            seed, min_depth=4, delta_base_cp=14,
            delta_divisor=12_900, expansion_factor_per_mille=1_740,
            max_fail_high_reductions=3), "anchor:plenty-like"),
        Proposal(replace(
            seed, max_fail_high_reductions=0),
            "anchor:no-fail-high-depth-reduction"),
        Proposal(replace(
            seed, mean_score_new_weight_per_mille=0),
            "anchor:no-mean-score"),
        Proposal(replace(
            seed, delta_base_cp=50, delta_divisor=40_000,
            expansion_factor_per_mille=2_000), "anchor:wide-window"),
    ]


def _integer_stratum(
    index: int, count: int, rng: random.Random, lower: int, upper: int
) -> int:
    point = (index + rng.random()) / count
    return min(upper, lower + int(point * (upper - lower + 1)))


def latin_hypercube_configs(
    count: int,
    seed: int,
    max_researches: int = 6,
    mean_score_clamp_cp: int = 1_500,
) -> list[Config]:
    if count < 0:
        raise ValueError("Latin-hypercube count must be non-negative")
    if count == 0:
        return []
    rng = random.Random(seed)
    columns: list[list[int]] = []
    builders = (
        lambda i: _integer_stratum(i, count, rng, 2, 5),
        lambda i: _integer_stratum(i, count, rng, 10, 90),
        lambda i: quantize(
            math.exp(
                math.log(4_000)
                + (i + rng.random()) / count
                * (math.log(40_000) - math.log(4_000))),
            100, 4_000, 40_000),
        lambda i: quantize(
            1_300 + (i + rng.random()) / count * 1_700,
            10, 1_300, 3_000),
        lambda i: _integer_stratum(i, count, rng, 0, 3),
        lambda i: quantize(
            (i + rng.random()) / count * 1_000,
            10, 0, 1_000),
    )
    for builder in builders:
        values = [builder(index) for index in range(count)]
        rng.shuffle(values)
        columns.append(values)
    result: list[Config] = []
    seen: set[str] = set()
    for row in range(count):
        config = Config(
            min_depth=columns[0][row],
            delta_base_cp=columns[1][row],
            delta_divisor=columns[2][row],
            expansion_factor_per_mille=columns[3][row],
            max_fail_high_reductions=columns[4][row],
            mean_score_new_weight_per_mille=columns[5][row],
            max_researches=max_researches,
            mean_score_clamp_cp=mean_score_clamp_cp,
        )
        if config.hash not in seen:
            seen.add(config.hash)
            result.append(config)
    # Integer dimensions can theoretically collide.  Fill deterministically
    # instead of silently shrinking the exploration plan.
    while len(result) < count:
        config = random_config(rng, max_researches, mean_score_clamp_cp)
        if config.hash not in seen:
            seen.add(config.hash)
            result.append(config)
    return result


def random_config(
    rng: random.Random,
    max_researches: int,
    mean_score_clamp_cp: int,
) -> Config:
    return Config(
        min_depth=rng.randint(2, 5),
        delta_base_cp=rng.randint(10, 90),
        delta_divisor=quantize(
            math.exp(rng.uniform(math.log(4_000), math.log(40_000))),
            100, 4_000, 40_000),
        expansion_factor_per_mille=quantize(
            rng.uniform(1_300, 3_000), 10, 1_300, 3_000),
        max_fail_high_reductions=rng.randint(0, 3),
        mean_score_new_weight_per_mille=quantize(
            rng.uniform(0, 1_000), 10, 0, 1_000),
        max_researches=max_researches,
        mean_score_clamp_cp=mean_score_clamp_cp,
    )


def global_proposals(
    total: int,
    seed: int,
    max_researches: int,
    mean_score_clamp_cp: int,
) -> list[Proposal]:
    result = anchors(max_researches, mean_score_clamp_cp)
    if total < len(result):
        raise ValueError(
            f"--global-candidates must be at least {len(result)}")
    lhs = latin_hypercube_configs(
        total - len(result), seed ^ 0x4C4853,
        max_researches, mean_score_clamp_cp)
    result.extend(Proposal(config, "global:latin-hypercube") for config in lhs)
    result = unique_proposals(result)
    rng = random.Random(seed ^ 0x474C4F42414C)
    seen = {proposal.config.hash for proposal in result}
    while len(result) < total:
        config = random_config(
            rng, max_researches, mean_score_clamp_cp)
        if config.hash not in seen:
            seen.add(config.hash)
            result.append(Proposal(config, "global:collision-fill"))
    return result


def preflight_proposals(
    max_researches: int, mean_score_clamp_cp: int
) -> list[Proposal]:
    result = anchors(max_researches, mean_score_clamp_cp)
    guard = {
        "max_researches": max_researches,
        "mean_score_clamp_cp": mean_score_clamp_cp,
    }
    extremes = (
        Config( min_depth=2, delta_base_cp=10, delta_divisor=4_000,
            expansion_factor_per_mille=1_300, max_fail_high_reductions=0,
            mean_score_new_weight_per_mille=0, **guard),
        Config(min_depth=5, delta_base_cp=90, delta_divisor=40_000,
            expansion_factor_per_mille=3_000, max_fail_high_reductions=3,
            mean_score_new_weight_per_mille=1_000, **guard),
        Config(min_depth=2, delta_base_cp=10, delta_divisor=40_000,
            expansion_factor_per_mille=1_300, max_fail_high_reductions=3,
            mean_score_new_weight_per_mille=1_000, **guard),
        Config(min_depth=5, delta_base_cp=90, delta_divisor=4_000,
            expansion_factor_per_mille=3_000, max_fail_high_reductions=0,
            mean_score_new_weight_per_mille=0, **guard),
    )
    result.extend(
        Proposal(config, f"preflight:extreme-{index}")
        for index, config in enumerate(extremes))
    return unique_proposals(result)


def unique_proposals(proposals: Iterable[Proposal]) -> list[Proposal]:
    result: list[Proposal] = []
    seen: set[str] = set()
    for proposal in proposals:
        if proposal.config.hash not in seen:
            seen.add(proposal.config.hash)
            result.append(proposal)
    return result


MUTABLE_FIELDS = (
    "min_depth",
    "delta_base_cp",
    "delta_divisor",
    "expansion_factor_per_mille",
    "max_fail_high_reductions",
    "mean_score_new_weight_per_mille",
)


def mutate(config: Config, changed: int, rng: random.Random) -> Config:
    if not config.enabled:
        config = replace(config, enabled=True)
    values = config.canonical()
    for name in rng.sample(MUTABLE_FIELDS, changed):
        original = int(values[name])
        if name == "min_depth":
            candidates = [clamp(original + step, 2, 5) for step in (-1, 1)]
        elif name == "delta_base_cp":
            candidates = [
                clamp(original + step, 10, 90)
                for step in (-20, -10, -5, 5, 10, 20)
            ]
        elif name == "delta_divisor":
            candidates = [
                quantize(original * factor, 100, 4_000, 40_000)
                for factor in (0.70, 0.85, 1.15, 1.40)
            ]
        elif name == "expansion_factor_per_mille":
            candidates = [
                quantize(original + step, 10, 1_300, 3_000)
                for step in (-400, -200, -100, 100, 200, 400)
            ]
        elif name == "max_fail_high_reductions":
            candidates = [clamp(original + step, 0, 3) for step in (-1, 1)]
        else:
            candidates = [
                quantize(original + step, 10, 0, 1_000)
                for step in (-200, -100, 100, 200)
            ]
        candidates = sorted(set(value for value in candidates if value != original))
        if candidates:
            values[name] = rng.choice(candidates)
    return Config.from_dict(values)


def crossover(left: Config, right: Config, rng: random.Random) -> Config:
    values = left.canonical()
    values["enabled"] = True
    for name in MUTABLE_FIELDS:
        values[name] = rng.choice((getattr(left, name), getattr(right, name)))
    # Guards come from the run manifest, and parents are already checked to
    # share them.  They are never inherited independently.
    values["max_researches"] = left.max_researches
    values["mean_score_clamp_cp"] = left.mean_score_clamp_cp
    return Config.from_dict(values)


def config_from_record(record: dict) -> Config:
    config = Config.from_dict(record["config"])
    if record.get("config_hash") != config.hash:
        raise RuntimeError("record config hash does not match canonical config")
    return config


def result_metrics(entry: dict) -> tuple[float, float]:
    result = entry["result"]
    return float(result["mean_wdl_loss"]), float(result["node_ratio"])


def dominates(left: dict, right: dict) -> bool:
    left_loss, left_nodes = result_metrics(left)
    right_loss, right_nodes = result_metrics(right)
    return (
        left_loss <= right_loss
        and left_nodes <= right_nodes
        and (left_loss < right_loss or left_nodes < right_nodes)
    )


def frontier(entries: Sequence[dict]) -> list[dict]:
    """Return the complete exact WDL/node Pareto frontier (never cap it)."""
    return [
        entry for entry in entries
        if not any(
            other is not entry and dominates(other, entry)
            for other in entries)
    ]


def deduplicate_objectives(entries: Sequence[dict]) -> list[dict]:
    """Retain every config unless a future evaluator emits a full signature.

    Equal aggregate node counts and rounded mean WDL loss do not prove equal
    per-position behaviour.  Collapsing on those two numbers can discard a
    latent config which separates at the next rung.
    """
    return list(entries)


def adaptive_frontier(entries: Sequence[dict]) -> list[dict]:
    """Pareto-rank V42 configs independently from the legacy control.

    The legacy baseline is useful for reporting but must not erase every
    adaptive parent or promotion candidate when it dominates them offline.
    """
    adaptive = [
        entry for entry in entries if config_from_record(entry).enabled
    ]
    return deduplicate_objectives(frontier(adaptive))


def refinement_proposals(
    parent_entries: Sequence[dict],
    count: int,
    seed: int,
    already_seen: set[str],
    max_researches: int,
    mean_score_clamp_cp: int,
) -> list[Proposal]:
    if count < 0:
        raise ValueError("refinement count must be non-negative")
    parents = [
        config_from_record(entry) for entry in adaptive_frontier(parent_entries)
    ]
    if not parents:
        parents = [anchors(max_researches, mean_score_clamp_cp)[1].config]
    if any(
        parent.max_researches != max_researches
        or parent.mean_score_clamp_cp != mean_score_clamp_cp
        for parent in parents
    ):
        raise RuntimeError("frontier changed a frozen aspiration guard")

    rng = random.Random(seed ^ 0x524546494E45)
    seen = set(already_seen)
    proposals: list[Proposal] = []
    attempts = 0
    slot_attempts = 0
    while len(proposals) < count:
        attempts += 1
        slot_attempts += 1
        if attempts > max(10_000, count * 1_000):
            raise RuntimeError("could not generate enough unique refinements")
        fraction = (len(proposals) + 0.5) / max(1, count)
        if slot_attempts > 200:
            method = "random-restart"
            candidate = random_config(
                rng, max_researches, mean_score_clamp_cp)
        elif fraction < 0.30:
            method = "mutate-1"
            candidate = mutate(rng.choice(parents), 1, rng)
        elif fraction < 0.60:
            method = "mutate-2"
            candidate = mutate(rng.choice(parents), 2, rng)
        elif fraction < 0.75:
            method = "mutate-3"
            candidate = mutate(rng.choice(parents), 3, rng)
        elif fraction < 0.85 and len(parents) >= 2:
            method = "crossover"
            left, right = rng.sample(parents, 2)
            candidate = crossover(left, right, rng)
        else:
            method = "random-restart"
            candidate = random_config(
                rng, max_researches, mean_score_clamp_cp)
        if candidate.hash in seen:
            continue
        seen.add(candidate.hash)
        proposals.append(Proposal(candidate, f"refine:{method}"))
        slot_attempts = 0
    return proposals


def refinement_generation_counts(total: int) -> tuple[int, int]:
    if total < 0:
        raise ValueError("refinement count must be non-negative")
    return (total + 1) // 2, total // 2


def validate_result(
    result: dict,
    config: Config,
    depth: int,
    candidate_time_ms: int,
    control_cache: dict,
) -> None:
    required = {
        "count", "ranking_count", "safety_count",
        "ranking_target_abs_cp", "include_all_in_objective",
        "objective", "wdl_formula", "wdl_calibration_run",
        "critical_mistakes", "safety_critical_mistakes",
        "win_to_draw", "win_to_loss", "self_mated",
        "heuristics_reset_per_sample", "candidate_search_mode",
        "candidate_time_ms", "mean_wdl_loss", "p95_wdl_loss",
        "objective_loss", "node_ratio",
        "control_nodes", "candidate_nodes",
        "dataset_ply_schema", "dataset_explicit_ply_count",
        "dataset_explicit_ply_position_match_count",
        "dataset_calibration_ply_min", "dataset_calibration_ply_max",
        "dataset_calibration_ply_mean", "selective_config",
        "twofold_search_draw_enabled", "aspiration_config",
        "control_exact_searches", "control_exact_full_window_fallbacks",
        "control_exact_range_conflicts", "control_exact_unresolved_ranges",
        "control_root_source", "control_time_source",
        "control_cache_identity_sha256", "control_live_root_searches",
        "control_live_child_searches", "control_cached_root_results",
        "time_ratio",
        "strict_best_score_violations", "strict_best_score_max_excess_cp",
        "candidate_repetition_history_aware_searches",
        "candidate_repetition_threefold_draws",
        "candidate_repetition_search_cycle_draws",
        "candidate_repetition_fifty_move_draws",
        "candidate_repetition_tt_score_suppressions",
        *TELEMETRY_FIELDS,
    }
    missing = sorted(required - set(result))
    if missing:
        raise RuntimeError(f"evaluator result missing fields: {missing}")
    count = int(result["count"])
    ranking_count = int(result["ranking_count"])
    if count <= 0:
        raise RuntimeError("evaluator returned no positions")
    if ranking_count != count:
        raise RuntimeError("evaluator excluded positions from all-position objective")
    if result["objective"] != "wdl":
        raise RuntimeError("evaluator objective is not WDL")
    if result["wdl_formula"] != WDL_FORMULA:
        raise RuntimeError("evaluator WDL formula differs from frozen profile")
    if result["wdl_calibration_run"] != WDL_CALIBRATION_RUN:
        raise RuntimeError(
            "evaluator WDL calibration run differs from frozen profile")
    if int(result["ranking_target_abs_cp"]) != RANKING_TARGET_ABS_CP:
        raise RuntimeError("evaluator safety threshold differs from frozen profile")
    safety_count = int(result["safety_count"])
    critical_mistakes = int(result["critical_mistakes"])
    safety_critical_mistakes = int(result["safety_critical_mistakes"])
    if not 0 <= safety_count <= count:
        raise RuntimeError("evaluator returned invalid safety_count")
    if not 0 <= critical_mistakes <= count:
        raise RuntimeError("evaluator returned invalid critical_mistakes")
    if not 0 <= safety_critical_mistakes <= min(
        safety_count, critical_mistakes
    ):
        raise RuntimeError(
            "evaluator returned invalid safety_critical_mistakes")
    for metric in ("win_to_draw", "win_to_loss", "self_mated"):
        if not 0 <= int(result[metric]) <= critical_mistakes:
            raise RuntimeError(f"evaluator returned invalid {metric}")
    if result["dataset_ply_schema"] != DATASET_SCHEMA:
        raise RuntimeError(
            "evaluator did not consume the required explicit-ply dataset schema")
    if int(result["dataset_explicit_ply_count"]) != int(result["count"]):
        raise RuntimeError("evaluator silently derived ply from payload/FEN")
    if int(result["dataset_explicit_ply_position_match_count"]) != int(
        result["count"]
    ):
        raise RuntimeError("explicit dataset ply disagrees with parsed position")
    calibration_ply_min = int(result["dataset_calibration_ply_min"])
    calibration_ply_max = int(result["dataset_calibration_ply_max"])
    calibration_ply_mean = float(result["dataset_calibration_ply_mean"])
    if (
        calibration_ply_min < 0
        or calibration_ply_max < calibration_ply_min
        or not math.isfinite(calibration_ply_mean)
        or not calibration_ply_min <= calibration_ply_mean <= calibration_ply_max
    ):
        raise RuntimeError("evaluator returned invalid explicit-ply statistics")
    if result["include_all_in_objective"] is not True:
        raise RuntimeError("evaluator did not enable the all-position objective")
    if result["heuristics_reset_per_sample"] is not True:
        raise RuntimeError("evaluator leaked search heuristics across samples")
    if result["candidate_search_mode"] != "iterative_depth":
        raise RuntimeError("V42 evaluator bypassed iterative aspiration search")
    if int(result["candidate_time_ms"]) != candidate_time_ms:
        raise RuntimeError("evaluator did not honor candidate-time mode")
    if result["control_root_source"] != "immutable_cache":
        raise RuntimeError("evaluator did not consume the immutable root cache")
    if result["control_time_source"] != "not_measured_cached":
        raise RuntimeError("evaluator reported a cached control wall time")
    if result["time_ratio"] is not None:
        raise RuntimeError("cached evaluator must report time_ratio=null")
    if (
        result["control_cache_identity_sha256"]
        != control_cache["identity_sha256"]
    ):
        raise RuntimeError("evaluator consumed the wrong root-cache identity")
    effective_selective = result["selective_config"]
    if (
        not isinstance(effective_selective, dict)
        or set(effective_selective) != set(SELECTIVE_CONFIG_FIELDS)
        or effective_selective != PRODUCTION_SELECTIVE_CONFIG
    ):
        raise RuntimeError(
            "evaluator effective selective config differs from production V41")
    if result["twofold_search_draw_enabled"] is not TWOFOLD_SEARCH_DRAW_ENABLED:
        raise RuntimeError("evaluator effective twofold-search policy differs")
    effective = Config.from_dict(result["aspiration_config"])
    if effective.canonical() != config.canonical():
        raise RuntimeError(
            "evaluator effective aspiration config differs from requested config")
    for metric in (
        "mean_wdl_loss", "p95_wdl_loss", "objective_loss", "node_ratio"
    ):
        if not math.isfinite(float(result[metric])):
            raise RuntimeError(f"non-finite evaluator metric: {metric}")
    if not 0.0 <= float(result["mean_wdl_loss"]) <= 1.0:
        raise RuntimeError("mean_wdl_loss is outside [0, 1]")
    if not 0.0 <= float(result["p95_wdl_loss"]) <= 1.0:
        raise RuntimeError("p95_wdl_loss is outside [0, 1]")
    if not math.isclose(
        float(result["objective_loss"]), float(result["mean_wdl_loss"]),
        rel_tol=2e-6, abs_tol=1e-12,
    ):
        raise RuntimeError("objective_loss differs from mean_wdl_loss")
    control_nodes = int(result["control_nodes"])
    candidate_nodes = int(result["candidate_nodes"])
    if control_nodes <= 0 or candidate_nodes <= 0:
        raise RuntimeError("evaluator returned non-positive node totals")
    cache_summary = control_cache.get("summary")
    required_cache_summary = {
        "count", "depth", "control_nodes", "control_exact_searches",
        "control_exact_full_window_fallbacks",
        "control_exact_range_conflicts", "control_exact_unresolved_ranges",
    }
    if (
        not isinstance(cache_summary, dict)
        or not required_cache_summary <= set(cache_summary)
        or int(cache_summary.get("control_nodes", -1)) != control_nodes
    ):
        raise RuntimeError(
            "evaluator control nodes differ from immutable root cache")
    if (
        int(cache_summary.get("count", -1)) != count
        or int(cache_summary.get("depth", -1)) != depth
        or int(cache_summary.get("control_exact_searches", -1)) != count
        or int(cache_summary.get(
            "control_exact_unresolved_ranges", -1)) != 0
    ):
        raise RuntimeError("immutable root-cache summary is inconsistent")
    expected_node_ratio = candidate_nodes / control_nodes
    if not math.isclose(
        float(result["node_ratio"]), expected_node_ratio,
        # The evaluator emits max_digits10, so its JSON double must round-trip
        # to the quotient implied by the two exact integer node totals.
        rel_tol=1e-12, abs_tol=1e-15,
    ):
        raise RuntimeError("evaluator node_ratio disagrees with node totals")
    if int(result["aspiration_unresolved_ranges"]) != 0:
        raise RuntimeError("adaptive aspiration returned an unresolved range")
    control_exact_searches = int(result["control_exact_searches"])
    control_exact_full_window_fallbacks = int(
        result["control_exact_full_window_fallbacks"])
    control_exact_range_conflicts = int(result["control_exact_range_conflicts"])
    control_exact_unresolved_ranges = int(
        result["control_exact_unresolved_ranges"])
    control_live_root_searches = int(result["control_live_root_searches"])
    control_live_child_searches = int(result["control_live_child_searches"])
    control_cached_root_results = int(result["control_cached_root_results"])
    if control_exact_searches < count:
        raise RuntimeError("strict teacher skipped one or more exact root searches")
    if control_cached_root_results != count:
        raise RuntimeError("evaluator skipped one or more cached root results")
    if control_live_root_searches != 0:
        raise RuntimeError("cached evaluator unexpectedly reran a live root")
    if control_live_child_searches < 0:
        raise RuntimeError("evaluator returned a negative live-child count")
    if (
        control_exact_searches
        != control_cached_root_results
            + control_live_root_searches
            + control_live_child_searches
    ):
        raise RuntimeError(
            "strict exact-search accounting does not match cached roots and "
            "fresh candidate-child searches")
    if (
        control_exact_full_window_fallbacks < 0
        or control_exact_range_conflicts < 0
        or control_exact_unresolved_ranges < 0
    ):
        raise RuntimeError("strict teacher returned invalid exactness counters")
    if control_exact_unresolved_ranges != 0:
        raise RuntimeError("strict teacher returned an unresolved range")
    if (
        control_exact_full_window_fallbacks
            < int(cache_summary["control_exact_full_window_fallbacks"])
        or control_exact_range_conflicts
            < int(cache_summary["control_exact_range_conflicts"])
    ):
        raise RuntimeError(
            "strict exactness totals omit cached root diagnostics")
    if (
        int(result["strict_best_score_violations"]) != 0
        or int(result["strict_best_score_max_excess_cp"]) != 0
    ):
        raise RuntimeError("strict move score exceeded strict best score")
    if int(result["candidate_repetition_history_aware_searches"]) != int(
        result["count"]
    ):
        raise RuntimeError("candidate search bypassed repetition-history overload")
    if candidate_time_ms == 0:
        if float(result.get("candidate_stopped_pct", 0.0)) != 0.0:
            raise RuntimeError("fixed-depth V42 evaluation stopped early")
        if config.enabled:
            completed_iterations = int(
                result["aspiration_completed_iterations"])
            final_mean_score_count = int(
                result["aspiration_final_mean_score_count"])
            narrow_iterations = int(result["aspiration_narrow_iterations"])
            narrow_attempts = int(result["aspiration_narrow_attempts"])
            initial_successes = int(
                result["aspiration_initial_window_successes"])
            accepted_reduced = int(
                result["aspiration_accepted_reduced_depth_iterations"])
            nominal_depth_sum = int(
                result["aspiration_accepted_narrow_nominal_depth_sum"])
            search_depth_sum = int(
                result["aspiration_accepted_narrow_search_depth_sum"])
            max_reduction = int(
                result["aspiration_max_accepted_depth_reduction"])
            if completed_iterations != count * depth:
                raise RuntimeError(
                    "fixed-depth aspiration iteration accounting mismatch")
            if final_mean_score_count != count:
                raise RuntimeError(
                    "fixed-depth aspiration mean-score accounting mismatch")
            if narrow_attempts < narrow_iterations:
                raise RuntimeError(
                    "aspiration attempts are fewer than narrow iterations")
            if initial_successes > narrow_iterations:
                raise RuntimeError(
                    "aspiration initial successes exceed narrow iterations")
            if accepted_reduced > narrow_iterations:
                raise RuntimeError(
                    "accepted reduced-depth count exceeds narrow iterations")
            if search_depth_sum > nominal_depth_sum:
                raise RuntimeError(
                    "accepted aspiration search depth exceeds nominal depth")
            if not 0 <= max_reduction <= config.max_fail_high_reductions:
                raise RuntimeError(
                    "accepted aspiration depth reduction exceeds configured cap")
            if depth >= config.min_depth and narrow_iterations <= 0:
                raise RuntimeError("enabled aspiration never opened a narrow window")
            if depth >= config.min_depth and narrow_attempts <= 0:
                raise RuntimeError("enabled aspiration never attempted narrow search")
        else:
            nonzero = {
                name: result[name]
                for name in TELEMETRY_FIELDS
                if int(result[name]) != 0
            }
            if nonzero:
                raise RuntimeError(
                    "disabled aspiration returned adaptive telemetry: "
                    f"{sorted(nonzero)}")


def evaluate(
    binary: Path,
    dataset: Path,
    model: Path,
    depth: int,
    config: Config,
    candidate_time_ms: int,
    candidate_max_depth: int,
    control_cache: dict,
    deadline: float | None = None,
) -> dict:
    command = [
        str(binary),
        "--dataset", str(dataset),
        "--model", str(model),
        "--depth", str(depth),
        "--objective", "wdl",
        "--include-all-in-objective",
        "--ranking-target-abs-cp", str(RANKING_TARGET_ABS_CP),
        *control_cache_command_args(control_cache),
        *selective_evaluator_args(),
        *config.evaluator_args(),
    ]
    if candidate_time_ms:
        command.extend([
            "--candidate-time-ms", str(candidate_time_ms),
            "--candidate-max-depth", str(candidate_max_depth),
        ])
    try:
        timeout = None
        if deadline is not None:
            timeout = max(0.001, deadline - time.monotonic())
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        if deadline is not None:
            raise EvaluationBudgetExhausted(
                "candidate evaluator reached the shared budget deadline"
            ) from error
        raise EvaluationFailure(
            "evaluator timed out",
            command=command,
            stdout=stdout,
            stderr=stderr,
        ) from error
    except OSError as error:
        raise EvaluationFailure(
            f"failed to launch evaluator: {error}", command=command) from error
    if completed.returncode:
        message = (
            f"evaluator terminated by signal {-completed.returncode}"
            if completed.returncode < 0
            else "evaluation failed"
        )
        raise EvaluationFailure(
            message,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise EvaluationFailure(
            "evaluator produced invalid JSON",
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        ) from error
    try:
        validate_result(
            result, config, depth, candidate_time_ms, control_cache)
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise EvaluationFailure(
            f"evaluator result validation failed: {error}",
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        ) from error
    return result


def proposal_payload(proposal: Proposal) -> dict:
    return {
        "config": proposal.config.canonical(),
        "config_hash": proposal.config.hash,
        "label": proposal.config.label,
        "origin": proposal.origin,
    }


def ensure_plan(
    records: list[dict],
    log_path: Path,
    stage: str,
    proposals: Sequence[Proposal],
    dataset: Path,
    depth: int,
    candidate_time_ms: int,
    budget: Budget,
    control_cache: dict,
) -> dict:
    expected = {
        "kind": "plan",
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "dataset": file_fingerprint(dataset),
        "depth": depth,
        "candidate_time_ms": candidate_time_ms,
        "control_cache": control_cache,
        "candidates": [proposal_payload(item) for item in unique_proposals(proposals)],
    }
    matches = [
        record for record in records
        if record.get("kind") == "plan" and record.get("stage") == stage
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple plans found for stage {stage}")
    if matches:
        actual = {key: matches[0].get(key) for key in expected}
        if canonical_json(actual) != canonical_json(expected):
            raise RuntimeError(f"saved plan mismatch for stage {stage}")
        return matches[0]
    record = {**expected, "budget_elapsed_sec": budget.elapsed_sec}
    append_jsonl(log_path, record)
    records.append(record)
    return record


def completed_results(records: Sequence[dict], stage: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for record in records:
        if record.get("kind") != "result" or record.get("stage") != stage:
            continue
        config = config_from_record(record)
        previous = result.get(config.hash)
        if previous is not None and canonical_json(previous) != canonical_json(record):
            raise RuntimeError(
                f"conflicting duplicate result for {stage}/{config.hash}")
        result[config.hash] = record
    return result


def legacy_rejected_results(records: Sequence[dict], stage: str) -> dict[str, dict]:
    """Load old reject records solely so fail-closed resume can refuse them."""
    rejected: dict[str, dict] = {}
    for record in records:
        if record.get("kind") != "reject" or record.get("stage") != stage:
            continue
        config = config_from_record(record)
        previous = rejected.get(config.hash)
        if previous is not None and canonical_json(previous) != canonical_json(record):
            raise RuntimeError(
                f"conflicting duplicate reject for {stage}/{config.hash}")
        rejected[config.hash] = record
    return rejected


def run_stage(
    records: list[dict],
    log_path: Path,
    plan: dict,
    binary: Path,
    model: Path,
    workers: int,
    candidate_max_depth: int,
    budget: Budget,
) -> tuple[list[dict], bool]:
    stage = plan["stage"]
    done = completed_results(records, stage)
    rejected = legacy_rejected_results(records, stage)
    if rejected:
        first_reject = next(iter(rejected))
        raise RuntimeError(
            "saved per-config evaluator reject is incompatible with the "
            f"fail-closed policy: {stage}/{first_reject}; use a new run directory")
    proposals = [
        Proposal(Config.from_dict(item["config"]), item["origin"])
        for item in plan["candidates"]
    ]
    for item, proposal in zip(plan["candidates"], proposals):
        if item["config_hash"] != proposal.config.hash:
            raise RuntimeError(f"plan config hash mismatch in stage {stage}")
    plan_hashes = {proposal.config.hash for proposal in proposals}
    unexpected = set(done) - plan_hashes
    if unexpected:
        raise RuntimeError(
            f"stage {stage} contains outcome outside saved plan: "
            f"{sorted(unexpected)[0]}")
    for outcome in done.values():
        if (
            outcome.get("schema_version") != SCHEMA_VERSION
            or outcome.get("dataset_sha256") != plan["dataset"]["sha256"]
            or int(outcome.get("depth", -1)) != int(plan["depth"])
            or int(outcome.get("candidate_time_ms", -1))
                != int(plan["candidate_time_ms"])
            or outcome.get("control_cache_sha256")
                != plan["control_cache"]["artifact"]["sha256"]
            or outcome.get("control_cache_identity_sha256")
                != plan["control_cache"]["identity_sha256"]
        ):
            raise RuntimeError(
                f"saved outcome metadata mismatch for "
                f"{stage}/{outcome.get('config_hash')}")
    resolved = set(done)
    pending = [item for item in proposals if item.config.hash not in resolved]
    dataset = Path(plan["dataset"]["path"])
    depth = int(plan["depth"])
    candidate_time_ms = int(plan["candidate_time_ms"])
    control_cache = plan["control_cache"]
    cache_path = Path(control_cache["path"])
    if file_fingerprint(cache_path) != control_cache["artifact"]:
        raise RuntimeError(
            f"immutable control cache changed before stage {stage}")
    validate_control_cache(cache_path, control_cache, dataset)

    while pending and not budget.exhausted:
        batch = pending[:workers]
        pending = pending[workers:]
        # Every worker in this batch shares one absolute wall deadline.  This
        # bounds batch overrun without accidentally granting each subprocess
        # a fresh copy of the remaining cumulative budget.
        batch_deadline = time.monotonic() + budget.remaining_sec

        def run(proposal: Proposal) -> tuple[Proposal, dict, float]:
            started = time.monotonic()
            try:
                result = evaluate(
                    binary, dataset, model, depth, proposal.config,
                    candidate_time_ms, candidate_max_depth, control_cache,
                    batch_deadline)
            except EvaluationBudgetExhausted:
                raise
            except EvaluationFailure as error:
                raise EvaluationFailure(
                    f"{stage}/{proposal.config.label}: {error}",
                    command=error.command,
                    returncode=error.returncode,
                    stdout=error.stdout,
                    stderr=error.stderr,
                ) from error
            return proposal, result, time.monotonic() - started

        budget_expired: list[Proposal] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(batch))) as pool:
            futures = {
                pool.submit(run, proposal): proposal for proposal in batch
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    proposal, result, wall_sec = future.result()
                except EvaluationBudgetExhausted:
                    budget_expired.append(futures[future])
                    continue
                entry = {
                    "kind": "result",
                    "schema_version": SCHEMA_VERSION,
                    "stage": stage,
                    "config": proposal.config.canonical(),
                    "config_hash": proposal.config.hash,
                    "label": proposal.config.label,
                    "origin": proposal.origin,
                    "dataset_sha256": plan["dataset"]["sha256"],
                    "depth": depth,
                    "candidate_time_ms": candidate_time_ms,
                    "control_cache_sha256":
                        control_cache["artifact"]["sha256"],
                    "control_cache_identity_sha256":
                        control_cache["identity_sha256"],
                    "wall_sec": wall_sec,
                    "budget_elapsed_sec": budget.elapsed_sec,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                }
                append_jsonl(log_path, entry)
                records.append(entry)
                done[proposal.config.hash] = entry
                print(
                    f"{stage}_progress label={proposal.config.label}"
                    f" origin={proposal.origin}"
                    f" nodes={result['node_ratio']:.6f}"
                    f" wdl={result['mean_wdl_loss']:.9f}"
                    f" attempts={result['aspiration_narrow_attempts']}"
                    f" fail_low={result['aspiration_fail_lows']}"
                    f" fail_high={result['aspiration_fail_highs']}"
                    f" reduced_accepted="
                    f"{result['aspiration_accepted_reduced_depth_iterations']}"
                    f" frontier={len(frontier(list(done.values())))}",
                    flush=True)
        if budget_expired:
            pending = [*budget_expired, *pending]
            break
    complete = not pending
    if complete and not any(
        record.get("kind") == "stage_complete"
        and record.get("stage") == stage for record in records
    ):
        entry = {
            "kind": "stage_complete",
            "stage": stage,
            "candidate_count": len(proposals),
            "result_count": len(done),
            "reject_count": 0,
            "frontier_count": len(frontier(list(done.values()))),
            "budget_elapsed_sec": budget.elapsed_sec,
        }
        append_jsonl(log_path, entry)
        records.append(entry)
    ordered = [done[item.config.hash] for item in proposals if item.config.hash in done]
    return ordered, complete


def manifest_for(
    args: argparse.Namespace,
    rung_paths: dict[str, Path],
    control_cache_specs: dict[str, dict],
) -> dict:
    dataset_sources = {
        "tune": args.dataset_dir / "tune.tsv",
        "selection": args.dataset_dir / "selection.tsv",
        "holdout": args.dataset_dir / "holdout.tsv",
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "dataset_schema": DATASET_SCHEMA,
        "sealed_dataset": args.sealed_dataset_identity,
        "objective": "all-position-wdl-node-pareto",
        "objective_sampling": "source-empirical-largest-remainder-v1",
        "wdl_formula": WDL_FORMULA,
        "wdl_calibration_run": WDL_CALIBRATION_RUN,
        "safety_target_abs_cp": RANKING_TARGET_ABS_CP,
        "tuner_source": file_fingerprint(Path(__file__)),
        "binary": file_fingerprint(args.binary),
        "model": file_fingerprint(args.model),
        "dataset_sources": {
            name: {
                **file_fingerprint(path),
                **validate_dataset_schema(path),
                "category_counts": category_counts(path),
            }
            for name, path in dataset_sources.items()
        },
        "rung_datasets": {
            name: {
                **file_fingerprint(path),
                **validate_dataset_schema(path),
                "category_counts": category_counts(path),
            }
            for name, path in rung_paths.items()
        },
        "control_cache_policy": {
            "schema": CONTROL_CACHE_SCHEMA,
            "prebuild": "once-per-rung-before-candidate-workers",
            "artifact": "immutable-atomic-sha256-verified",
            "cached_timing": False,
            "candidate_child_rescore": "fresh-strict-v36-depth-minus-one",
        },
        "control_cache_identities": control_cache_specs,
        "seed": args.seed,
        "budget_policy": BUDGET_POLICY,
        "workers": args.workers,
        "candidate_time_ms": args.candidate_time_ms,
        "candidate_max_depth": args.candidate_max_depth,
        "sizes": {
            "preflight": args.preflight_size,
            "exploration": args.exploration_size,
            "selection": args.selection_size,
            "holdout": args.holdout_size,
        },
        "depths": {
            "preflight": args.preflight_depth,
            "exploration": args.exploration_depth,
            "selection": args.selection_depth,
            "holdout": args.holdout_depth,
        },
        "candidate_counts": {
            "global": args.global_candidates,
            "refinement_total": args.refinement_candidates,
            "refinement_generations": list(
                refinement_generation_counts(args.refinement_candidates)),
        },
        "selective_config": PRODUCTION_SELECTIVE_CONFIG,
        "twofold_search_draw_enabled": TWOFOLD_SEARCH_DRAW_ENABLED,
        "config_fields": list(CONFIG_FIELDS),
        "telemetry_fields": list(TELEMETRY_FIELDS),
        "search_bounds": {
            "min_depth": [2, 5],
            "delta_base_cp": [10, 90],
            "delta_divisor": [4_000, 40_000, "log-sampled"],
            "expansion_factor_per_mille": [1_300, 3_000],
            "max_fail_high_reductions": [0, 3],
            "mean_score_new_weight_per_mille": [0, 1_000],
        },
        "frozen_guards": {
            "max_researches": args.max_researches,
            "mean_score_clamp_cp": args.mean_score_clamp_cp,
        },
    }
    return {**body, "manifest_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def ensure_manifest(path: Path, expected: dict) -> int | None:
    if not path.exists():
        atomic_write_json(path, expected)
        return None
    try:
        actual = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid manifest: {path}") from error

    # Schema-1 runs created before the cumulative-budget policy stored the
    # mutable duration ceiling directly in the otherwise immutable manifest.
    # Migrate only that exact legacy shape; every experiment-defining field
    # must still compare byte-for-byte after canonicalization.
    legacy_duration_sec: int | None = None
    migrated = False
    if (
        "budget_policy" in expected
        and "budget_policy" not in actual
        and "duration_sec" in actual
    ):
        legacy_duration_sec = int(actual["duration_sec"])
        actual_body = {
            key: value for key, value in actual.items()
            if key not in {"manifest_sha256", "duration_sec"}
        }
        actual_body["budget_policy"] = expected["budget_policy"]
        actual = {
            **actual_body,
            "manifest_sha256": hashlib.sha256(
                canonical_json(actual_body)).hexdigest(),
        }
        migrated = True
    if canonical_json(actual) != canonical_json(expected):
        differing = sorted(
            key for key in set(actual) | set(expected)
            if canonical_json(actual.get(key)) != canonical_json(expected.get(key)))
        raise RuntimeError(
            "run manifest mismatch; refusing unsafe resume; "
            f"changed top-level fields: {differing}")

    if migrated:
        atomic_write_json(path, expected)
    return legacy_duration_sec


def ensure_budget_limit(
    records: list[dict],
    log_path: Path,
    requested_limit_sec: int,
    legacy_limit_sec: int | None = None,
) -> None:
    """Record a monotonic cumulative budget ceiling.

    The ceiling is intentionally outside the immutable run manifest.  This
    lets an operator explicitly extend an exhausted run while preserving an
    append-only audit trail and refusing accidental budget reductions.
    """
    previous_limit = legacy_limit_sec or 0
    saw_event = False
    for record in records:
        if record.get("kind") != "budget_limit":
            continue
        saw_event = True
        if record.get("budget_policy_version") != BUDGET_POLICY_VERSION:
            raise RuntimeError("unsupported budget policy version in results log")
        try:
            limit = int(record["cumulative_limit_sec"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("invalid budget_limit record") from error
        if limit <= 0 or limit < previous_limit:
            raise RuntimeError("non-monotonic budget_limit record")
        previous_limit = limit

    if requested_limit_sec < previous_limit:
        raise RuntimeError(
            "--duration-sec is a cumulative ceiling and may not decrease; "
            f"previous={previous_limit} requested={requested_limit_sec}")
    if saw_event and requested_limit_sec == previous_limit:
        return

    event = {
        "kind": "budget_limit",
        "budget_policy_version": BUDGET_POLICY_VERSION,
        "cumulative_limit_sec": requested_limit_sec,
        "previous_limit_sec": previous_limit,
        "budget_elapsed_sec": max_budget_elapsed(records),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(log_path, event)
    records.append(event)


def max_budget_elapsed(records: Sequence[dict]) -> float:
    return max(
        (float(record.get("budget_elapsed_sec", 0.0)) for record in records),
        default=0.0)


def stage_complete(records: Sequence[dict], stage: str) -> bool:
    return any(
        record.get("kind") == "stage_complete"
        and record.get("stage") == stage for record in records)


def serializable_profile(entries: Sequence[dict]) -> list[dict]:
    return [
        {
            "label": entry["label"],
            "config_hash": entry["config_hash"],
            "config": entry["config"],
            "origin": entry["origin"],
            "result": entry["result"],
        }
        for entry in sorted(entries, key=lambda item: result_metrics(item)[1])
    ]


def relative_profile(entry: dict, legacy: dict) -> dict:
    payload = serializable_profile([entry])[0]
    result = entry["result"]
    legacy_result = legacy["result"]
    legacy_nodes = float(legacy_result["candidate_nodes"])
    payload["node_ratio_vs_legacy"] = (
        float(result["candidate_nodes"]) / legacy_nodes
        if legacy_nodes > 0.0 else math.inf)
    payload["wdl_loss_delta_vs_legacy"] = (
        float(result["mean_wdl_loss"])
        - float(legacy_result["mean_wdl_loss"]))
    return payload


def gauntlet_aspiration_profile(name: str, config: Config) -> str:
    values = (
        name,
        "1" if config.enabled else "0",
        str(config.min_depth),
        str(config.delta_base_cp),
        str(config.delta_divisor),
        str(config.expansion_factor_per_mille),
        str(config.max_fail_high_reductions),
        str(config.mean_score_new_weight_per_mille),
        str(config.max_researches),
        str(config.mean_score_clamp_cp),
    )
    return ",".join(values)


def selfplay_confirmation_args(
    profile_label: str,
    candidate_entry: dict,
    legacy_entry: dict,
) -> dict:
    candidate = config_from_record(candidate_entry)
    legacy = config_from_record(legacy_entry)
    candidate_name = f"v42_{profile_label}_{candidate.hash[:10]}"
    control_name = "v41_legacy_control"
    candidate_profile = ",".join(
        (candidate_name, *PRODUCTION_GAUNTLET_PROFILE_FIELDS))
    control_profile = ",".join(
        (control_name, *PRODUCTION_GAUNTLET_PROFILE_FIELDS))
    arguments = [
        "--profile", candidate_profile,
        "--profile", control_profile,
        "--aspiration-profile",
        gauntlet_aspiration_profile(candidate_name, candidate),
        "--aspiration-profile",
        gauntlet_aspiration_profile(control_name, legacy),
        "--twofold-search-profile", candidate_name,
        "--twofold-search-profile", control_name,
    ]
    return {
        "candidate_name": candidate_name,
        "control_name": control_name,
        "suggested_output_filename": (
            f"games_{profile_label}_{candidate.hash[:12]}_vs_"
            f"{legacy.hash[:12]}.jsonl"),
        "arguments": arguments,
        "twofold_search_draw_enabled_for_both": True,
    }


def knee(entries: Sequence[dict]) -> dict:
    if not entries:
        raise ValueError("cannot choose knee of empty frontier")
    if len(entries) <= 2:
        return min(entries, key=lambda item: sum(result_metrics(item)))
    losses = [result_metrics(entry)[0] for entry in entries]
    nodes = [result_metrics(entry)[1] for entry in entries]
    loss_min, loss_max = min(losses), max(losses)
    node_min, node_max = min(nodes), max(nodes)

    def score(entry: dict) -> tuple[float, float, float]:
        loss, node = result_metrics(entry)
        normalized_loss = (
            (loss - loss_min) / (loss_max - loss_min)
            if loss_max > loss_min else 0.0)
        normalized_node = (
            (node - node_min) / (node_max - node_min)
            if node_max > node_min else 0.0)
        # Endpoints lie near x+y=1.  The largest positive bend is the knee.
        return (
            1.0 - normalized_loss - normalized_node,
            -normalized_loss,
            -normalized_node,
        )
    return max(entries, key=score)


def pause(
    args: argparse.Namespace,
    records: list[dict],
    log_path: Path,
    budget: Budget,
    stage: str,
) -> None:
    payload = {
        "kind": "budget_exhausted",
        "stage": stage,
        "budget_elapsed_sec": budget.elapsed_sec,
        "budget_policy_version": BUDGET_POLICY_VERSION,
        "cumulative_limit_sec": args.duration_sec,
        "completed_results": sum(
            record.get("kind") == "result" for record in records),
    }
    append_jsonl(log_path, payload)
    atomic_write_json(args.run_dir / "PAUSED.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-unsealed-dataset", action="store_true",
        help="testing only: permit a dataset directory without manifest.json",
    )
    parser.add_argument(
        "--duration-sec", type=int, default=28_800,
        help=("cumulative elapsed ceiling for this run; resume a PAUSED run "
              "by supplying a larger value"))
    parser.add_argument("--preflight-size", type=int, default=128)
    parser.add_argument("--exploration-size", type=int, default=750)
    parser.add_argument("--selection-size", type=int, default=3_000)
    parser.add_argument("--holdout-size", type=int, default=4_000)
    parser.add_argument("--preflight-depth", type=int, default=6)
    parser.add_argument("--exploration-depth", type=int, default=6)
    parser.add_argument("--selection-depth", type=int, default=7)
    parser.add_argument("--holdout-depth", type=int, default=8)
    parser.add_argument("--global-candidates", type=int, default=64)
    parser.add_argument("--refinement-candidates", type=int, default=64)
    parser.add_argument("--candidate-time-ms", type=int, default=0)
    parser.add_argument("--candidate-max-depth", type=int, default=64)
    parser.add_argument("--max-researches", type=int, default=6)
    parser.add_argument("--mean-score-clamp-cp", type=int, default=1_500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    for name in (
        "preflight_size", "exploration_size", "selection_size", "holdout_size"
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "preflight_depth", "exploration_depth", "selection_depth", "holdout_depth"
    ):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be at least 2")
    if args.candidate_time_ms < 0:
        parser.error("--candidate-time-ms must be non-negative")
    if args.candidate_max_depth < 2:
        parser.error("--candidate-max-depth must be at least 2")
    if args.max_researches < 1 or args.mean_score_clamp_cp < 1:
        parser.error("frozen guards must be positive")
    if args.refinement_candidates < 0:
        parser.error("--refinement-candidates must be non-negative")
    if args.global_candidates < len(anchors(
        args.max_researches, args.mean_score_clamp_cp)):
        parser.error("--global-candidates is smaller than the anchor set")
    return args


def main() -> None:
    args = parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    args.dataset_dir = args.dataset_dir.resolve(strict=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = args.run_dir.resolve()

    source_tune = args.dataset_dir / "tune.tsv"
    source_selection = args.dataset_dir / "selection.tsv"
    source_holdout = args.dataset_dir / "holdout.tsv"
    source_datasets = {
        "tune": source_tune,
        "selection": source_selection,
        "holdout": source_holdout,
    }
    for source in source_datasets.values():
        validate_dataset_schema(source)
    validate_dataset_hash_disjointness(source_datasets)
    validate_no_balanced_lineage_overlap(source_datasets)
    args.sealed_dataset_identity = validate_sealed_dataset_manifest(
        args.dataset_dir, source_datasets, args.allow_unsealed_dataset)
    rung_paths = {
        "preflight": args.run_dir / "preflight.tsv",
        "exploration": args.run_dir / "exploration.tsv",
        "selection": args.run_dir / "selection.tsv",
        "holdout": args.run_dir / "holdout.tsv",
    }
    ensure_subset(
        rung_paths["preflight"], source_tune,
        args.preflight_size, args.seed ^ 0x505245)
    ensure_subset(
        rung_paths["exploration"], source_tune,
        args.exploration_size, args.seed ^ 0x455850)
    ensure_subset(
        rung_paths["selection"], source_selection,
        args.selection_size, args.seed ^ 0x53454C)
    ensure_subset(
        rung_paths["holdout"], source_holdout,
        args.holdout_size, args.seed ^ 0x484F4C)
    rung_depths = {
        "preflight": args.preflight_depth,
        "exploration": args.exploration_depth,
        "selection": args.selection_depth,
        "holdout": args.holdout_depth,
    }
    control_cache_specs = {
        cache_key: control_cache_spec(
            args.run_dir,
            cache_key,
            args.binary,
            args.model,
            rung_paths[cache_key],
            rung_depths[cache_key],
        )
        for cache_key in rung_paths
    }
    legacy_budget_limit = ensure_manifest(
        args.run_dir / "manifest.json",
        manifest_for(args, rung_paths, control_cache_specs))

    log_path = args.run_dir / "results.jsonl"
    records = load_jsonl(log_path)
    ensure_budget_limit(
        records, log_path, args.duration_sec, legacy_budget_limit)

    # A PAUSED marker describes the previous invocation.  Once the immutable
    # inputs and the new cumulative budget have been accepted, RUNNING/status
    # are the authoritative state and the stale marker must not confuse
    # external monitors.  A later budget exhaustion writes a fresh marker.
    paused_path = args.run_dir / "PAUSED.json"
    if paused_path.exists():
        paused_path.unlink()

    budget = Budget(
        float(args.duration_sec), max_budget_elapsed(records), time.monotonic())

    def prepare_control_cache(cache_key: str) -> dict | None:
        try:
            return ensure_control_cache(
                records,
                log_path,
                control_cache_specs[cache_key],
                args.binary,
                args.model,
                rung_paths[cache_key],
                budget,
            )
        except CacheBudgetExhausted:
            pause(
                args, records, log_path, budget,
                f"control_cache:{cache_key}")
            return None

    preflight_cache = prepare_control_cache("preflight")
    if preflight_cache is None:
        return
    preflight_plan = ensure_plan(
        records, log_path, "preflight",
        preflight_proposals(args.max_researches, args.mean_score_clamp_cp),
        rung_paths["preflight"], args.preflight_depth,
        args.candidate_time_ms, budget, preflight_cache)
    _, completed = run_stage(
        records, log_path, preflight_plan, args.binary, args.model,
        args.workers, args.candidate_max_depth, budget)
    if not completed:
        pause(args, records, log_path, budget, "preflight")
        return

    exploration_cache = prepare_control_cache("exploration")
    if exploration_cache is None:
        return
    global_plan = ensure_plan(
        records, log_path, "exploration",
        global_proposals(
            args.global_candidates, args.seed,
            args.max_researches, args.mean_score_clamp_cp),
        rung_paths["exploration"], args.exploration_depth,
        args.candidate_time_ms, budget, exploration_cache)
    exploration, completed = run_stage(
        records, log_path, global_plan, args.binary, args.model,
        args.workers, args.candidate_max_depth, budget)
    if not completed:
        pause(args, records, log_path, budget, "exploration")
        return

    refinement_1_count, refinement_2_count = refinement_generation_counts(
        args.refinement_candidates)
    refinement_1_plan = ensure_plan(
        records, log_path, "refinement_1",
        refinement_proposals(
            exploration, refinement_1_count, args.seed ^ 0x52454631,
            {config_from_record(item).hash for item in exploration},
            args.max_researches, args.mean_score_clamp_cp),
        rung_paths["exploration"], args.exploration_depth,
        args.candidate_time_ms, budget, exploration_cache)
    refinement_1, completed = run_stage(
        records, log_path, refinement_1_plan, args.binary, args.model,
        args.workers, args.candidate_max_depth, budget)
    if not completed:
        pause(args, records, log_path, budget, "refinement_1")
        return

    first_generation = [*exploration, *refinement_1]
    refinement_2_plan = ensure_plan(
        records, log_path, "refinement_2",
        refinement_proposals(
            first_generation, refinement_2_count, args.seed ^ 0x52454632,
            {config_from_record(item).hash for item in first_generation},
            args.max_researches, args.mean_score_clamp_cp),
        rung_paths["exploration"], args.exploration_depth,
        args.candidate_time_ms, budget, exploration_cache)
    refinement_2, completed = run_stage(
        records, log_path, refinement_2_plan, args.binary, args.model,
        args.workers, args.candidate_max_depth, budget)
    if not completed:
        pause(args, records, log_path, budget, "refinement_2")
        return

    run_anchors = anchors(
        args.max_researches, args.mean_score_clamp_cp)
    control_proposals = run_anchors[:2]  # Legacy V41 and the V42 seed.
    all_exploration = [*exploration, *refinement_1, *refinement_2]
    full_exploration_frontier = deduplicate_objectives(
        frontier(all_exploration))
    exploration_frontier = adaptive_frontier(all_exploration)
    selection_proposals = unique_proposals([
        Proposal(config_from_record(entry), "pareto:exploration")
        for entry in exploration_frontier
    ] + control_proposals)
    selection_cache = prepare_control_cache("selection")
    if selection_cache is None:
        return
    selection_plan = ensure_plan(
        records, log_path, "selection", selection_proposals,
        rung_paths["selection"], args.selection_depth,
        args.candidate_time_ms, budget, selection_cache)
    selection, completed = run_stage(
        records, log_path, selection_plan, args.binary, args.model,
        args.workers, args.candidate_max_depth, budget)
    if not completed:
        pause(args, records, log_path, budget, "selection")
        return

    selection_frontier = deduplicate_objectives(frontier(selection))
    enabled_selection_frontier = adaptive_frontier(selection)
    full_frontier_hashes = {
        config_from_record(entry).hash for entry in selection_frontier
    }
    adaptive_on_full_frontier = [
        entry for entry in enabled_selection_frontier
        if config_from_record(entry).hash in full_frontier_hashes
    ]
    offline_promotion_status = (
        "adaptive_offline_promotion"
        if adaptive_on_full_frontier else "no_offline_promotion"
    )
    if enabled_selection_frontier:
        preselected_fastest = min(
            enabled_selection_frontier,
            key=lambda entry: (
                result_metrics(entry)[1], result_metrics(entry)[0]))
        preselected_safest = min(
            enabled_selection_frontier,
            key=lambda entry: (
                result_metrics(entry)[0], result_metrics(entry)[1]))
        preselected_balanced = knee(enabled_selection_frontier)
        preselected_profiles = (
            ("preselected:fastest", preselected_fastest),
            ("preselected:balanced-knee", preselected_balanced),
            ("preselected:safest", preselected_safest),
        )
    else:
        # This can occur only when resuming a legacy-only historical plan.
        # The V42 seed control remains sealed; holdout is never used to invent
        # an adaptive winner which selection did not produce.
        preselected_fastest = None
        preselected_balanced = None
        preselected_safest = None
        preselected_profiles = ()
    # Keep the sealed rung small and decide its candidates solely from
    # selection.  The holdout validates these named profiles; it must not be
    # used to search across the entire selection frontier again.
    holdout_proposals = unique_proposals([
        Proposal(config_from_record(entry), origin)
        for origin, entry in preselected_profiles
    ] + control_proposals)
    holdout_cache = prepare_control_cache("holdout")
    if holdout_cache is None:
        return
    holdout_plan = ensure_plan(
        records, log_path, "holdout", holdout_proposals,
        rung_paths["holdout"], args.holdout_depth,
        args.candidate_time_ms, budget, holdout_cache)
    holdout, completed = run_stage(
        records, log_path, holdout_plan, args.binary, args.model,
        args.workers, args.candidate_max_depth, budget)
    if not completed:
        pause(args, records, log_path, budget, "holdout")
        return

    holdout_frontier = deduplicate_objectives(frontier(holdout))
    holdout_by_hash = {
        config_from_record(entry).hash: entry for entry in holdout
    }

    def held_out(selection_entry: dict) -> dict:
        config_hash = config_from_record(selection_entry).hash
        try:
            return holdout_by_hash[config_hash]
        except KeyError as error:
            raise RuntimeError(
                f"missing sealed result for preselected config {config_hash}"
            ) from error

    legacy_holdout = holdout_by_hash[control_proposals[0].config.hash]
    seed_holdout = holdout_by_hash[control_proposals[1].config.hash]
    fastest_holdout = (
        held_out(preselected_fastest)
        if preselected_fastest is not None else seed_holdout)
    balanced_holdout = (
        held_out(preselected_balanced)
        if preselected_balanced is not None else seed_holdout)
    safest_holdout = (
        held_out(preselected_safest)
        if preselected_safest is not None else seed_holdout)
    summary = {
        "kind": "complete",
        "schema_version": SCHEMA_VERSION,
        "objective": "all-position WDL/node Pareto",
        "hard_safety_rejects": False,
        "frozen_guards": {
            "max_researches": args.max_researches,
            "mean_score_clamp_cp": args.mean_score_clamp_cp,
        },
        "budget_elapsed_sec": budget.elapsed_sec,
        "budget_policy_version": BUDGET_POLICY_VERSION,
        "cumulative_limit_sec": args.duration_sec,
        "exploration_frontier_count": len(full_exploration_frontier),
        "adaptive_exploration_frontier_count": len(exploration_frontier),
        "selection_frontier_count": len(selection_frontier),
        "adaptive_selection_frontier_count": len(
            enabled_selection_frontier),
        "adaptive_frontier_empty": not enabled_selection_frontier,
        "offline_promotion_status": offline_promotion_status,
        "holdout_frontier_count": len(holdout_frontier),
        "exploration_frontier": serializable_profile(
            full_exploration_frontier),
        "adaptive_exploration_frontier": serializable_profile(
            exploration_frontier),
        "selection_frontier": serializable_profile(selection_frontier),
        "adaptive_selection_frontier": serializable_profile(
            enabled_selection_frontier),
        "holdout_frontier": serializable_profile(holdout_frontier),
        "fastest": relative_profile(fastest_holdout, legacy_holdout),
        "balanced_knee": relative_profile(balanced_holdout, legacy_holdout),
        "safest": relative_profile(safest_holdout, legacy_holdout),
        "legacy_control": relative_profile(legacy_holdout, legacy_holdout),
        "v42_seed_control": relative_profile(seed_holdout, legacy_holdout),
        "sealed_candidate_count": len(holdout_proposals),
        "winner_selection_used_holdout": False,
        "requires_equal_time_selfplay": True,
        "equal_time_selfplay_artifacts": {
            "gauntlet_binary": "bin/nnue_v42_time_gauntlet",
            "model": "inputs/phase_quantized_nnue.bin",
        },
        "equal_time_selfplay_config_args": {
            "fastest": selfplay_confirmation_args(
                "fastest", fastest_holdout, legacy_holdout),
            "balanced_knee": selfplay_confirmation_args(
                "balanced", balanced_holdout, legacy_holdout),
            "safest": selfplay_confirmation_args(
                "safest", safest_holdout, legacy_holdout),
        },
    }
    append_jsonl(log_path, summary)
    atomic_write_json(args.run_dir / "summary.json", summary)
    (args.run_dir / "DONE").touch()
    paused = args.run_dir / "PAUSED.json"
    if paused.exists():
        paused.unlink()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
