#!/usr/bin/env python3
"""Validate and summarize Strict-milestone fixed-depth benchmark NDJSON.

This analyzer intentionally targets the approved D6/D7/D8 protocol emitted by
``benchmark_strict_milestones`` schema v2.  It treats elapsed search time as the
primary metric.  Nodes and NPS are retained as secondary diagnostics because a
version can process nodes faster while still taking longer after searching a
larger tree.

Confidence intervals are descriptive, paired percentile bootstraps.  The
primary interval resamples all 50 complete (leg, FEN) blocks in a round as one
unit, preserving within-round thermal/scheduler correlation.  A sensitivity
interval resamples the 400 complete (round, leg, FEN) blocks; every resampled
block always contains all eight versions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VERSIONS = ("V15", "V19", "V24", "V27", "V29", "V32", "V33", "V35")
EXPECTED_DEPTHS = (6, 7, 8)
ROUNDS = 8
LEGS = ("forward", "reverse")
POSITIONS = 25
ROWS_PER_DEPTH = ROUNDS * len(LEGS) * POSITIONS * len(VERSIONS)
SAMPLES_PER_VERSION = ROUNDS * len(LEGS) * POSITIONS
WILLIAMS_OFFSETS = (0, 1, 7, 2, 6, 3, 5, 4)
DEFAULT_BOOTSTRAP_REPLICATES = 50_000
DEFAULT_BOOTSTRAP_SEED = 20_260_829


# Duplicating the versioned corpus here makes the analyzer an independent
# protocol check: a self-consistent but accidentally edited benchmark corpus
# cannot silently be compared with the documented run.
EXPECTED_CORPUS: tuple[tuple[str, str], ...] = (
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("opening_knights", "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4"),
    ("tactical_castling", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
    ("closed_center", "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9"),
    ("middlegame_rooks", "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12"),
    ("advanced_pawns", "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("promotion_race", "8/P6k/8/8/8/8/6Kp/8 w - - 0 1"),
    ("pawn_structure", "8/8/2p5/3p4/3P4/2P5/8/4K1k1 w - - 0 1"),
    ("development", "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10"),
    ("queenside_pressure", "2r3k1/1p1bqppp/p3pn2/3p4/3P4/P1NBPN2/1PQ2PPP/2R2RK1 b - - 0 16"),
    ("simple_king_pawns", "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"),
    ("rook_endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("promotion_and_castling", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
    ("opposite_attacks", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("undeveloped_tension", "2b1qb1r/rpppkp1p/2n1p2n/p7/P3N3/1P3P2/2PNP1PP/R1BQKB1R w KQ - 3 14"),
    ("minimal_castling", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("en_passant", "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"),
    ("in_check_evasion", "4r1k1/8/8/8/8/8/8/2B1K3 w - - 0 1"),
    ("queen_vs_pawn", "5k2/2p5/8/8/1Q6/8/8/6K1 b - - 0 1"),
    ("bishop_vs_pawn", "k7/8/1b6/8/8/8/3P4/6K1 w - - 0 1"),
    ("single_pawn_endgame", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"),
    ("promotion_technique", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ("rook_technique", "8/8/2k5/8/8/8/4K3/3R4 w - - 0 1"),
    ("black_to_move_center", "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 b - - 0 9"),
)


class AnalysisError(ValueError):
    """Raised when an input is not a complete, trustworthy protocol run."""


def corpus_hash(corpus: Sequence[tuple[str, str]]) -> str:
    value = 14_695_981_039_346_656_037
    prime = 1_099_511_628_211
    for _position_id, fen in corpus:
        for byte in fen.encode("utf-8"):
            value ^= byte
            value = (value * prime) & 0xFFFF_FFFF_FFFF_FFFF
        value ^= ord("\n")
        value = (value * prime) & 0xFFFF_FFFF_FFFF_FFFF
    return f"fnv1a64:{value:016x}"


EXPECTED_CORPUS_HASH = corpus_hash(EXPECTED_CORPUS)
assert EXPECTED_CORPUS_HASH == "fnv1a64:4e71582cee758c77"


def _where(record: Mapping[str, Any]) -> str:
    return f"{record.get('_source', '<input>')}:{record.get('_line', '?')}"


def _fail(record: Mapping[str, Any], message: str) -> AnalysisError:
    return AnalysisError(f"{_where(record)}: {message}")


def _require_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise _fail(record, f"{key} must be an integer")
    return value


def _require_bool(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key)
    if type(value) is not bool:
        raise _fail(record, f"{key} must be a boolean")
    return value


def _require_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise _fail(record, f"{key} must be a string")
    return value


def _expect(record: Mapping[str, Any], key: str, expected: Any) -> None:
    if record.get(key) != expected:
        raise _fail(
            record,
            f"{key} must be {expected!r}, got {record.get(key)!r}",
        )


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnalysisError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise AnalysisError(f"{path}:{line_number}: every NDJSON value must be an object")
        value["_source"] = str(path)
        value["_line"] = line_number
        records.append(value)
    if not records:
        raise AnalysisError(f"{path}: no NDJSON records")
    return records


def _validate_meta(meta: Mapping[str, Any]) -> tuple[int, ...]:
    expected_fields = {
        "schema_version": 2,
        "benchmark": "strict_milestones_v2",
        "versions": list(VERSIONS),
        "rounds_per_depth": ROUNDS,
        "legs_per_round": len(LEGS),
        "positions": POSITIONS,
        "tt_mb": 64,
        "fixed_depth": True,
        "fresh_searcher_per_tuple": True,
        "tt_or_history_shared_across_tuples": False,
        "searcher_construction_timed": False,
        "schedule": "paired_williams_forward_mirrored_reverse_v2",
        "schedule_base_offsets": list(WILLIAMS_OFFSETS),
        "schedule_rotation": "(round+position_index)%8",
        "schedule_balance_cycle_rounds": ROUNDS,
        "schedule_balance_cycle_complete": True,
        "warmup": "unmeasured_full_corpus_correctness_preflight",
        "all_depth_preflights_complete_before_timing": True,
        "row_order_scope": "requested_depth",
        "score_gate_reference": "V15_per_position_at_requested_depth",
        "move_gate": "returned_move_must_be_legal; equivalent_best_moves_allowed",
        "primary_metric": "elapsed_ns",
        "corpus_embedded": True,
        "corpus_hash": EXPECTED_CORPUS_HASH,
    }
    for key, expected in expected_fields.items():
        _expect(meta, key, expected)
    raw_depths = meta.get("depths")
    if not isinstance(raw_depths, list) or not raw_depths:
        raise _fail(meta, "depths must be a non-empty array")
    if any(type(value) is not int for value in raw_depths):
        raise _fail(meta, "every depths entry must be an integer")
    depths = tuple(raw_depths)
    if len(set(depths)) != len(depths):
        raise _fail(meta, "depths contains duplicates")
    if any(depth not in EXPECTED_DEPTHS for depth in depths):
        raise _fail(meta, f"depths must be a subset of {EXPECTED_DEPTHS}")
    return depths


def _validate_corpus_records(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != POSITIONS:
        raise AnalysisError(
            f"an emitted corpus must contain exactly {POSITIONS} records, got {len(records)}"
        )
    by_position: dict[int, Mapping[str, Any]] = {}
    for record in records:
        position = _require_int(record, "position")
        if position in by_position:
            raise _fail(record, f"duplicate corpus position {position}")
        by_position[position] = record
    if set(by_position) != set(range(POSITIONS)):
        raise AnalysisError("emitted corpus positions must be contiguous 0..24")
    observed: list[tuple[str, str]] = []
    for position, expected in enumerate(EXPECTED_CORPUS):
        record = by_position[position]
        _expect(record, "corpus_hash", EXPECTED_CORPUS_HASH)
        _expect(record, "position_id", expected[0])
        _expect(record, "fen", expected[1])
        observed.append((_require_string(record, "position_id"), _require_string(record, "fen")))
    if corpus_hash(observed) != EXPECTED_CORPUS_HASH:
        raise AnalysisError("emitted corpus FENs do not reproduce corpus_hash")


def _scheduled_version(round_index: int, position: int, leg: str, slot: int) -> str:
    rotation = (round_index + position) % len(VERSIONS)
    offsets = WILLIAMS_OFFSETS if leg == "forward" else tuple(reversed(WILLIAMS_OFFSETS))
    return VERSIONS[(rotation + offsets[slot]) % len(VERSIONS)]


def _expected_position(round_index: int, leg: str, position_slot: int) -> int:
    if leg == "forward":
        return (round_index + position_slot) % POSITIONS
    return (round_index + POSITIONS - 1 - position_slot) % POSITIONS


def _validate_row_scalars(row: Mapping[str, Any], depth: int) -> None:
    _expect(row, "requested_depth", depth)
    _expect(row, "depth", depth)
    for key in ("order", "round", "slot", "position", "elapsed_ns", "nodes", "score", "expected_score"):
        _require_int(row, key)
    for key in ("legal", "stopped", "score_match", "depth_match", "gate_pass"):
        _require_bool(row, key)
    if _require_int(row, "elapsed_ns") <= 0:
        raise _fail(row, "elapsed_ns must be positive")
    if _require_int(row, "nodes") <= 0:
        raise _fail(row, "nodes must be positive")
    version = _require_string(row, "version")
    if version not in VERSIONS:
        raise _fail(row, f"unknown version {version!r}")
    if _require_string(row, "leg") not in LEGS:
        raise _fail(row, "leg must be forward or reverse")
    if not _require_string(row, "move"):
        raise _fail(row, "move must not be empty")
    position = _require_int(row, "position")
    if not 0 <= position < POSITIONS:
        raise _fail(row, "position out of range")
    _expect(row, "position_id", EXPECTED_CORPUS[position][0])
    if row["score"] != row["expected_score"]:
        raise _fail(row, "score differs from expected_score")
    expected_gates = {
        "legal": True,
        "stopped": False,
        "score_match": True,
        "depth_match": True,
        "gate_pass": True,
    }
    for key, expected in expected_gates.items():
        _expect(row, key, expected)


def _validate_depth_rows(depth: int, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != ROWS_PER_DEPTH:
        raise AnalysisError(
            f"depth {depth}: expected exactly {ROWS_PER_DEPTH} rows, got {len(rows)}"
        )
    by_key: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    for row in rows:
        _validate_row_scalars(row, depth)
        key = (row["round"], row["leg"], row["position"], row["slot"])
        if key in by_key:
            raise _fail(row, f"duplicate timed tuple {key}")
        by_key[key] = row

    version_samples: Counter[str] = Counter()
    slot_samples: Counter[tuple[str, int]] = Counter()
    predecessor_samples: Counter[tuple[str, str]] = Counter()
    expected_order = 0
    for round_index in range(ROUNDS):
        for leg in LEGS:
            for position_slot in range(POSITIONS):
                position = _expected_position(round_index, leg, position_slot)
                execution_order: list[str] = []
                for slot in range(len(VERSIONS)):
                    key = (round_index, leg, position, slot)
                    if key not in by_key:
                        raise AnalysisError(f"depth {depth}: missing timed tuple {key}")
                    row = by_key[key]
                    expected_version = _scheduled_version(round_index, position, leg, slot)
                    _expect(row, "version", expected_version)
                    _expect(row, "order", expected_order)
                    expected_order += 1
                    version_samples[row["version"]] += 1
                    slot_samples[(row["version"], slot)] += 1
                    execution_order.append(row["version"])
                predecessor_samples.update(zip(execution_order, execution_order[1:]))

    if any(version_samples[version] != SAMPLES_PER_VERSION for version in VERSIONS):
        raise AnalysisError(f"depth {depth}: every version must have {SAMPLES_PER_VERSION} samples")
    expected_per_slot = SAMPLES_PER_VERSION // len(VERSIONS)
    if any(
        slot_samples[(version, slot)] != expected_per_slot
        for version in VERSIONS
        for slot in range(len(VERSIONS))
    ):
        raise AnalysisError(f"depth {depth}: version execution slots are not balanced")
    expected_predecessors = 50
    if any(
        predecessor_samples[(first, second)] != expected_predecessors
        for first in VERSIONS
        for second in VERSIONS
        if first != second
    ):
        raise AnalysisError(f"depth {depth}: ordered predecessor pairs are not balanced")
    if any(predecessor_samples[(version, version)] for version in VERSIONS):
        raise AnalysisError(f"depth {depth}: a version is its own scheduled predecessor")

    node_values: dict[tuple[str, int], set[int]] = defaultdict(set)
    score_values: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        node_values[(row["version"], row["position"])].add(row["nodes"])
        score_values[row["position"]].add(row["score"])
    nondeterministic = [key for key, values in node_values.items() if len(values) != 1]
    if nondeterministic:
        raise AnalysisError(
            f"depth {depth}: nodes are not deterministic for {nondeterministic[0]}"
        )
    mismatched_scores = [position for position, values in score_values.items() if len(values) != 1]
    if mismatched_scores:
        raise AnalysisError(
            f"depth {depth}: versions disagree on score at position {mismatched_scores[0]}"
        )
    return {
        "rows": len(rows),
        "samples_per_version": SAMPLES_PER_VERSION,
        "samples_per_execution_slot": expected_per_slot,
        "ordered_predecessor_pair_count": expected_predecessors,
        "nodes_deterministic_per_version_position": True,
        "scores_match_across_versions": True,
    }


def _median_integer(values: Iterable[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return ordered[middle - 1] + (ordered[middle] - ordered[middle - 1]) // 2


def _validate_emitted_summaries(
    depth: int,
    rows: Sequence[dict[str, Any]],
    version_summaries: Sequence[Mapping[str, Any]],
    depth_summaries: Sequence[Mapping[str, Any]],
    gate_summaries: Sequence[Mapping[str, Any]],
) -> None:
    if len(gate_summaries) != 1:
        raise AnalysisError(f"depth {depth}: expected one preflight gate_summary")
    gate = gate_summaries[0]
    for key, expected in {
        "phase": "preflight_warmup",
        "requested_depth": depth,
        "measured": False,
        "tuples": POSITIONS * len(VERSIONS),
        "failures": 0,
        "status": "pass",
    }.items():
        _expect(gate, key, expected)

    if len(depth_summaries) != 1:
        raise AnalysisError(f"depth {depth}: expected one depth_run_summary")
    summary = depth_summaries[0]
    for key, expected in {
        "requested_depth": depth,
        "status": "pass",
        "timed_rows": ROWS_PER_DEPTH,
        "expected_timed_rows": ROWS_PER_DEPTH,
        "primary_metric": "elapsed_ns",
        "corpus_hash": EXPECTED_CORPUS_HASH,
    }.items():
        _expect(summary, key, expected)

    if len(version_summaries) != len(VERSIONS):
        raise AnalysisError(f"depth {depth}: expected eight depth_version_summary records")
    by_version: dict[str, Mapping[str, Any]] = {}
    baseline_elapsed = sum(row["elapsed_ns"] for row in rows if row["version"] == VERSIONS[0])
    for emitted in version_summaries:
        version = _require_string(emitted, "version")
        if version in by_version:
            raise _fail(emitted, f"duplicate depth_version_summary for {version}")
        by_version[version] = emitted
    if set(by_version) != set(VERSIONS):
        raise AnalysisError(f"depth {depth}: summary versions differ from protocol versions")
    for index, version in enumerate(VERSIONS):
        emitted = by_version[version]
        version_rows = [row for row in rows if row["version"] == version]
        elapsed = sum(row["elapsed_ns"] for row in version_rows)
        nodes = sum(row["nodes"] for row in version_rows)
        elapsed_values = [row["elapsed_ns"] for row in version_rows]
        exact = {
            "requested_depth": depth,
            "order": index,
            "samples": SAMPLES_PER_VERSION,
            "elapsed_ns": elapsed,
            "nodes": nodes,
            "median_row_elapsed_ns": _median_integer(elapsed_values),
            "min_row_elapsed_ns": min(elapsed_values),
            "max_row_elapsed_ns": max(elapsed_values),
            "legal_failures": 0,
            "stopped_failures": 0,
            "score_failures": 0,
            "depth_failures": 0,
            "status": "pass",
        }
        for key, expected in exact.items():
            _expect(emitted, key, expected)
        expected_nps = nodes * 1_000_000_000.0 / elapsed
        expected_ratio = elapsed / baseline_elapsed
        for key, expected in (("nps", expected_nps), ("elapsed_ratio_to_v15", expected_ratio)):
            value = emitted.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise _fail(emitted, f"{key} must be a finite number")
            if not math.isclose(float(value), expected, rel_tol=2e-10, abs_tol=1e-9):
                raise _fail(emitted, f"{key} does not match timed rows")


def load_and_validate(paths: Sequence[Path]) -> dict[str, Any]:
    """Load one combined file or three single-depth files and audit the run."""

    if not paths:
        raise AnalysisError("at least one NDJSON path is required")
    rows_by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    version_summaries_by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    depth_summaries_by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gate_summaries_by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    emitted_corpora = 0
    declared_depths: set[int] = set()
    run_documents: list[dict[str, Any]] = []

    allowed_types = {
        "meta",
        "corpus_position",
        "gate_error",
        "gate_summary",
        "row",
        "depth_version_summary",
        "depth_run_summary",
        "run_summary",
        "error",
    }
    for path in paths:
        records = _load_records(path)
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            record_type = record.get("type")
            if record_type not in allowed_types:
                raise _fail(record, f"unknown record type {record_type!r}")
            by_type[str(record_type)].append(record)
        if by_type["error"]:
            raise _fail(by_type["error"][0], "benchmark emitted an error record")
        if by_type["gate_error"]:
            raise _fail(by_type["gate_error"][0], "benchmark emitted a gate_error")
        if len(by_type["meta"]) != 1:
            raise AnalysisError(f"{path}: expected exactly one meta record")
        depths = _validate_meta(by_type["meta"][0])
        overlap = declared_depths.intersection(depths)
        if overlap:
            raise AnalysisError(f"depths appear in more than one input: {sorted(overlap)}")
        declared_depths.update(depths)

        corpus_records = by_type["corpus_position"]
        if corpus_records:
            _validate_corpus_records(corpus_records)
            emitted_corpora += 1
        if len(by_type["run_summary"]) != 1:
            raise AnalysisError(f"{path}: expected exactly one run_summary")
        run_summary = by_type["run_summary"][0]
        local_rows = by_type["row"]
        local_row_depths = {_require_int(row, "requested_depth") for row in local_rows}
        if local_row_depths != set(depths):
            raise AnalysisError(
                f"{path}: row depths {sorted(local_row_depths)} differ from meta depths {list(depths)}"
            )
        for record_type in (
            "gate_summary",
            "depth_version_summary",
            "depth_run_summary",
        ):
            for record in by_type[record_type]:
                record_depth = _require_int(record, "requested_depth")
                if record_depth not in depths:
                    raise _fail(
                        record,
                        f"{record_type} depth {record_depth} is not declared by this file's meta",
                    )
        expected_local_rows = ROWS_PER_DEPTH * len(depths)
        for key, expected in {
            "status": "pass",
            "depths_completed": len(depths),
            "timed_rows": expected_local_rows,
            "expected_timed_rows": expected_local_rows,
            "primary_metric": "elapsed_ns",
            "corpus_hash": EXPECTED_CORPUS_HASH,
        }.items():
            _expect(run_summary, key, expected)

        for row in local_rows:
            rows_by_depth[row["requested_depth"]].append(row)
        for record in by_type["depth_version_summary"]:
            version_summaries_by_depth[_require_int(record, "requested_depth")].append(record)
        for record in by_type["depth_run_summary"]:
            depth_summaries_by_depth[_require_int(record, "requested_depth")].append(record)
        for record in by_type["gate_summary"]:
            gate_summaries_by_depth[_require_int(record, "requested_depth")].append(record)
        run_documents.append({"path": str(path), "depths": list(depths)})

    if declared_depths != set(EXPECTED_DEPTHS):
        raise AnalysisError(
            f"inputs must contain exactly depths {EXPECTED_DEPTHS}, got {sorted(declared_depths)}"
        )
    if emitted_corpora == 0:
        raise AnalysisError("at least one input must include --emit-corpus records")

    depth_validation: dict[str, Any] = {}
    for depth in EXPECTED_DEPTHS:
        rows = rows_by_depth[depth]
        depth_validation[str(depth)] = _validate_depth_rows(depth, rows)
        _validate_emitted_summaries(
            depth,
            rows,
            version_summaries_by_depth[depth],
            depth_summaries_by_depth[depth],
            gate_summaries_by_depth[depth],
        )
    return {
        "rows_by_depth": dict(rows_by_depth),
        "validation": {
            "status": "pass",
            "inputs": run_documents,
            "corpus_hash": EXPECTED_CORPUS_HASH,
            "corpus_records_verified": True,
            "run_summaries_pass": True,
            "gates_pass": True,
            "depths": depth_validation,
            "schedule": {
                "name": "paired_williams_forward_mirrored_reverse_v2",
                "exact_order_verified": True,
                "slot_balance_verified": True,
                "ordered_predecessor_balance_verified": True,
            },
        },
    }


def _bootstrap_ratio_intervals(
    blocks: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 512,
) -> dict[str, list[list[float] | None]]:
    if replicates <= 0:
        raise AnalysisError("bootstrap replicates must be positive")
    if blocks.ndim != 2 or blocks.shape[1] != len(VERSIONS) or blocks.shape[0] == 0:
        raise AnalysisError("bootstrap blocks have an invalid shape")
    rng = np.random.default_rng(seed)
    ratios_v15 = np.empty((replicates, len(VERSIONS)), dtype=np.float64)
    ratios_previous = np.empty((replicates, len(VERSIONS) - 1), dtype=np.float64)
    for begin in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - begin)
        indices = rng.integers(0, blocks.shape[0], size=(size, blocks.shape[0]))
        totals = blocks[indices].sum(axis=1, dtype=np.int64)
        ratios_v15[begin : begin + size] = totals / totals[:, [0]]
        ratios_previous[begin : begin + size] = totals[:, 1:] / totals[:, :-1]
    v15_quantiles = np.percentile(ratios_v15, (2.5, 97.5), axis=0)
    previous_quantiles = np.percentile(ratios_previous, (2.5, 97.5), axis=0)
    return {
        "vs_v15": [
            [float(v15_quantiles[0, index]), float(v15_quantiles[1, index])]
            for index in range(len(VERSIONS))
        ],
        "vs_previous": [None]
        + [
            [float(previous_quantiles[0, index]), float(previous_quantiles[1, index])]
            for index in range(len(VERSIONS) - 1)
        ],
    }


def _ratio(value: int, baseline: int) -> float:
    return value / baseline


def _summarize_depth(
    depth: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_tuple_version: dict[tuple[int, str, int, str], Mapping[str, Any]] = {
        (row["round"], row["leg"], row["position"], row["version"]): row
        for row in rows
    }
    tuple_blocks = np.asarray(
        [
            [
                by_tuple_version[(round_index, leg, position, version)]["elapsed_ns"]
                for version in VERSIONS
            ]
            for round_index in range(ROUNDS)
            for leg in LEGS
            for position in range(POSITIONS)
        ],
        dtype=np.int64,
    )
    round_blocks = tuple_blocks.reshape(ROUNDS, len(LEGS) * POSITIONS, len(VERSIONS)).sum(axis=1)
    round_seed = bootstrap_seed + depth * 10 + 1
    tuple_seed = bootstrap_seed + depth * 10 + 2
    round_intervals = _bootstrap_ratio_intervals(
        round_blocks,
        replicates=bootstrap_replicates,
        seed=round_seed,
    )
    tuple_intervals = _bootstrap_ratio_intervals(
        tuple_blocks,
        replicates=bootstrap_replicates,
        seed=tuple_seed,
    )

    elapsed = {
        version: sum(row["elapsed_ns"] for row in rows if row["version"] == version)
        for version in VERSIONS
    }
    nodes = {
        version: sum(row["nodes"] for row in rows if row["version"] == version)
        for version in VERSIONS
    }
    half_elapsed: dict[str, dict[str, int]] = {"first": {}, "second": {}}
    half_nodes: dict[str, dict[str, int]] = {"first": {}, "second": {}}
    half_samples: dict[str, dict[str, int]] = {"first": {}, "second": {}}
    for half, selected_rounds in (
        ("first", range(0, ROUNDS // 2)),
        ("second", range(ROUNDS // 2, ROUNDS)),
    ):
        selected = [row for row in rows if row["round"] in selected_rounds]
        for version in VERSIONS:
            version_rows = [row for row in selected if row["version"] == version]
            half_elapsed[half][version] = sum(row["elapsed_ns"] for row in version_rows)
            half_nodes[half][version] = sum(row["nodes"] for row in version_rows)
            half_samples[half][version] = len(version_rows)

    version_results: list[dict[str, Any]] = []
    for index, version in enumerate(VERSIONS):
        elapsed_ns = elapsed[version]
        node_count = nodes[version]
        ratio_v15 = _ratio(elapsed_ns, elapsed[VERSIONS[0]])
        ratio_previous = None if index == 0 else _ratio(elapsed_ns, elapsed[VERSIONS[index - 1]])
        first_elapsed = half_elapsed["first"][version]
        second_elapsed = half_elapsed["second"][version]
        first_samples = half_samples["first"][version]
        second_samples = half_samples["second"][version]
        drift_ratio = (second_elapsed / second_samples) / (first_elapsed / first_samples)
        relative_first = first_elapsed / half_elapsed["first"][VERSIONS[0]]
        relative_second = second_elapsed / half_elapsed["second"][VERSIONS[0]]
        previous_comparison = None
        if index > 0:
            assert ratio_previous is not None
            previous_comparison = {
                "baseline": VERSIONS[index - 1],
                "elapsed_ratio": ratio_previous,
                "speedup": 1.0 / ratio_previous,
                "elapsed_change_percent": (ratio_previous - 1.0) * 100.0,
                "ci95_paired_complete_round": round_intervals["vs_previous"][index],
                "ci95_paired_round_leg_fen_block": tuple_intervals["vs_previous"][index],
            }
        version_results.append(
            {
                "version": version,
                "samples": SAMPLES_PER_VERSION,
                "elapsed_ns": elapsed_ns,
                "elapsed_seconds": elapsed_ns / 1_000_000_000.0,
                "nodes": node_count,
                "nps": node_count * 1_000_000_000.0 / elapsed_ns,
                "node_ratio_to_v15": node_count / nodes[VERSIONS[0]],
                "comparison_to_v15": {
                    "baseline": VERSIONS[0],
                    "elapsed_ratio": ratio_v15,
                    "speedup": 1.0 / ratio_v15,
                    "elapsed_change_percent": (ratio_v15 - 1.0) * 100.0,
                    "ci95_paired_complete_round": round_intervals["vs_v15"][index],
                    "ci95_paired_round_leg_fen_block": tuple_intervals["vs_v15"][index],
                },
                "comparison_to_previous": previous_comparison,
                "drift": {
                    "split": "rounds_0_to_3_vs_4_to_7",
                    "first_half_samples": first_samples,
                    "second_half_samples": second_samples,
                    "first_half_elapsed_ns": first_elapsed,
                    "second_half_elapsed_ns": second_elapsed,
                    "second_to_first_mean_elapsed_ratio": drift_ratio,
                    "elapsed_drift_percent": (drift_ratio - 1.0) * 100.0,
                    "first_half_nps": half_nodes["first"][version] * 1_000_000_000.0 / first_elapsed,
                    "second_half_nps": half_nodes["second"][version] * 1_000_000_000.0 / second_elapsed,
                    "elapsed_ratio_to_v15_first_half": relative_first,
                    "elapsed_ratio_to_v15_second_half": relative_second,
                    "relative_ratio_drift_percent": (relative_second / relative_first - 1.0) * 100.0,
                },
            }
        )
    return {
        "requested_depth": depth,
        "rows": len(rows),
        "versions": version_results,
        "bootstrap_seeds": {
            "paired_complete_round": round_seed,
            "paired_round_leg_fen_block": tuple_seed,
        },
    }


def analyze(
    validated: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise AnalysisError("bootstrap_replicates must be positive")
    rows_by_depth = validated["rows_by_depth"]
    return {
        "schema_version": 1,
        "analysis": "strict_milestones_fixed_depth_d6_d7_d8",
        "primary_metric": "pooled elapsed_ns (searcher construction excluded by benchmark)",
        "secondary_metrics": ["pooled nodes", "pooled NPS"],
        "validation": validated["validation"],
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed_base": bootstrap_seed,
            "interval": "paired percentile 95%",
            "primary_unit": "complete round (both legs and all 25 FENs; 8 units)",
            "sensitivity_unit": "complete (round, leg, FEN) block containing all versions (400 units)",
            "interpretation": "conditional descriptive interval for this corpus, host, and run",
        },
        "depths": {
            str(depth): _summarize_depth(
                depth,
                rows_by_depth[depth],
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            for depth in EXPECTED_DEPTHS
        },
        "limitations": [
            "Intervals describe timing variation in this one host run and fixed 25-position corpus; they are not Elo intervals or population-wide claims.",
            "Only eight complete rounds are available for the conservative round bootstrap, so its uncertainty estimate has limited resolution.",
            "Wall-clock search time can still reflect CPU frequency, thermals, and scheduler interference; this log contains no hardware-counter attribution.",
            "NPS is secondary because versions may visit different node counts even when they return the same fixed-depth score.",
            "Every tuple starts with a fresh 64 MiB searcher, TT, and history state, which is deliberately controlled but differs from a long-lived game process.",
        ],
    }


def _format_ratio(comparison: Mapping[str, Any] | None) -> str:
    if comparison is None:
        return "—"
    ratio = comparison["elapsed_ratio"]
    low, high = comparison["ci95_paired_complete_round"]
    return f"{ratio:.3f} [{low:.3f}, {high:.3f}]"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict milestone benchmark: D6/D7/D8",
        "",
        "Validation passed: 3,200 rows/depth, 400 samples/version, all correctness gates, "
        "the exact Williams order, slot/predecessor balance, and deterministic nodes were verified.",
        "",
        "Elapsed ratio below 1.0 is faster. Brackets are the primary 95% paired complete-round "
        "bootstrap interval. Nodes and NPS are secondary diagnostics.",
    ]
    for depth in EXPECTED_DEPTHS:
        lines.extend(
            [
                "",
                f"## Depth {depth}",
                "",
                "| Version | Search time (s) | Time / V15 [95% CI] | Time / previous [95% CI] | Nodes | NPS | 2nd / 1st half time |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for version in report["depths"][str(depth)]["versions"]:
            lines.append(
                "| {version} | {elapsed:.3f} | {v15} | {previous} | {nodes:,} | {nps:,.0f} | {drift:.3f} |".format(
                    version=version["version"],
                    elapsed=version["elapsed_seconds"],
                    v15=_format_ratio(version["comparison_to_v15"]),
                    previous=_format_ratio(version["comparison_to_previous"]),
                    nodes=version["nodes"],
                    nps=version["nps"],
                    drift=version["drift"]["second_to_first_mean_elapsed_ratio"],
                )
            )
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and analyze Strict milestone D6/D7/D8 benchmark NDJSON."
    )
    parser.add_argument("ndjson", type=Path, nargs="+")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(
        load_and_validate(args.ndjson),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.seed,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.write_text(encoded, encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if args.json_output is None and args.markdown_output is None:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
