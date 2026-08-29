from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_strict_milestones",
    ROOT / "tools" / "analyze_strict_milestones.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


FACTORS = {
    version: factor
    for version, factor in zip(MODULE.VERSIONS, (120, 110, 100, 95, 90, 85, 80, 75))
}


def _meta(depth: int) -> dict[str, Any]:
    return {
        "type": "meta",
        "schema_version": 2,
        "benchmark": "strict_milestones_v3",
        "versions": list(MODULE.VERSIONS),
        "depths": [depth],
        "rounds_per_depth": MODULE.ROUNDS,
        "legs_per_round": len(MODULE.LEGS),
        "positions": MODULE.POSITIONS,
        "tt_mb": 64,
        "fixed_depth": True,
        "fresh_searcher_per_tuple": True,
        "tt_or_history_shared_across_tuples": False,
        "searcher_construction_timed": False,
        "schedule": "paired_williams_forward_mirrored_reverse_v2",
        "schedule_base_offsets": list(MODULE.WILLIAMS_OFFSETS),
        "schedule_rotation": "(round+position_index)%8",
        "schedule_balance_cycle_rounds": 8,
        "schedule_balance_cycle_complete": True,
        "warmup": "unmeasured_full_corpus_correctness_preflight",
        "all_depth_preflights_complete_before_timing": True,
        "row_order_scope": "requested_depth",
        "score_gate_reference": "V15_per_position_at_requested_depth",
        "move_gate": "returned_move_must_be_legal; equivalent_best_moves_allowed",
        "primary_metric": "elapsed_ns",
        "corpus_embedded": True,
        "corpus_hash": MODULE.EXPECTED_CORPUS_HASH,
    }


def _corpus_records() -> list[dict[str, Any]]:
    return [
        {
            "type": "corpus_position",
            "corpus_hash": MODULE.EXPECTED_CORPUS_HASH,
            "position": position,
            "position_id": position_id,
            "fen": fen,
        }
        for position, (position_id, fen) in enumerate(MODULE.EXPECTED_CORPUS)
    ]


def _elapsed_ns(round_index: int, position: int, version: str) -> int:
    half_multiplier = 2 if round_index >= MODULE.ROUNDS // 2 else 1
    return (1_000 + position) * FACTORS[version] * half_multiplier


def _nodes(depth: int, position: int, version: str) -> int:
    return (MODULE.VERSIONS.index(version) + 1) * 10_000 + position * 10 + depth


def _depth_records(depth: int, *, emit_corpus: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [_meta(depth)]
    if emit_corpus:
        records.extend(_corpus_records())
    records.append(
        {
            "type": "gate_summary",
            "phase": "preflight_warmup",
            "requested_depth": depth,
            "measured": False,
            "tuples": MODULE.POSITIONS * len(MODULE.VERSIONS),
            "failures": 0,
            "status": "pass",
        }
    )

    rows: list[dict[str, Any]] = []
    order = 0
    for round_index in range(MODULE.ROUNDS):
        for leg in MODULE.LEGS:
            for position_slot in range(MODULE.POSITIONS):
                position = MODULE._expected_position(round_index, leg, position_slot)
                for slot in range(len(MODULE.VERSIONS)):
                    version = MODULE._scheduled_version(round_index, position, leg, slot)
                    score = position - depth
                    row = {
                        "type": "row",
                        "order": order,
                        "requested_depth": depth,
                        "round": round_index,
                        "leg": leg,
                        "slot": slot,
                        "position": position,
                        "position_id": MODULE.EXPECTED_CORPUS[position][0],
                        "version": version,
                        "depth": depth,
                        "elapsed_ns": _elapsed_ns(round_index, position, version),
                        "nodes": _nodes(depth, position, version),
                        "score": score,
                        "expected_score": score,
                        "move": "a2a3",
                        "legal": True,
                        "stopped": False,
                        "score_match": True,
                        "depth_match": True,
                        "gate_pass": True,
                    }
                    rows.append(row)
                    records.append(row)
                    order += 1

    baseline_elapsed = sum(
        row["elapsed_ns"] for row in rows if row["version"] == MODULE.VERSIONS[0]
    )
    for version_index, version in enumerate(MODULE.VERSIONS):
        version_rows = [row for row in rows if row["version"] == version]
        elapsed_values = [row["elapsed_ns"] for row in version_rows]
        elapsed = sum(elapsed_values)
        nodes = sum(row["nodes"] for row in version_rows)
        records.append(
            {
                "type": "depth_version_summary",
                "requested_depth": depth,
                "order": version_index,
                "version": version,
                "samples": MODULE.SAMPLES_PER_VERSION,
                "elapsed_ns": elapsed,
                "nodes": nodes,
                "nps": nodes * 1_000_000_000.0 / elapsed,
                "elapsed_ratio_to_v15": elapsed / baseline_elapsed,
                "median_row_elapsed_ns": MODULE._median_integer(elapsed_values),
                "min_row_elapsed_ns": min(elapsed_values),
                "max_row_elapsed_ns": max(elapsed_values),
                "legal_failures": 0,
                "stopped_failures": 0,
                "score_failures": 0,
                "depth_failures": 0,
                "status": "pass",
            }
        )
    records.extend(
        [
            {
                "type": "depth_run_summary",
                "requested_depth": depth,
                "status": "pass",
                "timed_rows": MODULE.ROWS_PER_DEPTH,
                "expected_timed_rows": MODULE.ROWS_PER_DEPTH,
                "primary_metric": "elapsed_ns",
                "corpus_hash": MODULE.EXPECTED_CORPUS_HASH,
            },
            {
                "type": "run_summary",
                "status": "pass",
                "depths_completed": 1,
                "timed_rows": MODULE.ROWS_PER_DEPTH,
                "expected_timed_rows": MODULE.ROWS_PER_DEPTH,
                "primary_metric": "elapsed_ns",
                "corpus_hash": MODULE.EXPECTED_CORPUS_HASH,
            },
        ]
    )
    return records


Mutation = Callable[[int, list[dict[str, Any]]], None]


def _write_inputs(
    directory: Path,
    *,
    mutation: Mutation | None = None,
    include_corpus: bool = True,
) -> list[Path]:
    paths: list[Path] = []
    for depth in MODULE.EXPECTED_DEPTHS:
        records = _depth_records(depth, emit_corpus=include_corpus and depth == 6)
        if mutation is not None:
            mutation(depth, records)
        path = directory / f"strict_d{depth}.ndjson"
        path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _first_row(records: list[dict[str, Any]]) -> dict[str, Any]:
    return next(record for record in records if record["type"] == "row")


def test_validates_split_depth_runs_and_analyzes_paired_ratios(tmp_path: Path) -> None:
    assert MODULE.VERSIONS == (
        "V15",
        "V19",
        "V23",
        "V25",
        "V27",
        "V29",
        "V30",
        "V35",
    )
    validated = MODULE.load_and_validate(_write_inputs(tmp_path))
    assert validated["validation"]["status"] == "pass"
    assert validated["validation"]["schedule"]["ordered_predecessor_balance_verified"] is True
    for depth in MODULE.EXPECTED_DEPTHS:
        audit = validated["validation"]["depths"][str(depth)]
        assert audit["rows"] == 3_200
        assert audit["samples_per_version"] == 400
        assert audit["samples_per_execution_slot"] == 50
        assert audit["ordered_predecessor_pair_count"] == 50
        assert audit["nodes_deterministic_per_version_position"] is True

    report = MODULE.analyze(validated, bootstrap_replicates=64, bootstrap_seed=7)
    d7 = report["depths"]["7"]
    v19 = d7["versions"][1]
    assert v19["comparison_to_v15"]["elapsed_ratio"] == pytest.approx(110 / 120)
    assert v19["comparison_to_previous"]["elapsed_ratio"] == pytest.approx(110 / 120)
    assert v19["comparison_to_v15"]["ci95_paired_complete_round"] == pytest.approx(
        [110 / 120, 110 / 120]
    )
    assert v19["drift"]["second_to_first_mean_elapsed_ratio"] == pytest.approx(2.0)
    assert v19["drift"]["relative_ratio_drift_percent"] == pytest.approx(0.0)
    assert "| V19 |" in MODULE.render_markdown(report)


def test_bootstrap_is_deterministic(tmp_path: Path) -> None:
    validated = MODULE.load_and_validate(_write_inputs(tmp_path))
    first = MODULE.analyze(validated, bootstrap_replicates=32, bootstrap_seed=123)
    second = MODULE.analyze(validated, bootstrap_replicates=32, bootstrap_seed=123)
    assert first == second


def test_rejects_wrong_williams_schedule_even_when_row_count_is_unchanged(
    tmp_path: Path,
) -> None:
    def mutate(depth: int, records: list[dict[str, Any]]) -> None:
        if depth == 7:
            _first_row(records)["version"] = "V35"

    with pytest.raises(MODULE.AnalysisError, match="version must be 'V15'"):
        MODULE.load_and_validate(_write_inputs(tmp_path, mutation=mutate))


def test_rejects_nondeterministic_nodes(tmp_path: Path) -> None:
    def mutate(depth: int, records: list[dict[str, Any]]) -> None:
        if depth == 8:
            _first_row(records)["nodes"] += 1

    with pytest.raises(MODULE.AnalysisError, match="nodes are not deterministic"):
        MODULE.load_and_validate(_write_inputs(tmp_path, mutation=mutate))


def test_rejects_failed_run_summary(tmp_path: Path) -> None:
    def mutate(depth: int, records: list[dict[str, Any]]) -> None:
        if depth == 6:
            next(record for record in records if record["type"] == "run_summary")["status"] = "fail"

    with pytest.raises(MODULE.AnalysisError, match="status must be 'pass'"):
        MODULE.load_and_validate(_write_inputs(tmp_path, mutation=mutate))


def test_requires_independently_verifiable_emitted_corpus(tmp_path: Path) -> None:
    with pytest.raises(MODULE.AnalysisError, match="--emit-corpus"):
        MODULE.load_and_validate(_write_inputs(tmp_path, include_corpus=False))


def test_rejects_corpus_content_that_disagrees_with_hash(tmp_path: Path) -> None:
    def mutate(depth: int, records: list[dict[str, Any]]) -> None:
        if depth == 6:
            next(record for record in records if record["type"] == "corpus_position")["fen"] += " "

    with pytest.raises(MODULE.AnalysisError, match="fen must be"):
        MODULE.load_and_validate(_write_inputs(tmp_path, mutation=mutate))
