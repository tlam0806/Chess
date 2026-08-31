from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
import subprocess

import pytest

from tools import tune_nnue_v42_aspiration as tuner


def result_record(
    config: tuner.Config,
    nodes: float,
    wdl: float,
    stage: str = "exploration",
) -> dict:
    return {
        "kind": "result",
        "stage": stage,
        "label": config.label,
        "origin": "test",
        "config": config.canonical(),
        "config_hash": config.hash,
        "result": {"node_ratio": nodes, "mean_wdl_loss": wdl},
    }


def evaluator_result(config: tuner.Config, narrow: int = 1) -> dict:
    value = {
        "count": 10,
        "ranking_count": 10,
        "safety_count": 3,
        "ranking_target_abs_cp": tuner.RANKING_TARGET_ABS_CP,
        "include_all_in_objective": True,
        "objective": "wdl",
        "wdl_formula": tuner.WDL_FORMULA,
        "wdl_calibration_run": tuner.WDL_CALIBRATION_RUN,
        "critical_mistakes": 2,
        "safety_critical_mistakes": 1,
        "win_to_draw": 1,
        "win_to_loss": 1,
        "self_mated": 0,
        "heuristics_reset_per_sample": True,
        "candidate_search_mode": "iterative_depth",
        "candidate_time_ms": 0,
        "candidate_stopped_pct": 0.0,
        "dataset_ply_schema": tuner.DATASET_SCHEMA,
        "dataset_explicit_ply_count": 10,
        "dataset_explicit_ply_position_match_count": 10,
        "dataset_calibration_ply_min": 3,
        "dataset_calibration_ply_max": 80,
        "dataset_calibration_ply_mean": 31.0,
        "mean_wdl_loss": 0.001,
        "p95_wdl_loss": 0.003,
        "objective_loss": 0.001,
        "control_nodes": 1_000,
        "candidate_nodes": 900,
        "node_ratio": 0.9,
        "selective_config": tuner.PRODUCTION_SELECTIVE_CONFIG.copy(),
        "twofold_search_draw_enabled": True,
        "aspiration_config": config.canonical(),
        "control_exact_searches": 20,
        "control_exact_full_window_fallbacks": 0,
        "control_exact_range_conflicts": 0,
        "control_exact_unresolved_ranges": 0,
        "control_root_source": "immutable_cache",
        "control_time_source": "not_measured_cached",
        "control_cache_identity_sha256": "test-cache-identity",
        "control_live_root_searches": 0,
        "control_live_child_searches": 10,
        "control_cached_root_results": 10,
        "time_ratio": None,
        "strict_best_score_violations": 0,
        "strict_best_score_max_excess_cp": 0,
        "candidate_repetition_history_aware_searches": 10,
        "candidate_repetition_threefold_draws": 0,
        "candidate_repetition_search_cycle_draws": 0,
        "candidate_repetition_fifty_move_draws": 0,
        "candidate_repetition_tt_score_suppressions": 0,
    }
    value.update({name: 0 for name in tuner.TELEMETRY_FIELDS})
    if config.enabled:
        value["aspiration_completed_iterations"] = 60
        value["aspiration_final_mean_score_count"] = 10
        value["aspiration_narrow_iterations"] = narrow
        value["aspiration_narrow_attempts"] = narrow
    return value


def evaluator_control_cache() -> dict:
    return {
        "identity_sha256": "test-cache-identity",
        "summary": {
            "count": 10,
            "depth": 6,
            "control_nodes": 1_000,
            "control_exact_searches": 10,
            "control_exact_full_window_fallbacks": 0,
            "control_exact_range_conflicts": 0,
            "control_exact_unresolved_ranges": 0,
        },
    }


def test_canonical_hash_covers_all_nine_fields() -> None:
    baseline = tuner.Config()
    assert tuple(baseline.canonical()) == tuner.CONFIG_FIELDS
    assert len(baseline.hash) == 64
    for field in tuner.CONFIG_FIELDS:
        value = getattr(baseline, field)
        if field == "enabled":
            changed = tuner.replace(baseline, **{field: not value})
        elif field == "max_researches":
            changed = tuner.replace(baseline, **{field: value + 1})
        elif field == "mean_score_clamp_cp":
            changed = tuner.replace(baseline, **{field: value + 1})
        else:
            # Every behavioural default is interior to its validated range.
            changed = tuner.replace(baseline, **{field: value + 1})
        assert changed.hash != baseline.hash


def test_evaluator_args_set_full_config_and_boolean_switch() -> None:
    enabled = tuner.Config().evaluator_args()
    disabled = tuner.replace(tuner.Config(), enabled=False).evaluator_args()
    assert enabled[0] == "--enable-adaptive-aspiration"
    assert disabled[0] == "--disable-adaptive-aspiration"
    assert "--aspiration-max-researches" in enabled
    assert "--aspiration-mean-score-clamp-cp" in enabled
    assert len(enabled) == 17


def test_selective_evaluator_args_pin_full_production_profile() -> None:
    args = tuner.selective_evaluator_args()
    assert "--enable-lmr" in args
    assert "--enable-null-move" in args
    assert "--enable-reverse-futility" in args
    assert "--enable-late-move-pruning" in args
    assert "--enable-qsearch-see-pruning" in args
    assert "--disable-main-search-see-pruning" in args
    assert "--enable-twofold-search-draw" in args

    def value(flag: str) -> str:
        return args[args.index(flag) + 1]

    assert value("--lmr-base") == "0.45"
    assert value("--lmr-divisor") == "2.9"
    assert value("--null-reduction") == "3"
    assert value("--qsearch-see-threshold") == "-75"


def test_selfplay_args_pin_both_profiles_and_twofold_policy() -> None:
    candidate = result_record(tuner.Config(), 0.8, 0.001)
    legacy_config = tuner.replace(tuner.Config(), enabled=False)
    legacy = result_record(legacy_config, 1.0, 0.001)
    payload = tuner.selfplay_confirmation_args(
        "balanced", candidate, legacy)
    arguments = payload["arguments"]

    assert arguments.count("--profile") == 2
    assert arguments.count("--aspiration-profile") == 2
    assert arguments.count("--twofold-search-profile") == 2
    assert payload["twofold_search_draw_enabled_for_both"] is True
    assert "balanced" in payload["suggested_output_filename"]
    assert candidate["config_hash"][:12] in payload["suggested_output_filename"]
    assert legacy["config_hash"][:12] in payload["suggested_output_filename"]
    aspiration_specs = [
        arguments[index + 1]
        for index, value in enumerate(arguments)
        if value == "--aspiration-profile"
    ]
    assert any(spec.split(",")[1] == "1" for spec in aspiration_specs)
    assert any(spec.split(",")[1] == "0" for spec in aspiration_specs)


def test_latin_hypercube_is_unique_and_freezes_guards() -> None:
    configs = tuner.latin_hypercube_configs(
        64, 123, max_researches=7, mean_score_clamp_cp=1_700)
    assert len(configs) == 64
    assert len({config.hash for config in configs}) == 64
    assert {config.max_researches for config in configs} == {7}
    assert {config.mean_score_clamp_cp for config in configs} == {1_700}
    assert all(4_000 <= config.delta_divisor <= 40_000 for config in configs)


def test_global_plan_contains_anchors_and_exact_requested_count() -> None:
    proposals = tuner.global_proposals(32, 99, 6, 1_500)
    assert len(proposals) == 32
    assert len({item.config.hash for item in proposals}) == 32
    origins = {item.origin for item in proposals}
    assert "anchor:legacy-v41" in origins
    assert "anchor:v42-seed" in origins
    assert "global:latin-hypercube" in origins


def test_refinement_uses_multiple_methods_and_never_mutates_guards() -> None:
    parents = [
        result_record(tuner.Config(), 0.90, 0.002),
        result_record(tuner.replace(
            tuner.Config(), min_depth=5, delta_base_cp=70,
            delta_divisor=30_000, expansion_factor_per_mille=2_500,
            max_fail_high_reductions=0,
            mean_score_new_weight_per_mille=800), 0.80, 0.004),
    ]
    proposals = tuner.refinement_proposals(
        parents, 40, 55,
        {entry["config_hash"] for entry in parents}, 6, 1_500)
    assert len(proposals) == 40
    assert len({item.config.hash for item in proposals}) == 40
    assert all(item.config.max_researches == 6 for item in proposals)
    assert all(item.config.mean_score_clamp_cp == 1_500 for item in proposals)
    methods = {item.origin for item in proposals}
    assert "refine:mutate-1" in methods
    assert "refine:mutate-2" in methods
    assert "refine:mutate-3" in methods
    assert "refine:crossover" in methods
    assert "refine:random-restart" in methods


def test_frontier_is_not_thinned_or_capped_at_twelve() -> None:
    entries = [
        result_record(
            tuner.replace(tuner.Config(), delta_base_cp=10 + index),
            nodes=0.60 + index * 0.01,
            wdl=0.020 - index * 0.0005,
        )
        for index in range(20)
    ]
    assert len(tuner.frontier(entries)) == 20


def test_frontier_removes_wdl_node_dominated_config() -> None:
    safe = result_record(tuner.Config(), 0.90, 0.001)
    fast = result_record(
        tuner.replace(tuner.Config(), delta_base_cp=40), 0.80, 0.002)
    dominated = result_record(
        tuner.replace(tuner.Config(), delta_base_cp=50), 0.95, 0.003)
    assert tuner.frontier([safe, fast, dominated]) == [safe, fast]


def test_legacy_dominating_all_does_not_erase_adaptive_frontier() -> None:
    legacy = result_record(
        tuner.replace(tuner.Config(), enabled=False), 0.50, 0.0005)
    adaptive_fast = result_record(
        tuner.replace(tuner.Config(), delta_base_cp=40), 0.70, 0.003)
    adaptive_safe = result_record(
        tuner.replace(tuner.Config(), delta_base_cp=50), 0.90, 0.001)
    entries = [legacy, adaptive_fast, adaptive_safe]

    assert tuner.frontier(entries) == [legacy]
    adaptive = tuner.adaptive_frontier(entries)
    assert adaptive == [adaptive_fast, adaptive_safe]
    assert {entry["config_hash"] for entry in adaptive} != {
        tuner.anchors(6, 1_500)[1].config.hash
    }

    refinements = tuner.refinement_proposals(
        entries, 8, 42, {entry["config_hash"] for entry in entries}, 6, 1_500)
    assert len(refinements) == 8
    assert all(proposal.config.enabled for proposal in refinements)


def test_adaptive_frontier_accepts_legacy_only_without_crashing() -> None:
    legacy = result_record(
        tuner.replace(tuner.Config(), enabled=False), 0.50, 0.0005)
    assert tuner.adaptive_frontier([legacy]) == []


def test_jsonl_resume_ignores_only_unterminated_trailing_record(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"kind":"result","n":1}\n{"kind":"res')
    assert tuner.load_jsonl(path) == [{"kind": "result", "n": 1}]
    assert path.read_bytes() == b'{"kind":"result","n":1}\n'
    tuner.append_jsonl(path, {"kind": "result", "n": 2})
    assert tuner.load_jsonl(path) == [
        {"kind": "result", "n": 1},
        {"kind": "result", "n": 2},
    ]

    path.write_bytes(b'{"kind":"result"}\nnot-json\n')
    with pytest.raises(RuntimeError, match="corrupt JSONL record 2"):
        tuner.load_jsonl(path)


def test_manifest_refuses_any_locked_field_change(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    expected = {"schema_version": 1, "seed": 7, "depths": {"gate": 6}}
    tuner.ensure_manifest(path, expected)
    tuner.ensure_manifest(path, expected)
    changed = {"schema_version": 1, "seed": 8, "depths": {"gate": 6}}
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        tuner.ensure_manifest(path, changed)


def test_legacy_manifest_migrates_duration_to_budget_policy(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    legacy_body = {
        "schema_version": 1,
        "seed": 7,
        "duration_sec": 100,
    }
    legacy = {
        **legacy_body,
        "manifest_sha256": hashlib.sha256(
            tuner.canonical_json(legacy_body)).hexdigest(),
    }
    path.write_text(json.dumps(legacy) + "\n")

    expected_body = {
        "schema_version": 1,
        "seed": 7,
        "budget_policy": tuner.BUDGET_POLICY,
    }
    expected = {
        **expected_body,
        "manifest_sha256": hashlib.sha256(
            tuner.canonical_json(expected_body)).hexdigest(),
    }
    assert tuner.ensure_manifest(path, expected) == 100
    assert json.loads(path.read_text()) == expected


def test_cumulative_budget_can_only_extend(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    records: list[dict] = []

    tuner.ensure_budget_limit(records, path, 100)
    assert records[-1]["previous_limit_sec"] == 0
    assert records[-1]["cumulative_limit_sec"] == 100
    tuner.ensure_budget_limit(records, path, 100)
    assert len(records) == 1

    elapsed = {"kind": "result", "budget_elapsed_sec": 105.0}
    tuner.append_jsonl(path, elapsed)
    records.append(elapsed)
    assert tuner.Budget(
        100.0, tuner.max_budget_elapsed(records), tuner.time.monotonic()
    ).exhausted

    tuner.ensure_budget_limit(records, path, 250)
    assert records[-1]["previous_limit_sec"] == 100
    assert records[-1]["cumulative_limit_sec"] == 250
    resumed = tuner.Budget(
        250.0, tuner.max_budget_elapsed(records), tuner.time.monotonic())
    assert not resumed.exhausted
    with pytest.raises(RuntimeError, match="may not decrease"):
        tuner.ensure_budget_limit(records, path, 200)

    persisted = tuner.load_jsonl(path)
    assert [
        entry["cumulative_limit_sec"]
        for entry in persisted if entry["kind"] == "budget_limit"
    ] == [100, 250]


def test_fixed_rung_subset_is_stable_and_refuses_rotation(tmp_path) -> None:
    source = tmp_path / "source.tsv"
    rows = []
    categories = ("random", "balanced", "tactical", "endgame")
    for category in categories:
        rows.extend(
            f"{category}\t{category}-{index}\t{index}\tfen-{index}"
            for index in range(100))
    source.write_text("\n".join(rows) + "\n")
    subset = tmp_path / "rung.tsv"
    tuner.ensure_subset(subset, source, 100, 123)
    first = subset.read_bytes()
    selected = subset.read_text().splitlines()
    assert len(selected) == 100
    category_counts = {
        category: sum(row.startswith(category + "\t") for row in selected)
        for category in categories
    }
    assert category_counts == {
        "random": 25,
        "balanced": 25,
        "tactical": 25,
        "endgame": 25,
    }
    tuner.ensure_subset(subset, source, 100, 123)
    assert subset.read_bytes() == first
    with pytest.raises(RuntimeError, match="does not match"):
        tuner.ensure_subset(subset, source, 100, 124)


def test_result_validation_requires_effective_config_and_real_aspiration() -> None:
    config = tuner.Config()
    control_cache = evaluator_control_cache()
    tuner.validate_result(evaluator_result(config), config, depth=6,
                          candidate_time_ms=0, control_cache=control_cache)

    no_narrow = evaluator_result(config, narrow=0)
    with pytest.raises(RuntimeError, match="never opened"):
        tuner.validate_result(no_narrow, config, 6, 0, control_cache)

    wrong = evaluator_result(config)
    wrong["aspiration_config"] = tuner.replace(
        config, delta_base_cp=31).canonical()
    with pytest.raises(RuntimeError, match="differs"):
        tuner.validate_result(wrong, config, 6, 0, control_cache)

    legacy_ply = evaluator_result(config)
    legacy_ply["dataset_ply_schema"] = "legacy_payload_derived_v0"
    legacy_ply["dataset_explicit_ply_count"] = 0
    with pytest.raises(RuntimeError, match="explicit-ply"):
        tuner.validate_result(legacy_ply, config, 6, 0, control_cache)

    wrong_selective = evaluator_result(config)
    wrong_selective["selective_config"]["lmr_base"] = 0.5
    with pytest.raises(RuntimeError, match="selective config"):
        tuner.validate_result(wrong_selective, config, 6, 0, control_cache)

    no_history = evaluator_result(config)
    no_history["candidate_repetition_history_aware_searches"] = 0
    with pytest.raises(RuntimeError, match="repetition-history"):
        tuner.validate_result(no_history, config, 6, 0, control_cache)

    skipped_teacher_root = evaluator_result(config)
    skipped_teacher_root["control_exact_searches"] = 9
    with pytest.raises(RuntimeError, match="skipped one or more exact root"):
        tuner.validate_result(skipped_teacher_root, config, 6, 0, control_cache)

    unresolved_teacher = evaluator_result(config)
    unresolved_teacher["control_exact_unresolved_ranges"] = 1
    with pytest.raises(RuntimeError, match="unresolved range"):
        tuner.validate_result(unresolved_teacher, config, 6, 0, control_cache)

    inconsistent_strict_best = evaluator_result(config)
    inconsistent_strict_best["strict_best_score_max_excess_cp"] = 1
    with pytest.raises(RuntimeError, match="strict best score"):
        tuner.validate_result(
            inconsistent_strict_best, config, 6, 0, control_cache)

    invalid_ply_stats = evaluator_result(config)
    invalid_ply_stats["dataset_calibration_ply_mean"] = 100.0
    with pytest.raises(RuntimeError, match="explicit-ply statistics"):
        tuner.validate_result(invalid_ply_stats, config, 6, 0, control_cache)

    wrong_control_nodes = evaluator_result(config)
    wrong_control_nodes["control_nodes"] = 999
    with pytest.raises(RuntimeError, match="immutable root cache"):
        tuner.validate_result(
            wrong_control_nodes, config, 6, 0, control_cache)

    wrong_ratio = evaluator_result(config)
    wrong_ratio["node_ratio"] = 0.89
    with pytest.raises(RuntimeError, match="node_ratio"):
        tuner.validate_result(wrong_ratio, config, 6, 0, control_cache)

    invalid_loss = evaluator_result(config)
    invalid_loss["mean_wdl_loss"] = -0.001
    with pytest.raises(RuntimeError, match="mean_wdl_loss"):
        tuner.validate_result(invalid_loss, config, 6, 0, control_cache)

    wrong_safety_threshold = evaluator_result(config)
    wrong_safety_threshold["ranking_target_abs_cp"] = 1499
    with pytest.raises(RuntimeError, match="safety threshold"):
        tuner.validate_result(
            wrong_safety_threshold, config, 6, 0, control_cache)

    wrong_objective = evaluator_result(config)
    wrong_objective["objective"] = "cp"
    with pytest.raises(RuntimeError, match="objective is not WDL"):
        tuner.validate_result(wrong_objective, config, 6, 0, control_cache)

    wrong_wdl_formula = evaluator_result(config)
    wrong_wdl_formula["wdl_formula"] = "other"
    with pytest.raises(RuntimeError, match="WDL formula"):
        tuner.validate_result(
            wrong_wdl_formula, config, 6, 0, control_cache)

    wrong_wdl_run = evaluator_result(config)
    wrong_wdl_run["wdl_calibration_run"] = "other"
    with pytest.raises(RuntimeError, match="WDL calibration run"):
        tuner.validate_result(wrong_wdl_run, config, 6, 0, control_cache)

    wrong_objective_loss = evaluator_result(config)
    wrong_objective_loss["objective_loss"] = 0.002
    with pytest.raises(RuntimeError, match="objective_loss"):
        tuner.validate_result(
            wrong_objective_loss, config, 6, 0, control_cache)

    invalid_safety_count = evaluator_result(config)
    invalid_safety_count["safety_count"] = 11
    with pytest.raises(RuntimeError, match="safety_count"):
        tuner.validate_result(
            invalid_safety_count, config, 6, 0, control_cache)

    invalid_safety_critical = evaluator_result(config)
    invalid_safety_critical["safety_critical_mistakes"] = 3
    with pytest.raises(RuntimeError, match="safety_critical_mistakes"):
        tuner.validate_result(
            invalid_safety_critical, config, 6, 0, control_cache)

    bad_iterations = evaluator_result(config)
    bad_iterations["aspiration_completed_iterations"] = 59
    with pytest.raises(RuntimeError, match="iteration accounting"):
        tuner.validate_result(bad_iterations, config, 6, 0, control_cache)

    bad_attempts = evaluator_result(config)
    bad_attempts["aspiration_narrow_attempts"] = 0
    with pytest.raises(RuntimeError, match="attempts are fewer"):
        tuner.validate_result(bad_attempts, config, 6, 0, control_cache)

    legacy = tuner.replace(config, enabled=False)
    legacy_result = evaluator_result(legacy, narrow=0)
    tuner.validate_result(legacy_result, legacy, 6, 0, control_cache)
    legacy_result["aspiration_fail_highs"] = 1
    with pytest.raises(RuntimeError, match="disabled aspiration"):
        tuner.validate_result(legacy_result, legacy, 6, 0, control_cache)


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("control_root_source", "live_search", "immutable root cache"),
        ("control_time_source", "measured", "cached control wall time"),
        ("time_ratio", 1.0, "time_ratio=null"),
        ("control_cache_identity_sha256", "wrong", "wrong root-cache"),
        ("control_cached_root_results", 9, "cached root results"),
        ("control_live_root_searches", 1, "reran a live root"),
        ("control_exact_searches", 19, "exact-search accounting"),
    ],
)
def test_result_validation_rejects_cache_provenance_or_accounting_violation(
    field, value, diagnostic,
) -> None:
    config = tuner.Config()
    result = evaluator_result(config)
    result[field] = value
    with pytest.raises(RuntimeError, match=diagnostic):
        tuner.validate_result(
            result, config, 6, 0, evaluator_control_cache())


def test_evaluate_classifies_launch_signal_timeout_json_and_schema_as_fatal(
    tmp_path, monkeypatch,
) -> None:
    config = tuner.Config()
    arguments = (
        tmp_path / "binary", tmp_path / "dataset.tsv", tmp_path / "model",
        6, config, 0, 64, {
            "path": str(tmp_path / "control-cache.tsv"),
            "identity_sha256": "test-cache-identity",
            "identity": {
                "evaluator": {"sha256": "evaluator"},
                "model": {"sha256": "model"},
                "ordered_dataset": {"sha256": "dataset"},
            },
        },
    )

    monkeypatch.setattr(
        tuner.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic launch failure")))
    with pytest.raises(tuner.EvaluationFailure, match="failed to launch"):
        tuner.evaluate(*arguments)

    monkeypatch.setattr(
        tuner.subprocess, "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], -9, stdout="", stderr="killed"))
    with pytest.raises(tuner.EvaluationFailure, match="signal 9"):
        tuner.evaluate(*arguments)

    monkeypatch.setattr(
        tuner.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(
                ["fake"], 1.0, output="partial", stderr="timed out")))
    with pytest.raises(tuner.EvaluationFailure, match="timed out"):
        tuner.evaluate(*arguments)

    monkeypatch.setattr(
        tuner.subprocess, "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout="not-json", stderr=""))
    with pytest.raises(tuner.EvaluationFailure, match="invalid JSON"):
        tuner.evaluate(*arguments)

    malformed = evaluator_result(config)
    del malformed["dataset_ply_schema"]
    monkeypatch.setattr(
        tuner.subprocess, "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(malformed), stderr=""))
    with pytest.raises(
        tuner.EvaluationFailure, match="result validation failed"
    ):
        tuner.evaluate(*arguments)


def test_evaluate_enforces_shared_budget_deadline(tmp_path, monkeypatch) -> None:
    config = tuner.Config()
    observed_timeout = None

    def timeout(_command, **kwargs):
        nonlocal observed_timeout
        observed_timeout = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(["fake"], observed_timeout)

    monkeypatch.setattr(tuner.subprocess, "run", timeout)
    cache = {
        "path": str(tmp_path / "cache.tsv"),
        "identity_sha256": "test-cache-identity",
        "identity": {
            "evaluator": {"sha256": "evaluator"},
            "model": {"sha256": "model"},
            "ordered_dataset": {"sha256": "dataset"},
        },
    }
    deadline = time.monotonic() + 0.1
    with pytest.raises(tuner.EvaluationBudgetExhausted, match="deadline"):
        tuner.evaluate(
            tmp_path / "binary", tmp_path / "dataset.tsv",
            tmp_path / "model", 6, config, 0, 64, cache, deadline)
    assert observed_timeout is not None
    assert 0.0 < observed_timeout <= 0.1


def test_config_from_dict_refuses_partial_state() -> None:
    partial = tuner.Config().canonical()
    del partial["max_researches"]
    with pytest.raises(ValueError, match="non-canonical"):
        tuner.Config.from_dict(partial)


def test_equal_aggregate_objectives_are_not_assumed_identical() -> None:
    first = result_record(tuner.Config(), 0.8, 0.001)
    second = result_record(
        tuner.replace(tuner.Config(), delta_base_cp=40), 0.8, 0.001)
    first["result"].update({
        "candidate_nodes": 800,
        "aspiration_accepted_depth_ratio": 0.95,
    })
    second["result"].update({
        "candidate_nodes": 800,
        "aspiration_accepted_depth_ratio": 1.0,
    })
    assert tuner.deduplicate_objectives([first, second]) == [first, second]


def test_relative_profile_uses_candidate_nodes_not_v36_ratio() -> None:
    legacy = result_record(tuner.replace(tuner.Config(), enabled=False), 0.1, 0.002)
    candidate = result_record(tuner.Config(), 0.05, 0.003)
    legacy["result"]["candidate_nodes"] = 1_000
    candidate["result"]["candidate_nodes"] = 600
    profile = tuner.relative_profile(candidate, legacy)
    assert profile["node_ratio_vs_legacy"] == pytest.approx(0.6)
    assert profile["wdl_loss_delta_vs_legacy"] == pytest.approx(0.001)


def test_dataset_schema_rejects_legacy_and_wrong_book_ply(tmp_path) -> None:
    legacy = tmp_path / "legacy.tsv"
    legacy.write_text("random\thash\t8/8/8/8/8/8/8/8 w - - 0 1\n")
    with pytest.raises(RuntimeError, match="four-column"):
        tuner.validate_dataset_schema(legacy)

    wrong_book = tmp_path / "wrong-book.tsv"
    wrong_book.write_text("balanced\thash\t3\tbook:e2e4 e7e5\n")
    with pytest.raises(RuntimeError, match="book ply"):
        tuner.validate_dataset_schema(wrong_book)


def test_balanced_lineage_cannot_cross_splits(tmp_path) -> None:
    tune = tmp_path / "tune.tsv"
    selection = tmp_path / "selection.tsv"
    holdout = tmp_path / "holdout.tsv"
    tune.write_text("balanced\ta\t4\tbook:e2e4 e7e5 g1f3 b8c6\n")
    selection.write_text(
        "balanced\tb\t5\tbook:e2e4 e7e5 g1f3 b8c6 f1b5\n")
    holdout.write_text("natural\tc\t20\t8/8/8/8/8/8/8/8 w - - 0 1\n")
    with pytest.raises(RuntimeError, match="lineage overlaps"):
        tuner.validate_no_balanced_lineage_overlap({
            "tune": tune, "selection": selection, "holdout": holdout})

    selection.write_text("balanced\tb\t4\tbook:d2d4 d7d5 c2c4 e7e6\n")
    tuner.validate_no_balanced_lineage_overlap({
        "tune": tune, "selection": selection, "holdout": holdout})


def test_dataset_hashes_are_unique_within_and_across_splits(tmp_path) -> None:
    tune = tmp_path / "tune.tsv"
    selection = tmp_path / "selection.tsv"
    holdout = tmp_path / "holdout.tsv"
    tune.write_text(
        "phase0\ta\t0\t8/8/8/8/8/8/8/K6k w - - 0 1\n"
        "phase1\ta\t1\t8/8/8/8/8/8/8/K6k b - - 0 1\n")
    with pytest.raises(RuntimeError, match="duplicate position hash"):
        tuner.validate_dataset_schema(tune)

    tune.write_text("phase0\ta\t0\t8/8/8/8/8/8/8/K6k w - - 0 1\n")
    selection.write_text("phase1\ta\t1\t8/8/8/8/8/8/8/K6k b - - 0 1\n")
    holdout.write_text("phase2\tc\t2\t8/8/8/8/8/8/8/K6k w - - 0 2\n")
    with pytest.raises(RuntimeError, match="overlaps dataset splits"):
        tuner.validate_dataset_hash_disjointness({
            "tune": tune, "selection": selection, "holdout": holdout})


def test_sealed_dataset_manifest_is_required_and_cross_checked(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    paths = {
        "tune": dataset_dir / "tune.tsv",
        "selection": dataset_dir / "selection.tsv",
        "holdout": dataset_dir / "holdout.tsv",
    }
    rows = {
        "tune": "phase0\t00000000000000000000000000000001\t0\t"
                "8/8/8/8/8/8/8/K6k w - - 0 1\n",
        "selection": "phase1\t00000000000000000000000000000002\t1\t"
                     "7k/8/8/8/8/8/QQ6/KQ6 b - - 0 1\n",
        "holdout": "phase2\t00000000000000000000000000000003\t2\t"
                   "7k/8/8/8/8/Q7/QQQ5/KQQQ4 w - - 0 2\n",
    }
    for split, path in paths.items():
        path.write_text(rows[split])

    with pytest.raises(RuntimeError, match="missing sealed"):
        tuner.validate_sealed_dataset_manifest(dataset_dir, paths)
    assert tuner.validate_sealed_dataset_manifest(
        dataset_dir, paths, allow_unsealed=True)["sealed"] is False

    manifest = {
        "schema_version": 2,
        "format": "nnue-v42-lichess-game-disjoint-v2",
        "selection_policy": {
            "split_unit": "game_id",
            "positions_per_game": 1,
            "output_ply": "zero-based absolute ply derived from six-field FEN",
        },
        "game_overlap": {
            "tune_selection": 0, "tune_holdout": 0,
            "selection_holdout": 0,
        },
        "position_overlap": {
            "tune_selection": 0, "tune_holdout": 0,
            "selection_holdout": 0,
        },
        "splits": {
            split: {
                "count": 1,
                "games": 1,
                "phase_counts": {str(index): 1},
            }
            for index, split in enumerate(paths)
        },
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    identity = tuner.validate_sealed_dataset_manifest(dataset_dir, paths)
    assert identity["sealed"] is True
    assert identity["manifest"]["sha256"] == tuner.sha256_file(
        dataset_dir / "manifest.json")

    paths["tune"].write_text(rows["tune"].replace("phase0", "phase1", 1))
    with pytest.raises(RuntimeError, match="piece-count phase"):
        tuner.validate_sealed_dataset_manifest(dataset_dir, paths)
    paths["tune"].write_text(rows["tune"])

    manifest["splits"]["tune"]["phase_counts"] = {"1": 1}
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="phase_counts mismatch"):
        tuner.validate_sealed_dataset_manifest(dataset_dir, paths)
    manifest["splits"]["tune"]["phase_counts"] = {"0": 1}

    manifest["game_overlap"]["tune_holdout"] = 1
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="game_overlap"):
        tuner.validate_sealed_dataset_manifest(dataset_dir, paths)


def test_refinement_is_split_into_two_deterministic_generations() -> None:
    assert tuner.refinement_generation_counts(64) == (32, 32)
    assert tuner.refinement_generation_counts(63) == (32, 31)
    parents = [result_record(tuner.Config(), 0.9, 0.001)]
    first = tuner.refinement_proposals(
        parents, 8, 123 ^ 0x52454631,
        {parents[0]["config_hash"]}, 6, 1_500)
    again = tuner.refinement_proposals(
        parents, 8, 123 ^ 0x52454631,
        {parents[0]["config_hash"]}, 6, 1_500)
    assert [item.config.hash for item in first] == [
        item.config.hash for item in again]
    combined_parents = parents + [
        result_record(item.config, 0.89 - index * 0.01, 0.001 + index * 0.0001)
        for index, item in enumerate(first)
    ]
    second = tuner.refinement_proposals(
        combined_parents, 8, 123 ^ 0x52454632,
        {entry["config_hash"] for entry in combined_parents}, 6, 1_500)
    assert not ({item.config.hash for item in first}
                & {item.config.hash for item in second})


def _cache_fixture(tmp_path: Path, rows: int = 1) -> tuple[Path, Path, Path, dict]:
    binary = tmp_path / "evaluator"
    model = tmp_path / "model.bin"
    dataset = tmp_path / "cache-dataset.tsv"
    binary.write_bytes(b"evaluator-v1")
    model.write_bytes(b"model-v1")
    dataset.write_text("".join(
        f"phase0\thash-{index}\t{index}\t"
        f"8/8/8/8/8/8/8/K6k {'w' if index % 2 == 0 else 'b'} - - 0 1\n"
        for index in range(rows)
    ))
    spec = tuner.control_cache_spec(
        tmp_path, "fixture", binary, model, dataset, 6)
    return binary, model, dataset, spec


def _write_fake_control_cache(spec: dict, dataset: Path) -> dict:
    identity = spec["identity"]
    lines = [
        tuner.CONTROL_CACHE_SCHEMA,
        f"identity_sha256\t{spec['identity_sha256']}",
        f"evaluator_sha256\t{identity['evaluator']['sha256']}",
        f"model_sha256\t{identity['model']['sha256']}",
        f"dataset_sha256\t{identity['ordered_dataset']['sha256']}",
        f"depth\t{identity['depth']}",
        f"offset\t{identity['offset']}",
        f"count\t{identity['count']}",
        "\t".join(tuner.CONTROL_CACHE_COLUMNS),
    ]
    for index, dataset_line in enumerate(dataset.read_text().splitlines()):
        _, sample_hash, ply, _ = tuner.parse_dataset_row(
            dataset_line, dataset, index + 1)
        lines.append(
            f"{index}\t{sample_hash}\t{ply}\t{index}\t1\t0\t"
            f"{100 + index}\t{identity['depth']}\t0\t0\t0\t0")
    Path(spec["path"]).write_text("\n".join(lines) + "\n")
    return tuner.validate_control_cache(Path(spec["path"]), spec, dataset)


def test_control_cache_identity_pins_every_semantic_input(tmp_path) -> None:
    binary, model, dataset, baseline = _cache_fixture(tmp_path, rows=2)
    baseline_id = baseline["identity_sha256"]

    assert tuner.control_cache_spec(
        tmp_path, "other", binary, model, dataset, 7
    )["identity_sha256"] != baseline_id
    binary.write_bytes(b"evaluator-v2")
    assert tuner.control_cache_spec(
        tmp_path, "other", binary, model, dataset, 6
    )["identity_sha256"] != baseline_id
    binary.write_bytes(b"evaluator-v1")
    model.write_bytes(b"model-v2")
    assert tuner.control_cache_spec(
        tmp_path, "other", binary, model, dataset, 6
    )["identity_sha256"] != baseline_id
    model.write_bytes(b"model-v1")
    dataset.write_text("\n".join(
        reversed(dataset.read_text().splitlines())) + "\n")
    assert tuner.control_cache_spec(
        tmp_path, "other", binary, model, dataset, 6
    )["identity_sha256"] != baseline_id


def test_control_cache_validator_rejects_tamper_reorder_and_truncation(
    tmp_path,
) -> None:
    _, _, dataset, spec = _cache_fixture(tmp_path, rows=2)
    _write_fake_control_cache(spec, dataset)
    path = Path(spec["path"])
    original = path.read_text().splitlines()

    tampered = original.copy()
    tampered[9] = tampered[9].replace("hash-0", "tamper", 1)
    path.write_text("\n".join(tampered) + "\n")
    with pytest.raises(RuntimeError, match="identity/order mismatch"):
        tuner.validate_control_cache(path, spec, dataset)

    reordered = original.copy()
    reordered[9], reordered[10] = reordered[10], reordered[9]
    path.write_text("\n".join(reordered) + "\n")
    with pytest.raises(RuntimeError, match="identity/order mismatch"):
        tuner.validate_control_cache(path, spec, dataset)

    path.write_text("\n".join(original[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="row count mismatch"):
        tuner.validate_control_cache(path, spec, dataset)


def test_control_cache_builds_once_and_resume_verifies_immutable_sha(
    tmp_path, monkeypatch,
) -> None:
    binary, model, dataset, spec = _cache_fixture(tmp_path)
    calls = 0

    def fake_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        summary = _write_fake_control_cache(spec, dataset)
        report = {
            "control_cache_written": True,
            "control_cache_schema": tuner.CONTROL_CACHE_SCHEMA,
            "control_cache_identity_sha256": spec["identity_sha256"],
            **summary,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(report), "")

    monkeypatch.setattr(tuner.subprocess, "run", fake_run)
    records: list[dict] = []
    log = tmp_path / "results.jsonl"
    budget = tuner.Budget(100.0, 0.0, time.monotonic())
    first = tuner.ensure_control_cache(
        records, log, spec, binary, model, dataset, budget)
    again = tuner.ensure_control_cache(
        records, log, spec, binary, model, dataset, budget)
    assert first == again
    assert calls == 1
    assert len([
        record for record in records
        if record.get("kind") == "control_cache_ready"
    ]) == 1

    Path(spec["path"]).write_text(Path(spec["path"]).read_text() + "tamper\n")
    with pytest.raises(RuntimeError, match="artifact changed"):
        tuner.ensure_control_cache(
            records, log, spec, binary, model, dataset, budget)


def test_control_cache_budget_timeout_cleans_temp_and_can_resume(
    tmp_path, monkeypatch,
) -> None:
    binary, model, dataset, spec = _cache_fixture(tmp_path)
    path = Path(spec["path"])
    temporary = Path(str(path) + ".tmp-killed")

    def timeout(_command, **_kwargs):
        temporary.write_text("partial")
        raise subprocess.TimeoutExpired(["fake"], 0.01)

    monkeypatch.setattr(tuner.subprocess, "run", timeout)
    budget = tuner.Budget(100.0, 0.0, time.monotonic())
    with pytest.raises(tuner.CacheBudgetExhausted, match="while building"):
        tuner.ensure_control_cache(
            [], tmp_path / "results.jsonl", spec,
            binary, model, dataset, budget)
    assert not path.exists()
    assert not temporary.exists()

    def succeed(command, **_kwargs):
        summary = _write_fake_control_cache(spec, dataset)
        report = {
            "control_cache_written": True,
            "control_cache_schema": tuner.CONTROL_CACHE_SCHEMA,
            "control_cache_identity_sha256": spec["identity_sha256"],
            **summary,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(report), "")

    monkeypatch.setattr(tuner.subprocess, "run", succeed)
    records: list[dict] = []
    descriptor = tuner.ensure_control_cache(
        records, tmp_path / "results.jsonl", spec,
        binary, model, dataset, budget)
    assert descriptor["artifact"] == tuner.file_fingerprint(path)


def test_control_cache_does_not_start_after_budget_exhaustion(
    tmp_path, monkeypatch,
) -> None:
    binary, model, dataset, spec = _cache_fixture(tmp_path)
    monkeypatch.setattr(
        tuner.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("cache builder was invoked"))
    with pytest.raises(tuner.CacheBudgetExhausted, match="before building"):
        tuner.ensure_control_cache(
            [], tmp_path / "results.jsonl", spec, binary, model, dataset,
            tuner.Budget(0.0, 0.0, time.monotonic()))


def test_torn_final_cache_is_never_accepted_or_overwritten(
    tmp_path, monkeypatch,
) -> None:
    binary, model, dataset, spec = _cache_fixture(tmp_path)
    Path(spec["path"]).write_text("partial-final")
    monkeypatch.setattr(
        tuner.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("torn final was overwritten"))
    with pytest.raises(RuntimeError, match="truncated control cache"):
        tuner.ensure_control_cache(
            [], tmp_path / "results.jsonl", spec, binary, model, dataset,
            tuner.Budget(100.0, 0.0, time.monotonic()))


def _test_plan(tmp_path, stage: str, proposals: list[tuner.Proposal]) -> dict:
    dataset = tmp_path / "dataset.tsv"
    dataset.write_text("natural\thash\t4\t8/8/8/8/8/8/8/8 w - - 0 1\n")
    cache_binary = tmp_path / "cache-binary"
    cache_model = tmp_path / "cache-model"
    cache_binary.write_bytes(b"binary")
    cache_model.write_bytes(b"model")
    cache = tuner.control_cache_spec(
        tmp_path, f"test-{stage}", cache_binary, cache_model, dataset, 6)
    identity = cache["identity"]
    cache_path = Path(cache["path"])
    headers = [
        tuner.CONTROL_CACHE_SCHEMA,
        f"identity_sha256\t{cache['identity_sha256']}",
        f"evaluator_sha256\t{identity['evaluator']['sha256']}",
        f"model_sha256\t{identity['model']['sha256']}",
        f"dataset_sha256\t{identity['ordered_dataset']['sha256']}",
        "depth\t6",
        "offset\t0",
        "count\t1",
        "\t".join(tuner.CONTROL_CACHE_COLUMNS),
        "0\thash\t4\t0\t1\t0\t1\t6\t0\t0\t0\t0",
    ]
    cache_path.write_text("\n".join(headers) + "\n")
    cache = {
        **cache,
        "artifact": tuner.file_fingerprint(cache_path),
        "summary": tuner.validate_control_cache(cache_path, cache, dataset),
    }
    return {
        "stage": stage,
        "dataset": tuner.file_fingerprint(dataset),
        "depth": 6,
        "candidate_time_ms": 0,
        "control_cache": cache,
        "candidates": [tuner.proposal_payload(item) for item in proposals],
    }


@pytest.mark.parametrize(
    "stage", ["preflight", "exploration", "selection", "holdout"])
def test_evaluator_failure_aborts_every_stage_without_reject(
    tmp_path, monkeypatch, stage,
) -> None:
    failed = tuner.Proposal(tuner.Config(), "test:failed")
    plan = _test_plan(tmp_path, stage, [failed])

    def fake_evaluate(_binary, _dataset, _model, _depth, config, *_args):
        raise tuner.EvaluationFailure(
            "fatal evaluator failure", command=["fake"], returncode=7,
            stderr="synthetic failure")

    monkeypatch.setattr(tuner, "evaluate", fake_evaluate)
    records: list[dict] = []
    log = tmp_path / "results.jsonl"
    budget = tuner.Budget(100.0, 0.0, time.monotonic())
    with pytest.raises(tuner.EvaluationFailure, match="fatal evaluator") as caught:
        tuner.run_stage(
            records, log, plan, tmp_path / "binary", tmp_path / "model",
            1, 64, budget)
    assert stage in str(caught.value)
    assert failed.config.label in str(caught.value)
    assert not any(entry.get("kind") == "reject" for entry in records)
    assert not any(entry.get("kind") == "stage_complete" for entry in records)


def test_unknown_evaluator_exception_is_fatal(tmp_path, monkeypatch) -> None:
    proposal = tuner.Proposal(tuner.Config(), "test:unknown-failure")
    plan = _test_plan(tmp_path, "exploration", [proposal])
    monkeypatch.setattr(
        tuner, "evaluate",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("unknown failure")))
    with pytest.raises(RuntimeError, match="unknown failure"):
        tuner.run_stage(
            [], tmp_path / "results.jsonl", plan,
            tmp_path / "binary", tmp_path / "model", 1, 64,
            tuner.Budget(100.0, 0.0, time.monotonic()))


def test_stage_workers_share_one_budget_deadline_and_pause_cleanly(
    tmp_path, monkeypatch,
) -> None:
    proposals = [
        tuner.Proposal(tuner.Config(), "test:first"),
        tuner.Proposal(
            tuner.replace(tuner.Config(), delta_base_cp=31), "test:second"),
    ]
    plan = _test_plan(tmp_path, "exploration", proposals)
    deadlines: list[float] = []

    def fake_evaluate(
        _binary, _dataset, _model, _depth, _config,
        _candidate_time_ms, _candidate_max_depth, _cache, deadline,
    ):
        deadlines.append(deadline)
        raise tuner.EvaluationBudgetExhausted("synthetic deadline")

    monkeypatch.setattr(tuner, "evaluate", fake_evaluate)
    records: list[dict] = []
    results, complete = tuner.run_stage(
        records, tmp_path / "results.jsonl", plan,
        tmp_path / "binary", tmp_path / "model", 2, 64,
        tuner.Budget(100.0, 0.0, time.monotonic()))
    assert results == []
    assert complete is False
    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]
    assert not any(record.get("kind") == "result" for record in records)
    assert not any(
        record.get("kind") == "stage_complete" for record in records)


def test_saved_per_config_reject_refuses_fail_closed_resume(tmp_path) -> None:
    proposal = tuner.Proposal(tuner.Config(), "test:legacy-reject")
    plan = _test_plan(tmp_path, "exploration", [proposal])
    reject = {
        "kind": "reject",
        "stage": "exploration",
        "config": proposal.config.canonical(),
        "config_hash": proposal.config.hash,
    }
    with pytest.raises(RuntimeError, match="fail-closed policy"):
        tuner.run_stage(
            [reject], tmp_path / "results.jsonl", plan,
            tmp_path / "binary", tmp_path / "model", 1, 64,
            tuner.Budget(100.0, 0.0, time.monotonic()))


def test_preflight_evaluator_failure_aborts(tmp_path, monkeypatch) -> None:
    proposal = tuner.Proposal(tuner.Config(), "preflight:test")
    plan = _test_plan(tmp_path, "preflight", [proposal])
    monkeypatch.setattr(
        tuner, "evaluate",
        lambda *_args: (_ for _ in ()).throw(
            tuner.EvaluationFailure("preflight correctness failure")))
    with pytest.raises(tuner.EvaluationFailure, match="correctness"):
        tuner.run_stage(
            [], tmp_path / "results.jsonl", plan,
            tmp_path / "binary", tmp_path / "model", 1, 64,
            tuner.Budget(100.0, 0.0, time.monotonic()))
