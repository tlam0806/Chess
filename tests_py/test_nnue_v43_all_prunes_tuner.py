from __future__ import annotations

import json
import hashlib
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tools import tune_nnue_v43_all_prunes as tuner


def result_record(
    candidate: tuner.Candidate,
    nodes: float,
    wdl: float,
    *,
    stage: str = "selection",
    detail: dict | None = None,
) -> dict:
    return {
        "kind": "result",
        "stage": stage,
        "label": candidate.label,
        "origin": "test",
        "config": candidate.canonical(),
        "config_hash": candidate.hash,
        "detail": detail,
        "result": {"node_ratio": nodes, "mean_wdl_loss": wdl},
    }


def detail_row(
    index: int,
    sample_hash: str,
    *,
    wdl_loss: float,
    candidate_nodes: int,
    control_root_nodes: int = 100,
) -> dict:
    """Exact v1 sidecar row; additions/removals must be explicit schema bumps."""
    return {
        "schema": "nnue_selective_position_details_jsonl_v1",
        "index": index,
        "global_index": index,
        "category": "phase0",
        "sample_hash": sample_hash,
        "calibration_ply": index,
        "phase": 0,
        "static_target_cp": 0,
        "ranking_sample": True,
        "safety_sample": False,
        "control_move": "e2e4",
        "candidate_move": "e2e4",
        "control_score": 0,
        "candidate_strict_score": 0,
        "control_root_nodes": control_root_nodes,
        "candidate_nodes": candidate_nodes,
        "candidate_depth": 7,
        "candidate_stopped": False,
        "move_match": True,
        "cp_regret": 0,
        "wdl_loss": wdl_loss,
        "win_to_draw": False,
        "win_to_loss": False,
        "self_mated": False,
        "critical_mistake": False,
    }


def write_dataset_and_details(
    tmp_path: Path,
    name: str,
    metrics: list[tuple[float, int]],
) -> tuple[Path, Path, dict]:
    hashes = [f"sample-{index}" for index in range(len(metrics))]
    dataset = tmp_path / "dataset.tsv"
    if not dataset.exists():
        dataset.write_text("".join(
            f"phase0\t{sample_hash}\t{index}\tfen-{index}\n"
            for index, sample_hash in enumerate(hashes)
        ))
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(
        json.dumps(detail_row(
            index,
            sample_hash,
            wdl_loss=metric[0],
            candidate_nodes=metric[1],
            control_root_nodes=100 + index,
        ), sort_keys=True) + "\n"
        for index, (sample_hash, metric) in enumerate(zip(hashes, metrics))
    ))
    descriptor = tuner.validate_detail_sidecar(path, dataset, len(metrics))
    return dataset, path, descriptor


def aspiration_telemetry(
    *,
    count: int = 2,
    depth: int = 6,
    nominal_depth_sum: int = 10,
    search_depth_sum: int = 9,
    accepted_reduced: int = 1,
    max_reduction: int = 1,
) -> dict:
    result = {name: 0 for name in tuner.infra.TELEMETRY_FIELDS}
    result.update({
        "aspiration_completed_iterations": count * depth,
        "aspiration_narrow_iterations": 2,
        "aspiration_narrow_attempts": 3,
        "aspiration_initial_window_successes": 1,
        "aspiration_accepted_reduced_depth_iterations": accepted_reduced,
        "aspiration_accepted_narrow_nominal_depth_sum": nominal_depth_sum,
        "aspiration_accepted_narrow_search_depth_sum": search_depth_sum,
        "aspiration_max_accepted_depth_reduction": max_reduction,
        "aspiration_final_mean_score_count": count,
        "aspiration_accepted_depth_ratio": (
            search_depth_sum / nominal_depth_sum
            if nominal_depth_sum else 1.0),
    })
    return result


def test_balanced_v43_is_exact_versioned_production_baseline() -> None:
    expected = {
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
    baseline = tuner.Candidate()

    assert tuner.SCHEMA_VERSION == 2
    assert tuner.EXPERIMENT == (
        "v43-final-joint-all-prunes-balanced-baseline-v2")
    assert tuner.PRODUCTION_BASELINE_ASPIRATION.canonical() == expected
    assert tuner.PRODUCTION_BASELINE_ASPIRATION.hash == (
        "906570ae8ede9e87540847228cd75eef707a64aa5873163be665e220c8b2515e")
    assert baseline.hash == (
        "e770be6261511ed0bc391852c27d69cbe203204c3b5db96e8cc329daf7772d48")
    assert baseline.label.startswith(
        "v43_production_fast_qsee_balanced_aspiration-")
    assert tuner.BALANCED_V43_ASPIRATION in tuner.ASPIRATION_ANCHORS


def test_sealed_overlap_proof_requires_exact_four_zero_keys() -> None:
    valid = {key: 0 for key in tuner.REQUIRED_DATASET_OVERLAP_KEYS}
    tuner.validate_dataset_overlap_proof(valid)

    invalid = (
        None,
        {},
        {key: 0 for key in list(tuner.REQUIRED_DATASET_OVERLAP_KEYS)[:-1]},
        {**valid, "unexpected_overlap_metric": 0},
        {**valid, "games_between_new_splits": 1},
    )
    for overlap in invalid:
        with pytest.raises(RuntimeError, match="overlap proof"):
            tuner.validate_dataset_overlap_proof(overlap)


def test_conditional_canonicalization_resets_disabled_children() -> None:
    baseline = tuner.PRODUCTION_FAST_QSEE
    disabled_default = replace(baseline, enable_lmr=False)
    disabled_arbitrary = replace(
        baseline,
        enable_lmr=False,
        lmr_base=0.8,
        lmr_divisor=1.8,
        lmr_min_depth=7,
        lmr_min_move_index=12,
    )

    assert tuple(baseline.canonical()) == tuner.SELECTIVE_CONFIG_FIELDS
    assert disabled_arbitrary.normalized() == disabled_default.normalized()
    assert disabled_arbitrary.canonical() == disabled_default.canonical()
    assert disabled_arbitrary.hash == disabled_default.hash
    assert tuner.behavior_signature(disabled_arbitrary) == tuner.behavior_signature(
        disabled_default)

    args = disabled_arbitrary.evaluator_args()
    assert args[args.index("--lmr-base") + 1] == "0.45"
    assert args[args.index("--lmr-divisor") + 1] == "2.9"
    assert "--disable-lmr" in args

    reenabled = replace(disabled_arbitrary, enable_lmr=True)
    assert reenabled.hash != disabled_default.hash


def test_behavior_signature_does_not_collapse_lmr_after_move_32() -> None:
    """Regression: these policies first differ at depth 4, move index 34."""
    left = replace(
        tuner.PRODUCTION_FAST_QSEE, lmr_base=0.2, lmr_divisor=2.75)
    right = replace(
        tuner.PRODUCTION_FAST_QSEE, lmr_base=0.25, lmr_divisor=2.8)

    def reduction(config: tuner.SelectiveConfig, depth: int, move: int) -> int:
        raw = int(
            config.lmr_base
            + math.log(depth) * math.log(move + 1) / config.lmr_divisor)
        return max(1, min(raw, depth - 2))

    assert reduction(left, 4, 34) == 1
    assert reduction(right, 4, 34) == 2
    assert tuner.lmr_behavior_signature(left) != tuner.lmr_behavior_signature(right)
    proposals = tuner.unique_proposals([
        tuner.Proposal(tuner.Candidate(left), "left"),
        tuner.Proposal(tuner.Candidate(right), "right"),
    ])
    assert len(proposals) == 2


def test_behavior_dedupe_collapses_only_truly_conditional_duplicates() -> None:
    baseline = tuner.PRODUCTION_FAST_QSEE
    left = replace(
        baseline, enable_late_move_pruning=False,
        late_move_pruning_base=0,
        late_move_pruning_depth_multiplier=0)
    right = replace(
        baseline, enable_late_move_pruning=False,
        late_move_pruning_base=16,
        late_move_pruning_depth_multiplier=8)
    proposals = tuner.unique_proposals([
        tuner.Proposal(tuner.Candidate(left), "left"),
        tuner.Proposal(tuner.Candidate(right), "right"),
    ])
    assert len(proposals) == 1
    assert proposals[0].candidate.selective == left.normalized()


def test_deterministic_maximin_is_repeatable_unique_and_keeps_anchors() -> None:
    first = tuner.global_proposals(12, 0xC0FFEE)
    second = tuner.global_proposals(12, 0xC0FFEE)
    assert [item.candidate.hash for item in first] == [
        item.candidate.hash for item in second]
    assert len(first) == 12
    assert len({item.candidate.hash for item in first}) == 12
    signatures = [
        tuner.candidate_behavior_signature(item.candidate) for item in first]
    assert len(set(signatures)) == len(signatures)
    hashes = {item.candidate.selective.hash for item in first}
    assert tuner.PRODUCTION_FAST_QSEE.hash in hashes
    assert tuner.all_prunes_disabled().hash in hashes
    assert all(
        item.candidate.aspiration == tuner.PRODUCTION_BASELINE_ASPIRATION
        for item in first)


def test_successive_halving_never_caps_an_exact_pareto_frontier() -> None:
    candidates = [
        tuner.Candidate(replace(
            tuner.PRODUCTION_FAST_QSEE,
            qsearch_see_threshold=threshold,
        ))
        for threshold in (-200, -150, -100, -50, 0)
    ]
    entries = [
        result_record(candidate, 0.50 + index * 0.10, 0.005 - index * 0.001)
        for index, candidate in enumerate(candidates)
    ]
    assert tuner.frontier(entries) == entries
    assert tuner.successive_halving_survivors(entries, 2) == entries


@pytest.mark.parametrize("total", [0, 1, 2, 6, 7, 31, 32, 33, 100])
def test_refinement_restart_allocation_has_a_real_fifteen_percent_floor(
    total: int,
) -> None:
    counts = tuner.refinement_method_counts(total)
    assert sum(counts.values()) == total
    assert all(value >= 0 for value in counts.values())
    assert counts["random-restart"] >= math.ceil(total * 0.15)


def test_refinement_proposals_realize_the_declared_method_allocation() -> None:
    parents = [
        result_record(tuner.Candidate(tuner.PRODUCTION_FAST_QSEE), 0.75, 0.003),
        result_record(tuner.Candidate(replace(
            tuner.PRODUCTION_FAST_QSEE,
            qsearch_see_threshold=-50,
        )), 0.85, 0.002),
    ]
    proposals = tuner.refinement_proposals(parents, 32, 1234)
    expected = tuner.refinement_method_counts(32)
    actual = {
        method: sum(item.origin == f"refine:{method}" for item in proposals)
        for method in expected
    }
    assert actual == expected
    assert len({
        tuner.behavior_signature(item.candidate.selective)
        for item in proposals
    }) == 32
    assert all(item.candidate.aspiration == tuner.PRODUCTION_BASELINE_ASPIRATION
               for item in proposals)


def test_local_closure_enumerates_one_step_children_and_enable_toggles() -> None:
    parent = tuner.Candidate(tuner.PRODUCTION_FAST_QSEE)
    entry = result_record(parent, 0.8, 0.002)
    proposals = tuner.local_closure_proposals(
        (("balanced", entry),),
        {tuner.behavior_signature(parent.selective)},
    )
    assert proposals
    assert len({
        tuner.behavior_signature(item.candidate.selective)
        for item in proposals
    }) == len(proposals)
    assert {
        f"closure:balanced:{field}:toggle"
        for field in tuner.ENABLE_FIELD.values()
    } <= {item.origin for item in proposals}
    assert all(item.candidate.aspiration == tuner.PRODUCTION_BASELINE_ASPIRATION
               for item in proposals)

    parent_values = asdict(parent.selective)
    for proposal in proposals:
        suffix = proposal.origin.rsplit(":", 1)[-1]
        child = proposal.candidate.selective
        if suffix == "toggle":
            enabled_field = proposal.origin.split(":")[-2]
            assert getattr(child, enabled_field) is not parent_values[enabled_field]
            continue
        grid = tuner.GRIDS[suffix]
        assert abs(
            grid.index(getattr(child, suffix))
            - grid.index(parent_values[suffix])
        ) == 1


def test_every_prune_stage_fails_closed_on_nonbalanced_aspiration() -> None:
    balanced = tuner.Proposal(tuner.Candidate(), "test:balanced")
    legacy = tuner.Proposal(
        tuner.Candidate(
            tuner.PRODUCTION_FAST_QSEE, tuner.LEGACY_FIXED50),
        "test:legacy",
    )
    stages = {
        *tuner.PRUNE_TUNING_STAGES,
        *(f"screen_{block}" for block in tuner.BLOCKS),
    }
    for stage in stages:
        tuner.validate_stage_aspiration_policy(stage, [balanced])
        with pytest.raises(RuntimeError, match="frozen V43 Balanced"):
            tuner.validate_stage_aspiration_policy(stage, [legacy])

    # These are deliberately post-selection and may vary aspiration.
    tuner.validate_stage_aspiration_policy("aspiration_refresh", [legacy])
    tuner.validate_stage_aspiration_policy("holdout", [legacy])


def test_full_accepted_depth_telemetry_invariants() -> None:
    valid = aspiration_telemetry()
    tuner.validate_aspiration_telemetry(
        valid, tuner.PRODUCTION_BASELINE_ASPIRATION, count=2, depth=6)

    too_many_reduced = {
        **valid,
        "aspiration_accepted_reduced_depth_iterations": 3,
    }
    with pytest.raises(RuntimeError, match="exceeds narrow iterations"):
        tuner.validate_aspiration_telemetry(
            too_many_reduced,
            tuner.PRODUCTION_BASELINE_ASPIRATION,
            count=2,
            depth=6,
        )

    deeper_than_nominal = {
        **valid,
        "aspiration_accepted_narrow_search_depth_sum": 11,
        "aspiration_accepted_depth_ratio": 1.1,
    }
    with pytest.raises(RuntimeError, match="accepted-depth ratio"):
        tuner.validate_aspiration_telemetry(
            deeper_than_nominal,
            tuner.PRODUCTION_BASELINE_ASPIRATION,
            count=2,
            depth=6,
        )

    wrong_ratio = {**valid, "aspiration_accepted_depth_ratio": 0.8}
    with pytest.raises(RuntimeError, match="disagrees with depth sums"):
        tuner.validate_aspiration_telemetry(
            wrong_ratio,
            tuner.PRODUCTION_BASELINE_ASPIRATION,
            count=2,
            depth=6,
        )

    above_balanced_cap = {
        **valid,
        "aspiration_max_accepted_depth_reduction": 2,
    }
    with pytest.raises(RuntimeError, match="configured cap"):
        tuner.validate_aspiration_telemetry(
            above_balanced_cap,
            tuner.PRODUCTION_BASELINE_ASPIRATION,
            count=2,
            depth=6,
        )


def test_selection_depth_guard_is_diagnostic_and_uses_same_rung_baseline() -> None:
    baseline = result_record(
        tuner.Candidate(), 1.0, 0.003, stage="selection")
    candidate = result_record(
        tuner.Candidate(replace(
            tuner.PRODUCTION_FAST_QSEE, qsearch_see_threshold=-50)),
        0.9,
        0.003,
        stage="selection",
    )
    baseline["result"].update({
        "aspiration_accepted_depth_ratio": 0.96,
        "aspiration_max_accepted_depth_reduction": 1,
    })
    candidate["result"].update({
        "aspiration_accepted_depth_ratio": 0.956,
        "aspiration_max_accepted_depth_reduction": 1,
    })
    lineage = tuner.pre_refresh_selection_lineage(candidate)
    assert lineage == {
        "pre_refresh_selection_config_hash": candidate["config_hash"],
        "pre_refresh_selective_hash": tuner.config_from_record(
            candidate).selective.hash,
    }
    passed = tuner.accepted_depth_safeguard(candidate, baseline)
    assert passed["same_rung_production_baseline_hash"] == baseline["config_hash"]
    assert passed["accepted_depth_ratio_delta_vs_production_baseline"] \
        == pytest.approx(-0.004)
    assert passed["pass"] is True

    candidate["result"]["aspiration_accepted_depth_ratio"] = 0.954
    failed = tuner.accepted_depth_safeguard(candidate, baseline)
    assert failed["accepted_depth_ratio_noninferior"] is False
    assert failed["pass"] is False

    candidate["stage"] = "holdout"
    with pytest.raises(RuntimeError, match="must come from selection"):
        tuner.pre_refresh_selection_lineage(candidate)
    with pytest.raises(RuntimeError, match="not same-rung"):
        tuner.accepted_depth_safeguard(candidate, baseline)


def test_selfplay_args_serialize_full_selective_and_balanced_control() -> None:
    selective = replace(
        tuner.PRODUCTION_FAST_QSEE,
        enable_lmr=False,
        enable_null_move=False,
        enable_main_search_see_pruning=True,
        main_search_see_max_depth=6,
        main_search_see_margin_per_depth=50,
    )
    candidate = result_record(
        tuner.Candidate(selective, tuner.ASPIRATION_ANCHORS[-1]),
        0.8,
        0.002,
        stage="holdout",
    )
    baseline = result_record(
        tuner.Candidate(), 1.0, 0.002, stage="holdout")
    payload = tuner.selfplay_confirmation_args(
        "fast", candidate, baseline, eligible=True)
    arguments = payload["arguments"]
    selective_specs = [
        arguments[index + 1]
        for index, value in enumerate(arguments)
        if value == "--profile"
    ]
    aspiration_specs = [
        arguments[index + 1]
        for index, value in enumerate(arguments)
        if value == "--aspiration-profile"
    ]

    assert len(selective_specs) == 2
    assert all(len(spec.split(",")) == 22 for spec in selective_specs)
    candidate_fields = selective_specs[0].split(",")
    assert candidate_fields[17:20] == ["1", "6", "50"]
    assert candidate_fields[20:] == ["0", "0"]
    assert len(aspiration_specs) == 2
    assert aspiration_specs[1].split(",")[1:] == [
        "1", "2", "68", "33700", "2290", "1", "370", "6", "1500",
    ]
    assert payload["control_name"] == (
        "v43_balanced_aspiration_baseline_control")
    assert payload["eligible_for_equal_time_selfplay"] is True
    assert arguments[arguments.index("--tt-mb") + 1] == "64"


def test_external_control_cache_requires_exact_teacher_semantic_digest(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "evaluator"
    binary.write_bytes(b"evaluator")
    model = tmp_path / "model"
    model.write_bytes(b"model")
    dataset = tmp_path / "dataset.tsv"
    dataset.write_text(
        "phase0\t00000000000000000000000000000001\t0\t"
        "8/8/8/8/8/8/8/K6k w - - 0 1\n")
    spec = tuner.control_cache_spec(
        tmp_path / "run", "selection", binary, model, dataset, 7)
    identity = spec["identity"]
    assert identity["experiment"] == "v43-final-joint-all-prunes-v1"
    assert identity["teacher_profile"] == tuner.infra.CONTROL_TEACHER_PROFILE
    external = tmp_path / "control-root-selection-d7.tsv"
    external.write_text("\n".join((
        tuner.infra.CONTROL_CACHE_SCHEMA,
        f"identity_sha256\t{spec['identity_sha256']}",
        f"evaluator_sha256\t{identity['evaluator']['sha256']}",
        f"model_sha256\t{identity['model']['sha256']}",
        f"dataset_sha256\t{identity['ordered_dataset']['sha256']}",
        "depth\t7",
        "offset\t0",
        "count\t1",
        "\t".join(tuner.infra.CONTROL_CACHE_COLUMNS),
        "placeholder-row",
    )) + "\n")
    assert tuner._external_cache_candidate(tmp_path, spec) == external

    drifted = json.loads(json.dumps(spec))
    drifted["identity"]["teacher_profile"]["root_semantics"] = "drifted"
    drifted["identity_sha256"] = hashlib.sha256(
        tuner.canonical_json(drifted["identity"])).hexdigest()
    assert tuner._external_cache_candidate(tmp_path, drifted) is None


def test_detail_sidecar_requires_zero_based_index_and_control_root_nodes(
    tmp_path: Path,
) -> None:
    dataset, path, descriptor = write_dataset_and_details(
        tmp_path, "valid", [(0.001, 90), (0.002, 95)])
    assert descriptor["count"] == 2
    assert descriptor["artifact"]["path"] == str(path.resolve())

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["control_nodes"] = rows[0].pop("control_root_nodes")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="detail row fields differ"):
        tuner.validate_detail_sidecar(path, dataset, 2)

    rows[0]["control_root_nodes"] = rows[0].pop("control_nodes")
    rows[0]["index"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="detail index mismatch"):
        tuner.validate_detail_sidecar(path, dataset, 2)


def test_confidence_frontier_uses_paired_detail_sidecars(tmp_path: Path) -> None:
    _, _, better_detail = write_dataset_and_details(
        tmp_path, "better", [(0.001, 80)] * 8)
    _, _, worse_detail = write_dataset_and_details(
        tmp_path, "worse", [(0.003, 100)] * 8)
    better = result_record(
        tuner.Candidate(tuner.PRODUCTION_FAST_QSEE),
        0.8,
        0.001,
        detail=better_detail,
    )
    worse = result_record(
        tuner.Candidate(replace(
            tuner.PRODUCTION_FAST_QSEE, qsearch_see_threshold=-50)),
        1.0,
        0.003,
        detail=worse_detail,
    )
    assert tuner.confidence_frontier(
        [better, worse], seed=99, resamples=100) == [better]


def test_selection_and_refresh_do_not_reselect_from_holdout() -> None:
    production = result_record(
        tuner.Candidate(tuner.PRODUCTION_FAST_QSEE), 1.0, 0.006)
    candidates = [
        tuner.Candidate(replace(
            tuner.PRODUCTION_FAST_QSEE,
            qsearch_see_threshold=threshold,
        ))
        for threshold in (-200, -100, 0)
    ]
    selection = [production, *(
        result_record(candidate, nodes, wdl)
        for candidate, nodes, wdl in zip(
            candidates,
            (0.55, 0.72, 0.90),
            (0.005, 0.0025, 0.001),
        )
    )]
    profiles = tuner.named_profiles(selection)
    assert {entry["config_hash"] for _, entry in profiles} <= {
        entry["config_hash"] for entry in selection}

    refresh: list[dict] = []
    for _, entry in profiles:
        selective = tuner.config_from_record(entry).selective
        refresh.extend((
            result_record(
                tuner.Candidate(selective, tuner.LEGACY_FIXED50),
                0.90,
                0.003,
                stage="aspiration_refresh",
            ),
            result_record(
                tuner.Candidate(selective, tuner.ASPIRATION_ANCHORS[1]),
                0.80,
                0.002,
                stage="aspiration_refresh",
            ),
        ))
    chosen = tuner.choose_refreshed_profiles(profiles, refresh)
    assert all(tuner.config_from_record(entry).aspiration.enabled
               for _, entry in chosen)
    assert {entry["config_hash"] for _, entry in chosen} <= {
        entry["config_hash"] for entry in refresh}

    # A deliberately inverted holdout is diagnostic only.  The selection API
    # has no holdout input, so it cannot silently promote a different profile.
    holdout = [
        result_record(
            tuner.config_from_record(entry),
            nodes=2.0 - index,
            wdl=0.1 - index * 0.01,
            stage="holdout",
        )
        for index, (_, entry) in enumerate(chosen)
    ]
    assert holdout  # make the adversarial diagnostic explicit
    assert tuner.choose_refreshed_profiles(profiles, refresh) == chosen


def test_saved_stage_plan_is_immutable_across_resume(tmp_path: Path) -> None:
    dataset = tmp_path / "rung.tsv"
    dataset.write_text("phase0\tsample\t0\tfen\n")
    cache = {
        "identity_sha256": "cache-identity",
        "artifact": {"path": str(tmp_path / "cache"), "sha256": "cache"},
    }
    records: list[dict] = []
    log = tmp_path / "results.jsonl"
    proposal = tuner.Proposal(tuner.Candidate(), "test")
    budget = tuner.Budget(100.0, 0.0, time.monotonic())
    first = tuner.ensure_plan(
        records, log, "screen", [proposal], dataset, 6, cache, budget, False)
    assert tuner.ensure_plan(
        records, log, "screen", [proposal], dataset, 6, cache, budget, False
    ) == first

    dataset.write_text("phase0\tsample\t0\tchanged-fen\n")
    with pytest.raises(RuntimeError, match="saved plan mismatch"):
        tuner.ensure_plan(
            records, log, "screen", [proposal], dataset, 6, cache, budget, False)


def test_immutable_run_manifest_rejects_experiment_drift(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    body = {
        "schema_version": tuner.SCHEMA_VERSION,
        "experiment": tuner.EXPERIMENT,
        "seed": 7,
        "candidate_counts": {"global": 64},
        "budget_policy": tuner.infra.BUDGET_POLICY,
    }
    expected = {
        **body,
        "manifest_sha256": hashlib.sha256(
            tuner.canonical_json(body)).hexdigest(),
    }
    tuner.infra.ensure_manifest(path, expected)
    tuner.infra.ensure_manifest(path, expected)

    changed_body = {
        **body,
        "candidate_counts": {"global": 65},
    }
    changed = {
        **changed_body,
        "manifest_sha256": hashlib.sha256(
            tuner.canonical_json(changed_body)).hexdigest(),
    }
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        tuner.infra.ensure_manifest(path, changed)


def test_resume_rejects_conflicting_duplicate_results() -> None:
    record = result_record(tuner.Candidate(), 0.8, 0.002, stage="screen")
    conflict = {
        **record,
        "result": {"node_ratio": 0.81, "mean_wdl_loss": 0.002},
    }
    with pytest.raises(RuntimeError, match="conflicting duplicate result"):
        tuner.completed_results([record, conflict], "screen")


def test_cumulative_budget_is_fail_closed_and_can_only_extend(
    tmp_path: Path,
) -> None:
    log = tmp_path / "results.jsonl"
    records: list[dict] = []
    tuner.ensure_budget_limit(records, log, 100)
    tuner.ensure_budget_limit(records, log, 100)
    assert len(records) == 1

    elapsed = {"kind": "result", "budget_elapsed_sec": 101.0}
    tuner.append_jsonl(log, elapsed)
    records.append(elapsed)
    assert tuner.Budget(
        100.0, tuner.max_budget_elapsed(records), time.monotonic()).exhausted

    with pytest.raises(RuntimeError, match="may not decrease"):
        tuner.ensure_budget_limit(records, log, 99)
    tuner.ensure_budget_limit(records, log, 200)
    assert records[-1]["cumulative_limit_sec"] == 200
