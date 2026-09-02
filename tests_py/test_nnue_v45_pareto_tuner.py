from __future__ import annotations

from dataclasses import replace

from tools import tune_nnue_v45_pareto as tuner


def entry(candidate, loss: float, nodes: float) -> dict:
    return {
        "config": candidate.canonical(),
        "config_hash": candidate.hash,
        "result": {"mean_wdl_loss": loss, "node_ratio": nodes},
    }


def test_production_v43_identity_is_exact() -> None:
    assert tuner.production_v43_candidate().hash == tuner.PRODUCTION_V43_HASH


def test_deployed_v45_identity_is_exact() -> None:
    candidate = tuner.deployed_v45_candidate()
    assert candidate.hash == tuner.DEPLOYED_V45_HASH
    assert tuner.baseline_candidate("v45-deployed") == candidate


def test_pareto_insert_rejects_child_dominated_by_current_candidate() -> None:
    baseline = tuner.production_v43_candidate()
    child = tuner.legacy.Candidate(replace(
        baseline.selective, qsearch_see_threshold=-50
    ), baseline.aspiration)
    current = entry(baseline, 0.004, 0.8)
    proposed = entry(child, 0.005, 0.9)

    frontier, status, removed = tuner.pareto_insert([current], proposed)

    assert status == "dominated"
    assert removed == []
    assert frontier == [current]


def test_pareto_insert_removes_every_candidate_dominated_by_child() -> None:
    baseline = tuner.production_v43_candidate()
    first = tuner.legacy.Candidate(replace(
        baseline.selective, qsearch_see_threshold=-50
    ), baseline.aspiration)
    second = tuner.legacy.Candidate(replace(
        baseline.selective, qsearch_see_threshold=0
    ), baseline.aspiration)
    child = tuner.legacy.Candidate(replace(
        baseline.selective, qsearch_see_threshold=-100
    ), baseline.aspiration)
    current = [entry(first, 0.006, 0.8), entry(second, 0.005, 0.9)]
    proposed = entry(child, 0.004, 0.7)

    frontier, status, removed = tuner.pareto_insert(current, proposed)

    assert status == "inserted"
    assert removed == sorted([first.hash, second.hash])
    assert [value["config_hash"] for value in frontier] == [child.hash]


def test_pareto_insert_keeps_tradeoff_candidates() -> None:
    baseline = tuner.production_v43_candidate()
    child = tuner.legacy.Candidate(replace(
        baseline.selective, qsearch_see_threshold=-50
    ), baseline.aspiration)
    current = entry(baseline, 0.004, 0.9)
    proposed = entry(child, 0.005, 0.7)

    frontier, status, removed = tuner.pareto_insert([current], proposed)

    assert status == "inserted"
    assert removed == []
    assert {value["config_hash"] for value in frontier} == {
        baseline.hash, child.hash
    }


def test_generated_batch_is_deterministic_unique_and_frontier_derived() -> None:
    baseline = tuner.production_v43_candidate()
    seen = {tuner.legacy.candidate_behavior_signature(baseline)}

    first = tuner.generate_batch([baseline], seen, 6, 123)
    second = tuner.generate_batch([baseline], seen, 6, 123)

    assert first == second
    assert len({item["config_hash"] for item in first}) == 6
    assert {item["parent_hash"] for item in first} == {baseline.hash}
    assert all(
        tuner.candidate_from_payload(item).aspiration == baseline.aspiration
        for item in first
    )


def test_memory_worker_cap_degrades_without_stopping(monkeypatch) -> None:
    monkeypatch.setattr(tuner, "memory_free_percent", lambda: 9)
    assert tuner.memory_bounded_workers(4) == (1, 9)
    monkeypatch.setattr(tuner, "memory_free_percent", lambda: 15)
    assert tuner.memory_bounded_workers(4) == (2, 15)
    monkeypatch.setattr(tuner, "memory_free_percent", lambda: 30)
    assert tuner.memory_bounded_workers(3) == (3, 30)
