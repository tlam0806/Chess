from __future__ import annotations

import json
from dataclasses import replace

from tools import run_nnue_v46_timed_pipeline as pipeline


def pool_entry(candidate, loss: float, node_ratio: float) -> dict:
    evaluation = {
        "config_hash": candidate.hash,
        "config": candidate.canonical(),
        "result": {
            "mean_wdl_loss": loss,
            "candidate_nodes": int(node_ratio * 1_000_000),
            "node_ratio": node_ratio,
        },
    }
    return {
        "config_hash": candidate.hash,
        "config": candidate.canonical(),
        "mean_wdl_loss": loss,
        "production_node_ratio": node_ratio,
        "critical_mistakes": 0,
        "evaluation": evaluation,
    }


def test_schedule_has_expected_timed_budget() -> None:
    settings, confirmation = pipeline.schedule(False)
    assert [item["games_per_candidate"] for item in settings] == [16, 64, 128]
    assert [item["advance"] for item in settings] == [8, 3, 1]
    total = 24 * 16 + 8 * 64 + 3 * 128 + confirmation["games_per_candidate"]
    assert total == 1_880


def test_diverse_pool_is_deterministic_and_keeps_both_objective_extremes() -> None:
    baseline = pipeline.teacher.deployed_v45_candidate()
    candidates = [
        pipeline.teacher.legacy.Candidate(
            replace(baseline.selective, qsearch_see_threshold=value),
            baseline.aspiration,
        )
        for value in (-50, -25, 0, 25, 50)
    ]
    entries = [
        pool_entry(candidate, 0.004 + index * 0.001, 1.1 - index * 0.05)
        for index, candidate in enumerate(candidates)
    ]

    first = pipeline.choose_diverse_pool(entries, 3)
    second = pipeline.choose_diverse_pool(entries, 3)

    assert first == second
    hashes = {entry["config_hash"] for entry in first}
    assert candidates[0].hash in hashes
    assert candidates[-1].hash in hashes

    one = pipeline.choose_diverse_pool(entries, 1)
    assert len(one) == 1


def test_game_counter_does_not_repair_live_partial_tail(tmp_path) -> None:
    path = tmp_path / "timed" / "screen" / "matches" / "x.jsonl"
    path.parent.mkdir(parents=True)
    original = (
        json.dumps({"kind": "game", "key": "one"}).encode() +
        b"\n{\"kind\":\"game\""
    )
    path.write_bytes(original)

    assert pipeline.count_games(tmp_path) == 1
    assert path.read_bytes() == original
