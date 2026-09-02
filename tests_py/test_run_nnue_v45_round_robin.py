from __future__ import annotations

import json

from tools import run_nnue_v45_round_robin as runner


def profile(name: str, candidate_hash: str, loss: float, nodes: float) -> dict:
    return {
        "name": name,
        "config_hash": candidate_hash,
        "source_selection_metrics": {
            "mean_wdl_loss": loss, "node_ratio": nodes,
        },
    }


def game(first: str, second: str, outcome: str) -> dict:
    return {
        "profile": first, "opponent": second,
        "candidate_outcome": outcome, "candidate_nodes": 100,
        "control_nodes": 100, "candidate_time_ms": 10,
        "control_time_ms": 10, "reason": "checkmate",
    }


def test_frozen_schedule_has_expected_halving_and_game_budget() -> None:
    assert [value["entrants"] for value in runner.ROUNDS] == [17, 9, 5, 3, 2]
    assert [value["survivors"] for value in runner.ROUNDS] == [9, 5, 3, 2, 1]
    assert [value["games_per_pair"] for value in runner.ROUNDS] == [8, 16, 32, 64, 128]
    assert runner.EXPECTED_TOURNAMENT_GAMES == 2_304


def test_round_ranking_uses_points_before_offline_tiebreaks() -> None:
    profiles = [
        profile("a", "a" * 64, 0.001, 0.7),
        profile("b", "b" * 64, 0.010, 1.2),
        profile("c", "c" * 64, 0.002, 0.8),
    ]
    games = [
        game("a", "b", "loss"), game("a", "b", "loss"),
        game("a", "c", "win"), game("a", "c", "draw"),
        game("b", "c", "win"), game("b", "c", "win"),
    ]
    ranked = runner.rank_round(games, profiles, runner.ROUNDS[0])
    assert [row["name"] for row in ranked] == ["b", "a", "c"]


def test_exact_point_tie_uses_head_to_head_then_selection_metrics() -> None:
    profiles = [
        profile("a", "a" * 64, 0.004, 0.9),
        profile("b", "b" * 64, 0.003, 0.8),
    ]
    games = [game("a", "b", "draw"), game("a", "b", "draw")]
    ranked = runner.rank_round(games, profiles, runner.ROUNDS[0])
    assert [row["name"] for row in ranked] == ["b", "a"]


def test_completed_round_reloads_saved_resource_on_resume(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "r1" / "games.jsonl"
    output.parent.mkdir()
    output.write_text("{}\n")
    resource = {"wall_sec": 10.0, "child_cpu_sec": 9.5}
    (output.parent / "resource.json").write_text(json.dumps(resource))
    monkeypatch.setattr(runner, "expected_round_games", lambda setting, count: 1)
    monkeypatch.setattr(runner, "read_complete_games", lambda path: [{}])
    monkeypatch.setattr(
        runner, "validate_round_games",
        lambda path, profiles, setting: ([{}], {"run_fingerprint": "test"}),
    )

    resumed = runner.run_round_process({}, tmp_path, {}, [{}], output)

    assert resumed["resource"] == resource
