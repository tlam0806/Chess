import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "race", ROOT / "tools" / "match" / "run_nnue_v43_selfplay_race.py")
race = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(race)


ASPIRATION = {
    "enabled": True, "min_depth": 2, "delta_base_cp": 68,
    "delta_divisor": 33700, "expansion_factor_per_mille": 2290,
    "max_fail_high_reductions": 1,
    "mean_score_new_weight_per_mille": 370,
    "max_researches": 6, "mean_score_clamp_cp": 1500,
}
SELECTIVE = {
    "enable_lmr": True, "lmr_base": 0.45, "lmr_divisor": 2.9,
    "lmr_min_depth": 3, "lmr_min_move_index": 6,
    "enable_null_move": True, "null_move_min_depth": 2,
    "null_move_reduction": 3, "enable_reverse_futility": True,
    "reverse_futility_max_depth": 2, "reverse_futility_base_margin": 175,
    "reverse_futility_margin_per_depth": 275,
    "enable_late_move_pruning": True, "late_move_pruning_max_depth": 3,
    "late_move_pruning_base": 4, "late_move_pruning_depth_multiplier": 2,
    "enable_qsearch_see_pruning": True, "qsearch_see_threshold": -75,
    "enable_main_search_see_pruning": False,
    "main_search_see_max_depth": 5,
    "main_search_see_margin_per_depth": 100,
}


def profile(name: str, hash_char: str, loss: float = 0.01, nodes: float = 0.1):
    return {
        "name": name, "config_hash": hash_char * 64,
        "config": {"selective": SELECTIVE, "aspiration": ASPIRATION},
        "source_selection_metrics": {
            "mean_wdl_loss": loss, "node_ratio": nodes,
        },
    }


def game(first, second, index, color, outcome):
    return {
        "kind": "game", "key": f"fp:{first['name']}_vs_{second['name']}:{index}:{color}",
        "logical_key": f"{first['name']}_vs_{second['name']}:{index}:{color}",
        "profile": first["name"], "opponent": second["name"],
        "candidate_outcome": outcome, "reason": "rule_draw",
        "candidate_nodes": 10, "control_nodes": 20,
        "candidate_time_ms": 5, "control_time_ms": 5,
        "candidate_twofold_search_draw": True,
        "candidate_reuse_stale_tt_scores": False,
        "candidate_reuse_deeper_tt_scores": False,
        "opponent_reuse_stale_tt_scores": False,
        "opponent_reuse_deeper_tt_scores": False,
        "candidate_aspiration_config": ASPIRATION,
        "opponent_aspiration_config": ASPIRATION,
    }


def test_frozen_pool_and_protocol_sizes():
    pool = json.loads((ROOT / "tools" / "match" / "nnue_v43_selfplay_race_pool.json").read_text())
    assert len(pool["selection_candidate_hashes"]) == 22
    assert len(set(pool["selection_candidate_hashes"])) == 22
    assert pool["balanced_aspiration"] == ASPIRATION
    assert race.EXPECTED_TOTAL_GAMES == 2404
    assert [race.ROUNDS[key]["openings"] for key in ("r1", "r2", "r3", "final")] == [16, 48, 50, 300]


def test_profile_and_command_pin_every_policy(tmp_path):
    first, second = profile("candidate_a", "a"), profile("production_b", "b")
    binary, model, book = (tmp_path / name for name in ("binary", "model", "book"))
    contract = {
        "frozen_files": {"binary": {"path": str(binary)}, "model": {"path": str(model)}},
        "round_books": {"r1": {"path": str(book)}}, "rounds": {"r1": race.ROUNDS["r1"]},
        "protocol": {"overhead_ms": 20, "max_plies": 200, "tt_mb": 64},
        "balanced_aspiration": ASPIRATION,
    }
    command = race.build_command(contract, "r1", [first, second], tmp_path / "games.jsonl")
    joined = " ".join(command)
    assert "--stop-on-ci" not in command
    assert joined.count("--aspiration-profile") == 2
    assert joined.count("--twofold-search-profile") == 2
    assert ",1,2,68,33700,2290,1,370,6,1500" in joined
    assert race.profile_spec(second) == (
        "production_b,0.45,2.9,3,6,2,3,1,2,175,275,1,3,4,2,1,-75,0,5,100,1,1")


def test_paired_summary_and_ranking_are_color_paired(tmp_path):
    first, second = profile("candidate_a", "a", 0.02, 0.08), profile("production_b", "b")
    path = tmp_path / "games.jsonl"
    values = [game(first, second, 0, "w", "win"), game(first, second, 0, "b", "draw"),
              game(first, second, 1, "w", "loss"), game(first, second, 1, "b", "draw")]
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    summary = race.paired_summary(path, first, second, 4,
                                  require_manifest=False)
    assert summary["complete_pairs"] == 2
    assert summary["first_score"] == 0.5
    assert summary["node_ratio"] == 0.5
    faster = profile("candidate_c", "c", 0.01, 0.04)
    other = dict(summary, first="candidate_c", first_hash="c" * 64,
                 ci95=[0.4, 0.6])
    ranked = race.rank_control([summary, other], {
        first["config_hash"]: first, faster["config_hash"]: faster})
    assert ranked[0]["first_hash"] == faster["config_hash"]


def test_split_openings_is_exact_and_disjoint(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text("# book\n" + "\n".join(f"e2e4 e7e5 g1f3 b8c6 {n}" for n in range(500)) + "\n")
    split = race.split_openings(source)
    assert {key: len(value) for key, value in split.items()} == {
        "r1": 16, "r2": 48, "r3": 50, "final": 300}
    flattened = [line for lines in split.values() for line in lines]
    assert len(flattened) == len(set(flattened)) == 414


def test_only_non_newline_tail_is_repaired(tmp_path):
    path = tmp_path / "partial.jsonl"
    complete = {"kind": "gauntlet_manifest"}
    path.write_bytes((json.dumps(complete) + "\n{\"kind\":\"game\"").encode())
    assert race.read_games(path, repair_partial_tail=True) == []
    assert path.read_text() == json.dumps(complete) + "\n"


def test_final_round_illegal_move_is_a_hard_failure():
    clean = {"first_illegal_games": 0, "second_illegal_games": 0}
    race.require_legal_match(clean, "clean")
    for field in ("first_illegal_games", "second_illegal_games"):
        summary = dict(clean)
        summary[field] = 1
        with pytest.raises(RuntimeError, match="illegal engine move"):
            race.require_legal_match(summary, "final")


def test_gauntlet_manifest_proves_both_profile_policies(tmp_path):
    first, second = profile("candidate_a", "a"), profile("production_b", "b")
    games = tmp_path / "games.jsonl"
    games.write_text("")
    emitted = []
    for value in (first, second):
        emitted.append({
            "name": value["name"], "twofold_search_draw": True,
            "reuse_stale_tt_scores": False,
            "reuse_deeper_tt_scores": False,
            "selective_config": value["config"]["selective"],
            "aspiration_config": value["config"]["aspiration"],
        })
    manifest_path = Path(str(games) + ".manifest.json")
    manifest_path.write_text(json.dumps({
        "kind": "gauntlet_manifest", "run_fingerprint": "abc",
        "identity": {"profiles": emitted},
    }))
    assert race.validate_gauntlet_manifest(games, [first, second])["run_fingerprint"] == "abc"
    emitted[1]["twofold_search_draw"] = False
    manifest_path.write_text(json.dumps({
        "kind": "gauntlet_manifest", "run_fingerprint": "abc",
        "identity": {"profiles": emitted},
    }))
    try:
        race.validate_gauntlet_manifest(games, [first, second])
    except RuntimeError as error:
        assert "policy mismatch" in str(error)
    else:
        raise AssertionError("opponent twofold mismatch was accepted")
