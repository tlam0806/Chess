import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v43_v44_match",
    ROOT / "tools" / "run_nnue_v43_v44_selfplay.py",
)
match = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(match)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    binary = tmp_path / "match"
    model = tmp_path / "model.bin"
    book = tmp_path / "book.txt"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    model.write_bytes(b"model fixture")
    book.write_bytes(b"e2e4 e7e5\nd2d4 d7d5\n")
    monkeypatch.setattr(match, "EXPECTED_MODEL_SHA256", sha256(model.read_bytes()))
    monkeypatch.setattr(match, "EXPECTED_BOOK_SHA256", sha256(book.read_bytes()))
    args = match.parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--binary", str(binary),
        "--model", str(model),
        "--book", str(book),
        "--openings", "2",
        "--base-ms", "50",
        "--increment-ms", "1",
        "--max-plies", "240",
        "--init-only",
    ])
    return args


def game(index: int, color: str, outcome: str, nodes: tuple[int, int]):
    fingerprint = "abc123"
    move = "e2e4" if color == "white" else "e7e5"
    return {
        "kind": "game",
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "key": f"{fingerprint}:opening:{index}:{color[0]}",
        "logical_key": f"opening:{index}:{color[0]}",
        "candidate": match.CANDIDATE_NAME,
        "control": match.CONTROL_NAME,
        "combined_config_hash": match.COMBINED_CONFIG_HASH,
        "candidate_color": color,
        "candidate_outcome": outcome,
        "reason": "rule_draw",
        "candidate_nodes": nodes[0],
        "control_nodes": nodes[1],
        "candidate_time_ms": 10,
        "control_time_ms": 10,
        "played_moves": [move],
        "searches": [{
            "engine": match.CANDIDATE_NAME,
            "move": move,
            "depth": 8,
            "score": 12,
            "nodes": nodes[0],
            "time_ms": 10,
            "applied": True,
        }],
    }


def test_official_protocol_and_exact_production_profile_are_frozen():
    assert match.DEFAULT_PROTOCOL == {
        "openings": 300,
        "expected_games": 600,
        "base_ms": 10_000,
        "increment_ms": 100,
        "overhead_ms": 20,
        "max_plies": 240,
        "tt_mb": 64,
        "seed": 2_026_090_155,
    }
    assert match.SELECTIVE_CONFIG["null_move_reduction"] == 4
    assert match.SELECTIVE_CONFIG["qsearch_see_threshold"] == -25
    assert match.ASPIRATION_CONFIG["delta_base_cp"] == 68
    assert match.ASPIRATION_CONFIG["max_fail_high_reductions"] == 1
    assert match.COMBINED_CONFIG_HASH.startswith("98b7732c9587")


def test_contract_snapshots_every_input_and_locks_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = make_args(tmp_path, monkeypatch)
    contract = match.initialize(args)
    match.verify_contract(contract)
    assert contract["protocol"]["expected_games"] == 4
    assert contract["protocol"]["max_plies"] == 240
    assert contract["engines"]["candidate"]["generation_fields"] is False
    assert contract["engines"]["candidate"]["clear_tt_each_top_level_search"]
    assert not contract["engines"]["control"]["clear_tt_each_top_level_search"]
    assert contract["common_policy"]["per_ply_search_telemetry"]
    assert contract["common_policy"]["no_early_stop"]
    assert contract["execution"]["detached_worker"]
    assert contract["execution"]["sleep_preventer"] in {
        None, "/usr/bin/caffeinate",
    }
    assert "--stop-after-games" not in contract["command"]
    for value in contract["frozen_files"].values():
        assert Path(value["path"]).is_file()
        assert match.fingerprint(Path(value["path"])) == {
            "size": value["size"], "sha256": value["sha256"]}
    assert (args.run_dir / "contract.sha256").read_text().strip() == (
        match.fingerprint(args.run_dir / "contract.json")["sha256"]
    )


def test_changed_frozen_artifact_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = make_args(tmp_path, monkeypatch)
    contract = match.initialize(args)
    Path(contract["frozen_files"]["book"]["path"]).write_text("changed\n")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        match.verify_contract(contract)


def test_jsonl_tail_repair_preserves_only_complete_games(tmp_path: Path):
    path = tmp_path / "games.jsonl"
    complete = json.dumps(game(0, "white", "draw", (10, 20))) + "\n"
    path.write_bytes((complete + "{\"kind\":\"game\"").encode())
    assert match.repair_partial_jsonl_tail(path)
    assert path.read_text() == complete
    games = match.read_games(path)
    assert len(games) == 1
    assert not match.repair_partial_jsonl_tail(path)


def test_summary_is_color_paired_and_reports_time_and_nodes():
    games = [
        game(0, "white", "win", (10, 20)),
        game(0, "black", "draw", (30, 40)),
        game(1, "black", "loss", (20, 20)),
    ]
    summary = match.summarize_games(games, 4)
    assert summary["completed_games"] == 3
    assert summary["complete_pairs"] == 1
    assert summary["partial_pairs"] == 1
    assert summary["paired_score"] == 0.75
    assert summary["candidate_score"] == 0.5
    assert summary["node_ratio"] == 0.75
    assert summary["time_ratio"] == 1.0
    assert summary["search_records"] == 3


def test_non_mirrored_complete_pair_is_rejected():
    first = game(0, "white", "win", (10, 20))
    second = game(0, "white", "draw", (10, 20))
    second["key"] += ":duplicate-color"
    with pytest.raises(RuntimeError, match="not color-reversed"):
        match.summarize_games([first, second], 2)


def test_worker_identity_uses_the_private_lifetime_lock(tmp_path: Path):
    assert not match.runner_lock_held(tmp_path)
    lock_path = tmp_path / ".runner.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert match.runner_lock_held(tmp_path)
        assert match.worker_process_matches(tmp_path, os.getpid())
    assert not match.runner_lock_held(tmp_path)


def test_cpp_harness_times_the_top_level_v44_call_and_records_every_search():
    source = (ROOT / "tools" / "nnue_v43_v44_time_match.cpp").read_text()
    play_game = source[source.index("GameResult play_game("):]
    timer = play_game.index("const auto start = std::chrono::steady_clock::now()")
    candidate_call = play_game.index("search_with_history(candidate")
    elapsed = play_game.index("const auto elapsed")
    assert timer < candidate_call < elapsed
    assert "append_searches(output, game.searches)" in source
    assert "int max_plies = 240" in source
    assert "generation_fields\\\":false" in source
