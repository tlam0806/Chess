import argparse
import sys
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_uci_platform import (
    BenchmarkError,
    EngineSession,
    fixed_depth_integrity,
    measurement_schedule,
    observation_count_integrity,
    parse_info_line,
    read_config_uci_options,
    scheduled_positions,
    summarize_observations,
)
from tools.run_heroku_uci_platform_benchmark import (
    cross_run_integrity,
    remote_command,
)


class UciPlatformBenchmarkTests(unittest.TestCase):
    def test_remote_command_combines_heroku_environment_flag(self) -> None:
        args = argparse.Namespace(
            app="staging-app",
            process_type="benchmark",
            size="basic",
            backend="vnni",
            accumulator_backend="avx2",
            remote_engine="/app/engine",
            remote_model="/app/model",
            remote_config="/app/config",
            remote_engine_cwd="/app",
            source_label="test",
            harness_sha256="harness",
            runner_sha256="runner",
            profile="smoke",
        )
        command = remote_command(args, "spec", 1)
        self.assertEqual(command.count("--env"), 1)
        environment = command[command.index("--env") + 1]
        self.assertEqual(
            environment,
            "CHESS_NNUE_BACKEND=vnni;"
            "CHESS_NNUE_ACCUMULATOR_BACKEND=avx2",
        )

    def test_parse_info_accepts_token_reordering(self) -> None:
        parsed = parse_info_line("info nodes 123 score cp -17 depth 8")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["reported_depth"], 8)
        self.assertEqual(parsed["score_type"], "cp")
        self.assertEqual(parsed["score_value"], -17)
        self.assertEqual(parsed["nodes"], 123)

    def test_schedule_rotates_then_reverses(self) -> None:
        positions = [{"id": value, "fen": value} for value in "abcd"]
        self.assertEqual(
            [value["id"] for value in scheduled_positions(positions, 0)],
            list("abcd"),
        )
        self.assertEqual(
            [value["id"] for value in scheduled_positions(positions, 1)],
            list("dcba"),
        )
        self.assertEqual(
            [value["id"] for value in scheduled_positions(positions, 2)],
            list("bcda"),
        )
        self.assertEqual(
            [value["id"] for value in scheduled_positions(positions, 3)],
            list("adcb"),
        )

    def test_summary_uses_ratio_of_totals(self) -> None:
        observations = [
            {
                "mode": "fixed_depth",
                "limit": 7,
                "round": 0,
                "position_id": "p1",
                "nodes": 100,
                "elapsed_ns": 1_000_000_000,
            },
            {
                "mode": "fixed_depth",
                "limit": 7,
                "round": 1,
                "position_id": "p1",
                "nodes": 900,
                "elapsed_ns": 3_000_000_000,
            },
        ]
        protocol = {
            "bootstrap_replicates": 100,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 123,
        }
        summary = summarize_observations(observations, protocol)[0]
        self.assertEqual(summary["aggregate_nps"], 250.0)
        self.assertNotEqual(summary["aggregate_nps"], 200.0)

    def test_integrity_detects_history_dependent_signature(self) -> None:
        rows = []
        for round_index, nodes in enumerate((100, 101)):
            rows.append(
                {
                    "mode": "fixed_depth",
                    "limit": 7,
                    "round": round_index,
                    "position_id": "p1",
                    "reported_depth": 7,
                    "score_type": "cp",
                    "score_value": 12,
                    "nodes": nodes,
                    "bestmove": "e2e4",
                }
            )
        integrity = fixed_depth_integrity(rows, expected_positions=1)
        self.assertEqual(integrity["status"], "fail")
        self.assertEqual(len(integrity["mismatches"]), 1)

    def test_observation_count_gate_detects_missing_case(self) -> None:
        observations = [
            {"mode": "fixed_depth"},
            {"mode": "movetime_ms"},
        ]
        protocol = {
            "fixed_depths": [7],
            "fixed_rounds": 2,
            "movetimes_ms": [1000],
            "movetime_rounds": 1,
        }
        integrity = observation_count_integrity(
            observations, protocol, position_count=1
        )
        self.assertEqual(integrity["status"], "fail")
        self.assertEqual(integrity["fixed_depth"], {"actual": 1, "expected": 2})

    def test_measurement_schedule_spreads_timed_rounds(self) -> None:
        protocol = {
            "fixed_depths": [7, 8],
            "fixed_rounds": 10,
            "movetimes_ms": [1000, 3000],
            "movetime_rounds": 5,
        }
        schedule = measurement_schedule(protocol)
        timed_rounds = [
            round_index for mode, _, round_index in schedule if mode == "movetime_ms"
        ]
        self.assertEqual(sorted(set(timed_rounds)), [0, 1, 2, 3, 4])
        self.assertEqual(len(schedule), 30)
        self.assertTrue(any(mode == "movetime_ms" for mode, _, _ in schedule[:4]))

    def test_deployment_config_options_are_parseable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        options = read_config_uci_options(
            repo_root / "deploy/lichess/config-nnue-v41.yml"
        )
        self.assertEqual(options["LmrBase"], "0.45")
        self.assertEqual(options["QseeThreshold"], -75)
        self.assertTrue(options["QseeEnabled"])

    def test_rejected_uci_option_fails_handshake(self) -> None:
        fake_engine = r"""import sys
for raw in sys.stdin:
    line = raw.strip()
    if line == "uci":
        print("id name FakeReject", flush=True)
        print("option name Bad type spin default 0 min 0 max 1", flush=True)
        print("uciok", flush=True)
    elif line.startswith("setoption"):
        print("info string ignored invalid option Bad", flush=True)
    elif line == "isready":
        print("readyok", flush=True)
    elif line == "quit":
        break
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fake_reject.py"
            path.write_text(fake_engine, encoding="utf-8")
            with EngineSession(
                [sys.executable, str(path)], cwd=None, timeout_seconds=5
            ) as session:
                with self.assertRaises(BenchmarkError):
                    session.initialize("FakeReject", {"Bad": 9})

    def test_cross_run_integrity_requires_non_null_code_hashes(self) -> None:
        result = {
            "provenance": {
                "engine_sha256": "engine",
                "model_sha256": "model",
                "spec_sha256": "spec",
                "effective_spec_sha256": "effective",
                "config_sha256": "config",
                "config_validation": {"status": "match"},
                "harness_sha256": None,
                "runner_sha256": None,
            },
            "observations": [
                {
                    "mode": "fixed_depth",
                    "limit": 7,
                    "position_id": "p1",
                    "reported_depth": 7,
                    "score_type": "cp",
                    "score_value": 0,
                    "nodes": 10,
                    "bestmove": "e2e4",
                }
            ],
        }
        self.assertEqual(cross_run_integrity([result])["status"], "fail")

    def test_fake_uci_process_round_trip(self) -> None:
        fake_engine = r"""import sys
for raw in sys.stdin:
    line = raw.strip()
    if line == "uci":
        print("id name FakeBench", flush=True)
        print("info string nnue_kernel=x86_avx2_exact", flush=True)
        print("info string nnue_accumulator_kernel=x86_avx2", flush=True)
        print("uciok", flush=True)
    elif line == "isready":
        print("readyok", flush=True)
    elif line.startswith("go depth "):
        depth = int(line.split()[-1])
        print(f"info depth {depth} score cp 9 nodes {depth * 100}", flush=True)
        print("bestmove e2e4", flush=True)
    elif line.startswith("go movetime "):
        print("info depth 3 score cp 9 nodes 300", flush=True)
        print("bestmove e2e4", flush=True)
    elif line == "quit":
        break
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fake_engine.py"
            path.write_text(fake_engine, encoding="utf-8")
            with EngineSession(
                [sys.executable, str(path)], cwd=None, timeout_seconds=5
            ) as session:
                handshake = session.initialize("FakeBench", {})
                result = session.search(
                    "8/8/8/8/8/8/4P3/4K2k w - - 0 1", "fixed_depth", 4
                )
        self.assertEqual(handshake["engine_name"], "FakeBench")
        self.assertEqual(handshake["nnue_kernel"], "x86_avx2_exact")
        self.assertEqual(
            handshake["nnue_accumulator_kernel"], "x86_avx2"
        )
        self.assertEqual(result["reported_depth"], 4)
        self.assertEqual(result["nodes"], 400)
        self.assertEqual(result["bestmove"], "e2e4")
        self.assertGreater(result["elapsed_ns"], 0)


if __name__ == "__main__":
    unittest.main()
