import base64
import contextlib
import hashlib
import io
import json
import unittest
import zlib
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import compare_nnue_backend_benchmarks as comparator
from tools import run_heroku_nnue_backend_matrix as local_matrix
from tools import run_nnue_backend_matrix_remote as remote_matrix


class HerokuNnueBackendMatrixTests(unittest.TestCase):
    def test_smoke_matrix_keeps_both_fixed_depths(self) -> None:
        args = Namespace(
            engine=Path("engine"),
            model=Path("model"),
            config=Path("config"),
            engine_cwd=Path("."),
            spec=Path("spec"),
            host_label="host",
            source_label="source",
            harness_sha256="harness",
            runner_sha256="runner",
            profile="smoke",
        )
        arguments = remote_matrix.benchmark_arguments(args, "scalar")
        depths = [
            arguments[index + 1]
            for index, value in enumerate(arguments)
            if value == "--fixed-depth"
        ]
        self.assertEqual(depths, ["7", "8"])

    def test_remote_command_uses_one_matrix_dyno_without_forced_parent_env(
        self,
    ) -> None:
        args = local_matrix.build_parser().parse_args(
            ["--app", "chess-simd-staging", "--source-label", "tree:abc"]
        )
        command = local_matrix.remote_command(
            args,
            local_matrix.LATIN_SQUARE[0],
            ordinal=1,
            harness_sha256="harness",
            spec_sha256="spec",
            runner_sha256="runner",
        )
        self.assertEqual(command[command.index("--type") + 1], "benchmark")
        self.assertEqual(command[command.index("--order") + 1], "scalar,avx2,vnni")
        self.assertNotIn("CHESS_NNUE_BACKEND", " ".join(command))

    def test_production_app_is_rejected_before_running_heroku(self) -> None:
        with self.assertRaises(SystemExit):
            local_matrix.main(
                [
                    "--app",
                    "stormy-garden-92984",
                    "--source-label",
                    "test",
                ]
            )

    def test_full_run_requires_complete_latin_square(self) -> None:
        with self.assertRaises(SystemExit):
            local_matrix.main(
                [
                    "--app",
                    "chess-simd-staging",
                    "--profile",
                    "full",
                    "--runs",
                    "2",
                    "--source-label",
                    "test",
                ]
            )

    def test_chunked_remote_result_round_trip(self) -> None:
        expected = {"schema_version": 1, "status": "valid", "data": "x" * 20_000}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            remote_matrix.emit_result(expected)
        chunks = {}
        metadata = None
        for line in output.getvalue().splitlines():
            if line.startswith(remote_matrix.CHUNK_SENTINEL):
                descriptor, value = line.removeprefix(
                    remote_matrix.CHUNK_SENTINEL
                ).split(":", 1)
                index, count = (int(part) for part in descriptor.split("/", 1))
                chunks[index] = value
            elif line.startswith(remote_matrix.END_SENTINEL):
                metadata = json.loads(line.removeprefix(remote_matrix.END_SENTINEL))
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(len(chunks), metadata["chunks"])
        compressed = base64.b64decode(
            "".join(chunks[index] for index in range(1, len(chunks) + 1)),
            validate=True,
        )
        self.assertEqual(
            hashlib.sha256(compressed).hexdigest(), metadata["compressed_sha256"]
        )
        decoded = zlib.decompress(compressed)
        self.assertEqual(hashlib.sha256(decoded).hexdigest(), metadata["json_sha256"])
        self.assertEqual(json.loads(decoded), expected)

    def test_same_dyno_fixed_depth_parity_detects_backend_drift(self) -> None:
        def result(nodes: int) -> dict:
            return {
                "observations": [
                    {
                        "mode": "fixed_depth",
                        "limit": 7,
                        "position_id": "startpos",
                        "reported_depth": 7,
                        "score_type": "cp",
                        "score_value": 1,
                        "nodes": nodes,
                        "bestmove": "e2e4",
                    }
                ]
            }

        matching = {backend: result(100) for backend in remote_matrix.BACKENDS}
        self.assertEqual(remote_matrix.fixed_depth_parity(matching)["status"], "pass")
        matching["vnni"] = result(101)
        self.assertEqual(remote_matrix.fixed_depth_parity(matching)["status"], "fail")

    @mock.patch("tools.run_nnue_backend_matrix_remote.subprocess.run")
    def test_parity_requires_all_8192_fixture_positions(self, run: mock.Mock) -> None:
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=(
                "python_cpp_parity_positions=8192\n"
                "phase quantized NNUE incremental parity passed\n"
            ),
            stderr="",
        )
        result = remote_matrix.run_parity(
            Path("test"), Path("model"), Path("parity"), "vnni"
        )
        self.assertEqual(result["python_cpp_parity_positions"], 8192)
        run.return_value = SimpleNamespace(returncode=0, stdout="passed\n", stderr="")
        with self.assertRaises(RuntimeError):
            remote_matrix.run_parity(
                Path("test"), Path("model"), Path("parity"), "vnni"
            )

    def test_provenance_rejects_null_hash_and_unmatched_config(self) -> None:
        provenance = {key: "same" for key in comparator.PROVENANCE_KEYS}
        provenance["config_validation"] = {"status": "match"}
        all_runs = {
            backend: [{"provenance": dict(provenance)}]
            for backend in comparator.EXPECTED_KERNEL
        }
        self.assertEqual(comparator.provenance_check(all_runs)["status"], "pass")
        all_runs["vnni"][0]["provenance"]["engine_sha256"] = None
        self.assertEqual(comparator.provenance_check(all_runs)["status"], "fail")
        all_runs["vnni"][0]["provenance"]["engine_sha256"] = "same"
        all_runs["vnni"][0]["provenance"]["config_validation"] = {
            "status": "not_checked"
        }
        self.assertEqual(comparator.provenance_check(all_runs)["status"], "fail")

    def test_paired_bootstrap_keeps_backends_separate(self) -> None:
        elapsed = {"scalar": 100, "avx2": 50, "vnni": 25}
        all_runs = {}
        for backend in local_matrix.BACKENDS:
            all_runs[backend] = [
                {
                    "observations": [
                        {
                            "mode": "fixed_depth",
                            "limit": 8,
                            "round": 0,
                            "position_id": "startpos",
                            "nodes": 100,
                            "elapsed_ns": elapsed[backend],
                        }
                    ]
                }
            ]
        paired = local_matrix.paired_backend_speedups(all_runs, replicates=50, seed=1)
        self.assertEqual(paired["status"], "pass")
        ratios = paired["metrics"][0]["ratios"]
        self.assertEqual(ratios["avx2_over_scalar"]["estimate"], 2.0)
        self.assertEqual(ratios["vnni_over_scalar"]["estimate"], 4.0)

    def test_full_promotion_gate_requires_every_dyno_and_interval(self) -> None:
        elapsed = {"scalar": 400, "avx2": 100, "vnni": 50}
        all_runs = {
            backend: [
                {
                    "observations": [
                        {
                            "mode": "fixed_depth",
                            "limit": depth,
                            "round": 0,
                            "position_id": "startpos",
                            "nodes": 100,
                            "elapsed_ns": elapsed[backend],
                        }
                        for depth in local_matrix.PROMOTION_DEPTHS
                    ]
                }
                for _ in range(3)
            ]
            for backend in local_matrix.BACKENDS
        }
        paired = local_matrix.paired_backend_speedups(
            all_runs, replicates=50, seed=1
        )
        gate = local_matrix.full_promotion_gate(all_runs, paired)
        self.assertEqual(gate["status"], "pass")

        all_runs["vnni"][1]["observations"][0]["elapsed_ns"] = 500
        gate = local_matrix.full_promotion_gate(all_runs, paired)
        self.assertEqual(gate["status"], "fail")
        self.assertTrue(any("dyno 2, depth 7" in item for item in gate["failures"]))

        all_runs["vnni"][1]["observations"][0]["elapsed_ns"] = 50
        paired["metrics"][0]["ratios"]["vnni_over_avx2"][
            "paired_bootstrap_interval95"
        ][0] = 1.0
        gate = local_matrix.full_promotion_gate(all_runs, paired)
        self.assertEqual(gate["status"], "fail")
        self.assertTrue(any("lower95=1.000000" in item for item in gate["failures"]))


if __name__ == "__main__":
    unittest.main()
