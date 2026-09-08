import copy
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "analyze" / "analyze_nnue_forward_stage_crosshost.py"
SPEC = importlib.util.spec_from_file_location("nnue_forward_stage_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MODEL_SHA256 = "a" * 64
CORPUS = {
    "samples": 256,
    "seed": 20260825,
    "checksum": "0xf5399956b2fa9ea8",
    "phase_histogram": [32, 32, 16, 16, 16, 32, 32, 80],
    "nonzero_aux_samples": 112,
}


def make_run(host, ordinal, wall_ns_per_eval, cpu_ns_per_eval):
    batches = []
    elapsed = {stage: [] for stage in MODULE.STAGES}
    process_cpu = {stage: [] for stage in MODULE.STAGES}
    for repeat in range(MODULE.REPEATS):
        for order in range(len(MODULE.STAGES)):
            stage = MODULE.STAGES[(repeat + order) % len(MODULE.STAGES)]
            evaluations = MODULE.OFFICIAL_ITERATIONS[stage]
            # Balanced order noise exercises stratification while preserving a
            # simple, exactly known cluster median.
            adjustment = (order - 1) * 2
            wall_ns = int((wall_ns_per_eval[stage] + adjustment) * evaluations)
            cpu_ns = int((cpu_ns_per_eval[stage] + adjustment) * evaluations)
            batches.append(
                {
                    "repeat": repeat,
                    "order": order,
                    "stage": stage,
                    "evaluations": evaluations,
                    "wall_ns": wall_ns,
                    "process_cpu_ns": cpu_ns,
                }
            )
            elapsed[stage].append(wall_ns)
            process_cpu[stage].append(cpu_ns)

    stages = {}
    for stage in MODULE.STAGES:
        evaluations = MODULE.OFFICIAL_ITERATIONS[stage]
        wall_values = [value / evaluations for value in elapsed[stage]]
        cpu_values = [value / evaluations for value in process_cpu[stage]]
        stages[stage] = {
            "evaluations_per_repeat": evaluations,
            "elapsed_ns": elapsed[stage],
            "process_cpu_ns": process_cpu[stage],
            "median_ns_per_evaluation": statistics.median(wall_values),
            "median_process_cpu_ns_per_evaluation": statistics.median(cpu_values),
            "checksum": MODULE.OFFICIAL_STAGE_CHECKSUMS[stage],
            "timed_sink": "0x1234" + str(MODULE.STAGES.index(stage)),
        }

    result = {
        "schema_version": 2,
        "protocol": MODULE.PROTOCOL,
        "status": "valid",
        "backend_requested": "neon" if host == "local" else "vnni",
        "kernel": (
            "arm_neon_dotprod_i8mm"
            if host == "local"
            else "x86_avx512vnni_256"
        ),
        "model_sha256": MODEL_SHA256,
        "corpus": copy.deepcopy(CORPUS),
        "correctness": {
            "status": "pass",
            "scalar_parity": True,
            "stage_parity": True,
            "timed_sink_parity": True,
            "checksum": "0x61bbf9ec6191a24b",
        },
        "timing": copy.deepcopy(MODULE.EXPECTED_TIMING),
        "warmup_evaluations": 10_000,
        "repeats": MODULE.REPEATS,
        "stages": stages,
        "batches": batches,
    }
    if host == "local":
        result["local_run"] = {
            "ordinal": ordinal,
            "independent_process": True,
            "source_label": "clean-stage-" + "b" * 40,
        }
    else:
        result["heroku_run"] = {
            "dyno_ordinal": ordinal,
            "profile": "full",
            "size": "basic",
            "source_label": "source-commit:" + "b" * 40,
        }
    return result


def fixture_documents():
    stage_offsets = {stage: index * 10 for index, stage in enumerate(MODULE.STAGES)}
    local = []
    heroku = []
    for ordinal, cluster_offset in enumerate((0, 10, 20), start=1):
        local_wall = {
            stage: 100 + offset + cluster_offset for stage, offset in stage_offsets.items()
        }
        local_cpu = {stage: value - 10 for stage, value in local_wall.items()}
        heroku_wall = {stage: value * 3 for stage, value in local_wall.items()}
        heroku_cpu = {stage: value * 3 for stage, value in local_cpu.items()}
        local.append(make_run("local", ordinal, local_wall, local_cpu))
        heroku.append(make_run("heroku", ordinal, heroku_wall, heroku_cpu))
    return local, heroku


class AnalyzeNnueForwardStageCrossHostTests(unittest.TestCase):
    def test_primary_estimator_and_bootstrap_are_deterministic(self):
        local, heroku = fixture_documents()
        result = MODULE.analyze_documents(
            local,
            heroku,
            bootstrap_replicates=200,
            bootstrap_seed=20260825,
        )
        row = result["primary"]["stages"]["s1_input_to_hidden2"]["wall"]
        # Per-cluster medians have +1 ns from the four balanced order strata.
        self.assertEqual(row["local_cluster_medians_ns_per_evaluation"], [101, 111, 121])
        self.assertEqual(row["heroku_cluster_medians_ns_per_evaluation"], [301, 331, 361])
        self.assertEqual(row["local_median_of_cluster_medians_ns_per_evaluation"], 111)
        self.assertEqual(row["heroku_median_of_cluster_medians_ns_per_evaluation"], 331)
        self.assertAlmostEqual(row["ratio_heroku_over_local"], 331 / 111)
        self.assertEqual(row["delta_heroku_minus_local_ns_per_evaluation"], 220)
        self.assertEqual(result["bootstrap"]["replicates"], 200)
        self.assertEqual(
            result["bootstrap"]["design"]["within_cluster_strata"],
            "stage_order_0_to_3",
        )
        repeated = MODULE.analyze_documents(
            local,
            heroku,
            bootstrap_replicates=200,
            bootstrap_seed=20260825,
        )
        self.assertEqual(result["bootstrap"], repeated["bootstrap"])

    def test_requires_three_independent_clusters(self):
        local, heroku = fixture_documents()
        with self.assertRaisesRegex(MODULE.AnalysisError, "exactly 3"):
            MODULE.analyze_documents(local[:2], heroku, bootstrap_replicates=2)
        local[0]["local_run"]["independent_process"] = False
        with self.assertRaisesRegex(MODULE.AnalysisError, "independence"):
            MODULE.analyze_documents(local, heroku, bootstrap_replicates=2)

    def test_rejects_parity_schedule_and_cross_host_identity_failures(self):
        local, heroku = fixture_documents()
        broken_parity = copy.deepcopy(local)
        broken_parity[0]["correctness"]["timed_sink_parity"] = False
        with self.assertRaisesRegex(MODULE.AnalysisError, "timed_sink_parity"):
            MODULE.analyze_documents(broken_parity, heroku, bootstrap_replicates=2)

        broken_schedule = copy.deepcopy(local)
        broken_schedule[0]["batches"][0]["stage"] = "full"
        with self.assertRaisesRegex(MODULE.AnalysisError, "stage set|schedule"):
            MODULE.analyze_documents(broken_schedule, heroku, bootstrap_replicates=2)

        broken_model = copy.deepcopy(heroku)
        broken_model[2]["model_sha256"] = "c" * 64
        with self.assertRaisesRegex(MODULE.AnalysisError, "model_sha256 identity"):
            MODULE.analyze_documents(local, broken_model, bootstrap_replicates=2)

        broken_checksum = copy.deepcopy(heroku)
        broken_checksum[1]["stages"]["full"]["checksum"] = "0xdeadbeef"
        with self.assertRaisesRegex(MODULE.AnalysisError, "full checksum"):
            MODULE.analyze_documents(local, broken_checksum, bootstrap_replicates=2)

        broken_corpus = copy.deepcopy(local)
        broken_corpus[1]["corpus"]["checksum"] = "0xdeadbeef"
        with self.assertRaisesRegex(MODULE.AnalysisError, "corpus checksum"):
            MODULE.analyze_documents(broken_corpus, heroku, bootstrap_replicates=2)

    def test_directory_loader_uses_only_canonical_run_files(self):
        local, heroku = fixture_documents()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_dir = root / "local"
            heroku_dir = root / "heroku"
            local_dir.mkdir()
            heroku_dir.mkdir()
            for ordinal, document in enumerate(local, start=1):
                (local_dir / f"local-forward-stages-run-{ordinal}.json").write_text(
                    json.dumps(document), encoding="utf-8"
                )
                # Must not be mistaken for an independent cluster.
                (local_dir / f"local-forward-stages-run-{ordinal}.stdout.json").write_text(
                    json.dumps(document), encoding="utf-8"
                )
            for ordinal, document in enumerate(heroku, start=1):
                (heroku_dir / f"heroku-forward-stages-run-{ordinal}.json").write_text(
                    json.dumps(document), encoding="utf-8"
                )
            result = MODULE.analyze_directories(
                local_dir,
                heroku_dir,
                bootstrap_replicates=5,
                bootstrap_seed=7,
            )
            self.assertEqual(result["status"], "valid")
            self.assertEqual(len(result["clusters"]["local"]), 3)


if __name__ == "__main__":
    unittest.main()
