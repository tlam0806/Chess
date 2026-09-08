import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "analyze" / "analyze_cross_host_profile.py"
SPEC = importlib.util.spec_from_file_location("cross_host_profile_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def symbol_document(host, cpu_seconds, nn_samples, tt_samples):
    return {
        "host": host,
        "rounds": [
            {
                "round": 1,
                "mode": "profile",
                "nodes": 100,
                "process_cpu_seconds": cpu_seconds[0],
            },
            {
                "round": 2,
                "mode": "profile",
                "nodes": 100,
                # Exercise the Heroku runner's historical alias.
                "cpu_seconds": cpu_seconds[1],
            },
        ],
        "profile_rounds": [
            {
                "round": 1,
                "sample_summary": {
                    "total_samples": nn_samples[0] + tt_samples[0],
                    "symbols": [
                        {
                            "symbol": "chess::phase_nnue_detail::VnniNetwork::evaluate",
                            "flat_samples": nn_samples[0],
                        },
                        {
                            "symbol": "chess::LowerMoveRangeBucketTranspositionTable::probe",
                            "flat_samples": tt_samples[0],
                        },
                    ],
                },
            },
            {
                "round": 2,
                "sample_summary": {
                    "total_samples": nn_samples[1] + tt_samples[1],
                    "symbols": [
                        {
                            "symbol": "chess::PhaseQuantizedNnueModel::forward_positional_scalar",
                            "flat_samples": nn_samples[1],
                        },
                        {
                            "symbol": "chess::LowerMoveRangeBucketTranspositionTable::store",
                            "flat_samples": tt_samples[1],
                        },
                    ],
                },
            },
        ],
    }


def abba_document(host, cpu_seconds):
    round_ids = (1, 2, 5, 6)
    return {
        "host": host,
        "rounds": [
            {
                "round": round_id,
                "mode": "profile",
                "nodes": 100,
                "process_cpu_seconds": cpu,
                "sample_summary": {
                    "total_samples": 10,
                    "symbols": [
                        {
                            "symbol": "chess::phase_nnue_detail::VnniNetwork::evaluate",
                            "flat_samples": 5,
                        },
                        {
                            "symbol": "chess::NnueSearcherV38::negamax",
                            "flat_samples": 5,
                        },
                    ],
                },
            }
            for round_id, cpu in zip(round_ids, cpu_seconds)
        ],
    }


def thermal_abba_document(host, control_cpu, profile_cpu):
    """Two blocks whose profiler arm runs faster than surrounding controls."""

    rounds = []
    profiles = []
    control_by_round = {0: control_cpu[0], 3: control_cpu[1], 4: control_cpu[2], 7: control_cpu[3]}
    profile_by_round = {1: profile_cpu[0], 2: profile_cpu[1], 5: profile_cpu[2], 6: profile_cpu[3]}
    for round_id in range(8):
        profiled = round_id % 4 in (1, 2)
        rounds.append(
            {
                "round": round_id,
                "mode": "profile" if profiled else "control",
                "nodes": 100,
                "process_cpu_seconds": (
                    profile_by_round[round_id]
                    if profiled
                    else control_by_round[round_id]
                ),
            }
        )
        if profiled:
            # The slow first block is NN-heavy; the hotter second block is
            # search-heavy. Correct attribution must retain this block-local
            # weighting rather than use one global control multiplier.
            nn_samples = 75 if round_id < 4 else 25
            profiles.append(
                {
                    "round": round_id,
                    "sample_summary": {
                        "total_samples": 100,
                        "symbols": [
                            {
                                "symbol": "chess::phase_nnue_detail::VnniNetwork::evaluate",
                                "flat_samples": nn_samples,
                            },
                            {
                                "symbol": "chess::NnueSearcherV38::negamax",
                                "flat_samples": 100 - nn_samples,
                            },
                        ],
                    },
                }
            )
    return {"host": host, "rounds": rounds, "profile_rounds": profiles}


class AnalyzeCrossHostProfileTests(unittest.TestCase):
    def test_symbol_classification_uses_exclusive_semantic_buckets(self):
        cases = {
            "chess::phase_nnue_detail::VnniNetwork::evaluate": "nn_forward",
            "chess::PhaseQuantizedNnueAccumulator::rebuild_perspective": "nn_accumulator",
            "chess::LowerMoveRangeBucketTranspositionTable::probe": "transposition_table",
            "chess::generate_legal_quiet_non_promotion_moves_with_info": "move_generation_attacks",
            "chess::Position::unmake_move": "make_unmake_king_safety",
            "chess::static_exchange_eval": "static_exchange_evaluation",
            "chess::CounterHistoryTable::get_score": "move_ordering",
            "chess::NnueSearcherV38::negamax": "search_control",
            "[libm.so.6]": "runtime",
            "chess::Position::piece_type_on_occupied": "other",
            "chess::(anonymous namespace)::clear_see_piece": "static_exchange_evaluation",
            (
                "chess::(anonymous namespace)::find_least_valuable_pseudo_attacker"
                "(chess::(anonymous namespace)::SeeState const&, int)"
            ): "static_exchange_evaluation",
            "chess::detail::try_emit_legal_noisy_move": "move_generation_attacks",
            (
                "chess::nnue_v38_detail::"
                "is_quiet_non_promotion_move_legal_by_attack_check"
            ): "move_generation_attacks",
            "_platform_memmove": "runtime",
            "___chkstk_darwin": "runtime",
            "log": "runtime",
            "chess::RepetitionStack::push_impl": "search_control",
            "chess::ScopedRepetitionPush::~ScopedRepetitionPush": "search_control",
            "chess::PhaseQuantizedNnueModel::uses_accelerated_kernel": "nn_forward",
        }
        for symbol, expected in cases.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(MODULE.classify_symbol(symbol), expected)

    def test_classifier_uses_outer_leaf_not_nested_template_callback(self):
        generate = (
            "chess::generate_legal_quiet_non_promotion_moves_with_info<"
            "chess::FixedList<int> chess::NnueSearcherV38::"
            "ordered_moves_for_stage<3>()::'lambda'&>"
            "(chess::Position const&)"
        )
        emit = (
            "chess::detail::try_emit_legal_non_king_non_capture_move<"
            "chess::NnueSearcherV38::ordered_moves_for_stage<3>()::'lambda'&>"
            "(chess::Position const&)"
        )
        forward = (
            "chess::PhaseCandidateKernel<2, 8, 128>::evaluate"
            "(int const*, int const*, unsigned long, unsigned long) const"
        )
        self.assertEqual(
            MODULE.classify_symbol(generate), "move_generation_attacks"
        )
        self.assertEqual(MODULE.classify_symbol(emit), "move_generation_attacks")
        self.assertEqual(MODULE.classify_symbol(forward), "nn_forward")

    def test_component_cost_slowdown_and_gap_share(self):
        local = symbol_document("m4", [1.0, 3.0], [50, 50], [50, 50])
        heroku = symbol_document("heroku", [4.0, 4.0], [25, 25], [75, 75])
        result = MODULE.analyze(
            local,
            heroku,
            bootstrap_replicates=200,
            bootstrap_seed=17,
        )

        # 4 CPU-s / 200 nodes locally and 8 / 200 on Heroku.
        self.assertAlmostEqual(result["local"]["cpu_ns_per_node"], 20_000_000)
        self.assertAlmostEqual(result["heroku"]["cpu_ns_per_node"], 40_000_000)
        self.assertAlmostEqual(result["comparison"]["total"]["slowdown_ratio"], 2.0)

        # NN costs are equal: local 50% of 4s, Heroku 25% of 8s.
        nn = result["comparison"]["nn_forward"]
        self.assertAlmostEqual(nn["local_cpu_ns_per_node"], 10_000_000)
        self.assertAlmostEqual(nn["heroku_cpu_ns_per_node"], 10_000_000)
        self.assertAlmostEqual(nn["slowdown_ratio"], 1.0)
        self.assertAlmostEqual(nn["gap_cpu_ns_per_node"], 0.0)
        self.assertAlmostEqual(nn["gap_share"], 0.0)

        # All 20 ms/node of excess is attributed to TT.
        tt = result["comparison"]["transposition_table"]
        self.assertAlmostEqual(tt["slowdown_ratio"], 3.0)
        self.assertAlmostEqual(tt["gap_cpu_ns_per_node"], 20_000_000)
        self.assertAlmostEqual(tt["gap_share"], 1.0)
        for host in (result["local"], result["heroku"]):
            self.assertAlmostEqual(
                sum(
                    row["cpu_ns_per_node"]
                    for row in host["buckets"].values()
                ),
                host["cpu_ns_per_node"],
            )
        self.assertAlmostEqual(
            sum(
                row["gap_share"]
                for name, row in result["comparison"].items()
                if name != "total" and row["gap_share"] is not None
            ),
            1.0,
        )
        self.assertEqual(result["bootstrap"]["replicates"], 200)
        self.assertEqual(result["bootstrap"]["seed"], 17)

    def test_flat_bucket_counts_and_unresolved_remainder(self):
        document = {
            "rounds": [
                {
                    "round": "r1",
                    "nodes": 10,
                    "process_cpu_seconds": 0.1,
                    "sample_summary": {
                        "total_samples": 10,
                        "flat_bucket_counts": {"forward": 4, "tt": 3},
                    },
                }
            ]
        }
        parsed = MODULE.parse_host_document(document, "local")
        shares = parsed["rounds"][0]["bucket_sample_shares"]
        self.assertAlmostEqual(shares["nn_forward"], 0.4)
        self.assertAlmostEqual(shares["transposition_table"], 0.3)
        self.assertAlmostEqual(shares["other"], 0.3)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_xctrace_nanosecond_weights_override_audit_row_counts(self):
        document = {
            "rounds": [
                {
                    "round": 1,
                    "nodes": 10,
                    "process_cpu_seconds": 0.1,
                    "sample_summary": {
                        "total_weight_ns": 300,
                        "total_sampled_seconds": 300e-9,
                        # Deliberately opposite to the duration distribution.
                        "total_samples": 3,
                        "symbols": [
                            {
                                "symbol": (
                                    "chess::phase_nnue_detail::"
                                    "VnniNetwork::evaluate"
                                ),
                                "flat_samples": 2,
                                "flat_weight_ns": 100,
                            },
                            {
                                "symbol": "chess::NnueSearcherV38::negamax",
                                "flat_samples": 1,
                                "flat_weight_ns": 200,
                            },
                        ],
                    },
                }
            ]
        }
        parsed = MODULE.parse_host_document(document, "local")
        row = parsed["rounds"][0]
        self.assertEqual(row["sample_weight_unit"], "nanoseconds")
        self.assertAlmostEqual(row["bucket_sample_shares"]["nn_forward"], 1 / 3)
        self.assertAlmostEqual(
            row["bucket_sample_shares"]["search_control"], 2 / 3
        )

    def test_xctrace_rejects_inconsistent_weight_and_seconds_totals(self):
        document = {
            "rounds": [
                {
                    "round": 1,
                    "nodes": 10,
                    "process_cpu_seconds": 0.1,
                    "sample_summary": {
                        "total_weight_ns": 100,
                        "total_sampled_seconds": 1.0,
                        "symbols": [
                            {
                                "symbol": "chess::NnueSearcherV38::negamax",
                                "flat_weight_ns": 100,
                            }
                        ],
                    },
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "total_weight_ns is inconsistent"):
            MODULE.parse_host_document(document, "local")

    def test_bootstrap_is_deterministic(self):
        local = symbol_document("m4", [1.0, 3.0], [50, 50], [50, 50])
        heroku = symbol_document("heroku", [4.0, 4.0], [25, 25], [75, 75])
        first = MODULE.analyze(
            local, heroku, bootstrap_replicates=250, bootstrap_seed=99
        )["bootstrap"]
        second = MODULE.analyze(
            local, heroku, bootstrap_replicates=250, bootstrap_seed=99
        )["bootstrap"]
        self.assertEqual(first, second)
        self.assertEqual(
            first["method"],
            "independent complete-ABBA-block percentile bootstrap within each host",
        )
        self.assertEqual(first["requested_unit"], "auto")
        self.assertEqual(first["unit"], "abba-block")
        self.assertEqual(first["local_sampling_units"], 1)
        self.assertEqual(first["profiled_rounds_per_unit"], 2)

    def test_round_bootstrap_remains_available_as_sensitivity_mode(self):
        local = abba_document("m4", [1.0, 1.0, 9.0, 9.0])
        heroku = abba_document("heroku", [2.0, 2.0, 10.0, 10.0])
        block = MODULE.analyze(
            local,
            heroku,
            bootstrap_replicates=250,
            bootstrap_seed=23,
        )["bootstrap"]
        repeated = MODULE.analyze(
            local,
            heroku,
            bootstrap_replicates=250,
            bootstrap_seed=23,
        )["bootstrap"]
        by_round = MODULE.analyze(
            local,
            heroku,
            bootstrap_replicates=250,
            bootstrap_seed=23,
            bootstrap_unit="round",
        )["bootstrap"]
        self.assertEqual(block, repeated)
        self.assertEqual(block["unit"], "abba-block")
        self.assertEqual(block["local_sampling_units"], 2)
        self.assertEqual(by_round["unit"], "round")
        self.assertEqual(by_round["requested_unit"], "round")
        self.assertEqual(by_round["local_sampling_units"], 4)
        self.assertEqual(
            by_round["method"],
            "independent complete-round percentile bootstrap within each host",
        )

    def test_abba_controls_calibrate_profiler_frequency_and_thermal_drift(self):
        local = thermal_abba_document(
            "m4",
            # Pooled controls are 40 ms/node in block 0 and 80 ms/node in block 1.
            control_cpu=[3.0, 5.0, 6.0, 10.0],
            profile_cpu=[2.0, 2.0, 3.0, 3.0],
        )
        heroku = thermal_abba_document(
            "heroku",
            control_cpu=[8.0, 12.0, 16.0, 24.0],
            profile_cpu=[9.0, 9.0, 18.0, 18.0],
        )
        result = MODULE.analyze(
            local,
            heroku,
            bootstrap_replicates=200,
            bootstrap_seed=101,
        )

        self.assertEqual(result["cpu_calibration"]["requested"], "auto")
        self.assertEqual(result["cpu_calibration"]["resolved"], "abba-control")
        self.assertTrue(result["local"]["calibration"]["enabled"])
        self.assertEqual(len(result["local"]["calibration"]["blocks"]), 2)
        # Local B rounds measured only 10 CPU-s / 400 nodes, but the four
        # surrounding-control equivalents total 24 CPU-s / 400 nodes.
        self.assertAlmostEqual(
            result["local"]["measured_profile_cpu_ns_per_node"], 25_000_000
        )
        self.assertAlmostEqual(result["local"]["cpu_ns_per_node"], 60_000_000)
        self.assertAlmostEqual(
            result["local"]["profile_to_control_cpu_ns_per_node_ratio"],
            25 / 60,
        )
        blocks = result["local"]["calibration"]["blocks"]
        self.assertAlmostEqual(blocks[0]["control_cpu_ns_per_node"], 40_000_000)
        self.assertAlmostEqual(blocks[1]["control_cpu_ns_per_node"], 80_000_000)
        self.assertAlmostEqual(
            blocks[0]["profile_to_control_cpu_ns_per_node_ratio"], 0.5
        )
        self.assertAlmostEqual(
            blocks[1]["profile_to_control_cpu_ns_per_node_ratio"], 0.375
        )
        self.assertAlmostEqual(
            result["local"]["calibration"][
                "block_profile_to_control_ratio_median"
            ],
            0.4375,
        )
        self.assertEqual(
            result["local"]["calibration"][
                "block_control_cpu_ns_per_node_range"
            ],
            [40_000_000, 80_000_000],
        )
        # Block-local weighting: 8 control-equivalent seconds * 75% plus
        # 16 seconds * 25% = 10 seconds over 400 nodes.
        self.assertAlmostEqual(
            result["local"]["buckets"]["nn_forward"]["cpu_ns_per_node"],
            25_000_000,
        )
        self.assertEqual(result["bootstrap"]["unit"], "abba-block")
        self.assertEqual(result["bootstrap"]["local_sampling_units"], 2)

        sensitivity = MODULE.analyze(
            local,
            heroku,
            bootstrap_replicates=20,
            bootstrap_seed=101,
            cpu_calibration="profile-measured",
        )
        self.assertEqual(
            sensitivity["cpu_calibration"]["resolved"], "profile-measured"
        )
        self.assertAlmostEqual(
            sensitivity["local"]["cpu_ns_per_node"], 25_000_000
        )
        self.assertIsNone(
            sensitivity["local"]["profile_to_control_cpu_ns_per_node_ratio"]
        )

    def test_auto_calibration_falls_back_for_both_hosts_consistently(self):
        complete = thermal_abba_document(
            "m4", [3.0, 5.0, 6.0, 10.0], [2.0, 2.0, 3.0, 3.0]
        )
        incomplete = symbol_document("heroku", [4.0, 4.0], [25, 25], [75, 75])
        result = MODULE.analyze(
            complete,
            incomplete,
            bootstrap_replicates=20,
            bootstrap_seed=7,
        )
        self.assertEqual(result["cpu_calibration"]["resolved"], "profile-measured")
        self.assertFalse(result["local"]["calibration"]["enabled"])
        self.assertFalse(result["heroku"]["calibration"]["enabled"])
        with self.assertRaisesRegex(ValueError, "available on both hosts"):
            MODULE.analyze(
                complete,
                incomplete,
                bootstrap_replicates=20,
                bootstrap_seed=7,
                cpu_calibration="abba-control",
            )

    def test_auto_bootstrap_falls_back_and_explicit_incomplete_block_rejects(self):
        local = symbol_document("m4", [1.0, 3.0], [50, 50], [50, 50])
        heroku = symbol_document("heroku", [4.0, 4.0], [25, 25], [75, 75])
        for document in (local, heroku):
            document["rounds"][0]["round"] = 0
            document["profile_rounds"][0]["round"] = 0
        automatic = MODULE.analyze(
            local, heroku, bootstrap_replicates=20, bootstrap_seed=3
        )["bootstrap"]
        self.assertEqual(automatic["requested_unit"], "auto")
        self.assertEqual(automatic["unit"], "round")
        with self.assertRaisesRegex(ValueError, "profiled round ids"):
            MODULE.analyze(
                local,
                heroku,
                bootstrap_replicates=20,
                bootstrap_seed=3,
                bootstrap_unit="abba-block",
            )

    def test_rejects_flat_weights_larger_than_declared_total(self):
        document = {
            "rounds": [
                {
                    "round": 0,
                    "nodes": 10,
                    "process_cpu_seconds": 1.0,
                    "sample_summary": {
                        "total_samples": 2,
                        "flat_buckets": {"nn_forward": 3},
                    },
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "exceed declared sample total"):
            MODULE.parse_host_document(document, "local")

    def test_artifact_directory_adapter_joins_manifest_and_pprof_rounds(self):
        report = """\
File: engine
Type: cpu
Showing nodes accounting for 100ms, 100% of 100ms total
      flat  flat%   sum%        cum   cum%
      60ms 60.00% 60.00%       60ms 60.00%  chess::phase_nnue_detail::VnniNetwork::evaluate
      40ms 40.00%   100%       40ms 40.00%  chess::LowerMoveRangeBucketTranspositionTable::probe
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pprof-rounds").mkdir()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "rounds": [
                            {
                                "round": 0,
                                "mode": "control",
                                "nodes": 100,
                                "cpu_seconds": 0.9,
                            },
                            {
                                "round": 1,
                                "mode": "profile",
                                "nodes": 100,
                                "cpu_seconds": 1.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "pprof-rounds" / "profile-r01-functions.txt").write_text(
                report, encoding="utf-8"
            )
            document = MODULE.load_host_input(root, "heroku")

        parsed = MODULE.parse_host_document(document, "heroku")
        self.assertEqual(len(parsed["rounds"]), 1)
        shares = parsed["rounds"][0]["bucket_sample_shares"]
        self.assertAlmostEqual(shares["nn_forward"], 0.6)
        self.assertAlmostEqual(shares["transposition_table"], 0.4)


if __name__ == "__main__":
    unittest.main()
