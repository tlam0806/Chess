import random

from tools import build_nnue_selective_safety_bank as safety_bank
from tools import tune_nnue_lmr_nmp_adversarial as tuner
from tools import tune_nnue_v39_config7_wdl as v39_wdl_tuner
from tools import tune_nnue_v38_selective as v38_tuner


def test_safety_detail_classification() -> None:
    line = (
        "detail\tindex=1\tcategory=random\thash=abc"
        "\tpayload=8/8/8/8/8/8/8/8 w - - 0 1"
        "\tstatic_target_cp=500\tmetric_group=ranking"
        "\tcontrol_move=a1a2\tcandidate_move=a1b1"
        "\tcontrol_score=600\tcandidate_strict_score=-200"
        "\tregret=800\tcontrol_mate=0\tcandidate_mate=0"
    )
    detail = safety_bank.parse_detail(line)
    assert detail is not None
    assert safety_bank.classify(detail) == "win_to_loss"
    detail["candidate_strict_score"] = 100
    assert safety_bank.classify(detail) == "win_to_draw"
    detail["control_score"] = 300
    detail["candidate_strict_score"] = 0
    assert safety_bank.classify(detail) == "near_miss"


def test_initial_population_contains_all_lineages() -> None:
    anchors = tuner.initial_configs()
    assert anchors[0].lineage == "control"
    assert {config.lineage for config in anchors} == {
        "control", "lmr", "nmp", "joint",
    }


def test_mutations_stay_inside_declared_space() -> None:
    anchors = tuner.initial_configs()
    for lineage in tuner.LINEAGES:
        base = next(
            config for config in anchors if config.lineage == lineage)
        for seed in range(200):
            config = tuner.mutate(
                base, lineage, random.Random(seed))
            assert config.lineage == lineage
            assert 0.0 <= config.lmr_base <= 1.0
            assert 1.8 <= config.lmr_divisor <= 3.5
            assert 3 <= config.lmr_min_depth <= 8
            assert 2 <= config.lmr_min_move_index <= 12
            assert 3 <= config.null_min_depth <= 7
            assert 1 <= config.null_reduction <= 3


def test_hash_partition_is_stable() -> None:
    assert (
        safety_bank.stable_fraction(123, "abc")
        == safety_bank.stable_fraction(123, "abc")
    )


def test_v38_frontier_uses_declared_objective_loss() -> None:
    faster_worse = {"node_ratio": 0.2, "mean_root_regret": 1.0,
                    "objective_loss": 0.03}
    slower_better = {"node_ratio": 0.3, "mean_root_regret": 100.0,
                     "objective_loss": 0.01}
    dominated = {"node_ratio": 0.4, "mean_root_regret": 0.0,
                 "objective_loss": 0.04}
    assert not v38_tuner.dominated(faster_worse, slower_better)
    assert not v38_tuner.dominated(slower_better, faster_worse)
    assert v38_tuner.dominated(dominated, faster_worse)


def test_v38_frontier_deduplicates_equal_objectives() -> None:
    first = {
        "config": v38_tuner.Config().__dict__,
        "result": {
            "node_ratio": 0.3,
            "objective_loss": 0.01,
            "mean_root_regret": 5.0,
        },
    }
    second = {
        "config": v38_tuner.Config(lmr_base=0.6).__dict__,
        "result": {
            "node_ratio": 0.3,
            "objective_loss": 0.01,
            "mean_root_regret": 5.0,
        },
    }
    assert len(v38_tuner.deduplicate_objectives([first, second])) == 1


def test_v38_frontier_can_keep_critical_candidates_as_diagnostics() -> None:
    entry = {
        "config": v38_tuner.Config().__dict__,
        "result": {
            "node_ratio": 0.3,
            "objective_loss": 0.01,
            "mean_root_regret": 5.0,
            "critical_mistakes": 1,
        },
    }
    assert v38_tuner.frontier([entry]) == []
    assert v38_tuner.frontier([entry], allow_critical=True) == [entry]
    assert (
        safety_bank.stable_fraction(123, "abc")
        != safety_bank.stable_fraction(124, "abc")
    )


def test_v39_config7_wdl_anchors_cover_ablation_lineages() -> None:
    anchors = v39_wdl_tuner.initial_configs()
    assert {config.lineage for config in anchors} == {
        "off", "rfp", "lmp", "joint",
    }
    baseline = next(config for config in anchors if config.lineage == "off")
    args = baseline.args()
    assert args[:12] == list(v39_wdl_tuner.BASELINE_ARGS)
    assert "--disable-reverse-futility" in args
    assert "--disable-late-move-pruning" in args


def test_v39_config7_wdl_mutations_stay_in_conservative_space() -> None:
    anchors = v39_wdl_tuner.initial_configs()
    for lineage in v39_wdl_tuner.LINEAGES:
        base = next(config for config in anchors if config.lineage == lineage)
        for seed in range(200):
            config = v39_wdl_tuner.mutate(base, random.Random(seed))
            assert config.lineage == lineage
            assert 1 <= config.reverse_futility_max_depth <= 3
            assert 150 <= config.reverse_futility_base_margin <= 500
            assert 150 <= config.reverse_futility_margin_per_depth <= 400
            assert 2 <= config.late_move_pruning_max_depth <= 4
            assert 4 <= config.late_move_pruning_base <= 16
            assert 2 <= config.late_move_pruning_depth_multiplier <= 8
            if config.reverse_futility_max_depth == 3:
                assert config.reverse_futility_base_margin >= 300
                assert config.reverse_futility_margin_per_depth >= 250
            if config.late_move_pruning_max_depth == 4:
                assert config.late_move_pruning_base >= 8
                assert config.late_move_pruning_depth_multiplier >= 4


def test_v39_config7_wdl_frontier_uses_wdl_and_keeps_critical() -> None:
    fast = {
        "config_obj": v39_wdl_tuner.Config("joint"),
        "result": {
            "node_ratio": 0.10,
            "objective_loss": 0.004,
            "critical_mistakes": 2,
        },
    }
    safe = {
        "config_obj": v39_wdl_tuner.Config("off"),
        "result": {
            "node_ratio": 0.15,
            "objective_loss": 0.003,
            "critical_mistakes": 0,
        },
    }
    dominated = {
        "config_obj": v39_wdl_tuner.Config("rfp"),
        "result": {
            "node_ratio": 0.16,
            "objective_loss": 0.005,
            "critical_mistakes": 0,
        },
    }
    assert v39_wdl_tuner.frontier([fast, safe, dominated]) == [fast, safe]
