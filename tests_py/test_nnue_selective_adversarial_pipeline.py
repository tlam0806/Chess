import random

from tools import build_nnue_selective_safety_bank as safety_bank
from tools import tune_nnue_lmr_nmp_adversarial as tuner


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
    assert (
        safety_bank.stable_fraction(123, "abc")
        != safety_bank.stable_fraction(124, "abc")
    )
