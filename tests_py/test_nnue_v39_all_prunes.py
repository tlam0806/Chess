import random

from tools import tune_nnue_v39_all_prunes as tuner


def entry(config: tuner.Config, nodes: float, cp: float, wdl: float) -> dict:
    return {
        "config_obj": config,
        "result": {
            "node_ratio": nodes,
            "mean_root_regret": cp,
            "mean_wdl_loss": wdl,
            "p95_root_regret": cp,
            "p95_wdl_loss": wdl,
            "critical_mistakes": 0,
            "ranking_move_agreement_pct": 50.0,
        },
    }


def test_initial_configs_preserve_proven_fast_and_control() -> None:
    configs = tuner.initial_configs()
    assert tuner.FAST in configs
    assert tuner.BALANCED in configs
    assert tuner.BASELINE7 in configs
    assert not tuner.BASELINE7.enable_reverse_futility
    assert not tuner.BASELINE7.enable_late_move_pruning


def test_all_mutation_modes_stay_on_declared_grids() -> None:
    for mode in ("single", "pair", "multi"):
        for seed in range(100):
            config = tuner.mutate(tuner.FAST, mode, random.Random(seed))
            assert config != tuner.FAST
            assert config.enable_reverse_futility
            assert config.enable_late_move_pruning
            for name, grid in tuner.GRIDS.items():
                assert getattr(config, name) in grid
            if config.reverse_futility_max_depth >= 3:
                assert config.reverse_futility_base_margin >= 300
                assert config.reverse_futility_margin_per_depth >= 250
            if config.late_move_pruning_max_depth >= 4:
                assert config.late_move_pruning_base >= 8
                assert config.late_move_pruning_depth_multiplier >= 4


def test_crossover_inherits_whole_blocks_and_enables_all_prunes() -> None:
    child = tuner.crossover(
        [tuner.FAST, tuner.BALANCED], random.Random(4))
    assert child.enable_reverse_futility
    assert child.enable_late_move_pruning
    for block in tuner.BLOCKS.values():
        values = tuple(getattr(child, name) for name in block)
        assert values in {
            tuple(getattr(tuner.FAST, name) for name in block),
            tuple(getattr(tuner.BALANCED, name) for name in block),
        }


def test_fixed_time_frontier_does_not_reward_lower_node_count() -> None:
    high_nodes = entry(tuner.FAST, 0.9, 4.0, 0.002)
    low_nodes = entry(tuner.BALANCED, 0.1, 5.0, 0.003)
    assert tuner.frontier([high_nodes, low_nodes], fixed_time=True) == [
        high_nodes]
    assert tuner.frontier([high_nodes, low_nodes], fixed_time=False) == [
        high_nodes, low_nodes]
