from __future__ import annotations

import pytest

from tools.tune import tune_nnue_v41_main_see as tuner


def entry(
    depth: int | None,
    margin: int | None,
    nodes: float,
    wdl: float,
    cp: float = 1.0,
    searched_depth: float = 7.0,
) -> dict:
    config = tuner.Config(depth, margin)
    return {
        "label": config.label,
        "max_depth": depth,
        "margin_per_depth": margin,
        "result": {
            "node_ratio": nodes,
            "mean_wdl_loss": wdl,
            "mean_root_regret": cp,
            "candidate_mean_depth": searched_depth,
        },
    }


def test_grid_has_baseline_plus_sixteen_unique_candidates() -> None:
    assert len(tuner.CONFIGS) == 17
    assert tuner.CONFIGS[0] == tuner.BASELINE
    assert len(set(tuner.CONFIGS)) == 17
    assert {
        (config.max_depth, config.margin_per_depth)
        for config in tuner.CONFIGS[1:]
    } == {
        (depth, margin)
        for depth in tuner.MAX_DEPTHS
        for margin in tuner.MARGINS_PER_DEPTH
    }


def test_partial_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        tuner.Config(4, None)
    with pytest.raises(ValueError):
        tuner.Config(None, 100)


def test_arguments_keep_fast_and_qsee_fixed() -> None:
    enabled = tuner.Config(4, 100).args()
    disabled = tuner.BASELINE.args()
    assert enabled[:len(tuner.FAST_ARGS)] == list(tuner.FAST_ARGS)
    assert enabled[-5:] == [
        "--enable-main-search-see-pruning",
        "--main-search-see-max-depth", "4",
        "--main-search-see-margin-per-depth", "100",
    ]
    assert "--qsearch-see-threshold" in enabled
    assert disabled[-1] == "--disable-main-search-see-pruning"


def test_fixed_depth_frontier_uses_wdl_and_nodes() -> None:
    safe = entry(3, 75, 0.90, 0.001)
    fast = entry(5, 125, 0.80, 0.002)
    dominated = entry(4, 100, 0.92, 0.003)
    assert tuner.frontier([safe, fast, dominated]) == [safe, fast]


def test_neighbors_are_manhattan_only() -> None:
    selected = tuner.expand_neighbors([entry(4, 100, 0.9, 0.001)])
    assert set(selected) == {
        tuner.Config(4, 100),
        tuner.Config(3, 100),
        tuner.Config(5, 100),
        tuner.Config(4, 75),
        tuner.Config(4, 125),
    }
    assert tuner.Config(3, 75) not in selected


def test_baseline_survives_neighbor_expansion() -> None:
    assert tuner.expand_neighbors([
        entry(None, None, 1.0, 0.0),
    ]) == [tuner.BASELINE]


def test_fixed_time_frontier_rewards_depth_without_ignoring_loss() -> None:
    accurate = entry(3, 75, 0.9, 0.001, cp=1.0, searched_depth=7.0)
    deep = entry(5, 125, 0.8, 0.002, cp=2.0, searched_depth=8.0)
    dominated = entry(4, 100, 0.85, 0.003, cp=3.0, searched_depth=6.0)
    assert tuner.frontier(
        [accurate, deep, dominated], fixed_time=True) == [accurate, deep]
