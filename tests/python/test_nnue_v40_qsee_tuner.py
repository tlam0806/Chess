from __future__ import annotations

from tools.tune import tune_nnue_v40_qsee as tuner


def entry(threshold: int | None, nodes: float, wdl: float,
          cp: float = 1.0, depth: float = 7.0) -> dict:
    return {
        "label": tuner.Config(threshold).label,
        "threshold": threshold,
        "result": {
            "node_ratio": nodes,
            "mean_wdl_loss": wdl,
            "mean_root_regret": cp,
            "candidate_mean_depth": depth,
        },
    }


def test_threshold_grid_is_exhaustive_and_ordered() -> None:
    assert tuner.THRESHOLDS[0] is None
    numeric = tuner.THRESHOLDS[1:]
    assert tuple(sorted(numeric)) == numeric
    assert numeric[0] == -600
    assert numeric[-1] == 0
    assert len(set(tuner.THRESHOLDS)) == len(tuner.THRESHOLDS)


def test_qsee_arguments_keep_fast_fixed() -> None:
    enabled = tuner.Config(-200).args()
    disabled = tuner.Config(None).args()
    assert enabled[:len(tuner.FAST_ARGS)] == list(tuner.FAST_ARGS)
    assert enabled[-3:] == [
        "--enable-qsearch-see-pruning", "--qsearch-see-threshold", "-200"]
    assert disabled[-1] == "--disable-qsearch-see-pruning"


def test_fixed_depth_frontier_uses_wdl_and_nodes() -> None:
    safe = entry(-400, 0.90, 0.001)
    fast = entry(-100, 0.80, 0.002)
    dominated = entry(-200, 0.92, 0.003)
    assert tuner.frontier([safe, fast, dominated]) == [safe, fast]


def test_frontier_neighbors_are_retained() -> None:
    selected = tuner.expand_neighbors([entry(-200, 0.9, 0.001)])
    assert [config.threshold for config in selected] == [-225, -200, -175]


def test_fixed_time_frontier_rewards_depth_without_ignoring_loss() -> None:
    accurate = entry(-300, 0.9, 0.001, cp=1.0, depth=7.0)
    deep = entry(-100, 0.8, 0.002, cp=2.0, depth=8.0)
    dominated = entry(-200, 0.85, 0.003, cp=3.0, depth=6.0)
    assert tuner.frontier(
        [accurate, deep, dominated], fixed_time=True) == [accurate, deep]
