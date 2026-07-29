from tools.select_nnue_wdl_balanced_candidate import select_candidate
from tools.summarize_nnue_v38_paired_match import summarize


def entry(node_ratio: float, objective_loss: float, critical: int = 0) -> dict:
    return {
        "config": {
            "lmr_base": 0.5,
            "lmr_divisor": 2.6,
            "lmr_min_depth": 4,
            "lmr_min_move_index": 4,
            "null_min_depth": 3,
            "null_reduction": 2,
        },
        "result": {
            "node_ratio": node_ratio,
            "objective_loss": objective_loss,
            "critical_mistakes": critical,
        },
    }


def test_selects_lowest_loss_inside_old_node_budget() -> None:
    selected = select_candidate(
        [
            entry(0.20, 0.03),
            entry(0.29, 0.02),
            entry(0.31, 0.01),
            entry(0.25, 0.001, critical=1),
        ],
        {"node_ratio": 0.292},
        1.02,
    )
    assert selected["candidate"]["result"]["node_ratio"] == 0.29


def test_summarizes_complete_color_reversed_pairs(tmp_path) -> None:
    games = tmp_path / "games.jsonl"
    games.write_text(
        '{"key":"x:0:w","candidate_outcome":"win"}\n'
        '{"key":"x:0:b","candidate_outcome":"draw"}\n'
        '{"key":"x:1:w","candidate_outcome":"loss"}\n'
        '{"key":"x:1:b","candidate_outcome":"draw"}\n'
    )
    result = summarize(games)
    assert result["games"] == 4
    assert result["complete_pairs"] == 2
    assert result["candidate_score"] == 0.5
