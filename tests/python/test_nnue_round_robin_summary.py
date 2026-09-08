import json

from tools.analyze.summarize_nnue_round_robin import summarize


def game(key: str, first: str, second: str, outcome: str) -> str:
    return json.dumps(
        {
            "key": key,
            "profile": first,
            "opponent": second,
            "candidate_outcome": outcome,
        }
    )


def test_groups_matchups_and_ignores_incomplete_pairs(tmp_path) -> None:
    games = tmp_path / "games.jsonl"
    games.write_text(
        "\n".join(
            [
                game("fast_vs_balanced:0:w", "fast", "balanced", "win"),
                game("fast_vs_balanced:0:b", "fast", "balanced", "draw"),
                game("fast_vs_balanced:1:w", "fast", "balanced", "loss"),
                game("fast_vs_baseline7:0:w", "fast", "baseline7", "loss"),
                game("fast_vs_baseline7:0:b", "fast", "baseline7", "loss"),
            ]
        )
        + "\n"
    )

    result = summarize(games)

    assert len(result["matchups"]) == 2
    by_second = {item["second"]: item for item in result["matchups"]}
    assert by_second["balanced"]["games"] == 3
    assert by_second["balanced"]["complete_pairs"] == 1
    assert by_second["balanced"]["first_score"] == 0.75
    assert by_second["baseline7"]["complete_pairs"] == 1
    assert by_second["baseline7"]["first_score"] == 0.0
