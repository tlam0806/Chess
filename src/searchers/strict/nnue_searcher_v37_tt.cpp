#include "nnue_searcher_v37.hpp"

#include "nnue_searcher_v37_detail.hpp"

namespace chess {

int NnueSearcherV37::score_to_tt(int score, int ply) const {
    if (score >= MateScoreThreshold) {
        return score + ply;
    }
    if (score <= -MateScoreThreshold) {
        return score - ply;
    }
    return score;
}

NnueSearcherV37::TTProbeResult NnueSearcherV37::probe_tt(
    HashKey key,
    int depth,
    int& alpha,
    int& beta,
    int ply,
    bool allow_probe
) const {
    TTProbeResult result;
    if (!allow_probe || !EnableTt) {
        return result;
    }

    ScoreRange search_range{alpha, beta};
    result.hit = tt_.probe(
        key,
        depth,
        search_range,
        result.range,
        result.moves,
        ply,
        result.has_score,
        TTDepthPolicy::Exact
    );
    alpha = search_range.lower;
    beta = search_range.upper;
    return result;
}

bool NnueSearcherV37::should_store_tt(ScoreRange range) const {
    return EnableTt
        && (range.lower != range.upper || EnableTtExactStore);
}

void NnueSearcherV37::update_best_range(
    ScoreRange& node_range,
    Move& best_lower_move,
    Move& best_upper_move,
    int& alpha,
    Move move,
    ScoreRange move_range
) const {
    if (!is_valid_move(best_lower_move) && move_range.lower != -Infinity) {
        best_lower_move = move;
    }
    if (!is_valid_move(best_upper_move) && move_range.upper != -Infinity) {
        best_upper_move = move;
    }
    if (move_range.lower > node_range.lower) {
        node_range.lower = move_range.lower;
        best_lower_move = move;
    }
    if (move_range.upper > node_range.upper) {
        node_range.upper = move_range.upper;
        best_upper_move = move;
    }
    if (move_range.lower != -Infinity && move_range.lower > alpha) {
        alpha = move_range.lower;
    }
}

void NnueSearcherV37::store_tt_if_needed(
    HashKey key,
    int depth,
    int ply,
    ScoreRange range,
    Move best_lower_move,
    Move best_upper_move,
    Move fallback_best_move
) {
    if (!should_store_tt(range)) {
        return;
    }
    if (!is_valid_move(best_lower_move)) {
        best_lower_move = fallback_best_move;
    }
    if (!is_valid_move(best_upper_move)) {
        best_upper_move = fallback_best_move;
    }

    ScoreRange tt_range = range;
    if (tt_range.lower != -Infinity) {
        tt_range.lower = score_to_tt(tt_range.lower, ply);
    }
    if (tt_range.upper != Infinity) {
        tt_range.upper = score_to_tt(tt_range.upper, ply);
    }
    tt_.store(key, depth, tt_range, MoveRange{best_lower_move, best_upper_move});
}

} // namespace chess
