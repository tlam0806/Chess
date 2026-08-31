#include "nnue_searcher_v43.hpp"

#include "nnue_searcher_v43_detail.hpp"

namespace chess {

NnueSearcherV43::TTProbeResult NnueSearcherV43::probe_tt(
    HashKey key,
    int depth,
    int alpha,
    int beta,
    int ply,
    bool allow_probe,
    bool allow_score
) const {
    TTProbeResult result;
    if (!allow_probe || !EnableTt) {
        return result;
    }

    const auto generation_policy = reuse_stale_tt_scores_
        ? V43SingleBoundTranspositionTable::GenerationPolicy::Any
        : V43SingleBoundTranspositionTable::GenerationPolicy::CurrentOnly;
    const auto depth_policy = reuse_deeper_tt_scores_
        ? V43SingleBoundTranspositionTable::DepthPolicy::AtLeast
        : V43SingleBoundTranspositionTable::DepthPolicy::Exact;
    const auto probe = tt_.probe(
        key,
        depth,
        alpha,
        beta,
        ply,
        depth_policy,
        generation_policy);
    result.move = probe.move;
    result.score = probe.score;
    result.bound = probe.bound;
    result.cutoff = allow_score && probe.cutoff;
    return result;
}

void NnueSearcherV43::update_best_score(
    int& best_score,
    Move& best_move,
    int& alpha,
    Move move,
    int move_score
) const {
    if (!is_valid_move(best_move) || move_score > best_score) {
        best_score = move_score;
        best_move = move;
    }
    alpha = std::max(alpha, move_score);
}

void NnueSearcherV43::store_tt_if_needed(
    HashKey key,
    int depth,
    int ply,
    int score,
    int original_alpha,
    int original_beta,
    Move best_move,
    Move fallback_best_move,
    bool allow_store
) {
    if (!allow_store || !EnableTt) {
        return;
    }
    if (!is_valid_move(best_move)) {
        best_move = fallback_best_move;
    }

    V43SingleBoundTranspositionTable::Bound bound;
    if (score <= original_alpha) {
        bound = V43SingleBoundTranspositionTable::Bound::Upper;
    } else if (score >= original_beta) {
        bound = V43SingleBoundTranspositionTable::Bound::Lower;
    } else {
        bound = V43SingleBoundTranspositionTable::Bound::Exact;
    }
    tt_.store(key, depth, score, bound, best_move, ply);
}

} // namespace chess
