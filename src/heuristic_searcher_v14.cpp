#include "heuristic_searcher_v14.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <vector>

namespace chess {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int MaxQuiescenceDepth = 16;
constexpr int MateScoreThreshold = CheckmateScore - 1024;

constexpr bool EnablePvs = false;
constexpr bool EnablePvsSkipFailLow = false;
constexpr bool EnableAspirationWindow = true;
constexpr bool EnableTt = true;
constexpr bool EnableTtBounds = true;
constexpr bool EnableTtExactStore = true;
constexpr bool EnableTtMateScores = false;
constexpr bool EnableTtBoundTighten = true;

constexpr int TtBonus = 1'000'000'000;
constexpr int PromotionBonus = 900'000'000;
constexpr int GoodCaptureBonus = 50'000'000;
constexpr int BadCaptureBonus = 30'000;
constexpr int SeeWeight = 10;
constexpr int Killer1Bonus = 40'000;
constexpr int Killer2Bonus = 10'000;
constexpr int CheckBonus = 160'000;

bool is_valid_move(Move move) {
    return move.value != 0;
}

bool is_promotion(Move move) {
    return promotion_piece(move) != PieceType::None;
}

bool is_noisy_move(Move move) {
    return is_capture(move) || is_promotion(move);
}

TTBound negated_bound(TTBound bound) {
    if (bound == TTBound::Lower) {
        return TTBound::Upper;
    }
    if (bound == TTBound::Upper) {
        return TTBound::Lower;
    }
    return TTBound::Exact;
}

} // namespace

struct HeuristicSearcherV14::SearchState {
    std::uint64_t nodes = 0;
    Clock::time_point deadline{};
    bool has_deadline = false;
    bool stopped = false;
    bool in_pvs_scout = false;
    bool tt_use_exact = true;
    bool tt_allow_bound_cutoff = true;
    TTBound root_bound = TTBound::Exact;
};

enum class HeuristicSearcherV14::ScoringMode {
    MainSearch,
    Quiescence
};

struct HeuristicSearcherV14::ScoredMove {
    Move move{};
    int order_score = 0;
    bool gives_check = false;
    bool capture = false;
    bool promotion = false;
};

struct HeuristicSearcherV14::SearchValue {
    int score = 0;
    TTBound bound = TTBound::Exact;
};

struct HeuristicSearcherV14::TTProbeResult {
    int score = 0;
    TTBound bound = TTBound::Exact;
    Move best_move{};
    bool hit = false;
};

int HeuristicSearcherV14::score_to_tt(int score, int ply) const {
    if (score >= MateScoreThreshold) {
        return score + ply;
    }
    if (score <= -MateScoreThreshold) {
        return score - ply;
    }
    return score;
}

TTProbeOptions HeuristicSearcherV14::make_tt_probe_options(int ply, const SearchState& state) const {
    return TTProbeOptions{
        .use_exact = state.tt_use_exact,
        .use_bounds = EnableTtBounds,
        .allow_bound_cutoff = state.tt_allow_bound_cutoff,
        .allow_bound_tighten = EnableTtBoundTighten,
        .normalize_mate_scores = true,
        .use_mate_scores = EnableTtMateScores,
        .ply = ply,
    };
}

HeuristicSearcherV14::TTProbeResult HeuristicSearcherV14::probe_tt(
    HashKey key,
    int depth,
    int& alpha,
    int& beta,
    int ply,
    bool allow_probe,
    const SearchState& state
) const {
    TTProbeResult result;
    if (!allow_probe || !EnableTt) {
        return result;
    }

    result.hit = tt_.probe(
        key,
        depth,
        alpha,
        beta,
        result.score,
        result.best_move,
        make_tt_probe_options(ply, state),
        &result.bound
    );
    return result;
}

bool HeuristicSearcherV14::should_store_tt(TTBound bound) const {
    return EnableTt
        && (bound != TTBound::Exact || EnableTtExactStore);
}

bool HeuristicSearcherV14::should_stop(SearchState& state) const {
    if (!state.has_deadline) {
        return false;
    }
    if ((state.nodes & 1023ULL) != 0) {
        return false;
    }
    if (Clock::now() >= state.deadline) {
        state.stopped = true;
        return true;
    }
    return false;
}

int HeuristicSearcherV14::move_order_score(
    Move move,
    const Position& pos,
    int ply,
    Move tt_move,
    bool gives_check,
    bool capture,
    bool promotion,
    int see_score,
    ScoringMode stage
) const {
    const int weighted_see = SeeWeight * see_score;
    if (stage == ScoringMode::Quiescence) {
        return promotion ? PromotionBonus : weighted_see;
    }

    if (is_valid_move(tt_move) && move == tt_move) {
        return TtBonus;
    }

    int score = history_table_.get_score(pos, move);
    if (promotion) {
        return PromotionBonus + score + weighted_see;
    }
    if (capture) {
        if (see_score >= 0) {
            return GoodCaptureBonus + score + weighted_see;
        }
        return BadCaptureBonus + score + weighted_see;
    }
    if (gives_check) {
        score += CheckBonus;
    }

    const int killer_score = killer_table_.score(ply, move);
    if (killer_score == 2) {
        score += Killer1Bonus;
    } else if (killer_score == 1) {
        score += Killer2Bonus;
    }

    return score;
}

HeuristicSearcherV14::ScoredMove HeuristicSearcherV14::make_scored_move(
    const Position& pos,
    Move move,
    int ply,
    Move tt_move,
    ScoringMode stage
) const {
    Position next = pos;
    next.make_move(move);
    const bool gives_check = in_check(next, next.side_to_move);
    const bool capture = is_capture(move);
    const bool promotion = is_promotion(move);
    const int see_score = capture ? static_exchange_eval(pos, move) : 0;

    if (stage == ScoringMode::Quiescence) {
        return ScoredMove{
            move,
            move_order_score(
                move,
                pos,
                ply,
                tt_move,
                gives_check,
                capture,
                promotion,
                see_score,
                stage),
            gives_check,
            capture,
            promotion
        };
    }

    return ScoredMove{
        move,
        move_order_score(
            move,
            pos,
            ply,
            tt_move,
            gives_check,
            capture,
            promotion,
            see_score,
            stage),
        gives_check,
        capture,
        promotion
    };
}

bool HeuristicSearcherV14::better_scored_move(
    const ScoredMove& lhs,
    const ScoredMove& rhs
) const {
    return lhs.order_score > rhs.order_score;
}

std::vector<HeuristicSearcherV14::ScoredMove> HeuristicSearcherV14::ordered_moves(
    const Position& pos,
    int ply,
    Move tt_move
) const {
    std::vector<ScoredMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        ordered.push_back(make_scored_move(pos, move, ply, tt_move, ScoringMode::MainSearch));
    }

    std::stable_sort(ordered.begin(), ordered.end(), [this](const ScoredMove& lhs, const ScoredMove& rhs) {
        return better_scored_move(lhs, rhs);
    });

    return ordered;
}

std::vector<HeuristicSearcherV14::ScoredMove> HeuristicSearcherV14::ordered_noisy_moves(
    const Position& pos
) const {
    std::vector<ScoredMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        if (!is_noisy_move(move)) {
            continue;
        }

        ordered.push_back(make_scored_move(pos, move, 0, Move{}, ScoringMode::Quiescence));
    }

    std::stable_sort(ordered.begin(), ordered.end(), [this](const ScoredMove& lhs, const ScoredMove& rhs) {
        return better_scored_move(lhs, rhs);
    });

    return ordered;
}

HeuristicSearcherV14::SearchValue HeuristicSearcherV14::quiescence(
    Position pos,
    int alpha,
    int beta,
    int ply,
    int q_depth,
    SearchState& state
) {
    assert(alpha < beta);
    const int alpha_original = alpha;
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{0, TTBound::Exact};
    }

    const std::vector<Move> legal_moves = generate_legal_moves(pos);
    if (legal_moves.empty()) {
        return SearchValue{in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0, TTBound::Exact};
    }

    if (in_check(pos, pos.side_to_move)) {
        if (q_depth >= MaxQuiescenceDepth) {
            return SearchValue{evaluate_for_side_to_move(pos), TTBound::Exact};
        }

        int best_score = -Infinity;
        std::vector<ScoredMove> moves;
        moves.reserve(legal_moves.size());
        for (Move move : legal_moves) {
            moves.push_back(make_scored_move(pos, move, ply, Move{}, ScoringMode::Quiescence));
        }
        std::stable_sort(moves.begin(), moves.end(), [this](const ScoredMove& lhs, const ScoredMove& rhs) {
            return better_scored_move(lhs, rhs);
        });

        for (const ScoredMove& scored_move : moves) {
            Position next = pos;
            next.make_move(scored_move.move);

            SearchValue child = quiescence(next, -beta, -alpha, ply + 1, q_depth + 1, state);
            const int score = -child.score;
            if (state.stopped) {
                return SearchValue{0, TTBound::Exact};
            }

            if (score > best_score) {
                best_score = score;
                if (score >= beta) {
                    return SearchValue{score, TTBound::Lower};
                }
                if (score > alpha) {
                    alpha = score;
                }
            }
        }

        return SearchValue{best_score, best_score <= alpha_original ? TTBound::Upper : TTBound::Exact};
    }

    int best_score = evaluate_for_side_to_move(pos);
    if (best_score >= beta) {
        return SearchValue{best_score, TTBound::Lower};
    }
    if (best_score > alpha) {
        alpha = best_score;
    }

    if (q_depth >= MaxQuiescenceDepth) {
        return SearchValue{best_score, best_score <= alpha_original ? TTBound::Upper : TTBound::Exact};
    }

    const std::vector<ScoredMove> moves = ordered_noisy_moves(pos);
    for (const ScoredMove& scored_move : moves) {
        Position next = pos;
        next.make_move(scored_move.move);

        SearchValue child = quiescence(next, -beta, -alpha, ply + 1, q_depth + 1, state);
        const int score = -child.score;
        if (state.stopped) {
            return SearchValue{0, TTBound::Exact};
        }

            if (score > best_score) {
            best_score = score;
            if (score >= beta) {
                return SearchValue{score, TTBound::Lower};
            }
                if (score > alpha) {
                    alpha = score;
            }
        }
    }

    return SearchValue{best_score, best_score <= alpha_original ? TTBound::Upper : TTBound::Exact};
}

bool HeuristicSearcherV14::is_quiet_move(const ScoredMove& scored_move) const {
    return !scored_move.capture
        && !scored_move.promotion
        && !scored_move.gives_check;
}

HeuristicSearcherV14::SearchValue HeuristicSearcherV14::negamax(
    Position pos,
    int depth,
    int ply,
    int alpha,
    int beta,
    SearchState& state
) {
    assert(depth >= 0);
    assert(alpha < beta);
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{0, TTBound::Exact};
    }

    const int alpha_original = alpha;
    const int beta_original = beta;
    TTProbeResult tt_probe = probe_tt(pos.zobrist_key, depth, alpha, beta, ply, true, state);
    if (tt_probe.hit) {
        return SearchValue{tt_probe.score, tt_probe.bound};
    }

    const int alpha_after_tt = alpha;
    const int beta_after_tt = beta;
    const bool alpha_tightened_by_tt = alpha_after_tt > alpha_original;
    const bool beta_tightened_by_tt = beta_after_tt < beta_original;

    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply, 0, state);
    }

    const std::vector<ScoredMove> moves = ordered_moves(pos, ply, tt_probe.best_move);

    if (moves.empty()) {
        return SearchValue{in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0, TTBound::Exact};
    }

    int best_proven_score = -Infinity;
    TTBound best_proven_bound = TTBound::Exact;
    Move best_proven_move{};
    bool has_proven_score = false;

    int best_upper_score = -Infinity;
    Move best_upper_move{};
    bool has_upper_score = false;

    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];

        Position next = pos;
        next.make_move(scored_move.move);

        int score = -Infinity;
        TTBound move_bound = TTBound::Exact;
        if (EnablePvs && move_index > 0 && beta > alpha + 1) {
            const bool previous_in_pvs_scout = state.in_pvs_scout;
            state.in_pvs_scout = true;
            SearchValue scout = negamax(next, depth - 1, ply + 1, -alpha - 1, -alpha, state);
            score = -scout.score;
            move_bound = negated_bound(scout.bound);
            state.in_pvs_scout = previous_in_pvs_scout;
            if (state.stopped) {
                return SearchValue{0, TTBound::Exact};
            }

            const bool scout_proves_fail_low =
                move_bound != TTBound::Lower && score <= alpha;
            const bool scout_proves_fail_high =
                move_bound != TTBound::Upper && score >= beta;
            if (EnablePvsSkipFailLow && scout_proves_fail_high) {
            } else if (!EnablePvsSkipFailLow || !scout_proves_fail_low) {
                SearchValue full = negamax(next, depth - 1, ply + 1, -beta, -alpha, state);
                score = -full.score;
                move_bound = negated_bound(full.bound);
            }
        } else {
            SearchValue child = negamax(next, depth - 1, ply + 1, -beta, -alpha, state);
            score = -child.score;
            move_bound = negated_bound(child.bound);
        }
        if (state.stopped) {
            return SearchValue{0, TTBound::Exact};
        }

        if (move_bound == TTBound::Upper) {
            if (!has_upper_score || score > best_upper_score) {
                best_upper_score = score;
                best_upper_move = scored_move.move;
                has_upper_score = true;
            }
            continue;
        }

        if (!has_proven_score || score > best_proven_score) {
            best_proven_score = score;
            best_proven_bound = move_bound;
            best_proven_move = scored_move.move;
            has_proven_score = true;
        }

        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            if (!state.in_pvs_scout && is_quiet_move(scored_move)) {
                killer_table_.store(ply, scored_move.move);
                history_table_.store(pos, scored_move.move, depth);
            }
            break;
        }
    }

    TTBound bound = TTBound::Exact;
    int node_score = best_upper_score;
    Move best_move = best_upper_move;
    if (has_proven_score) {
        node_score = best_proven_score;
        best_move = best_proven_move;
    }

    if (!has_proven_score) {
        bound = TTBound::Upper;
    } else if (best_proven_bound == TTBound::Lower) {
        bound = TTBound::Lower;
    } else if (has_upper_score && best_upper_score > best_proven_score) {
        bound = TTBound::Lower;
    } else if (node_score <= alpha_original) {
        bound = TTBound::Upper;
    } else if (node_score >= beta_original) {
        bound = TTBound::Lower;
    } else if (alpha_tightened_by_tt && node_score <= alpha_after_tt) {
        bound = TTBound::Upper;
    } else if (beta_tightened_by_tt && node_score >= beta_after_tt) {
        bound = TTBound::Lower;
    }
    if (!state.in_pvs_scout && should_store_tt(bound)) {
        tt_.store(pos.zobrist_key, depth, score_to_tt(node_score, ply), bound, best_move);
    }

    return SearchValue{node_score, bound};
}

SearchResult HeuristicSearcherV14::make_fallback_result(const Position& pos) const {
    SearchResult result;
    const std::vector<ScoredMove> moves = ordered_moves(pos, 0);
    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }
    result.best_move = moves.front().move;
    result.score = evaluate_for_side_to_move(pos);
    result.nodes = 1;
    return result;
}

SearchResult HeuristicSearcherV14::search_fixed_depth(
    const Position& pos,
    int depth,
    SearchState& state,
    int alpha,
    int beta,
    bool allow_root_tt_probe
) {
    assert(depth >= 0);

    SearchResult result;
    result.depth = depth;

    if (depth == 0) {
        result.score = quiescence(pos, -Infinity, Infinity, 0, 0, state).score;
        result.nodes = state.nodes;
        state.root_bound = TTBound::Exact;
        return result;
    }

    const int alpha_original = alpha;
    const int beta_original = beta;
    TTProbeResult tt_probe = probe_tt(pos.zobrist_key, depth, alpha, beta, 0, allow_root_tt_probe, state);
    if (tt_probe.hit) {
        result.score = tt_probe.score;
        result.best_move = tt_probe.best_move;
        result.nodes = 1;
        state.root_bound = tt_probe.bound;
        return result;
    }

    const int alpha_after_tt = alpha;
    const int beta_after_tt = beta;
    const bool alpha_tightened_by_tt = alpha_after_tt > alpha_original;
    const bool beta_tightened_by_tt = beta_after_tt < beta_original;

    const std::vector<ScoredMove> moves = ordered_moves(pos, 0, tt_probe.best_move);

    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        state.root_bound = TTBound::Exact;
        return result;
    }

    int best_proven_score = -Infinity;
    TTBound best_proven_bound = TTBound::Exact;
    Move best_proven_move{};
    bool has_proven_score = false;

    int best_upper_score = -Infinity;
    Move best_upper_move{};
    bool has_upper_score = false;

    for (const ScoredMove& scored_move : moves) {
        Position next = pos;
        next.make_move(scored_move.move);

        SearchValue child = negamax(next, depth - 1, 1, -beta, -alpha, state);
        int score = -child.score;
        TTBound score_bound = negated_bound(child.bound);
        if (state.stopped) {
            result.stopped = true;
            result.nodes = state.nodes;
            return result;
        }

        if (score_bound == TTBound::Upper) {
            if (!has_upper_score || score > best_upper_score) {
                best_upper_score = score;
                best_upper_move = scored_move.move;
                has_upper_score = true;
            }
            continue;
        }

        if (!has_proven_score || score > best_proven_score) {
            best_proven_score = score;
            best_proven_bound = score_bound;
            best_proven_move = scored_move.move;
            has_proven_score = true;
        }

        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) break;
    }

    result.score = has_proven_score ? best_proven_score : best_upper_score;
    result.best_move = has_proven_score ? best_proven_move : best_upper_move;

    TTBound bound = TTBound::Exact;
    if (!has_proven_score) {
        bound = TTBound::Upper;
    } else if (best_proven_bound == TTBound::Lower) {
        bound = TTBound::Lower;
    } else if (has_upper_score && best_upper_score > best_proven_score) {
        bound = TTBound::Lower;
    } else if (result.score <= alpha_original) {
        bound = TTBound::Upper;
    } else if (result.score >= beta_original) {
        bound = TTBound::Lower;
    } else if (alpha_tightened_by_tt && result.score <= alpha_after_tt) {
        bound = TTBound::Upper;
    } else if (beta_tightened_by_tt && result.score >= beta_after_tt) {
        bound = TTBound::Lower;
    }
    if (should_store_tt(bound)) {
        tt_.store(pos.zobrist_key, depth, score_to_tt(result.score, 0), bound, result.best_move);
    }

    result.nodes = state.nodes;
    state.root_bound = bound;
    return result;
}

SearchResult HeuristicSearcherV14::search_root_without_tt_cutoff(
    const Position& pos,
    int depth,
    SearchState& state
) {
    const bool previous_tt_exact = state.tt_use_exact;
    const bool previous_tt_bound_cutoff = state.tt_allow_bound_cutoff;
    state.tt_use_exact = false;
    state.tt_allow_bound_cutoff = false;

    SearchResult result = search_fixed_depth(pos, depth, state, -Infinity, Infinity, false);

    state.tt_use_exact = previous_tt_exact;
    state.tt_allow_bound_cutoff = previous_tt_bound_cutoff;
    return result;
}

HeuristicSearcherV14::HeuristicSearcherV14(std::size_t tt_mb)
    : tt_(tt_mb) {
}

SearchResult HeuristicSearcherV14::search_best_move(const Position& pos, int depth) {
    killer_table_.clear();
    SearchState state;
    return search_fixed_depth(pos, depth, state);
}

SearchResult HeuristicSearcherV14::search_best_move(const Position& pos, const SearchLimits& limits) {
    assert(limits.max_depth >= 0);

    killer_table_.clear();
    SearchResult best = make_fallback_result(pos);
    best.depth = 0;

    SearchState state;
    state.has_deadline = limits.move_time.count() > 0;
    if (state.has_deadline) {
        state.deadline = Clock::now() + limits.move_time;
    }

    const int aspiration_window_cp = 50;
    
    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        if (depth == 1) {
            SearchResult current = search_fixed_depth(pos, depth, state);
            if (current.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return best;
            }
            best = current;
        } else if (EnableAspirationWindow) {
            int alpha = best.score - aspiration_window_cp;
            int beta = best.score + aspiration_window_cp;
            SearchResult current;
            for (;;) {
                current = search_fixed_depth(pos, depth, state, alpha, beta);
                if (current.stopped || state.stopped) {
                    best.stopped = true;
                    best.nodes = state.nodes;
                    return best;
                }
                if (state.root_bound == TTBound::Exact || (alpha == -Infinity && beta == Infinity)) {
                    break;
                }
                if (state.root_bound == TTBound::Upper) {
                    beta = std::min(beta, current.score);
                    alpha = -Infinity;
                } else {
                    alpha = std::max(alpha, current.score);
                    beta = Infinity;
                }
            }
            current = search_root_without_tt_cutoff(pos, depth, state);
            if (current.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return best;
            }
            best = current;
        } else {
            SearchResult current = search_fixed_depth(pos, depth, state);
            if (current.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return best;
            }
            best = current;
        }
    }

    best.nodes = state.nodes;
    return best;
}

std::string_view HeuristicSearcherV14::name() const {
    return "heuristic_v14";
}

void HeuristicSearcherV14::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV14::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV14::clear_tt_stats() {
    tt_.clear_stats();
}

const TranspositionTableStats& HeuristicSearcherV14::tt_stats() const {
    return tt_.stats();
}

} // namespace chess
