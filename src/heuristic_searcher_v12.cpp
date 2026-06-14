#include "heuristic_searcher_v12.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"
#include "zobrist.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <memory>
#include <vector>

namespace chess {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int MaxQuiescenceDepth = 16;
constexpr int LmrMinDepth = 3;
constexpr int LmrMoveIndex = 4;
constexpr int LmrReduction = 1;
constexpr int NullMoveMinDepth = 3;
constexpr int NullMoveReduction = 2;

bool is_valid_move(Move move) {
    return move.value != 0;
}

bool is_promotion(Move move) {
    return promotion_piece(move) != PieceType::None;
}

bool is_noisy_move(Move move) {
    return is_capture(move) || is_promotion(move);
}

bool has_non_pawn_material(const Position& pos, Color color) {
    const int color_index = static_cast<int>(color);
    for (PieceType piece :
         {PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen}) {
        if (pos.pieces[color_index][static_cast<int>(piece)] != EmptyBB) {
            return true;
        }
    }
    return false;
}

} // namespace

void HeuristicSearcherV12::make_null_move(Position& pos) const {
    if (pos.en_passant_square != NoSquare) {
        pos.zobrist_key ^= zobrist::en_passant_file_key(file_of(pos.en_passant_square));
        pos.en_passant_square = NoSquare;
    }

    pos.zobrist_key ^= zobrist::side_key();
    pos.side_to_move = opposite(pos.side_to_move);
}

bool HeuristicSearcherV12::should_stop(SearchContext& context) const {
    if (!context.has_deadline) {
        return false;
    }
    if ((context.nodes & 1023ULL) != 0) {
        return false;
    }
    if (Clock::now() >= context.deadline) {
        context.stopped = true;
        return true;
    }
    return false;
}

int HeuristicSearcherV12::move_order_score(
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
    const int weighted_see = weights_.see_weight * see_score;
    if (stage == ScoringMode::Quiescence) {
        return promotion ? weights_.promotion_bonus : weighted_see;
    }

    if (is_valid_move(tt_move) && move == tt_move) {
        return weights_.tt_bonus;
    }

    int score = history_table_.get_score(pos, move);
    if (promotion) {
        return weights_.promotion_bonus + score + weighted_see;
    }
    if (capture) {
        if (see_score >= 0) {
            return weights_.good_capture_bonus + score + weighted_see;
        }
        return weights_.bad_capture_bonus + score + weighted_see;
    }
    if (gives_check) {
        score += weights_.check_bonus;
    }

    const int killer_score = killer_table_.score(ply, move);
    if (killer_score == 2) {
        score += weights_.killer1_bonus;
    } else if (killer_score == 1) {
        score += weights_.killer2_bonus;
    }

    return score;
}

HeuristicSearcherV12::ScoredMove HeuristicSearcherV12::make_scored_move(
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

bool HeuristicSearcherV12::better_scored_move(
    const ScoredMove& lhs,
    const ScoredMove& rhs
) const {
    return lhs.order_score > rhs.order_score;
}

std::vector<HeuristicSearcherV12::ScoredMove> HeuristicSearcherV12::ordered_moves(
    const Position& pos,
    int ply,
    Move tt_move
) const {
    std::vector<ScoredMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        ordered.push_back(make_scored_move(pos, move, ply, tt_move));
    }

    std::stable_sort(ordered.begin(), ordered.end(), [this](const ScoredMove& lhs, const ScoredMove& rhs) {
        return better_scored_move(lhs, rhs);
    });

    return ordered;
}

std::vector<HeuristicSearcherV12::ScoredMove> HeuristicSearcherV12::ordered_noisy_moves(
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

bool HeuristicSearcherV12::should_reduce_late_move(
    const Position& pos,
    const ScoredMove& scored_move,
    int depth,
    int move_index
) const {
    return depth >= LmrMinDepth
        && move_index >= LmrMoveIndex
        && is_quiet_move(scored_move)
        && !in_check(pos, pos.side_to_move);
}

bool HeuristicSearcherV12::can_null_move_prune(
    const Position& pos,
    int depth,
    const SearchContext& context
) const {
    return depth >= NullMoveMinDepth
        && !context.in_null_move
        && !in_check(pos, pos.side_to_move)
        && has_non_pawn_material(pos, pos.side_to_move);
}

int HeuristicSearcherV12::quiescence(
    Position pos,
    int alpha,
    int beta,
    int ply,
    int q_depth,
    SearchContext& context
) {
    assert(alpha < beta);
    ++context.nodes;

    if (should_stop(context)) {
        return 0;
    }

    const std::vector<Move> legal_moves = generate_legal_moves(pos);
    if (legal_moves.empty()) {
        return in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0;
    }

    const int stand_pat = evaluate_for_side_to_move(pos);
    if (stand_pat >= beta) {
        return beta;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    if (q_depth >= MaxQuiescenceDepth) {
        return alpha;
    }

    const std::vector<ScoredMove> moves = ordered_noisy_moves(pos);
    for (const ScoredMove& scored_move : moves) {
        Position next = pos;
        next.make_move(scored_move.move);

        const int score = -quiescence(next, -beta, -alpha, ply + 1, q_depth + 1, context);
        if (context.stopped) {
            return 0;
        }

        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    return alpha;
}

bool HeuristicSearcherV12::is_quiet_move(const ScoredMove& scored_move) const {
    return !scored_move.capture
        && !scored_move.promotion
        && !scored_move.gives_check;
}

void HeuristicSearcherV12::record_beta_cutoff(
    const ScoredMove& scored_move,
    int ply,
    std::size_t move_index,
    bool lmr,
    bool pvs_scout
) {
    ++move_ordering_stats_.beta_cutoffs;
    const std::size_t bucket = move_index < 8 ? move_index : 8;
    ++move_ordering_stats_.cutoff_index_buckets[bucket];

    if (scored_move.capture) {
        ++move_ordering_stats_.cutoff_by_capture;
    }
    if (scored_move.gives_check) {
        ++move_ordering_stats_.cutoff_by_check;
    }
    if (scored_move.promotion) {
        ++move_ordering_stats_.cutoff_by_promotion;
    }
    if (is_quiet_move(scored_move)) {
        ++move_ordering_stats_.cutoff_by_quiet;
    }

    const int killer_score = killer_table_.score(ply, scored_move.move);
    if (killer_score == 2) {
        ++move_ordering_stats_.cutoff_by_killer1;
    } else if (killer_score == 1) {
        ++move_ordering_stats_.cutoff_by_killer2;
    }
    if (lmr) {
        ++move_ordering_stats_.cutoff_by_lmr;
    }
    if (pvs_scout) {
        ++move_ordering_stats_.cutoff_by_pvs_scout;
    }
}

int HeuristicSearcherV12::negamax(
    Position pos,
    int depth,
    int ply,
    int alpha,
    int beta,
    SearchContext& context
) {
    assert(depth >= 0);
    assert(alpha < beta);
    ++context.nodes;

    if (should_stop(context)) {
        return 0;
    }

    int tt_score = 0;
    Move tt_best_move{};
    if (tt_.probe(pos.zobrist_key, depth, alpha, beta, tt_score, tt_best_move)) {
        return tt_score;
    }

    const int alpha_before_search = alpha;
    const int beta_before_search = beta;

    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply, 0, context);
    }

    if (can_null_move_prune(pos, depth, context)) {
        Position null_pos = pos;
        make_null_move(null_pos);

        const bool previous_in_null_move = context.in_null_move;
        context.in_null_move = true;
        const int null_depth = std::max(0, depth - 1 - NullMoveReduction);
        const int null_score = -negamax(
            null_pos,
            null_depth,
            ply + 1,
            -beta,
            -beta + 1,
            context);
        context.in_null_move = previous_in_null_move;

        if (context.stopped) {
            return 0;
        }
        if (null_score >= beta) {
            return beta;
        }
    }

    const std::vector<ScoredMove> moves = ordered_moves(pos, ply, tt_best_move);

    if (moves.empty()) {
        return in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0;
    }

    int best_score = -Infinity;
    Move best_move{};
    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];

        Position next = pos;
        next.make_move(scored_move.move);

        int score = -Infinity;
        bool lmr_search = false;
        bool pvs_scout_cutoff = false;
        if (should_reduce_late_move(pos, scored_move, depth, static_cast<int>(move_index))) {
            lmr_search = true;
            const int reduced_depth = std::max(0, depth - 1 - LmrReduction);
            score = -negamax(next, reduced_depth, ply + 1, -alpha - 1, -alpha, context);
            if (context.stopped) {
                return 0;
            }

            if (score > alpha) {
                score = -negamax(next, depth - 1, ply + 1, -beta, -alpha, context);
                if (context.stopped) {
                    return 0;
                }
            }
        } else if (move_index > 0) {
            std::unique_ptr<HeuristicSearcherV12> shadow_searcher;
            if (pvs_shadow_enabled_
                && pvs_shadow_stats_.fail_low_samples < pvs_shadow_max_fail_low_samples_) {
                shadow_searcher = std::make_unique<HeuristicSearcherV12>(*this);
                shadow_searcher->pvs_shadow_enabled_ = false;
                shadow_searcher->clear_move_ordering_stats();
                shadow_searcher->clear_pvs_shadow_stats();
            }

            const std::uint64_t scout_nodes_before = context.nodes;
            ++move_ordering_stats_.pvs_scouts;
            score = -negamax(next, depth - 1, ply + 1, -alpha - 1, -alpha, context);
            const std::uint64_t scout_nodes = context.nodes - scout_nodes_before;
            if (context.stopped) {
                return 0;
            }
            if (alpha < score && score < beta) {
                ++move_ordering_stats_.pvs_researches;
                score = -negamax(next, depth - 1, ply + 1, -beta, -alpha, context);
                if (context.stopped) {
                    return 0;
                }
            } else if (score >= beta) {
                pvs_scout_cutoff = true;
            } else if (shadow_searcher
                && pvs_shadow_stats_.fail_low_samples < pvs_shadow_max_fail_low_samples_) {
                SearchContext shadow_context;
                const int shadow_score = -shadow_searcher->negamax(
                    next,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    shadow_context);
                (void) shadow_score;
                ++pvs_shadow_stats_.fail_low_samples;
                pvs_shadow_stats_.scout_nodes += scout_nodes;
                pvs_shadow_stats_.shadow_full_nodes += shadow_context.nodes;
                const std::size_t depth_bucket = depth < static_cast<int>(pvs_shadow_stats_.fail_low_by_depth.size())
                    ? static_cast<std::size_t>(depth)
                    : pvs_shadow_stats_.fail_low_by_depth.size() - 1;
                ++pvs_shadow_stats_.fail_low_by_depth[depth_bucket];
                pvs_shadow_stats_.scout_nodes_by_depth[depth_bucket] += scout_nodes;
                pvs_shadow_stats_.shadow_full_nodes_by_depth[depth_bucket] += shadow_context.nodes;
            }
        } else {
            score = -negamax(next, depth - 1, ply + 1, -beta, -alpha, context);
            if (context.stopped) {
                return 0;
            }
        }

        if (score > best_score) {
            best_score = score;
            best_move = scored_move.move;
        }

        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            record_beta_cutoff(scored_move, ply, move_index, lmr_search, pvs_scout_cutoff);
            if (is_quiet_move(scored_move)) {
                killer_table_.store(ply, scored_move.move);
                history_table_.store(pos, scored_move.move, depth);
            }
            break;
        }
    }

    TTBound bound = TTBound::Exact;
    if (best_score <= alpha_before_search) {
        bound = TTBound::Upper;
    } else if (best_score >= beta_before_search) {
        bound = TTBound::Lower;
    }
    tt_.store(pos.zobrist_key, depth, best_score, bound, best_move);

    return best_score;
}

SearchResult HeuristicSearcherV12::make_fallback_result(const Position& pos) const {
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

SearchResult HeuristicSearcherV12::search_fixed_depth(
    const Position& pos,
    int depth,
    SearchContext& context
) {
    assert(depth >= 0);

    SearchResult result;
    result.depth = depth;

    if (depth == 0) {
        result.score = quiescence(pos, -Infinity, Infinity, 0, 0, context);
        result.nodes = context.nodes;
        return result;
    }

    int alpha = -Infinity;
    int beta = Infinity;
    int tt_score = 0;
    Move tt_best_move{};
    if (tt_.probe(pos.zobrist_key, depth, alpha, beta, tt_score, tt_best_move)) {
        result.score = tt_score;
        result.best_move = tt_best_move;
        result.nodes = 1;
        return result;
    }

    const int alpha_before_search = alpha;
    const int beta_before_search = beta;

    const std::vector<ScoredMove> moves = ordered_moves(pos, 0, tt_best_move);

    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }

    result.score = -Infinity;
    for (const ScoredMove& scored_move : moves) {
        Position next = pos;
        next.make_move(scored_move.move);

        const int score = -negamax(next, depth - 1, 1, -beta, -alpha, context);
        if (context.stopped) {
            result.stopped = true;
            result.nodes = context.nodes;
            return result;
        }
        if (score > result.score) {
            result.score = score;
            result.best_move = scored_move.move;
        }
        alpha = std::max(alpha, score);
    }

    TTBound bound = TTBound::Exact;
    if (result.score <= alpha_before_search) {
        bound = TTBound::Upper;
    } else if (result.score >= beta_before_search) {
        bound = TTBound::Lower;
    }
    tt_.store(pos.zobrist_key, depth, result.score, bound, result.best_move);

    result.nodes = context.nodes;
    return result;
}

HeuristicSearcherV12::HeuristicSearcherV12(
    std::size_t tt_mb,
    V12MoveOrderWeights weights
)
    : tt_(tt_mb),
      weights_(weights) {
}

SearchResult HeuristicSearcherV12::search_best_move(const Position& pos, int depth) {
    killer_table_.clear();
    SearchContext context;
    return search_fixed_depth(pos, depth, context);
}

SearchResult HeuristicSearcherV12::search_best_move(const Position& pos, const SearchLimits& limits) {
    assert(limits.max_depth >= 0);

    killer_table_.clear();
    SearchResult best = make_fallback_result(pos);
    best.depth = 0;

    SearchContext context;
    context.has_deadline = limits.move_time.count() > 0;
    if (context.has_deadline) {
        context.deadline = Clock::now() + limits.move_time;
    }

    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        SearchResult current = search_fixed_depth(pos, depth, context);
        if (current.stopped || context.stopped) {
            best.stopped = true;
            best.nodes = context.nodes;
            return best;
        }
        best = current;
    }

    best.nodes = context.nodes;
    return best;
}

std::string_view HeuristicSearcherV12::name() const {
    return "heuristic_v12";
}

void HeuristicSearcherV12::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV12::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV12::clear_tt_stats() {
    tt_.clear_stats();
}

const TranspositionTableStats& HeuristicSearcherV12::tt_stats() const {
    return tt_.stats();
}

void HeuristicSearcherV12::clear_move_ordering_stats() {
    move_ordering_stats_ = {};
}

const HeuristicSearcherV12::MoveOrderingStats& HeuristicSearcherV12::move_ordering_stats() const {
    return move_ordering_stats_;
}

void HeuristicSearcherV12::enable_pvs_shadow_measurement(std::uint64_t max_fail_low_samples) {
    pvs_shadow_enabled_ = max_fail_low_samples > 0;
    pvs_shadow_max_fail_low_samples_ = max_fail_low_samples;
    clear_pvs_shadow_stats();
}

void HeuristicSearcherV12::clear_pvs_shadow_stats() {
    pvs_shadow_stats_ = {};
}

const HeuristicSearcherV12::PvsShadowStats& HeuristicSearcherV12::pvs_shadow_stats() const {
    return pvs_shadow_stats_;
}

} // namespace chess
