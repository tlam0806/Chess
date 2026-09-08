#include "heuristic_searcher_v20.hpp"

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

constexpr bool EnableAspirationWindow = true;
constexpr bool EnableTt = true;
constexpr bool EnableTtExactStore = true;

bool is_valid_move(Move move) {
    return move.value != 0;
}

bool is_promotion(Move move) {
    return promotion_piece(move) != PieceType::None;
}

int ordering_piece_value(PieceType piece) {
    switch (piece) {
    case PieceType::Pawn:
        return 100;
    case PieceType::Knight:
        return 320;
    case PieceType::Bishop:
        return 330;
    case PieceType::Rook:
        return 500;
    case PieceType::Queen:
        return 900;
    default:
        return 0;
    }
}

ScoreRange exact_range(int score) {
    return ScoreRange{score, score};
}

ScoreRange lower_range(int score) {
    return ScoreRange{score, Infinity};
}

ScoreRange negate_range(ScoreRange range) {
    return ScoreRange{-range.upper, -range.lower};
}

ScoreRange intersect_ranges(ScoreRange lhs, ScoreRange rhs) {
    return ScoreRange{
        std::max(lhs.lower, rhs.lower),
        std::min(lhs.upper, rhs.upper)
    };
}

int representative_score(ScoreRange range) {
    return range.lower != -Infinity ? range.lower : range.upper;
}

enum class WindowStatus {
    Inside,
    Below,
    Above
};

int representative_score_for_status(ScoreRange range, WindowStatus status) {
    if (status == WindowStatus::Below) {
        return range.upper;
    }
    return representative_score(range);
}

WindowStatus status_from_range(ScoreRange range) {
    if (range.lower == range.upper) {
        return WindowStatus::Inside;
    }
    if (range.lower != -Infinity) {
        return WindowStatus::Above;
    }
    return WindowStatus::Below;
}

WindowStatus status_for_window(ScoreRange range, int alpha, int beta) {
    if (range.lower == range.upper) {
        return WindowStatus::Inside;
    }
    if (range.upper <= alpha) {
        return WindowStatus::Below;
    }
    if (range.lower >= beta) {
        return WindowStatus::Above;
    }
    return status_from_range(range);
}

Move preferred_tt_move(MoveRange moves) {
    return moves.lower.value != 0 ? moves.lower : moves.upper;
}

} // namespace

struct HeuristicSearcherV20::SearchState {
    std::uint64_t nodes = 0;
    Clock::time_point deadline{};
    bool has_deadline = false;
    bool stopped = false;
    WindowStatus root_status = WindowStatus::Inside;
};

enum class HeuristicSearcherV20::ScoringMode {
    MainSearch,
    Quiescence
};

struct HeuristicSearcherV20::SearchValue {
    ScoreRange range{};
};

struct HeuristicSearcherV20::TTProbeResult {
    ScoreRange range{};
    MoveRange moves{};
    bool hit = false;
    bool has_score = false;
};

int HeuristicSearcherV20::score_to_tt(int score, int ply) const {
    if (score >= MateScoreThreshold) {
        return score + ply;
    }
    if (score <= -MateScoreThreshold) {
        return score - ply;
    }
    return score;
}

HeuristicSearcherV20::TTProbeResult HeuristicSearcherV20::probe_tt(
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

bool HeuristicSearcherV20::should_store_tt(ScoreRange range) const {
    return EnableTt
        && (range.lower != range.upper || EnableTtExactStore);
}

bool HeuristicSearcherV20::should_stop(SearchState& state) const {
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

PieceType HeuristicSearcherV20::moved_piece_for(const Position& pos, Move move) const {
    return pos.piece_type_on_occupied(pos.side_to_move, from_square(move));
}

PieceType HeuristicSearcherV20::captured_piece_for(const Position& pos, Move move) const {
    const MoveFlag flag = move_flag(move);
    if (!is_capture(flag)) {
        return PieceType::None;
    }
    if (flag == MoveFlag::EnPassant) {
        return PieceType::Pawn;
    }
    return pos.piece_type_on_occupied(opposite(pos.side_to_move), to_square(move));
}

int HeuristicSearcherV20::move_order_score(
    Move move,
    const Position& pos,
    PieceType moved_piece,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    bool gives_check,
    bool capture,
    bool promotion,
    PieceType captured_piece,
    int see_score,
    ScoringMode stage
) const {
    const MoveOrderingWeights& weights = move_ordering_weights_;
    const int weighted_see = weights.see_weight * see_score;
    if (stage == ScoringMode::Quiescence) {
        int score = weights.qsearch_see_weight * see_score;
        if (promotion) {
            score += weights.qsearch_promotion_bonus;
        }
        if (capture) {
            score += weights.qsearch_captured_value_weight * ordering_piece_value(captured_piece);
            score += see_score >= 0
                ? weights.qsearch_good_capture_bonus
                : weights.qsearch_bad_capture_bonus;
        }
        return score;
    }

    if (is_valid_move(tt_moves.lower) && move == tt_moves.lower) {
        return weights.tt_lower_bonus;
    }
    if (is_valid_move(tt_moves.upper) && move == tt_moves.upper) {
        return weights.tt_upper_bonus;
    }

    int score = history_table_.get_score(pos.side_to_move, moved_piece, move);
    if (promotion) {
        return weights.promotion_bonus + score + weighted_see;
    }
    if (capture) {
        if (see_score >= 0) {
            return weights.good_capture_bonus + score + weighted_see;
        }
        return weights.bad_capture_bonus + score + weighted_see;
    }
    if (gives_check) {
        score += weights.check_bonus;
    }

    if (!gives_check
        && is_valid_move(prev_move)
        && prev_moved_piece != PieceType::None) {
        const int counter_history_score = counter_history_table_.get_score(
            pos.side_to_move,
            prev_moved_piece,
            prev_move,
            moved_piece,
            move);
        score += static_cast<int>(
            static_cast<long long>(weights.counter_history_bonus) * counter_history_score
                / CounterHistoryTable::MaxScore
        );
    }

    const int killer_score = killer_table_.score(ply, move);
    if (killer_score == 2) {
        score += weights.killer1_bonus;
    } else if (killer_score == 1) {
        score += weights.killer2_bonus;
    }

    return score;
}

HeuristicSearcherV20::ScoredMove HeuristicSearcherV20::make_scored_move(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Move move,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    ScoringMode stage
) const {
    const PieceType moved_piece = moved_piece_for(pos, move);
    const bool capture = is_capture(move);
    const PieceType captured_piece = captured_piece_for(pos, move);
    if (!is_pseudo_move_legal(pos, king_safety, move, moved_piece, captured_piece)) {
        return ScoredMove{};
    }

    const bool gives_check = stage == ScoringMode::MainSearch
        && gives_check_fast(pos, move, moved_piece, captured_piece);
    const bool promotion = is_promotion(move);
    const int see_score = capture
        ? static_exchange_eval(pos, move, moved_piece, captured_piece)
        : 0;

    if (stage == ScoringMode::Quiescence) {
        return ScoredMove{
            move_order_score(
                move,
                pos,
                moved_piece,
                ply,
                tt_moves,
                prev_move,
                prev_moved_piece,
                gives_check,
                capture,
                promotion,
                captured_piece,
                see_score,
                stage),
            move,
            gives_check,
            capture,
            promotion
        };
    }

    return ScoredMove{
        move_order_score(
            move,
            pos,
            moved_piece,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            gives_check,
            capture,
            promotion,
            captured_piece,
            see_score,
            stage),
        move,
        gives_check,
        capture,
        promotion
    };
}

bool HeuristicSearcherV20::better_scored_move(
    const ScoredMove& lhs,
    const ScoredMove& rhs
) const {
    return lhs.order_score > rhs.order_score;
}

void HeuristicSearcherV20::sort_scored_moves(std::vector<ScoredMove>& moves) const {
    for (std::size_t i = 1; i < moves.size(); ++i) {
        ScoredMove current = moves[i];
        std::size_t j = i;
        while (j > 0 && better_scored_move(current, moves[j - 1])) {
            moves[j] = moves[j - 1];
            --j;
        }
        moves[j] = current;
    }
}

std::vector<HeuristicSearcherV20::ScoredMove> HeuristicSearcherV20::ordered_moves(
    const Position& pos,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece
) const {
    std::vector<ScoredMove> ordered;
    const KingSafetyContext king_safety = make_king_safety_context(pos);
    MoveList moves;
    generate_pseudo_legal_moves(pos, moves);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        ScoredMove scored_move =
            make_scored_move(
                pos,
                king_safety,
                move,
                ply,
                tt_moves,
                prev_move,
                prev_moved_piece,
                ScoringMode::MainSearch);
        if (is_valid_move(scored_move.move)) {
            ordered.push_back(scored_move);
        }
    }

    sort_scored_moves(ordered);

    return ordered;
}

HeuristicSearcherV20::SearchValue HeuristicSearcherV20::quiescence(
    Position pos,
    int alpha,
    int beta,
    int ply,
    int q_depth,
    SearchState& state
) {
    assert(alpha < beta);
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{exact_range(0)};
    }

    const KingSafetyContext king_safety = make_king_safety_context(pos);
    const bool side_in_check = king_safety.checkers != EmptyBB;
    std::vector<ScoredMove> moves;
    bool has_legal_move = false;
    MoveList pseudo_moves;
    if (side_in_check) {
        generate_pseudo_legal_moves(pos, pseudo_moves);
    } else {
        generate_pseudo_noisy_moves(pos, pseudo_moves);
    }
    moves.reserve(pseudo_moves.size());
    for (Move move : pseudo_moves) {
        ScoredMove scored_move =
            make_scored_move(
                pos,
                king_safety,
                move,
                ply,
                MoveRange{},
                Move{},
                PieceType::None,
                ScoringMode::Quiescence);
        if (!is_valid_move(scored_move.move)) {
            continue;
        }
        has_legal_move = true;
        moves.push_back(scored_move);
    }

    if (!has_legal_move && side_in_check) {
        return SearchValue{exact_range(-CheckmateScore + ply)};
    }
    // Non-check quiescence intentionally does not detect stalemate.
    // Root/negamax handle terminal positions when depth remains.

    if (side_in_check) {
        if (q_depth >= MaxQuiescenceDepth) {
            return SearchValue{exact_range(evaluate_for_side_to_move(pos))};
        }

        ScoreRange node_range;
        node_range.lower = -Infinity;
        node_range.upper = -Infinity;
        sort_scored_moves(moves);

        for (const ScoredMove& scored_move : moves) {
            const PieceType moved_piece = moved_piece_for(pos, scored_move.move);
            const PieceType captured_piece = captured_piece_for(pos, scored_move.move);
            Position next = pos;
            next.make_move(scored_move.move, moved_piece, captured_piece);

            SearchValue child = quiescence(next, -beta, -alpha, ply + 1, q_depth + 1, state);
            const ScoreRange move_range = negate_range(child.range);
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }

            node_range.lower = std::max(node_range.lower, move_range.lower);
            node_range.upper = std::max(node_range.upper, move_range.upper);
            if (node_range.lower >= beta) {
                return SearchValue{ScoreRange{node_range.lower, Infinity}};
            }
            if (move_range.lower != -Infinity && move_range.lower > alpha) {
                alpha = move_range.lower;
            }
        }

        return SearchValue{node_range};
    }

    int best_score = evaluate_for_side_to_move(pos);
    if (best_score >= beta) {
        return SearchValue{lower_range(best_score)};
    }
    if (best_score > alpha) {
        alpha = best_score;
    }

    if (q_depth >= MaxQuiescenceDepth) {
        return SearchValue{exact_range(best_score)};
    }

    ScoreRange node_range = exact_range(best_score);
    sort_scored_moves(moves);
    for (const ScoredMove& scored_move : moves) {
        const PieceType moved_piece = moved_piece_for(pos, scored_move.move);
        const PieceType captured_piece = captured_piece_for(pos, scored_move.move);
        Position next = pos;
        next.make_move(scored_move.move, moved_piece, captured_piece);

        SearchValue child = quiescence(next, -beta, -alpha, ply + 1, q_depth + 1, state);
        const ScoreRange move_range = negate_range(child.range);
        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }

        node_range.lower = std::max(node_range.lower, move_range.lower);
        node_range.upper = std::max(node_range.upper, move_range.upper);
        if (node_range.lower >= beta) {
            return SearchValue{ScoreRange{node_range.lower, Infinity}};
        }
        if (move_range.lower != -Infinity && move_range.lower > alpha) {
            alpha = move_range.lower;
        }
    }

    return SearchValue{node_range};
}

bool HeuristicSearcherV20::is_quiet_move(const ScoredMove& scored_move) const {
    return !scored_move.capture
        && !scored_move.promotion
        && !scored_move.gives_check;
}

HeuristicSearcherV20::SearchValue HeuristicSearcherV20::negamax(
    Position pos,
    int depth,
    int ply,
    Move prev_move,
    PieceType prev_moved_piece,
    int alpha,
    int beta,
    SearchState& state
) {
    assert(depth >= 0);
    assert(alpha < beta);
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{exact_range(0)};
    }

    TTProbeResult tt_probe = probe_tt(pos.zobrist_key, depth, alpha, beta, ply, true);
    if (tt_probe.hit) {
        return SearchValue{tt_probe.range};
    }

    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply, 0, state);
    }

    const std::vector<ScoredMove> moves = ordered_moves(pos, ply, tt_probe.moves, prev_move, prev_moved_piece);

    if (moves.empty()) {
        return SearchValue{exact_range(in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0)};
    }

    ScoreRange node_range;
    node_range.lower = -Infinity;
    node_range.upper = -Infinity;
    Move best_lower_move{};
    Move best_upper_move{};
    std::vector<ScoredMove> failed_quiet_moves;

    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];
        const PieceType moved_piece = moved_piece_for(pos, scored_move.move);
        const PieceType captured_piece = captured_piece_for(pos, scored_move.move);

        Position next = pos;
        next.make_move(scored_move.move, moved_piece, captured_piece);

        ScoreRange move_range;
        if (move_index > 0 && beta > alpha + 1) {
            SearchValue scout = negamax(
                next,
                depth - 1,
                ply + 1,
                scored_move.move,
                moved_piece,
                -alpha - 1,
                -alpha,
                state
            );
            move_range = negate_range(scout.range);
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }

            const bool scout_proves_fail_low = move_range.upper <= alpha;
            const bool scout_proves_fail_high = move_range.lower >= beta;
            if (!scout_proves_fail_low && !scout_proves_fail_high) {
                SearchValue full = negamax(
                    next,
                    depth - 1,
                    ply + 1,
                    scored_move.move,
                    moved_piece,
                    -beta,
                    -alpha,
                    state
                );
                move_range = negate_range(full.range);
            }
        } else {
            SearchValue child = negamax(
                next,
                depth - 1,
                ply + 1,
                scored_move.move,
                moved_piece,
                -beta,
                -alpha,
                state
            );
            move_range = negate_range(child.range);
        }
        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }

        if (move_range.lower > node_range.lower) {
            node_range.lower = move_range.lower;
            best_lower_move = scored_move.move;
        }
        if (move_range.upper > node_range.upper) {
            node_range.upper = move_range.upper;
            best_upper_move = scored_move.move;
        }

        if (move_range.lower != -Infinity && move_range.lower > alpha) {
            alpha = move_range.lower;
        }
        if (node_range.lower >= beta) {
            if (is_quiet_move(scored_move)) {
                killer_table_.store(ply, scored_move.move);
                history_table_.store(pos.side_to_move, moved_piece, scored_move.move, depth);
                if (is_valid_move(prev_move) && prev_moved_piece != PieceType::None) {
                    counter_history_table_.store(
                        pos.side_to_move,
                        prev_moved_piece,
                        prev_move,
                        moved_piece,
                        scored_move.move,
                        depth
                    );
                }
            }
            for (const ScoredMove& failed_quiet : failed_quiet_moves) {
                const PieceType failed_moved_piece = moved_piece_for(pos, failed_quiet.move);
                history_table_.penalize(pos.side_to_move, failed_moved_piece, failed_quiet.move, depth);
                if (is_valid_move(prev_move) && prev_moved_piece != PieceType::None) {
                    counter_history_table_.penalize(
                        pos.side_to_move,
                        prev_moved_piece,
                        prev_move,
                        failed_moved_piece,
                        failed_quiet.move,
                        depth
                    );
                }
            }
            node_range.upper = move_index == moves.size() - 1 ? node_range.upper : Infinity;
            break;
        }
        if (is_quiet_move(scored_move)) {
            failed_quiet_moves.push_back(scored_move);
        }
    }
    assert(node_range.lower <= node_range.upper);
    if (tt_probe.has_score) {
        node_range = intersect_ranges(node_range, tt_probe.range);
    }
    assert(node_range.lower <= node_range.upper);

    if (should_store_tt(node_range)) {
        ScoreRange tt_range = node_range;
        if (tt_range.lower != -Infinity) {
            tt_range.lower = score_to_tt(tt_range.lower, ply);
        }
        if (tt_range.upper != Infinity) {
            tt_range.upper = score_to_tt(tt_range.upper, ply);
        }
        tt_.store(pos.zobrist_key, depth, tt_range, MoveRange{best_lower_move, best_upper_move});
    }

    return SearchValue{node_range};
}

SearchResult HeuristicSearcherV20::make_fallback_result(const Position& pos) const {
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

SearchResult HeuristicSearcherV20::search_fixed_depth(
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
        result.score = representative_score(quiescence(pos, -Infinity, Infinity, 0, 0, state).range);
        result.nodes = state.nodes;
        state.root_status = WindowStatus::Inside;
        return result;
    }

    const int alpha_original = alpha;
    const int beta_original = beta;
    TTProbeResult tt_probe = probe_tt(pos.zobrist_key, depth, alpha, beta, 0, allow_root_tt_probe);
    if (tt_probe.hit) {
        state.root_status = status_for_window(tt_probe.range, alpha_original, beta_original);
        result.score = representative_score_for_status(tt_probe.range, state.root_status);
        result.best_move = preferred_tt_move(tt_probe.moves);
        result.nodes = 1;
        return result;
    }

    const std::vector<ScoredMove> moves = ordered_moves(pos, 0, tt_probe.moves);

    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        state.root_status = WindowStatus::Inside;
        return result;
    }

    ScoreRange root_range;
    root_range.lower = -Infinity;
    root_range.upper = -Infinity;
    Move best_lower_move{};
    Move best_upper_move{};

    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];
        const PieceType moved_piece = moved_piece_for(pos, scored_move.move);
        const PieceType captured_piece = captured_piece_for(pos, scored_move.move);

        Position next = pos;
        next.make_move(scored_move.move, moved_piece, captured_piece);

        SearchValue child = negamax(
            next,
            depth - 1,
            1,
            scored_move.move,
            moved_piece,
            -beta,
            -alpha,
            state
        );
        const ScoreRange move_range = negate_range(child.range);
        if (state.stopped) {
            result.stopped = true;
            result.nodes = state.nodes;
            return result;
        }

        if (move_range.lower > root_range.lower) {
            root_range.lower = move_range.lower;
            best_lower_move = scored_move.move;
        }
        if (move_range.upper > root_range.upper) {
            root_range.upper = move_range.upper;
            best_upper_move = scored_move.move;
        }

        if (move_range.lower != -Infinity && move_range.lower > alpha) {
            alpha = move_range.lower;
        }
        if (root_range.lower >= beta) {
            root_range.upper = move_index == moves.size() - 1 ? root_range.upper : Infinity;
            break;
        }
    }

    if (root_range.lower != -Infinity && root_range.upper <= root_range.lower) {
        root_range.upper = root_range.lower;
    }
    if (tt_probe.has_score) {
        root_range = intersect_ranges(root_range, tt_probe.range);
    }
    assert(root_range.lower <= root_range.upper);

    state.root_status = status_for_window(root_range, alpha_original, beta_original);
    result.score = representative_score_for_status(root_range, state.root_status);
    result.best_move = root_range.lower != -Infinity ? best_lower_move : best_upper_move;

    if (should_store_tt(root_range)) {
        ScoreRange tt_range = root_range;
        if (tt_range.lower != -Infinity) {
            tt_range.lower = score_to_tt(tt_range.lower, 0);
        }
        if (tt_range.upper != Infinity) {
            tt_range.upper = score_to_tt(tt_range.upper, 0);
        }
        tt_.store(pos.zobrist_key, depth, tt_range, MoveRange{best_lower_move, best_upper_move});
    }

    result.nodes = state.nodes;
    return result;
}

SearchResult HeuristicSearcherV20::search_root_without_tt_probe(
    const Position& pos,
    int depth,
    SearchState& state
) {
    return search_fixed_depth(pos, depth, state, -Infinity, Infinity, false);
}

HeuristicSearcherV20::HeuristicSearcherV20(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : HeuristicSearcherV20(
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

HeuristicSearcherV20::HeuristicSearcherV20(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : tt_(tt_mb, bucket_size),
      history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      counter_history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      move_ordering_weights_(weights) {
    move_ordering_weights_.counter_history_bonus = counter_history_bonus;
}

SearchResult HeuristicSearcherV20::search_best_move(const Position& pos, int depth) {
    killer_table_.clear();
    SearchState state;
    return search_fixed_depth(pos, depth, state);
}

SearchResult HeuristicSearcherV20::search_best_move(const Position& pos, const SearchLimits& limits) {
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
                if (state.root_status == WindowStatus::Inside || (alpha == -Infinity && beta == Infinity)) {
                    break;
                }
                if (state.root_status == WindowStatus::Below) {
                    beta = std::min(beta, current.score);
                    alpha = -Infinity;
                } else {
                    alpha = std::max(alpha, current.score);
                    beta = Infinity;
                }
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

std::string_view HeuristicSearcherV20::name() const {
    return "heuristic_v20";
}

void HeuristicSearcherV20::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV20::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV20::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& HeuristicSearcherV20::tt_stats() const {
    return tt_.stats();
}

} // namespace chess
