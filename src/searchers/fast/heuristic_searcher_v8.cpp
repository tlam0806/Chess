#include "heuristic_searcher_v8.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"
#include "zobrist.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
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

void HeuristicSearcherV8::make_null_move(Position& pos) const {
    if (pos.en_passant_square != NoSquare) {
        pos.zobrist_key ^= zobrist::en_passant_file_key(file_of(pos.en_passant_square));
        pos.en_passant_square = NoSquare;
    }

    pos.zobrist_key ^= zobrist::side_key();
    pos.side_to_move = opposite(pos.side_to_move);
}

bool HeuristicSearcherV8::should_stop(SearchContext& context) const {
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

int HeuristicSearcherV8::move_priority(
    Move move,
    const Position& cur,
    Move tt_move,
    bool gives_check
) const {
    if (is_valid_move(tt_move) && move == tt_move) {
        return 1e9;
    }
    if (is_promotion(move)) {
        return 1e9 - 1;
    }
    if (is_capture(move)) {
        return 2 + std::max(0, static_exchange_eval(cur, move));
    }
    if (gives_check) {
        return 2;
    }
    return 0;
}

HeuristicSearcherV8::ScoredMove HeuristicSearcherV8::make_scored_move(
    const Position& pos,
    Move move,
    Move tt_move,
    ScoringMode stage
) const {
    Position next = pos;
    next.make_move(move);
    const bool gives_check = in_check(next, next.side_to_move);

    if (stage == ScoringMode::Quiescence) {
        return ScoredMove{
            move,
            is_promotion(move) ? static_cast<int>(1e9) : static_exchange_eval(pos, move),
            -evaluate_for_side_to_move(next),
            gives_check
        };
    }

    return ScoredMove{
        move,
        move_priority(move, pos, tt_move, gives_check),
        history_table_.get_score(pos, move),
        gives_check
    };
}

std::vector<HeuristicSearcherV8::ScoredMove> HeuristicSearcherV8::ordered_moves(
    const Position& pos,
    Move tt_move
) const {
    std::vector<ScoredMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        ordered.push_back(make_scored_move(pos, move, tt_move));
    }

    std::stable_sort(ordered.begin(), ordered.end(), [](const ScoredMove& lhs, const ScoredMove& rhs) {
        if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
        }
        return lhs.static_score > rhs.static_score;
    });

    return ordered;
}

std::vector<HeuristicSearcherV8::ScoredMove> HeuristicSearcherV8::ordered_noisy_moves(
    const Position& pos
) const {
    std::vector<ScoredMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        if (!is_noisy_move(move)) {
            continue;
        }

        ordered.push_back(make_scored_move(pos, move, Move{}, ScoringMode::Quiescence));
    }

    std::stable_sort(ordered.begin(), ordered.end(), [](const ScoredMove& lhs, const ScoredMove& rhs) {
        if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
        }
        return lhs.static_score > rhs.static_score;
    });

    return ordered;
}

bool HeuristicSearcherV8::should_reduce_late_move(
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

bool HeuristicSearcherV8::can_null_move_prune(
    const Position& pos,
    int depth,
    const SearchContext& context
) const {
    return depth >= NullMoveMinDepth
        && !context.in_null_move
        && !in_check(pos, pos.side_to_move)
        && has_non_pawn_material(pos, pos.side_to_move);
}

int HeuristicSearcherV8::quiescence(
    Position pos,
    int alpha,
    int beta,
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
        return in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
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

        const int score = -quiescence(next, -beta, -alpha, q_depth + 1, context);
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

bool HeuristicSearcherV8::is_quiet_move(const ScoredMove& scored_move) const {
    return !is_capture(scored_move.move)
        && !is_promotion(scored_move.move)
        && !scored_move.gives_check;
}

int HeuristicSearcherV8::negamax(
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
        return quiescence(pos, alpha, beta, 0, context);
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

    const std::vector<ScoredMove> moves = ordered_moves(pos, tt_best_move);

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
        if (should_reduce_late_move(pos, scored_move, depth, static_cast<int>(move_index))) {
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
            if (is_quiet_move(scored_move)) {
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

SearchResult HeuristicSearcherV8::make_fallback_result(const Position& pos) const {
    SearchResult result;
    const std::vector<ScoredMove> moves = ordered_moves(pos);
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

SearchResult HeuristicSearcherV8::search_fixed_depth(
    const Position& pos,
    int depth,
    SearchContext& context
) {
    assert(depth >= 0);

    SearchResult result;
    result.depth = depth;

    if (depth == 0) {
        result.score = quiescence(pos, -Infinity, Infinity, 0, context);
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

    const std::vector<ScoredMove> moves = ordered_moves(pos, tt_best_move);

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

HeuristicSearcherV8::HeuristicSearcherV8(std::size_t tt_mb)
    : tt_(tt_mb) {
}

SearchResult HeuristicSearcherV8::search_best_move(const Position& pos, int depth) {
    SearchContext context;
    return search_fixed_depth(pos, depth, context);
}

SearchResult HeuristicSearcherV8::search_best_move(const Position& pos, const SearchLimits& limits) {
    assert(limits.max_depth >= 0);

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

std::string_view HeuristicSearcherV8::name() const {
    return "heuristic_v8";
}

void HeuristicSearcherV8::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV8::tt_entry_count() const {
    return tt_.entry_count();
}

} // namespace chess
