#include "heuristic_searcher_v7.hpp"

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

struct SearchContext {
    std::uint64_t nodes = 0;
    Clock::time_point deadline{};
    bool has_deadline = false;
    bool stopped = false;
    bool in_null_move = false;
};

struct OrderedMove {
    Move move{};
    int priority = 0;
    int static_score = 0;
};

bool should_stop(SearchContext& context) {
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

void make_null_move(Position& pos) {
    if (pos.en_passant_square != NoSquare) {
        pos.zobrist_key ^= zobrist::en_passant_file_key(file_of(pos.en_passant_square));
        pos.en_passant_square = NoSquare;
    }

    pos.zobrist_key ^= zobrist::side_key();
    pos.side_to_move = opposite(pos.side_to_move);
}

int move_priority(Move move, const Position& cur, const Position& next, Move tt_move) {
    if (is_valid_move(tt_move) && move == tt_move) {
        return 1e9;
    }
    if (is_promotion(move)) {
        return 1e9 - 1;
    }
    if (is_capture(move)) {
        return 2 + std::max(0, static_exchange_eval(cur, move));
    }
    if (in_check(next, next.side_to_move)) {
        return 2;
    }
    return 0;
}

std::vector<OrderedMove> ordered_moves(const Position& pos, Move tt_move = Move{}) {
    std::vector<OrderedMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        Position next = pos;
        next.make_move(move);

        ordered.push_back(OrderedMove{
            move,
            move_priority(move, pos, next, tt_move),
            -evaluate_for_side_to_move(next)
        });
    }

    std::stable_sort(ordered.begin(), ordered.end(), [](const OrderedMove& lhs, const OrderedMove& rhs) {
        if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
        }
        return lhs.static_score > rhs.static_score;
    });

    return ordered;
}

std::vector<OrderedMove> ordered_noisy_moves(const Position& pos) {
    std::vector<OrderedMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        if (!is_noisy_move(move)) {
            continue;
        }

        Position next = pos;
        next.make_move(move);

        ordered.push_back(OrderedMove{
            move,
            is_promotion(move) ? (int) 1e9 : static_exchange_eval(pos, move),
            -evaluate_for_side_to_move(next)
        });
    }

    std::stable_sort(ordered.begin(), ordered.end(), [](const OrderedMove& lhs, const OrderedMove& rhs) {
        if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
        }
        return lhs.static_score > rhs.static_score;
    });

    return ordered;
}

bool should_reduce_late_move(
    const Position& pos,
    const OrderedMove& ordered_move,
    int depth,
    int move_index
) {
    return depth >= LmrMinDepth
        && move_index >= LmrMoveIndex
        && ordered_move.priority == 0
        && !in_check(pos, pos.side_to_move);
}

bool can_null_move_prune(const Position& pos, int depth, const SearchContext& context) {
    return depth >= NullMoveMinDepth
        && !context.in_null_move
        && !in_check(pos, pos.side_to_move)
        && has_non_pawn_material(pos, pos.side_to_move);
}

int quiescence(Position pos, int alpha, int beta, int q_depth, SearchContext& context) {
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

    const std::vector<OrderedMove> moves = ordered_noisy_moves(pos);
    for (const OrderedMove& ordered_move : moves) {
        Position next = pos;
        next.make_move(ordered_move.move);

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

int negamax_impl(
    Position pos,
    int depth,
    int ply,
    int alpha,
    int beta,
    SearchContext& context,
    TranspositionTable& tt
) {
    assert(depth >= 0);
    assert(alpha < beta);
    ++context.nodes;

    if (should_stop(context)) {
        return 0;
    }

    int tt_score = 0;
    Move tt_best_move{};
    if (tt.probe(pos.zobrist_key, depth, alpha, beta, tt_score, tt_best_move)) {
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
        const int null_score = -negamax_impl(
            null_pos,
            null_depth,
            ply + 1,
            -beta,
            -beta + 1,
            context,
            tt);
        context.in_null_move = previous_in_null_move;

        if (context.stopped) {
            return 0;
        }
        if (null_score >= beta) {
            return beta;
        }
    }

    const std::vector<OrderedMove> moves = ordered_moves(pos, tt_best_move);

    if (moves.empty()) {
        return in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0;
    }

    int best_score = -Infinity;
    Move best_move{};
    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const OrderedMove& ordered_move = moves[move_index];

        Position next = pos;
        next.make_move(ordered_move.move);

        int score = -Infinity;
        if (should_reduce_late_move(pos, ordered_move, depth, static_cast<int>(move_index))) {
            const int reduced_depth = std::max(0, depth - 1 - LmrReduction);
            score = -negamax_impl(next, reduced_depth, ply + 1, -alpha - 1, -alpha, context, tt);
            if (context.stopped) {
                return 0;
            }

            if (score > alpha) {
                score = -negamax_impl(next, depth - 1, ply + 1, -beta, -alpha, context, tt);
                if (context.stopped) {
                    return 0;
                }
            }
        } else {
            score = -negamax_impl(next, depth - 1, ply + 1, -beta, -alpha, context, tt);
            if (context.stopped) {
                return 0;
            }
        }

        if (score > best_score) {
            best_score = score;
            best_move = ordered_move.move;
        }

        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            break;
        }
    }

    TTBound bound = TTBound::Exact;
    if (best_score <= alpha_before_search) {
        bound = TTBound::Upper;
    } else if (best_score >= beta_before_search) {
        bound = TTBound::Lower;
    }
    tt.store(pos.zobrist_key, depth, best_score, bound, best_move);

    return best_score;
}

SearchResult make_fallback_result(const Position& pos) {
    SearchResult result;
    const std::vector<OrderedMove> moves = ordered_moves(pos);
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

SearchResult search_fixed_depth(
    const Position& pos,
    int depth,
    TranspositionTable& tt,
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
    if (tt.probe(pos.zobrist_key, depth, alpha, beta, tt_score, tt_best_move)) {
        result.score = tt_score;
        result.best_move = tt_best_move;
        result.nodes = 1;
        return result;
    }

    const int alpha_before_search = alpha;
    const int beta_before_search = beta;

    const std::vector<OrderedMove> moves = ordered_moves(pos, tt_best_move);

    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }

    result.score = -Infinity;
    for (const OrderedMove& ordered_move : moves) {
        Position next = pos;
        next.make_move(ordered_move.move);

        const int score = -negamax_impl(next, depth - 1, 1, -beta, -alpha, context, tt);
        if (context.stopped) {
            result.stopped = true;
            result.nodes = context.nodes;
            return result;
        }
        if (score > result.score) {
            result.score = score;
            result.best_move = ordered_move.move;
        }
        alpha = std::max(alpha, score);
    }

    TTBound bound = TTBound::Exact;
    if (result.score <= alpha_before_search) {
        bound = TTBound::Upper;
    } else if (result.score >= beta_before_search) {
        bound = TTBound::Lower;
    }
    tt.store(pos.zobrist_key, depth, result.score, bound, result.best_move);

    result.nodes = context.nodes;
    return result;
}

} // namespace

HeuristicSearcherV7::HeuristicSearcherV7(std::size_t tt_mb)
    : tt_(tt_mb) {
}

SearchResult HeuristicSearcherV7::search_best_move(const Position& pos, int depth) {
    SearchContext context;
    return search_fixed_depth(pos, depth, tt_, context);
}

SearchResult HeuristicSearcherV7::search_best_move(const Position& pos, const SearchLimits& limits) {
    assert(limits.max_depth >= 0);

    SearchResult best = make_fallback_result(pos);
    best.depth = 0;

    SearchContext context;
    context.has_deadline = limits.move_time.count() > 0;
    if (context.has_deadline) {
        context.deadline = Clock::now() + limits.move_time;
    }

    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        SearchResult current = search_fixed_depth(pos, depth, tt_, context);
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

std::string_view HeuristicSearcherV7::name() const {
    return "heuristic_v7";
}

void HeuristicSearcherV7::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV7::tt_entry_count() const {
    return tt_.entry_count();
}

} // namespace chess
