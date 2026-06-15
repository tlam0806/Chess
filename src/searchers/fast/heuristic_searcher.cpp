#include "heuristic_searcher.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <vector>

namespace chess {

namespace {

using Clock = std::chrono::steady_clock;

struct SearchContext {
    std::uint64_t nodes = 0;
    Clock::time_point deadline{};
    bool has_deadline = false;
    bool stopped = false;
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

int negamax_impl(Position pos, int depth, int ply, int alpha, int beta, SearchContext& context) {
    assert(depth >= 0);
    assert(alpha < beta);
    ++context.nodes;

    if (should_stop(context)) {
        return 0;
    }

    const std::vector<Move> moves = generate_legal_moves(pos);

    if (moves.empty()) {
        return in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0;
    }

    if (depth == 0) {
        return evaluate_for_side_to_move(pos);
    }

    int best_score = -Infinity;
    for (Move move : moves) {
        Position next = pos;
        next.make_move(move);

        const int score = -negamax_impl(next, depth - 1, ply + 1, -beta, -alpha, context);
        if (context.stopped) {
            return 0;
        }
        best_score = std::max(best_score, score);

        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            break;
        }
    }

    return best_score;
}

SearchResult make_fallback_result(const Position& pos) {
    SearchResult result;
    const std::vector<Move> moves = generate_legal_moves(pos);
    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }
    result.best_move = moves.front();
    result.score = evaluate_for_side_to_move(pos);
    result.nodes = 1;
    return result;
}

SearchResult search_fixed_depth(const Position& pos, int depth, SearchContext& context) {
    assert(depth >= 0);

    SearchResult result;
    result.depth = depth;

    if (depth == 0) {
        result.score = evaluate_for_side_to_move(pos);
        result.nodes = 1;
        return result;
    }

    const std::vector<Move> moves = generate_legal_moves(pos);

    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }

    result.score = -Infinity;
    int alpha = -Infinity;
    for (Move move : moves) {
        Position next = pos;
        next.make_move(move);

        const int score = -negamax_impl(next, depth - 1, 1, -Infinity, -alpha, context);
        if (context.stopped) {
            result.stopped = true;
            result.nodes = context.nodes;
            return result;
        }
        if (score > result.score) {
            result.score = score;
            result.best_move = move;
        }
        alpha = std::max(alpha, score);
    }

    result.nodes = context.nodes;
    return result;
}

} // namespace

SearchResult HeuristicSearcher::search_best_move(const Position& pos, int depth) {
    SearchContext context;
    return search_fixed_depth(pos, depth, context);
}

SearchResult HeuristicSearcher::search_best_move(const Position& pos, const SearchLimits& limits) {
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

std::string_view HeuristicSearcher::name() const {
    return "heuristic";
}

} // namespace chess
