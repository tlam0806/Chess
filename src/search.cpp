#include "search.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

namespace chess {

namespace {

int negamax_impl(Position pos, int depth, int ply, int alpha, int beta, std::uint64_t& nodes) {
    assert(depth >= 0);
    assert(alpha < beta);
    ++nodes;

    const std::vector<Move> moves = generate_legal_moves(pos);

    if (moves.empty()) {
        if (in_check(pos, pos.side_to_move)) {
            return -CheckmateScore + ply;
        }
        return 0;
    }

    if (depth == 0) {
        return evaluate_for_side_to_move(pos);
    }

    int best_score = -Infinity;
    for (Move move : moves) {
        Position next = pos;
        next.make_move(move);

        const int score = -negamax_impl(next, depth - 1, ply + 1, -beta, -alpha, nodes);
        best_score = std::max(best_score, score);

        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            break;
        }
    }

    return best_score;
}

} // namespace

int negamax(Position pos, int depth) {
    return negamax(pos, depth, -Infinity, Infinity);
}

int negamax(Position pos, int depth, int alpha, int beta) {
    std::uint64_t nodes = 0;
    return negamax_impl(pos, depth, 0, alpha, beta, nodes);
}

SearchResult search_best_move(Position pos, int depth) {
    assert(depth >= 0);

    SearchResult result;
    std::uint64_t nodes = 0;

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

        const int score = -negamax_impl(next, depth - 1, 1, -Infinity, -alpha, nodes);
        if (score > result.score) {
            result.score = score;
            result.best_move = move;
        }
        alpha = std::max(alpha, score);
    }

    result.nodes = nodes;
    return result;
}

} // namespace chess
