#include "nn_search.hpp"

#include "attacks.hpp"
#include "move.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

namespace chess {

namespace {

int negamax_nn_impl(
    Position pos,
    int depth,
    int ply,
    int alpha,
    int beta,
    const NnValueModel& model,
    std::uint64_t& nodes
) {
    assert(depth >= 0);
    assert(alpha < beta);
    assert(model.loaded());
    ++nodes;

    const std::vector<Move> moves = generate_legal_moves(pos);

    if (moves.empty()) {
        if (in_check(pos, pos.side_to_move)) {
            return -CheckmateScore + ply;
        }
        return 0;
    }

    if (depth == 0) {
        return model.evaluate_cp_rounded(pos);
    }

    int best_score = -Infinity;
    for (Move move : moves) {
        Position next = pos;
        next.make_move(move);

        const int score = -negamax_nn_impl(next, depth - 1, ply + 1, -beta, -alpha, model, nodes);
        best_score = std::max(best_score, score);

        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            break;
        }
    }

    return best_score;
}

} // namespace

int negamax_nn(Position pos, int depth, const NnValueModel& model) {
    return negamax_nn(pos, depth, -Infinity, Infinity, model);
}

int negamax_nn(Position pos, int depth, int alpha, int beta, const NnValueModel& model) {
    std::uint64_t nodes = 0;
    return negamax_nn_impl(pos, depth, 0, alpha, beta, model, nodes);
}

SearchResult search_best_move_nn(Position pos, int depth, const NnValueModel& model) {
    assert(depth >= 0);
    assert(model.loaded());

    SearchResult result;
    std::uint64_t nodes = 0;

    if (depth == 0) {
        result.score = model.evaluate_cp_rounded(pos);
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

        const int score = -negamax_nn_impl(next, depth - 1, 1, -Infinity, -alpha, model, nodes);
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
