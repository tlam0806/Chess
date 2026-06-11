#include "perft.hpp"

#include "move.hpp"

#include <cassert>

namespace chess {

std::uint64_t perft(Position pos, int depth) {
    assert(depth >= 0);

    if (depth == 0) {
        return 1;
    }

    const std::vector<Move> moves = generate_legal_moves(pos);

    if (depth == 1) {
        return moves.size();
    }

    std::uint64_t nodes = 0;
    for (Move move : moves) {
        Position next = pos;
        next.make_move(move);
        nodes += perft(next, depth - 1);
    }

    return nodes;
}

std::vector<PerftDivideEntry> perft_divide(Position pos, int depth) {
    assert(depth > 0);

    std::vector<PerftDivideEntry> result;
    const std::vector<Move> moves = generate_legal_moves(pos);
    result.reserve(moves.size());

    for (Move move : moves) {
        Position next = pos;
        next.make_move(move);
        result.push_back({move, perft(next, depth - 1)});
    }

    return result;
}

} // namespace chess
