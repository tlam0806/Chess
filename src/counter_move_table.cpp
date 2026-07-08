#include "counter_move_table.hpp"

#include <algorithm>
#include <cassert>

namespace chess {

void CounterMoveTable::clear() {
    std::fill(moves_.begin(), moves_.end(), Move{});
}

std::size_t CounterMoveTable::index(
    Color side_to_move,
    PieceType previous_piece,
    Move previous_move
) const {
    assert(previous_piece != PieceType::None);
    std::size_t result = static_cast<std::size_t>(side_to_move);
    result = result * PieceTypeCount + static_cast<std::size_t>(previous_piece);
    result = result * SquareCount + static_cast<std::size_t>(previous_move.to());
    return result;
}

void CounterMoveTable::store(
    Color side_to_move,
    PieceType previous_piece,
    Move previous_move,
    Move counter_move
) {
    if (previous_piece == PieceType::None
        || previous_move.value == 0
        || counter_move.value == 0) {
        return;
    }
    moves_[index(side_to_move, previous_piece, previous_move)] = counter_move;
}

Move CounterMoveTable::move(
    Color side_to_move,
    PieceType previous_piece,
    Move previous_move
) const {
    if (previous_piece == PieceType::None || previous_move.value == 0) {
        return Move{};
    }
    return moves_[index(side_to_move, previous_piece, previous_move)];
}

} // namespace chess
