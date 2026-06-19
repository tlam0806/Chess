#include "history_table.hpp"

#include <cassert>

namespace chess {

namespace {

int history_bonus(int depth) {
    return depth * depth;
}

int piece_to_square_index(Color color, PieceType piece, Square square) {
    assert(piece != PieceType::None);
    return (static_cast<int>(color) * 6 * 64)
        + (static_cast<int>(piece) * 64)
        + square;
}

} // namespace

void HistoryTable::reset() {
    piece_to_square_.fill(0);
}


void HistoryTable::store(const Position& pos, Move move, int depth) {
    const Square from = from_square(move);
    const PieceType piece = pos.piece_type_on_occupied(pos.side_to_move, from);
    assert(piece != PieceType::None);

    piece_to_square_[piece_to_square_index(pos.side_to_move, piece, to_square(move))]
        += history_bonus(depth);
}

HistoryTable::Score HistoryTable::get_score(const Position& pos, Move move) const {
    const Square from = from_square(move);
    const PieceType piece = pos.piece_type_on_occupied(pos.side_to_move, from);
    assert (piece != PieceType::None);

    return piece_to_square_[piece_to_square_index(pos.side_to_move, piece, to_square(move))];
}

} // namespace chess
