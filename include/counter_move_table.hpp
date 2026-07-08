#pragma once

#include "move.hpp"
#include "position.hpp"

#include <array>
#include <cstddef>

namespace chess {

class CounterMoveTable {
public:
    void clear();
    void store(Color side_to_move, PieceType previous_piece, Move previous_move, Move counter_move);
    Move move(Color side_to_move, PieceType previous_piece, Move previous_move) const;

private:
    static constexpr int ColorCount = 2;
    static constexpr int PieceTypeCount = 6;
    static constexpr int SquareCount = 64;

    std::size_t index(Color side_to_move, PieceType previous_piece, Move previous_move) const;

    std::array<Move, ColorCount * PieceTypeCount * SquareCount> moves_{};
};

} // namespace chess
