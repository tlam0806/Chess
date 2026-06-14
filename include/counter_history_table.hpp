#pragma once
#include "move.hpp"
#include "position.hpp"

#include <array>
#include <cstddef>

namespace chess {
class CounterHistoryTable {
public:
    using Score = int;

    static constexpr Score MaxScore = 16'384;

    explicit CounterHistoryTable(int penalty_divisor_numerator = 6, int penalty_divisor_denominator = 5);

    void store(Color color, PieceType previous_piece, Move previous_move, PieceType counter_piece, Move counter_move, int depth);
    void penalize(Color color, PieceType previous_piece, Move previous_move, PieceType counter_piece, Move counter_move, int depth);
    Score get_score(Color color, PieceType previous_piece, Move previous_move, PieceType counter_piece, Move counter_move) const;

private:
    static constexpr int ColorEncoder = 2;
    static constexpr int PieceTypeEncoder = 6;
    static constexpr int SquareEncoder = 64;

    std::array<Score, ColorEncoder * PieceTypeEncoder * SquareEncoder * PieceTypeEncoder * SquareEncoder> scores{};
    int penalty_divisor_numerator_ = 1;
    int penalty_divisor_denominator_ = 1;

    std::size_t get_index(Color color, PieceType previous_piece, Move previous_move, PieceType counter_piece, Move counter_move) const;
};
} // namespace chess
