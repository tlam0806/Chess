#pragma once

#include "move.hpp"
#include "position.hpp"

#include <array>

namespace chess {

class HistoryTableV16 {
public:
    using Score = int;

    static constexpr Score MaxScore = 16'384;

    explicit HistoryTableV16(int penalty_divisor_numerator = 6, int penalty_divisor_denominator = 5);

    void reset();

    void store(const Position& pos, Move move, int depth);
    void store(Color color, PieceType piece, Move move, int depth);
    void penalize(const Position& pos, Move move, int depth);
    void penalize(Color color, PieceType piece, Move move, int depth);
    Score get_score(const Position& pos, Move move) const;
    Score get_score(Color color, PieceType piece, Move move) const;

private:
    static constexpr int ColorEncoder = 2;
    static constexpr int PieceEncoder = 6;
    static constexpr int SquareEncoder = 64;

    std::array<Score, ColorEncoder * PieceEncoder * SquareEncoder> piece_to_square_{};
    int penalty_divisor_numerator_ = 1;
    int penalty_divisor_denominator_ = 1;
};

} // namespace chess
