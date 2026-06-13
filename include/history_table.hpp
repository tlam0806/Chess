#pragma once

#include "move.hpp"
#include "position.hpp"

#include <array>

namespace chess {

class HistoryTable {
public:
    using Score = int;

    void reset();

    void store(const Position& pos, Move move, int depth);
    Score get_score(const Position& pos, Move move) const;

private:
    static constexpr int ColorEncoder = 2;
    static constexpr int PieceEncoder = 6;
    static constexpr int SquareEncoder = 128;

    std::array<Score, ColorEncoder * PieceEncoder * SquareEncoder> piece_to_square_{};
};

} // namespace chess
