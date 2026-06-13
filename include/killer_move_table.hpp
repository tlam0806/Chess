#pragma once

#include "move.hpp"

#include <array>

namespace chess {

class KillerMoveTable {
public:
    static constexpr int MaxPly = 128;
    static constexpr int Slots = 2;

    void clear();
    void store(int ply, Move move);

    int score(int ply, Move move) const;

private:
    std::array<std::array<Move, Slots>, MaxPly> killers_{};
};

} // namespace chess
