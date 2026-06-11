#pragma once

#include "move.hpp"
#include "position.hpp"

#include <cstdint>
#include <vector>

namespace chess {

struct PerftDivideEntry {
    Move move;
    std::uint64_t nodes = 0;
};

std::uint64_t perft(Position pos, int depth);
std::vector<PerftDivideEntry> perft_divide(Position pos, int depth);

} // namespace chess
