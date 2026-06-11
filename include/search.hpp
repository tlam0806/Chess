#pragma once

#include "move.hpp"
#include "position.hpp"

#include <cstdint>

namespace chess {

constexpr int CheckmateScore = 100000;
constexpr int Infinity = 1000000;

struct SearchResult {
    Move best_move{};
    int score = 0;
    std::uint64_t nodes = 0;
};

int negamax(Position pos, int depth);
int negamax(Position pos, int depth, int alpha, int beta);
SearchResult search_best_move(Position pos, int depth);

} // namespace chess
