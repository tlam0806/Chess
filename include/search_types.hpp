#pragma once

#include "move.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>

namespace chess {

constexpr int CheckmateScore = 100000;
constexpr int Infinity = 1000000;

struct SearchResult {
    Move best_move{};
    Move ponder_move{};
    int score = 0;
    std::uint64_t nodes = 0;
    int depth = 0;
    bool stopped = false;
};

struct SearchLimits {
    int max_depth = 1;
    std::chrono::milliseconds move_time{0};
    const std::atomic<bool>* stop_requested = nullptr;
};

} // namespace chess
