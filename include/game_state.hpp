#pragma once

#include "position.hpp"
#include "types.hpp"

#include <vector>

namespace chess {

int repetition_count(HashKey key, const std::vector<HashKey>& position_hashes);
bool is_threefold_repetition(HashKey key, const std::vector<HashKey>& position_hashes);

int repetition_count(const Position& current, const std::vector<Position>& previous_positions);
bool is_threefold_repetition(const Position& current, const std::vector<Position>& previous_positions);

} // namespace chess
