#include "game_state.hpp"

namespace chess {

int repetition_count(HashKey key, const std::vector<HashKey>& position_hashes) {
    int count = 0;
    for (HashKey hash : position_hashes) {
        if (hash == key) {
            ++count;
        }
    }
    return count;
}

bool is_threefold_repetition(HashKey key, const std::vector<HashKey>& position_hashes) {
    return repetition_count(key, position_hashes) >= 3;
}

int repetition_count(const Position& current, const std::vector<Position>& previous_positions) {
    int count = 1;
    for (const Position& previous : previous_positions) {
        if (previous.zobrist_key == current.zobrist_key) {
            ++count;
        }
    }
    return count;
}

bool is_threefold_repetition(const Position& current, const std::vector<Position>& previous_positions) {
    return repetition_count(current, previous_positions) >= 3;
}

} // namespace chess
