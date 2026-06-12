#include "transposition_table.hpp"

#include <algorithm>

namespace chess {

namespace {

constexpr std::size_t BytesPerMegabyte = 1024 * 1024;

std::size_t entry_count_from_megabytes(std::size_t megabytes) {
    const std::size_t bytes = std::max<std::size_t>(megabytes, 1) * BytesPerMegabyte;
    return std::max<std::size_t>(bytes / sizeof(TTEntry), 1);
}

} // namespace

TranspositionTable::TranspositionTable(std::size_t megabytes)
    : entries_(entry_count_from_megabytes(megabytes)) {
}

void TranspositionTable::clear() {
    std::fill(entries_.begin(), entries_.end(), TTEntry{});
}

std::size_t TranspositionTable::entry_count() const {
    return entries_.size();
}

bool TranspositionTable::probe(
    HashKey key,
    int depth,
    int& alpha,
    int& beta,
    int& score,
    Move& best_move
) const {
    const TTEntry& entry = entries_[key % entries_.size()];

    if (!entry.valid || entry.key != key) {
        return false;
    }

    best_move = entry.best_move;

    if (entry.depth < depth) {
        return false;
    }

    if (entry.bound == TTBound::Exact) {
        score = entry.score;
        return true;
    }

    if (entry.bound == TTBound::Lower) {
        alpha = std::max(alpha, entry.score);
    } else if (entry.bound == TTBound::Upper) {
        beta = std::min(beta, entry.score);
    }

    if (alpha >= beta) {
        score = entry.score;
        return true;
    }

    return false;
}

void TranspositionTable::store(
    HashKey key,
    int depth,
    int score,
    TTBound bound,
    Move best_move
) {
    TTEntry& entry = entries_[key % entries_.size()];
    if (!entry.valid || entry.key == key || depth >= entry.depth) {
        entry.key = key;
        entry.valid = true;
        entry.depth = depth;
        entry.score = score;
        entry.bound = bound;
        entry.best_move = best_move;
    }
}

} // namespace chess
