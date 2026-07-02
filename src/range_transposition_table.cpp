#include "range_transposition_table.hpp"

#include <algorithm>
#include <cstdlib>

namespace chess {

namespace {

constexpr std::size_t BytesPerMegabyte = 1024 * 1024;
constexpr int MateScoreThreshold = CheckmateScore - 1024;

std::size_t entry_count_from_megabytes(std::size_t megabytes) {
    const std::size_t bytes = std::max<std::size_t>(megabytes, 1) * BytesPerMegabyte;
    return std::max<std::size_t>(bytes / sizeof(RangeTTEntry), 1);
}

int score_from_table(int score, int ply) {
    if (score >= Infinity / 2 || score <= -Infinity / 2) {
        return score;
    }
    if (score >= MateScoreThreshold) {
        return score - ply;
    }
    if (score <= -MateScoreThreshold) {
        return score + ply;
    }
    return score;
}

bool is_mate_score(int score) {
    return score != -Infinity && score != Infinity && std::abs(score) >= MateScoreThreshold;
}

bool depth_matches_policy(int entry_depth, int requested_depth, TTDepthPolicy policy) {
    switch (policy) {
    case TTDepthPolicy::Exact:
        return entry_depth == requested_depth;
    case TTDepthPolicy::AtLeast:
        return entry_depth >= requested_depth;
    }
    return false;
}

bool is_valid_entry(const RangeTTEntry& entry) {
    return entry.depth >= 0;
}

} // namespace

RangeTranspositionTable::RangeTranspositionTable(std::size_t megabytes)
    : entries_(entry_count_from_megabytes(megabytes)) {
}

void RangeTranspositionTable::clear() {
    std::fill(entries_.begin(), entries_.end(), RangeTTEntry{});
}

void RangeTranspositionTable::clear_stats() {
    stats_ = {};
}

std::size_t RangeTranspositionTable::entry_count() const {
    return entries_.size();
}

const RangeTranspositionTableStats& RangeTranspositionTable::stats() const {
    return stats_;
}

bool RangeTranspositionTable::probe(
    HashKey key,
    int depth,
    ScoreRange& search_window,
    ScoreRange& stored_score,
    MoveRange& stored_move,
    int ply,
    bool& score_available,
    TTDepthPolicy depth_policy
) const {
    score_available = false;
    ++stats_.probes;
    const RangeTTEntry& entry = entries_[key % entries_.size()];

    if (!is_valid_entry(entry)) {
        ++stats_.empty_misses;
        return false;
    }

    if (entry.key != key) {
        ++stats_.index_collisions;
        return false;
    }

    ++stats_.key_hits;
    stored_move = entry.move;
    if (stored_move.lower.value != 0 || stored_move.upper.value != 0) {
        ++stats_.move_hint_hits;
    }

    if (!depth_matches_policy(entry.depth, depth, depth_policy)) {
        ++stats_.depth_misses;
        return false;
    }

    stored_score.lower = score_from_table(entry.score.lower, ply);
    stored_score.upper = score_from_table(entry.score.upper, ply);

    if (is_mate_score(stored_score.lower) || is_mate_score(stored_score.upper)) {
        return false;
    }
    score_available = true;

    const bool exact = stored_score.lower == stored_score.upper;
    if (exact) {
        ++stats_.exact_hits;
    } else {
        if (stored_score.lower != -Infinity) {
            ++stats_.lower_score_hits;
        }
        if (stored_score.upper != Infinity) {
            ++stats_.upper_score_hits;
        }
    }

    const ScoreRange search_window_before = search_window;
    if (stored_score.lower != -Infinity) {
        search_window.lower = std::max(search_window.lower, stored_score.lower);
    }
    if (stored_score.upper != Infinity) {
        search_window.upper = std::min(search_window.upper, stored_score.upper);
    }
    if (search_window.lower != search_window_before.lower
        || search_window.upper != search_window_before.upper) {
        ++stats_.window_narrowings;
    }

    const bool enough_to_return =
        (stored_score.lower != -Infinity && stored_score.lower >= search_window_before.upper)
        || (stored_score.upper != Infinity && stored_score.upper <= search_window_before.lower);
    if (exact || enough_to_return) {
        if (!exact) {
            ++stats_.score_returns;
        }
        return true;
    }

    return false;
}

void RangeTranspositionTable::store(
    HashKey key,
    int depth,
    ScoreRange score,
    MoveRange move
) {
    ++stats_.stores;
    RangeTTEntry& entry = entries_[key % entries_.size()];
    if (!is_valid_entry(entry) || entry.key == key || depth >= entry.depth) {
        if (!is_valid_entry(entry)) {
            ++stats_.new_stores;
        } else if (entry.key == key) {
            ++stats_.same_key_updates;
        } else {
            ++stats_.replacement_collisions;
        }
        entry.key = key;
        entry.depth = depth;
        entry.score = score;
        entry.move = move;
    } else {
        ++stats_.skipped_shallow_replacements;
    }
}

} // namespace chess
