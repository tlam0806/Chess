#include "lower_move_range_bucket_transposition_table.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <limits>

namespace chess {

namespace {

constexpr std::size_t BytesPerMegabyte = 1024 * 1024;
constexpr int MateScoreThreshold = CheckmateScore - 1024;
constexpr std::uint8_t MissingDepth = std::numeric_limits<std::uint8_t>::max();

#ifdef CHESS_ENABLE_TT_STATS
#define CHESS_TT_STAT(counter) (++(counter))
#else
#define CHESS_TT_STAT(counter) ((void)0)
#endif // CHESS_ENABLE_TT_STATS

std::size_t entry_count_from_megabytes(std::size_t megabytes) {
    const std::size_t bytes = std::max<std::size_t>(megabytes, 1) * BytesPerMegabyte;
    const std::size_t bytes_per_entry =
        sizeof(HashKey) + sizeof(LowerMoveRangeBucketTranspositionTable::TTValue);
    return std::max<std::size_t>(bytes / bytes_per_entry, 1);
}

std::size_t floor_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result <= value / 2) {
        result *= 2;
    }
    return result;
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

bool depth_matches_policy(std::uint8_t entry_depth, int requested_depth, TTDepthPolicy policy) {
    if (entry_depth == MissingDepth) {
        return false;
    }
    const int stored_depth = static_cast<int>(entry_depth);
    switch (policy) {
    case TTDepthPolicy::Exact:
        return stored_depth == requested_depth;
    case TTDepthPolicy::AtLeast:
        return stored_depth >= requested_depth;
    }
    return false;
}

int replacement_depth(const LowerMoveRangeBucketTranspositionTable::TTValue& value) {
    const int lower_depth = value.lower_depth == MissingDepth
        ? -1
        : static_cast<int>(value.lower_depth);
    const int upper_depth = value.upper_depth == MissingDepth
        ? -1
        : static_cast<int>(value.upper_depth);
    return std::max(lower_depth, upper_depth);
}

std::uint8_t depth_to_table(int depth) {
    assert(depth >= 0);
    assert(depth < static_cast<int>(MissingDepth));
    return static_cast<std::uint8_t>(depth);
}

void store_lower_bound(
    LowerMoveRangeBucketTranspositionTable::TTValue& value,
    int depth,
    int score,
    Move move
) {
    if (score == -Infinity) {
        return;
    }
    if (value.lower_depth == MissingDepth || depth >= static_cast<int>(value.lower_depth)) {
        value.lower_depth = depth_to_table(depth);
        value.set_lower_score(score);
        value.lower_move = move;
    }
}

void store_upper_bound(
    LowerMoveRangeBucketTranspositionTable::TTValue& value,
    int depth,
    int score,
    Move move
) {
    (void)move;
    if (score == Infinity) {
        return;
    }
    if (value.upper_depth == MissingDepth || depth >= static_cast<int>(value.upper_depth)) {
        value.upper_depth = depth_to_table(depth);
        value.set_upper_score(score);
    }
}

LowerMoveRangeBucketTranspositionTable::TTValue make_tt_value(
    int depth,
    ScoreRange score,
    MoveRange move,
    std::uint8_t generation
) {
    LowerMoveRangeBucketTranspositionTable::TTValue value{};
    value.set_generation(generation);
    store_lower_bound(value, depth, score.lower, move.lower);
    store_upper_bound(value, depth, score.upper, move.upper);
    return value;
}

void merge_tt_value(
    LowerMoveRangeBucketTranspositionTable::TTValue& value,
    int depth,
    ScoreRange score,
    MoveRange move
) {
    store_lower_bound(value, depth, score.lower, move.lower);
    store_upper_bound(value, depth, score.upper, move.upper);
}

} // namespace

LowerMoveRangeBucketTranspositionTable::LowerMoveRangeBucketTranspositionTable(
    std::size_t megabytes,
    std::size_t bucket_size
) : bucket_size_(std::max<std::size_t>(bucket_size, 1)) {
    const std::size_t total_entries = entry_count_from_megabytes(megabytes);
    bucket_count_ = floor_power_of_two(std::max<std::size_t>(total_entries / bucket_size_, 1));
    bucket_mask_ = bucket_count_ - 1;
    const std::size_t entry_count = bucket_count_ * bucket_size_;
    keys_.resize(entry_count);
    values_.resize(entry_count);
}

void LowerMoveRangeBucketTranspositionTable::clear() {
    std::fill(keys_.begin(), keys_.end(), HashKey{});
    std::fill(values_.begin(), values_.end(), TTValue{});
    generation_ = 0;
}

void LowerMoveRangeBucketTranspositionTable::advance_generation() {
    if (generation_ == std::numeric_limits<std::uint8_t>::max()) {
        clear();
    }
    ++generation_;
}

void LowerMoveRangeBucketTranspositionTable::clear_stats() {
    stats_ = {};
}

std::size_t LowerMoveRangeBucketTranspositionTable::entry_count() const {
    return keys_.size();
}

std::size_t LowerMoveRangeBucketTranspositionTable::bucket_size() const {
    return bucket_size_;
}

std::size_t LowerMoveRangeBucketTranspositionTable::bucket_count() const {
    return bucket_count_;
}

const RangeTranspositionTableStats& LowerMoveRangeBucketTranspositionTable::stats() const {
    return stats_;
}

std::size_t LowerMoveRangeBucketTranspositionTable::bucket_offset(HashKey key) const {
    return (key & bucket_mask_) * bucket_size_;
}

bool LowerMoveRangeBucketTranspositionTable::probe(
    HashKey key,
    int depth,
    ScoreRange& search_window,
    ScoreRange& stored_score,
    MoveRange& stored_move,
    int ply,
    bool& score_available,
    TTDepthPolicy depth_policy
) const {
    assert(key != 0);
    score_available = false;
    CHESS_TT_STAT(stats_.probes);

    const std::size_t offset = bucket_offset(key);
    if (keys_[offset] == 0) {
        CHESS_TT_STAT(stats_.empty_misses);
        return false;
    }

    auto probe_hit = [&](std::size_t index) {
        CHESS_TT_STAT(stats_.key_hits);
        const TTValue& value = values_[index];
        stored_move = MoveRange{value.lower_move, Move{}};
        if (stored_move.lower.value != 0) {
            CHESS_TT_STAT(stats_.move_hint_hits);
        }

        // Scores can depend on the root game history (repetition and the
        // 50-move clock), which is not part of the Zobrist key.  Keep an old
        // entry's best move for ordering, but only use scores written during
        // the current search generation.
        if (value.generation() != generation_) {
            CHESS_TT_STAT(stats_.depth_misses);
            return false;
        }

        // A missing bound always has MissingDepth, so reject depth misses
        // before decoding either packed score.  Most failed probes avoid the
        // masks and subtracts entirely, while one-sided hits decode one word.
        const bool lower_depth_matches =
            depth_matches_policy(value.lower_depth, depth, depth_policy);
        const bool upper_depth_matches =
            depth_matches_policy(value.upper_depth, depth, depth_policy);
        if (!lower_depth_matches && !upper_depth_matches) {
            CHESS_TT_STAT(stats_.depth_misses);
            return false;
        }

        stored_score = ScoreRange{};
        if (lower_depth_matches) {
            stored_score.lower = score_from_table(value.lower_score(), ply);
        }
        if (upper_depth_matches) {
            stored_score.upper = score_from_table(value.upper_score(), ply);
        }

        if (is_mate_score(stored_score.lower) || is_mate_score(stored_score.upper)) {
            return false;
        }
        score_available = true;

        const bool exact = stored_score.lower == stored_score.upper;
        if (exact) {
            CHESS_TT_STAT(stats_.exact_hits);
        } else {
            if (stored_score.lower != -Infinity) {
                CHESS_TT_STAT(stats_.lower_score_hits);
            }
            if (stored_score.upper != Infinity) {
                CHESS_TT_STAT(stats_.upper_score_hits);
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
            CHESS_TT_STAT(stats_.window_narrowings);
        }

        const bool enough_to_return =
            (stored_score.lower != -Infinity && stored_score.lower >= search_window_before.upper)
            || (stored_score.upper != Infinity && stored_score.upper <= search_window_before.lower);
        if (exact || enough_to_return) {
            if (!exact) {
                CHESS_TT_STAT(stats_.score_returns);
            }
            return true;
        }

        return false;
    };

    if (bucket_size_ == 4) {
        const HashKey key0 = keys_[offset];
        if (key0 == key) {
            return probe_hit(offset);
        }
        const HashKey key1 = keys_[offset + 1];
        if (key1 == 0) {
            CHESS_TT_STAT(stats_.index_collisions);
            return false;
        }
        if (key1 == key) {
            return probe_hit(offset + 1);
        }
        const HashKey key2 = keys_[offset + 2];
        if (key2 == 0) {
            CHESS_TT_STAT(stats_.index_collisions);
            return false;
        }
        if (key2 == key) {
            return probe_hit(offset + 2);
        }
        const HashKey key3 = keys_[offset + 3];
        if (key3 == key) {
            return probe_hit(offset + 3);
        }
        CHESS_TT_STAT(stats_.index_collisions);
        return false;
    }

    bool saw_valid_entry = false;
    for (std::size_t i = 0; i < bucket_size_; ++i) {
        const std::size_t index = offset + i;
        const HashKey entry_key = keys_[index];
        if (entry_key == 0) {
            break;
        }
        saw_valid_entry = true;
        if (entry_key != key) {
            continue;
        }

        return probe_hit(index);
    }

    if (saw_valid_entry) {
        CHESS_TT_STAT(stats_.index_collisions);
    } else {
        CHESS_TT_STAT(stats_.empty_misses);
    }
    return false;
}

void LowerMoveRangeBucketTranspositionTable::store(
    HashKey key,
    int depth,
    ScoreRange score,
    MoveRange move
) {
    assert(key != 0);
    CHESS_TT_STAT(stats_.stores);
    const std::size_t offset = bucket_offset(key);

    std::size_t shallowest_index = offset;
    std::size_t shallowest_stale_index = offset;
    bool found_stale_entry = false;
    for (std::size_t i = 0; i < bucket_size_; ++i) {
        const std::size_t index = offset + i;
        const HashKey entry_key = keys_[index];
        if (entry_key == key) {
            CHESS_TT_STAT(stats_.same_key_updates);
            if (values_[index].generation() == generation_) {
                merge_tt_value(values_[index], depth, score, move);
            } else {
                values_[index] = make_tt_value(depth, score, move, generation_);
            }
            return;
        }
        if (entry_key == 0) {
            CHESS_TT_STAT(stats_.new_stores);
            keys_[index] = key;
            values_[index] = make_tt_value(depth, score, move, generation_);
            return;
        }
        if (values_[index].generation() != generation_) {
            if (!found_stale_entry
                || replacement_depth(values_[index])
                    < replacement_depth(values_[shallowest_stale_index])) {
                shallowest_stale_index = index;
            }
            found_stale_entry = true;
            continue;
        }
        if (replacement_depth(values_[index]) < replacement_depth(values_[shallowest_index])) {
            shallowest_index = index;
        }
    }

    if (found_stale_entry) {
        CHESS_TT_STAT(stats_.replacement_collisions);
        keys_[shallowest_stale_index] = key;
        values_[shallowest_stale_index] = make_tt_value(
            depth, score, move, generation_);
        return;
    }

    if (depth >= replacement_depth(values_[shallowest_index])) {
        CHESS_TT_STAT(stats_.replacement_collisions);
        keys_[shallowest_index] = key;
        values_[shallowest_index] = make_tt_value(depth, score, move, generation_);
    } else {
        CHESS_TT_STAT(stats_.skipped_shallow_replacements);
    }
}

} // namespace chess
