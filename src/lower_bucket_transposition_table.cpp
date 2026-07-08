#include "lower_bucket_transposition_table.hpp"

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
    const std::size_t bytes_per_entry = sizeof(HashKey) + sizeof(LowerBucketTranspositionTable::TTValue);
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

std::uint8_t depth_to_table(int depth) {
    assert(depth >= 0);
    assert(depth < static_cast<int>(MissingDepth));
    return static_cast<std::uint8_t>(depth);
}

void store_lower_bound(
    LowerBucketTranspositionTable::TTValue& value,
    int depth,
    int score,
    Move move
) {
    if (score == -Infinity) {
        return;
    }
    if (value.lower_depth == MissingDepth || depth >= static_cast<int>(value.lower_depth)) {
        value.lower_depth = depth_to_table(depth);
        value.lower_score = score;
        value.lower_move = move;
    }
}

LowerBucketTranspositionTable::TTValue make_tt_value(
    int depth,
    ScoreRange score,
    MoveRange move
) {
    LowerBucketTranspositionTable::TTValue value{};
    store_lower_bound(value, depth, score.lower, move.lower);
    return value;
}

void merge_tt_value(
    LowerBucketTranspositionTable::TTValue& value,
    int depth,
    ScoreRange score,
    MoveRange move
) {
    store_lower_bound(value, depth, score.lower, move.lower);
}

int replacement_depth(const LowerBucketTranspositionTable::TTValue& value) {
    return value.lower_depth == MissingDepth ? -1 : static_cast<int>(value.lower_depth);
}

} // namespace

LowerBucketTranspositionTable::LowerBucketTranspositionTable(
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

void LowerBucketTranspositionTable::clear() {
    std::fill(keys_.begin(), keys_.end(), HashKey{});
    std::fill(values_.begin(), values_.end(), TTValue{});
}

void LowerBucketTranspositionTable::clear_stats() {
    stats_ = {};
}

std::size_t LowerBucketTranspositionTable::entry_count() const {
    return keys_.size();
}

std::size_t LowerBucketTranspositionTable::bucket_size() const {
    return bucket_size_;
}

std::size_t LowerBucketTranspositionTable::bucket_count() const {
    return bucket_count_;
}

const RangeTranspositionTableStats& LowerBucketTranspositionTable::stats() const {
    return stats_;
}

std::size_t LowerBucketTranspositionTable::bucket_offset(HashKey key) const {
    return (key & bucket_mask_) * bucket_size_;
}

bool LowerBucketTranspositionTable::probe(
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

        const bool lower_depth_matches =
            value.lower_score != -Infinity
            && depth_matches_policy(value.lower_depth, depth, depth_policy);
        if (!lower_depth_matches) {
            CHESS_TT_STAT(stats_.depth_misses);
            return false;
        }

        stored_score = ScoreRange{};
        stored_score.lower = score_from_table(value.lower_score, ply);
        if (is_mate_score(stored_score.lower)) {
            return false;
        }
        score_available = true;
        CHESS_TT_STAT(stats_.lower_score_hits);

        const ScoreRange search_window_before = search_window;
        search_window.lower = std::max(search_window.lower, stored_score.lower);
        if (search_window.lower != search_window_before.lower) {
            CHESS_TT_STAT(stats_.window_narrowings);
        }

        if (stored_score.lower >= search_window_before.upper) {
            CHESS_TT_STAT(stats_.score_returns);
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

void LowerBucketTranspositionTable::store(
    HashKey key,
    int depth,
    ScoreRange score,
    MoveRange move
) {
    if (score.lower == -Infinity) {
        return;
    }

    assert(key != 0);
    CHESS_TT_STAT(stats_.stores);
    const std::size_t offset = bucket_offset(key);

    std::size_t shallowest_index = offset;
    for (std::size_t i = 0; i < bucket_size_; ++i) {
        const std::size_t index = offset + i;
        const HashKey entry_key = keys_[index];
        if (entry_key == key) {
            CHESS_TT_STAT(stats_.same_key_updates);
            merge_tt_value(values_[index], depth, score, move);
            return;
        }
        if (entry_key == 0) {
            CHESS_TT_STAT(stats_.new_stores);
            keys_[index] = key;
            values_[index] = make_tt_value(depth, score, move);
            return;
        }
        if (replacement_depth(values_[index]) < replacement_depth(values_[shallowest_index])) {
            shallowest_index = index;
        }
    }

    if (depth >= replacement_depth(values_[shallowest_index])) {
        CHESS_TT_STAT(stats_.replacement_collisions);
        keys_[shallowest_index] = key;
        values_[shallowest_index] = make_tt_value(depth, score, move);
    } else {
        CHESS_TT_STAT(stats_.skipped_shallow_replacements);
    }
}

} // namespace chess
