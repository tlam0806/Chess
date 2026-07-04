#include "range_bucket_transposition_table.hpp"

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
    return std::max<std::size_t>(bytes / sizeof(RangeTTEntry), 1);
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

int replacement_depth(const RangeBucketTranspositionTable::TTValue& value) {
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
    RangeBucketTranspositionTable::TTValue& value,
    int depth,
    int score,
    Move move
) {
    if (score == -Infinity) {
        return;
    }
    if (value.lower_depth == MissingDepth || depth >= static_cast<int>(value.lower_depth)) {
        value.lower_depth = depth_to_table(depth);
        value.score.lower = score;
        value.move.lower = move;
    }
}

void store_upper_bound(
    RangeBucketTranspositionTable::TTValue& value,
    int depth,
    int score,
    Move move
) {
    if (score == Infinity) {
        return;
    }
    if (value.upper_depth == MissingDepth || depth >= static_cast<int>(value.upper_depth)) {
        value.upper_depth = depth_to_table(depth);
        value.score.upper = score;
        value.move.upper = move;
    }
}

RangeBucketTranspositionTable::TTValue make_tt_value(
    int depth,
    ScoreRange score,
    MoveRange move
) {
    RangeBucketTranspositionTable::TTValue value{};
    store_lower_bound(value, depth, score.lower, move.lower);
    store_upper_bound(value, depth, score.upper, move.upper);
    return value;
}

void merge_tt_value(
    RangeBucketTranspositionTable::TTValue& value,
    int depth,
    ScoreRange score,
    MoveRange move
) {
    store_lower_bound(value, depth, score.lower, move.lower);
    store_upper_bound(value, depth, score.upper, move.upper);
}

} // namespace

RangeBucketTranspositionTable::RangeBucketTranspositionTable(
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

void RangeBucketTranspositionTable::clear() {
    std::fill(keys_.begin(), keys_.end(), HashKey{});
    std::fill(values_.begin(), values_.end(), TTValue{});
}

void RangeBucketTranspositionTable::clear_stats() {
    stats_ = {};
}

std::size_t RangeBucketTranspositionTable::entry_count() const {
    return keys_.size();
}

std::size_t RangeBucketTranspositionTable::bucket_size() const {
    return bucket_size_;
}

std::size_t RangeBucketTranspositionTable::bucket_count() const {
    return bucket_count_;
}

const RangeTranspositionTableStats& RangeBucketTranspositionTable::stats() const {
    return stats_;
}

std::size_t RangeBucketTranspositionTable::bucket_offset(HashKey key) const {
    return (key & bucket_mask_) * bucket_size_;
}

bool RangeBucketTranspositionTable::probe(
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

        CHESS_TT_STAT(stats_.key_hits);
        const TTValue& value = values_[index];
        stored_move = value.move;
        if (stored_move.lower.value != 0 || stored_move.upper.value != 0) {
            CHESS_TT_STAT(stats_.move_hint_hits);
        }

        const bool lower_depth_matches =
            value.score.lower != -Infinity
            && depth_matches_policy(value.lower_depth, depth, depth_policy);
        const bool upper_depth_matches =
            value.score.upper != Infinity
            && depth_matches_policy(value.upper_depth, depth, depth_policy);
        if (!lower_depth_matches && !upper_depth_matches) {
            CHESS_TT_STAT(stats_.depth_misses);
            return false;
        }

        stored_score = ScoreRange{};
        if (lower_depth_matches) {
            stored_score.lower = score_from_table(value.score.lower, ply);
        }
        if (upper_depth_matches) {
            stored_score.upper = score_from_table(value.score.upper, ply);
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
    }

    if (saw_valid_entry) {
        CHESS_TT_STAT(stats_.index_collisions);
    } else {
        CHESS_TT_STAT(stats_.empty_misses);
    }
    return false;
}

void RangeBucketTranspositionTable::store(
    HashKey key,
    int depth,
    ScoreRange score,
    MoveRange move
) {
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
