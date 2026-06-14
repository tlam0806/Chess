#include "range_bucket_transposition_table.hpp"

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

} // namespace

RangeBucketTranspositionTable::RangeBucketTranspositionTable(
    std::size_t megabytes,
    std::size_t bucket_size
) : bucket_size_(std::max<std::size_t>(bucket_size, 1)) {
    const std::size_t total_entries = entry_count_from_megabytes(megabytes);
    bucket_count_ = std::max<std::size_t>(total_entries / bucket_size_, 1);
    entries_.resize(bucket_count_ * bucket_size_);
}

void RangeBucketTranspositionTable::clear() {
    std::fill(entries_.begin(), entries_.end(), RangeTTEntry{});
}

void RangeBucketTranspositionTable::clear_stats() {
    stats_ = {};
}

std::size_t RangeBucketTranspositionTable::entry_count() const {
    return entries_.size();
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

RangeTTEntry* RangeBucketTranspositionTable::bucket_begin(HashKey key) {
    return entries_.data() + (key % bucket_count_) * bucket_size_;
}

const RangeTTEntry* RangeBucketTranspositionTable::bucket_begin(HashKey key) const {
    return entries_.data() + (key % bucket_count_) * bucket_size_;
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
    score_available = false;
    ++stats_.probes;

    const RangeTTEntry* bucket = bucket_begin(key);
    bool saw_valid_entry = false;
    for (std::size_t i = 0; i < bucket_size_; ++i) {
        const RangeTTEntry& entry = bucket[i];
        if (!entry.valid) {
            continue;
        }
        saw_valid_entry = true;
        if (entry.key != key) {
            continue;
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

    if (saw_valid_entry) {
        ++stats_.index_collisions;
    } else {
        ++stats_.empty_misses;
    }
    return false;
}

void RangeBucketTranspositionTable::store(
    HashKey key,
    int depth,
    ScoreRange score,
    MoveRange move
) {
    ++stats_.stores;
    RangeTTEntry* bucket = bucket_begin(key);

    RangeTTEntry* shallowest = &bucket[0];
    for (std::size_t i = 0; i < bucket_size_; ++i) {
        RangeTTEntry& entry = bucket[i];
        if (entry.valid && entry.key == key) {
            ++stats_.same_key_updates;
            entry.depth = depth;
            entry.score = score;
            entry.move = move;
            return;
        }
        if (!entry.valid) {
            ++stats_.new_stores;
            entry.key = key;
            entry.valid = true;
            entry.depth = depth;
            entry.score = score;
            entry.move = move;
            return;
        }
        if (entry.depth < shallowest->depth) {
            shallowest = &entry;
        }
    }

    if (depth >= shallowest->depth) {
        ++stats_.replacement_collisions;
        shallowest->key = key;
        shallowest->valid = true;
        shallowest->depth = depth;
        shallowest->score = score;
        shallowest->move = move;
    } else {
        ++stats_.skipped_shallow_replacements;
    }
}

} // namespace chess
