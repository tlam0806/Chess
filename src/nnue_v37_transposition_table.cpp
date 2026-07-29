#include "nnue_v37_transposition_table.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#if defined(CHESS_PROFILE_TT_TIMING) || defined(CHESS_PROFILE_TT_PATH_TIMING)
#include <chrono>
#endif
#include <cstdlib>
#include <limits>
#include <optional>

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

#ifdef CHESS_PROFILE_TT_TIMING
struct ScopedTiming {
    std::uint64_t& total_ns;
    std::uint64_t& calls;
    std::chrono::steady_clock::time_point start;

    ScopedTiming(std::uint64_t& total, std::uint64_t& count)
        : total_ns(total),
          calls(count),
          start(std::chrono::steady_clock::now()) {
        ++calls;
    }

    ~ScopedTiming() {
        total_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
    }
};
#define CHESS_TT_TIME(total, calls) ScopedTiming scoped_timing_##__LINE__((total), (calls))
#else
#define CHESS_TT_TIME(total, calls) ((void)0)
#endif

#ifdef CHESS_PROFILE_TT_PATH_TIMING
std::uint64_t elapsed_probe_path_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

void add_probe_path_timing(
    std::uint64_t& ns,
    std::uint64_t& calls,
    std::chrono::steady_clock::time_point start
) {
    ns += elapsed_probe_path_ns(start);
    ++calls;
}
#endif

using V37TT = NnueV37TranspositionTable;
using Bound = V37TT::Bound;

std::size_t entry_count_from_megabytes(std::size_t megabytes) {
    const std::size_t bytes = std::max<std::size_t>(megabytes, 1) * BytesPerMegabyte;
    const std::size_t bytes_per_entry = sizeof(V37TT::KeyLock) + sizeof(V37TT::TTValue);
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
    return score >= MateScoreThreshold || score <= -MateScoreThreshold;
}

std::uint8_t depth_to_table(int depth) {
    assert(depth >= 0);
    assert(depth < static_cast<int>(MissingDepth));
    return static_cast<std::uint8_t>(depth);
}

int bound_priority(Bound bound) {
    assert(bound != Bound::Missing);
    return static_cast<int>(bound);
}

int replacement_priority(const V37TT::TTValue& value) {
    assert(value.depth != MissingDepth);
    assert(value.bound != Bound::Missing);
    return static_cast<int>(value.depth) * 3 + bound_priority(value.bound);
}

V37TT::KeyLock make_tt_lock(HashKey key) {
    assert(key != 0);
    const auto lock = static_cast<V37TT::KeyLock>(key >> 32);
    return lock == 0 ? V37TT::KeyLock{1} : lock;
}

struct StoreValue {
    int score = 0;
    Move move{};
    std::uint8_t depth = MissingDepth;
    Bound bound = Bound::Missing;
};

std::optional<StoreValue> make_store_value(int depth, ScoreRange score, MoveRange move) {
    if (score.lower == score.upper) {
        return StoreValue{
            score.lower,
            move.lower.value != 0 ? move.lower : move.upper,
            depth_to_table(depth),
            Bound::Exact
        };
    }
    if (score.lower != -Infinity) {
        return StoreValue{score.lower, move.lower, depth_to_table(depth), Bound::Lower};
    }
    if (score.upper != Infinity) {
        return StoreValue{score.upper, move.upper, depth_to_table(depth), Bound::Upper};
    }
    return std::nullopt;
}

V37TT::TTValue to_tt_value(StoreValue value) {
    return V37TT::TTValue{value.score, value.move, value.depth, value.bound};
}

bool opposite_bounds(Bound lhs, Bound rhs) {
    return (lhs == Bound::Lower && rhs == Bound::Upper)
        || (lhs == Bound::Upper && rhs == Bound::Lower);
}

bool should_replace_same_key(const V37TT::TTValue& old_value, StoreValue new_value) {
    if (old_value.bound == Bound::Missing || old_value.depth == MissingDepth) {
        return true;
    }
    if (new_value.depth > old_value.depth) {
        return true;
    }
    if (new_value.depth < old_value.depth) {
        return false;
    }
    if (new_value.bound == Bound::Exact) {
        return true;
    }
    if (old_value.bound == Bound::Exact) {
        return false;
    }
    if (opposite_bounds(old_value.bound, new_value.bound)) {
        if (old_value.score == new_value.score) {
            return true;
        }
        return new_value.bound == Bound::Lower && old_value.bound == Bound::Upper;
    }
    if (new_value.bound == Bound::Lower && old_value.bound == Bound::Lower) {
        return new_value.score > old_value.score;
    }
    if (new_value.bound == Bound::Upper && old_value.bound == Bound::Upper) {
        return new_value.score < old_value.score;
    }
    return bound_priority(new_value.bound) > bound_priority(old_value.bound);
}

void merge_same_key(V37TT::TTValue& old_value, StoreValue new_value) {
    if (old_value.depth == new_value.depth
        && opposite_bounds(old_value.bound, new_value.bound)
        && old_value.score == new_value.score) {
        old_value = V37TT::TTValue{
            new_value.score,
            new_value.move.value != 0 ? new_value.move : old_value.move,
            new_value.depth,
            Bound::Exact
        };
        return;
    }
    if (should_replace_same_key(old_value, new_value)) {
        old_value = to_tt_value(new_value);
    }
}

} // namespace

NnueV37TranspositionTable::NnueV37TranspositionTable(
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

void NnueV37TranspositionTable::clear() {
    CHESS_TT_TIME(timing_stats_.clear_ns, timing_stats_.clear_calls);
    std::fill(keys_.begin(), keys_.end(), HashKey{});
    std::fill(values_.begin(), values_.end(), TTValue{});
}

void NnueV37TranspositionTable::clear_stats() {
    stats_ = {};
}

std::size_t NnueV37TranspositionTable::entry_count() const {
    return keys_.size();
}

std::size_t NnueV37TranspositionTable::bucket_size() const {
    return bucket_size_;
}

std::size_t NnueV37TranspositionTable::bucket_count() const {
    return bucket_count_;
}

const RangeTranspositionTableStats& NnueV37TranspositionTable::stats() const {
    return stats_;
}

#ifdef CHESS_PROFILE_TT_TIMING
void NnueV37TranspositionTable::clear_timing_stats() {
    timing_stats_ = {};
}

const TTFunctionTimingStats& NnueV37TranspositionTable::timing_stats() const {
    return timing_stats_;
}
#endif

#ifdef CHESS_PROFILE_TT_PATH_TIMING
void NnueV37TranspositionTable::clear_probe_path_timing_stats() {
    probe_path_timing_stats_ = {};
}

const TTProbePathTimingStats&
NnueV37TranspositionTable::probe_path_timing_stats() const {
    return probe_path_timing_stats_;
}
#endif

#ifdef CHESS_PROFILE_TT_CACHE_LINES
constexpr std::size_t ProfileCacheLineSize = 128;

void accumulate_cache_line_sample(
    V37TT::ProbeCacheLineStats& stats,
    const V37TT::ProbeCacheLineSample& sample
) {
    ++stats.probes;
    stats.key_hits += sample.key_hit ? 1 : 0;
    stats.empty_misses += sample.empty_miss ? 1 : 0;
    stats.index_collisions += sample.index_collision ? 1 : 0;
    stats.value_reads += sample.value_read ? 1 : 0;
    stats.key_lines += sample.key_lines;
    stats.value_lines += sample.value_lines;
    stats.total_lines += sample.total_lines;
    stats.key_slots_read += sample.key_slots_read;
    if (sample.empty_miss) {
        stats.empty_total_lines += sample.total_lines;
        stats.empty_key_slots_read += sample.key_slots_read;
    } else if (sample.index_collision) {
        stats.index_collision_total_lines += sample.total_lines;
        stats.index_collision_key_slots_read += sample.key_slots_read;
    } else if (sample.key_hit) {
        stats.hit_total_lines += sample.total_lines;
        stats.hit_key_slots_read += sample.key_slots_read;
    }
}

NnueV37TranspositionTable::ProbeCacheLineSample
NnueV37TranspositionTable::probe_cache_line_sample(
    HashKey key,
    std::size_t cache_line_size
) const {
    assert(cache_line_size > 0);
    ProbeCacheLineSample sample{};
    const std::size_t offset = bucket_offset(key);
    const KeyLock key_lock = make_tt_lock(key);

    auto line_of = [cache_line_size](const void* ptr) {
        return reinterpret_cast<std::uintptr_t>(ptr) / cache_line_size;
    };
    auto add_line = [](std::array<std::uintptr_t, 8>& lines, std::uint8_t& count, std::uintptr_t line) {
        for (std::uint8_t i = 0; i < count; ++i) {
            if (lines[i] == line) {
                return;
            }
        }
        lines[count++] = line;
    };

    std::array<std::uintptr_t, 8> key_lines{};
    std::array<std::uintptr_t, 8> value_lines{};

    if (bucket_size_ == 4) {
        for (std::size_t i = 0; i < 4; ++i) {
            const std::size_t index = offset + i;
            add_line(key_lines, sample.key_lines, line_of(&keys_[index]));
            ++sample.key_slots_read;
            const KeyLock entry_key = keys_[index];
            if (i > 0 && entry_key == 0) {
                sample.index_collision = true;
                break;
            }
            if (entry_key == 0) {
                sample.empty_miss = true;
                break;
            }
            if (entry_key == key_lock) {
                sample.key_hit = true;
                sample.value_read = true;
                add_line(value_lines, sample.value_lines, line_of(&values_[index]));
                break;
            }
            if (i == 3) {
                sample.index_collision = true;
            }
        }
    } else {
        bool saw_valid_entry = false;
        for (std::size_t i = 0; i < bucket_size_; ++i) {
            const std::size_t index = offset + i;
            add_line(key_lines, sample.key_lines, line_of(&keys_[index]));
            ++sample.key_slots_read;
            const KeyLock entry_key = keys_[index];
            if (entry_key == 0) {
                break;
            }
            saw_valid_entry = true;
            if (entry_key == key_lock) {
                sample.key_hit = true;
                sample.value_read = true;
                add_line(value_lines, sample.value_lines, line_of(&values_[index]));
                break;
            }
        }
        sample.empty_miss = !sample.key_hit && !saw_valid_entry;
        sample.index_collision = !sample.key_hit && saw_valid_entry;
    }

    sample.total_lines = sample.key_lines + sample.value_lines;
    return sample;
}

void NnueV37TranspositionTable::clear_cache_line_stats() {
    cache_line_stats_ = {};
}

const NnueV37TranspositionTable::ProbeCacheLineStats&
NnueV37TranspositionTable::cache_line_stats() const {
    return cache_line_stats_;
}
#endif

std::size_t NnueV37TranspositionTable::bucket_offset(HashKey key) const {
    return (key & bucket_mask_) * bucket_size_;
}

bool NnueV37TranspositionTable::probe(
    HashKey key,
    int depth,
    ScoreRange& search_window,
    ScoreRange& stored_score,
    MoveRange& stored_move,
    int ply,
    bool& score_available,
    TTDepthPolicy depth_policy
) const {
    CHESS_TT_TIME(timing_stats_.probe_ns, timing_stats_.probe_calls);
#ifdef CHESS_PROFILE_TT_PATH_TIMING
    const auto probe_path_start = std::chrono::steady_clock::now();
#endif
    assert(key != 0);
#ifdef CHESS_PROFILE_TT_CACHE_LINES
    accumulate_cache_line_sample(cache_line_stats_, probe_cache_line_sample(key, ProfileCacheLineSize));
#endif
    score_available = false;
    CHESS_TT_STAT(stats_.probes);

    const std::size_t offset = bucket_offset(key);
    const KeyLock key_lock = make_tt_lock(key);
    if (keys_[offset] == 0) {
        CHESS_TT_STAT(stats_.empty_misses);
#ifdef CHESS_PROFILE_TT_PATH_TIMING
        add_probe_path_timing(
            probe_path_timing_stats_.empty_miss_ns,
            probe_path_timing_stats_.empty_miss_calls,
            probe_path_start);
#endif
        return false;
    }

    auto probe_hit = [&](std::size_t index) {
#ifdef CHESS_PROFILE_TT_PATH_TIMING
        const auto probe_hit_body_start = std::chrono::steady_clock::now();
#endif
        CHESS_TT_STAT(stats_.key_hits);
        const TTValue& value = values_[index];
        assert(value.bound != Bound::Missing);
        assert(value.depth != MissingDepth);
        stored_move = value.bound == Bound::Upper
            ? MoveRange{Move{}, value.move}
            : MoveRange{value.move, Move{}};
        if (value.move.value != 0) {
            CHESS_TT_STAT(stats_.move_hint_hits);
        }

        const int stored_depth = static_cast<int>(value.depth);
        const bool depth_matches = depth_policy == TTDepthPolicy::Exact
            ? stored_depth == depth
            : stored_depth >= depth;
        if (!depth_matches) {
            CHESS_TT_STAT(stats_.depth_misses);
#ifdef CHESS_PROFILE_TT_PATH_TIMING
            add_probe_path_timing(
                probe_path_timing_stats_.probe_hit_body_ns,
                probe_path_timing_stats_.probe_hit_body_calls,
                probe_hit_body_start);
            add_probe_path_timing(
                probe_path_timing_stats_.key_hit_total_ns,
                probe_path_timing_stats_.key_hit_calls,
                probe_path_start);
#endif
            return false;
        }

        const int stored = score_from_table(value.score, ply);
        if (is_mate_score(stored)) {
#ifdef CHESS_PROFILE_TT_PATH_TIMING
            add_probe_path_timing(
                probe_path_timing_stats_.probe_hit_body_ns,
                probe_path_timing_stats_.probe_hit_body_calls,
                probe_hit_body_start);
            add_probe_path_timing(
                probe_path_timing_stats_.key_hit_total_ns,
                probe_path_timing_stats_.key_hit_calls,
                probe_path_start);
#endif
            return false;
        }
        score_available = true;
        stored_score = ScoreRange{};
        if (value.bound == Bound::Exact) {
            stored_score = ScoreRange{stored, stored};
            CHESS_TT_STAT(stats_.exact_hits);
        } else if (value.bound == Bound::Lower) {
            stored_score.lower = stored;
            CHESS_TT_STAT(stats_.lower_score_hits);
        } else if (value.bound == Bound::Upper) {
            stored_score.upper = stored;
            CHESS_TT_STAT(stats_.upper_score_hits);
        } else {
            assert(false);
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
        if (value.bound == Bound::Exact || enough_to_return) {
            if (value.bound != Bound::Exact) {
                CHESS_TT_STAT(stats_.score_returns);
            }
#ifdef CHESS_PROFILE_TT_PATH_TIMING
            add_probe_path_timing(
                probe_path_timing_stats_.probe_hit_body_ns,
                probe_path_timing_stats_.probe_hit_body_calls,
                probe_hit_body_start);
            add_probe_path_timing(
                probe_path_timing_stats_.key_hit_total_ns,
                probe_path_timing_stats_.key_hit_calls,
                probe_path_start);
#endif
            return true;
        }
#ifdef CHESS_PROFILE_TT_PATH_TIMING
        add_probe_path_timing(
            probe_path_timing_stats_.probe_hit_body_ns,
            probe_path_timing_stats_.probe_hit_body_calls,
            probe_hit_body_start);
        add_probe_path_timing(
            probe_path_timing_stats_.key_hit_total_ns,
            probe_path_timing_stats_.key_hit_calls,
            probe_path_start);
#endif
        return false;
    };

    if (bucket_size_ == 4) {
        const KeyLock key0 = keys_[offset];
        if (key0 == key_lock) {
            return probe_hit(offset);
        }
        const KeyLock key1 = keys_[offset + 1];
        if (key1 == 0) {
            CHESS_TT_STAT(stats_.index_collisions);
#ifdef CHESS_PROFILE_TT_PATH_TIMING
            add_probe_path_timing(
                probe_path_timing_stats_.index_collision_ns,
                probe_path_timing_stats_.index_collision_calls,
                probe_path_start);
#endif
            return false;
        }
        if (key1 == key_lock) {
            return probe_hit(offset + 1);
        }
        const KeyLock key2 = keys_[offset + 2];
        if (key2 == 0) {
            CHESS_TT_STAT(stats_.index_collisions);
#ifdef CHESS_PROFILE_TT_PATH_TIMING
            add_probe_path_timing(
                probe_path_timing_stats_.index_collision_ns,
                probe_path_timing_stats_.index_collision_calls,
                probe_path_start);
#endif
            return false;
        }
        if (key2 == key_lock) {
            return probe_hit(offset + 2);
        }
        const KeyLock key3 = keys_[offset + 3];
        if (key3 == key_lock) {
            return probe_hit(offset + 3);
        }
        CHESS_TT_STAT(stats_.index_collisions);
#ifdef CHESS_PROFILE_TT_PATH_TIMING
        add_probe_path_timing(
            probe_path_timing_stats_.index_collision_ns,
            probe_path_timing_stats_.index_collision_calls,
            probe_path_start);
#endif
        return false;
    }

    bool saw_valid_entry = false;
    for (std::size_t i = 0; i < bucket_size_; ++i) {
        const std::size_t index = offset + i;
        const KeyLock entry_key = keys_[index];
        if (entry_key == 0) {
            break;
        }
        saw_valid_entry = true;
        if (entry_key != key_lock) {
            continue;
        }
        return probe_hit(index);
    }

    if (saw_valid_entry) {
        CHESS_TT_STAT(stats_.index_collisions);
#ifdef CHESS_PROFILE_TT_PATH_TIMING
        add_probe_path_timing(
            probe_path_timing_stats_.index_collision_ns,
            probe_path_timing_stats_.index_collision_calls,
            probe_path_start);
#endif
    } else {
        CHESS_TT_STAT(stats_.empty_misses);
#ifdef CHESS_PROFILE_TT_PATH_TIMING
        add_probe_path_timing(
            probe_path_timing_stats_.empty_miss_ns,
            probe_path_timing_stats_.empty_miss_calls,
            probe_path_start);
#endif
    }
    return false;
}

void NnueV37TranspositionTable::store(
    HashKey key,
    int depth,
    ScoreRange score,
    MoveRange move
) {
    CHESS_TT_TIME(timing_stats_.store_ns, timing_stats_.store_calls);
    const std::optional<StoreValue> new_value = make_store_value(depth, score, move);
    if (!new_value) {
        return;
    }

    assert(key != 0);
    CHESS_TT_STAT(stats_.stores);
    const std::size_t offset = bucket_offset(key);
    const KeyLock key_lock = make_tt_lock(key);

    std::size_t weakest_index = offset;
    int weakest_priority = std::numeric_limits<int>::max();
    for (std::size_t i = 0; i < bucket_size_; ++i) {
        const std::size_t index = offset + i;
        const KeyLock entry_key = keys_[index];
        if (entry_key == key_lock) {
            CHESS_TT_STAT(stats_.same_key_updates);
            merge_same_key(values_[index], *new_value);
            return;
        }
        if (entry_key == 0) {
            CHESS_TT_STAT(stats_.new_stores);
            keys_[index] = key_lock;
            values_[index] = to_tt_value(*new_value);
            return;
        }
        const int priority = replacement_priority(values_[index]);
        if (priority < weakest_priority) {
            weakest_priority = priority;
            weakest_index = index;
        }
    }

    const int new_priority =
        static_cast<int>(new_value->depth) * 3 + bound_priority(new_value->bound);
    if (new_priority > weakest_priority) {
        CHESS_TT_STAT(stats_.replacement_collisions);
        keys_[weakest_index] = key_lock;
        values_[weakest_index] = to_tt_value(*new_value);
    } else {
        CHESS_TT_STAT(stats_.skipped_shallow_replacements);
    }
}

} // namespace chess
