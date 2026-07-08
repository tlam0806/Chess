#pragma once

#include "range_transposition_table.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class SingleBoundBucketTranspositionTable {
public:
    using KeyLock = std::uint32_t;

    enum class Bound : std::uint8_t {
        Upper = 0,
        Lower = 1,
        Exact = 2,
        Missing = 255
    };

    SingleBoundBucketTranspositionTable(std::size_t megabytes, std::size_t bucket_size);

    void clear();
    void clear_stats();

    std::size_t entry_count() const;
    std::size_t bucket_size() const;
    std::size_t bucket_count() const;
    const RangeTranspositionTableStats& stats() const;
#ifdef CHESS_PROFILE_TT_TIMING
    void clear_timing_stats();
    const TTFunctionTimingStats& timing_stats() const;
#endif
#ifdef CHESS_PROFILE_TT_PATH_TIMING
    void clear_probe_path_timing_stats();
    const TTProbePathTimingStats& probe_path_timing_stats() const;
#endif
#ifdef CHESS_PROFILE_TT_CACHE_LINES
    struct ProbeCacheLineSample {
        std::uint8_t key_lines = 0;
        std::uint8_t value_lines = 0;
        std::uint8_t total_lines = 0;
        std::uint8_t key_slots_read = 0;
        bool key_hit = false;
        bool value_read = false;
        bool empty_miss = false;
        bool index_collision = false;
    };
    struct ProbeCacheLineStats {
        std::uint64_t probes = 0;
        std::uint64_t key_hits = 0;
        std::uint64_t empty_misses = 0;
        std::uint64_t index_collisions = 0;
        std::uint64_t value_reads = 0;
        std::uint64_t key_lines = 0;
        std::uint64_t value_lines = 0;
        std::uint64_t total_lines = 0;
        std::uint64_t key_slots_read = 0;
        std::uint64_t empty_total_lines = 0;
        std::uint64_t index_collision_total_lines = 0;
        std::uint64_t hit_total_lines = 0;
        std::uint64_t empty_key_slots_read = 0;
        std::uint64_t index_collision_key_slots_read = 0;
        std::uint64_t hit_key_slots_read = 0;
    };
    void clear_cache_line_stats();
    const ProbeCacheLineStats& cache_line_stats() const;
    ProbeCacheLineSample probe_cache_line_sample(HashKey key, std::size_t cache_line_size) const;
#endif

    bool probe(
        HashKey key,
        int depth,
        ScoreRange& search_window,
        ScoreRange& stored_score,
        MoveRange& stored_move,
        int ply,
        bool& score_available,
        TTDepthPolicy depth_policy = TTDepthPolicy::AtLeast
    ) const;

    void store(
        HashKey key,
        int depth,
        ScoreRange score,
        MoveRange move
    );

    struct TTValue {
        int score = 0;
        Move move{};
        std::uint8_t depth = 255;
        Bound bound = Bound::Missing;
    };
    static_assert(sizeof(TTValue) == 8);

private:
    std::size_t bucket_offset(HashKey key) const;

    std::vector<KeyLock> keys_;
    std::vector<TTValue> values_;
    std::size_t bucket_size_ = 1;
    std::size_t bucket_count_ = 1;
    std::size_t bucket_mask_ = 0;
    mutable RangeTranspositionTableStats stats_{};
#ifdef CHESS_PROFILE_TT_TIMING
    mutable TTFunctionTimingStats timing_stats_{};
#endif
#ifdef CHESS_PROFILE_TT_PATH_TIMING
    mutable TTProbePathTimingStats probe_path_timing_stats_{};
#endif
#ifdef CHESS_PROFILE_TT_CACHE_LINES
    mutable ProbeCacheLineStats cache_line_stats_{};
#endif
};

} // namespace chess
