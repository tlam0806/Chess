#pragma once

#include "range_transposition_table.hpp"

#include <cstddef>
#include <vector>

namespace chess {

class RangeBucketTranspositionTable {
public:
    RangeBucketTranspositionTable(std::size_t megabytes, std::size_t bucket_size);

    void clear();
    void clear_stats();

    std::size_t entry_count() const;
    std::size_t bucket_size() const;
    std::size_t bucket_count() const;
    const RangeTranspositionTableStats& stats() const;

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

private:
    RangeTTEntry* bucket_begin(HashKey key);
    const RangeTTEntry* bucket_begin(HashKey key) const;

    std::vector<RangeTTEntry> entries_;
    std::size_t bucket_size_ = 1;
    std::size_t bucket_count_ = 1;
    std::size_t bucket_mask_ = 0;
    mutable RangeTranspositionTableStats stats_{};
};

} // namespace chess
