#pragma once

#include "range_transposition_table.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class LowerBucketTranspositionTable {
public:
    LowerBucketTranspositionTable(std::size_t megabytes, std::size_t bucket_size);

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

    struct TTValue {
        int lower_score = -Infinity;
        Move lower_move{};
        std::uint8_t lower_depth = 255;
    };

private:
    std::size_t bucket_offset(HashKey key) const;

    std::vector<HashKey> keys_;
    std::vector<TTValue> values_;
    std::size_t bucket_size_ = 1;
    std::size_t bucket_count_ = 1;
    std::size_t bucket_mask_ = 0;
    mutable RangeTranspositionTableStats stats_{};
};

} // namespace chess
