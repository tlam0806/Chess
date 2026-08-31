#pragma once

#include "range_transposition_table.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class LowerMoveRangeBucketTranspositionTable {
public:
    LowerMoveRangeBucketTranspositionTable(std::size_t megabytes, std::size_t bucket_size);

    void clear();
    void advance_generation();
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
        static constexpr std::uint32_t ScoreBits = 21;
        static constexpr std::uint32_t ScoreMask = (1u << ScoreBits) - 1u;
        static constexpr std::uint32_t GenerationShift = 24;
        static constexpr std::uint32_t GenerationMask = 0xffu << GenerationShift;

        // Every score stored by the search is in [-Infinity, Infinity], which
        // has 2,000,001 values and therefore fits losslessly in 21 bits.  Pack
        // the generation into the otherwise unused high bits of the lower
        // score word.  The upper score remains a directly readable int32_t;
        // the value still occupies 12 bytes without a side generation array.
        std::uint32_t lower_score_and_generation = 0;
        int upper_score_value = Infinity;
        Move lower_move{};
        std::uint8_t lower_depth = 255;
        std::uint8_t upper_depth = 255;

        int lower_score() const {
            return decode_score(lower_score_and_generation & ScoreMask);
        }

        int upper_score() const {
            return upper_score_value;
        }

        void set_lower_score(int score) {
            lower_score_and_generation =
                (lower_score_and_generation & GenerationMask) | encode_score(score);
        }

        void set_upper_score(int score) {
            assert(score >= -Infinity && score <= Infinity);
            upper_score_value = score;
        }

        std::uint8_t generation() const {
            return static_cast<std::uint8_t>(
                lower_score_and_generation >> GenerationShift);
        }

        void set_generation(std::uint8_t generation_value) {
            lower_score_and_generation =
                (lower_score_and_generation & ScoreMask)
                | (static_cast<std::uint32_t>(generation_value) << GenerationShift);
        }

    private:
        static std::uint32_t encode_score(int score) {
            assert(score >= -Infinity && score <= Infinity);
            return static_cast<std::uint32_t>(score + Infinity);
        }

        static int decode_score(std::uint32_t encoded_score) {
            assert(encoded_score <= 2u * static_cast<std::uint32_t>(Infinity));
            return static_cast<int>(encoded_score) - Infinity;
        }
    };
    static_assert(2u * static_cast<std::uint32_t>(Infinity) <= TTValue::ScoreMask);
    static_assert(sizeof(TTValue) == 12);

private:
    std::size_t bucket_offset(HashKey key) const;

    std::vector<HashKey> keys_;
    std::vector<TTValue> values_;
    std::size_t bucket_size_ = 1;
    std::size_t bucket_count_ = 1;
    std::size_t bucket_mask_ = 0;
    std::uint8_t generation_ = 0;
    mutable RangeTranspositionTableStats stats_{};
};

} // namespace chess
