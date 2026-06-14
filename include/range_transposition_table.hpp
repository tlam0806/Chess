#pragma once

#include "move.hpp"
#include "search_types.hpp"
#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

struct ScoreRange {
    int lower = -Infinity;
    int upper = Infinity;
};

struct MoveRange {
    Move lower{};
    Move upper{};
};

enum class TTDepthPolicy : std::uint8_t {
    Exact,
    AtLeast
};

struct RangeTTEntry {
    HashKey key = 0;
    bool valid = false;
    int depth = -1;
    ScoreRange score{};
    MoveRange move{};
};

struct RangeTranspositionTableStats {
    std::uint64_t probes = 0;
    std::uint64_t empty_misses = 0;
    std::uint64_t index_collisions = 0;
    std::uint64_t key_hits = 0;
    std::uint64_t move_hint_hits = 0;
    std::uint64_t depth_misses = 0;
    std::uint64_t exact_hits = 0;
    std::uint64_t lower_score_hits = 0;
    std::uint64_t upper_score_hits = 0;
    std::uint64_t window_narrowings = 0;
    std::uint64_t score_returns = 0;
    std::uint64_t stores = 0;
    std::uint64_t new_stores = 0;
    std::uint64_t same_key_updates = 0;
    std::uint64_t replacement_collisions = 0;
    std::uint64_t skipped_shallow_replacements = 0;
};

class RangeTranspositionTable {
public:
    explicit RangeTranspositionTable(std::size_t megabytes);

    void clear();
    void clear_stats();

    std::size_t entry_count() const;
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
    std::vector<RangeTTEntry> entries_;
    mutable RangeTranspositionTableStats stats_{};
};

} // namespace chess
