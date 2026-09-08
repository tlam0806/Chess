#pragma once

#include "move.hpp"
#include "search_types.hpp"
#include "types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace chess {

// V44 owns a fresh table for each top-level search. Entries therefore need no
// cross-search age metadata: all surviving entries belong to the current
// iterative-deepening search and are immediately eligible for move ordering
// and score cutoffs.
//
// This table is intentionally single-search-thread only. Entry key/payload
// writes are not atomic; callers must finish (join) an old search before a new
// search can probe or store in the same table. A separate thread may only set
// the searcher's atomic stop flag.
class V44SingleBoundTranspositionTable {
public:
    enum class Bound : std::uint8_t {
        None = 0,
        Upper = 1,
        Lower = 2,
        Exact = 3
    };

    enum class DepthPolicy : std::uint8_t {
        Exact,
        AtLeast
    };

    struct ProbeResult {
        Move move{};
        int score = 0;
        int depth = -1;
        Bound bound = Bound::None;
        bool key_hit = false;
        bool score_usable = false;
        bool cutoff = false;
    };

    struct DecodedEntry {
        HashKey key = 0;
        Move move{};
        int score = 0;
        int depth = -1;
        Bound bound = Bound::None;
    };

    struct alignas(16) Entry {
        HashKey key = 0;
        std::uint64_t payload = 0;
    };

    struct alignas(64) Bucket {
        std::array<Entry, 4> entries{};
    };

    static_assert(sizeof(Entry) == 16);
    static_assert(alignof(Entry) == 16);
    static_assert(sizeof(Bucket) == 64);
    static_assert(alignof(Bucket) == 64);

    explicit V44SingleBoundTranspositionTable(std::size_t megabytes);

    void clear();

    std::size_t entry_count() const noexcept;
    std::size_t occupied_entry_count() const noexcept;
    std::size_t bucket_count() const noexcept;
    std::size_t memory_bytes() const noexcept;

    ProbeResult probe(
        HashKey key,
        int depth,
        int alpha,
        int beta,
        int ply,
        DepthPolicy depth_policy = DepthPolicy::Exact
    ) const noexcept;

    // score is in the caller/root-relative mate convention. The table
    // normalizes mate scores on store and reverses that normalization on
    // probe, so V44 must not pre-adjust the score itself.
    bool store(
        HashKey key,
        int depth,
        int score,
        Bound bound,
        Move move,
        int ply
    ) noexcept;

    bool store_move(HashKey key, int depth, Move move) noexcept;

    // Read-only diagnostics for unit tests and microbenchmarks. They are not
    // used by the search hot path.
    std::optional<DecodedEntry> inspect(HashKey key) const noexcept;
    const Bucket* bucket_address(HashKey key) const noexcept;

private:
    std::size_t bucket_index(HashKey key) const noexcept;

    std::vector<Bucket> buckets_;
    std::size_t bucket_mask_ = 0;
};

} // namespace chess
