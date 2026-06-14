#pragma once

#include "move.hpp"
#include "search_types.hpp"
#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

enum class TTBound : std::uint8_t {
    Exact,
    Lower,
    Upper
};

struct TTEntry {
    HashKey key = 0;
    bool valid = false;
    int depth = -1;
    int score = 0;
    TTBound bound = TTBound::Exact;
    Move best_move{};
};

struct TranspositionTableStats {
    std::uint64_t probes = 0;
    std::uint64_t empty_misses = 0;
    std::uint64_t index_collisions = 0;
    std::uint64_t key_hits = 0;
    std::uint64_t move_hint_hits = 0;
    std::uint64_t depth_misses = 0;
    std::uint64_t exact_hits = 0;
    std::uint64_t lower_bound_hits = 0;
    std::uint64_t upper_bound_hits = 0;
    std::uint64_t bound_tightens = 0;
    std::uint64_t bound_cutoffs = 0;
    std::uint64_t stores = 0;
    std::uint64_t new_stores = 0;
    std::uint64_t same_key_updates = 0;
    std::uint64_t replacement_collisions = 0;
    std::uint64_t skipped_shallow_replacements = 0;
};

struct TTProbeOptions {
    bool use_exact = true;
    bool use_bounds = true;
    bool allow_bound_cutoff = true;
    bool allow_bound_tighten = true;
    bool normalize_mate_scores = false;
    bool use_mate_scores = true;
    int ply = 0;
};

class TranspositionTable {
public:
    explicit TranspositionTable(std::size_t megabytes);

    void clear();
    void clear_stats();

    std::size_t entry_count() const;
    const TranspositionTableStats& stats() const;

    bool probe(
        HashKey key,
        int depth,
        int& alpha,
        int& beta,
        int& score,
        Move& best_move
    ) const;

    bool probe(
        HashKey key,
        int depth,
        int& alpha,
        int& beta,
        int& score,
        Move& best_move,
        TTProbeOptions options,
        TTBound* hit_bound = nullptr
    ) const;

    void store(
        HashKey key,
        int depth,
        int score,
        TTBound bound,
        Move best_move
    );

private:
    std::vector<TTEntry> entries_;
    mutable TranspositionTableStats stats_{};
};

} // namespace chess
