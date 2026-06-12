#pragma once

#include "move.hpp"
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

class TranspositionTable {
public:
    explicit TranspositionTable(std::size_t megabytes);

    void clear();

    std::size_t entry_count() const;

    bool probe(
        HashKey key,
        int depth,
        int& alpha,
        int& beta,
        int& score,
        Move& best_move
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
};

} // namespace chess
