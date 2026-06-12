#include "searcher.hpp"

#include <algorithm>
#include <chrono>

namespace chess {

SearchResult Searcher::search(const Position& pos, int depth) {
    return search_best_move(pos, depth);
}

SearchResult Searcher::search(const Position& pos, const SearchLimits& limits) {
    return search_best_move(pos, limits);
}

SearchResult Searcher::search_best_move(const Position& pos, const SearchLimits& limits) {
    SearchResult best;
    const int max_depth = std::max(limits.max_depth, 0);

    const bool has_deadline = limits.move_time.count() > 0;
    const auto deadline = std::chrono::steady_clock::now() + limits.move_time;

    for (int depth = 0; depth <= max_depth; ++depth) {
        if (has_deadline && depth > 0 && std::chrono::steady_clock::now() >= deadline) {
            best.stopped = true;
            return best;
        }
        best = search_best_move(pos, depth);
        best.depth = depth;
    }
    return best;
}

} // namespace chess
