#pragma once

#include "searcher.hpp"

namespace chess {

class HeuristicSearcher final : public Searcher {
public:
    SearchResult search_best_move(const Position& pos, int depth) override;
    SearchResult search_best_move(const Position& pos, const SearchLimits& limits) override;
    std::string_view name() const override;
};

} // namespace chess
