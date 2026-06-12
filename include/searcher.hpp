#pragma once

#include "position.hpp"
#include "search_types.hpp"

#include <string_view>

namespace chess {

class Searcher {
public:
    virtual ~Searcher() = default;

    SearchResult search(const Position& pos, int depth);
    SearchResult search(const Position& pos, const SearchLimits& limits);

    virtual SearchResult search_best_move(const Position& pos, int depth) = 0;
    virtual SearchResult search_best_move(const Position& pos, const SearchLimits& limits);
    virtual std::string_view name() const = 0;
};

} // namespace chess
