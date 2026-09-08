#pragma once

#include "nn_value.hpp"
#include "searcher.hpp"
#include "transposition_table.hpp"

#include <cstddef>

namespace chess {

class NnSearcherV6 final : public Searcher {
public:
    explicit NnSearcherV6(const NnValueModel& model, std::size_t tt_mb = 64);

    SearchResult search_best_move(const Position& pos, int depth) override;
    SearchResult search_best_move(const Position& pos, const SearchLimits& limits) override;
    std::string_view name() const override;

    void clear_tt();
    std::size_t tt_entry_count() const;

private:
    const NnValueModel& model_;
    TranspositionTable tt_;
};

} // namespace chess
