#pragma once

#include "nn_value.hpp"
#include "searcher.hpp"

namespace chess {

class NnSearcher final : public Searcher {
public:
    explicit NnSearcher(const NnValueModel& model);

    SearchResult search_best_move(const Position& pos, int depth) override;
    std::string_view name() const override;

private:
    const NnValueModel& model_;
};

} // namespace chess
