#include "nn_searcher.hpp"

#include "nn_search.hpp"

namespace chess {

NnSearcher::NnSearcher(const NnValueModel& model)
    : model_(model) {
}

SearchResult NnSearcher::search_best_move(const Position& pos, int depth) {
    return search_best_move_nn(pos, depth, model_);
}

std::string_view NnSearcher::name() const {
    return "nn";
}

} // namespace chess
