#include "heuristic_searcher_v28.hpp"

namespace chess {

HeuristicSearcherV28::HeuristicSearcherV28(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : HeuristicSearcherV28(
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

HeuristicSearcherV28::HeuristicSearcherV28(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : tt_(tt_mb, bucket_size),
      history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      counter_history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      move_ordering_weights_(weights) {
    move_ordering_weights_.counter_history_bonus = counter_history_bonus;
}

std::string_view HeuristicSearcherV28::name() const {
    return "heuristic_v28";
}

void HeuristicSearcherV28::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV28::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV28::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& HeuristicSearcherV28::tt_stats() const {
    return tt_.stats();
}

} // namespace chess
