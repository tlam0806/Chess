#include "heuristic_searcher_v29_see.hpp"

namespace chess {

HeuristicSearcherV29See::HeuristicSearcherV29See(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : HeuristicSearcherV29See(
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

HeuristicSearcherV29See::HeuristicSearcherV29See(
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

std::string_view HeuristicSearcherV29See::name() const {
    return "heuristic_v29_see";
}

void HeuristicSearcherV29See::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV29See::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV29See::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& HeuristicSearcherV29See::tt_stats() const {
    return tt_.stats();
}

void HeuristicSearcherV29See::clear_see_timing_stats() {
    see_timing_stats_ = SeeTimingStats{};
}

HeuristicSearcherV29See::SeeTimingStats HeuristicSearcherV29See::see_timing_stats() const {
    return see_timing_stats_;
}

} // namespace chess
