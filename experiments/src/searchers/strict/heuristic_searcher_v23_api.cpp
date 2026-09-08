#include "heuristic_searcher_v23.hpp"

namespace chess {

HeuristicSearcherV23::HeuristicSearcherV23(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : HeuristicSearcherV23(
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

HeuristicSearcherV23::HeuristicSearcherV23(
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

std::string_view HeuristicSearcherV23::name() const {
    return "heuristic_v23";
}

void HeuristicSearcherV23::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV23::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV23::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& HeuristicSearcherV23::tt_stats() const {
    return tt_.stats();
}

void HeuristicSearcherV23::clear_move_ordering_stats() {
    move_ordering_stats_ = MoveOrderingStats{};
}

const HeuristicSearcherV23::MoveOrderingStats& HeuristicSearcherV23::move_ordering_stats() const {
    return move_ordering_stats_;
}

void HeuristicSearcherV23::set_move_ordering_stats_enabled(bool enabled) {
    move_ordering_stats_enabled_ = enabled;
}

} // namespace chess
