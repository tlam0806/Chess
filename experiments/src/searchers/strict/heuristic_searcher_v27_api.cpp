#include "heuristic_searcher_v27.hpp"

namespace chess {

HeuristicSearcherV27::HeuristicSearcherV27(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : HeuristicSearcherV27(
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

HeuristicSearcherV27::HeuristicSearcherV27(
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

std::string_view HeuristicSearcherV27::name() const {
    return "heuristic_v27";
}

void HeuristicSearcherV27::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV27::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV27::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& HeuristicSearcherV27::tt_stats() const {
    return tt_.stats();
}

void HeuristicSearcherV27::clear_move_ordering_stats() {
    move_ordering_stats_ = MoveOrderingStats{};
}

const HeuristicSearcherV27::MoveOrderingStats& HeuristicSearcherV27::move_ordering_stats() const {
    return move_ordering_stats_;
}

void HeuristicSearcherV27::set_move_ordering_stats_enabled(bool enabled) {
    move_ordering_stats_enabled_ = enabled;
}

void HeuristicSearcherV27::clear_timing_stats() {
    timing_stats_ = TimingStats{};
}

const HeuristicSearcherV27::TimingStats& HeuristicSearcherV27::timing_stats() const {
    return timing_stats_;
}

void HeuristicSearcherV27::set_timing_stats_enabled(bool enabled) {
    timing_stats_enabled_ = enabled;
}

} // namespace chess
