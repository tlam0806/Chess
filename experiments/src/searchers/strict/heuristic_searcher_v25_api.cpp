#include "heuristic_searcher_v25.hpp"

namespace chess {

HeuristicSearcherV25::HeuristicSearcherV25(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : HeuristicSearcherV25(
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

HeuristicSearcherV25::HeuristicSearcherV25(
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

std::string_view HeuristicSearcherV25::name() const {
    return "heuristic_v24";
}

void HeuristicSearcherV25::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV25::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV25::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& HeuristicSearcherV25::tt_stats() const {
    return tt_.stats();
}

void HeuristicSearcherV25::clear_move_ordering_stats() {
    move_ordering_stats_ = MoveOrderingStats{};
}

const HeuristicSearcherV25::MoveOrderingStats& HeuristicSearcherV25::move_ordering_stats() const {
    return move_ordering_stats_;
}

void HeuristicSearcherV25::set_move_ordering_stats_enabled(bool enabled) {
    move_ordering_stats_enabled_ = enabled;
}

void HeuristicSearcherV25::clear_timing_stats() {
    timing_stats_ = TimingStats{};
}

const HeuristicSearcherV25::TimingStats& HeuristicSearcherV25::timing_stats() const {
    return timing_stats_;
}

void HeuristicSearcherV25::set_timing_stats_enabled(bool enabled) {
    timing_stats_enabled_ = enabled;
}

} // namespace chess
