#include "nnue_searcher_v38.hpp"

namespace chess {

NnueSearcherV38::NnueSearcherV38(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : NnueSearcherV38(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{},
        SelectiveConfig{}
    ) {
}

NnueSearcherV38::NnueSearcherV38(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : NnueSearcherV38(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        weights,
        SelectiveConfig{}
    ) {
}

NnueSearcherV38::NnueSearcherV38(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights,
    SelectiveConfig selective_config
)
    : tt_(tt_mb, bucket_size),
      history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      counter_history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      model_(model),
      move_ordering_weights_(weights),
      selective_config_(selective_config) {
    move_ordering_weights_.counter_history_bonus = counter_history_bonus;
}

std::string_view NnueSearcherV38::name() const {
    return "nnue_selective_v38";
}

void NnueSearcherV38::clear_tt() {
    tt_.clear();
}

void NnueSearcherV38::clear_search_heuristics() {
    killer_table_.clear();
    counter_move_table_.clear();
    history_table_.reset();
    counter_history_table_.reset();
}

std::size_t NnueSearcherV38::tt_entry_count() const {
    return tt_.entry_count();
}

void NnueSearcherV38::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& NnueSearcherV38::tt_stats() const {
    return tt_.stats();
}

void NnueSearcherV38::clear_move_ordering_stats() {
    move_ordering_stats_ = MoveOrderingStats{};
}

const NnueSearcherV38::MoveOrderingStats&
NnueSearcherV38::move_ordering_stats() const {
    return move_ordering_stats_;
}

void NnueSearcherV38::set_move_ordering_stats_enabled(bool enabled) {
    move_ordering_stats_enabled_ = enabled;
}

void NnueSearcherV38::clear_selective_stats() {
    selective_stats_ = SelectiveStats{};
}

const NnueSearcherV38::SelectiveStats&
NnueSearcherV38::selective_stats() const {
    return selective_stats_;
}

const RepetitionStack::Stats& NnueSearcherV38::repetition_stats() const {
    return repetition_stats_;
}

void NnueSearcherV38::set_selective_config(SelectiveConfig config) {
    selective_config_ = config;
}

const NnueSearcherV38::SelectiveConfig&
NnueSearcherV38::selective_config() const {
    return selective_config_;
}

} // namespace chess
