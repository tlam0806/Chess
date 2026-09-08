#include "nnue_searcher_v36.hpp"

namespace chess {

NnueSearcherV36::NnueSearcherV36(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : NnueSearcherV36(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

NnueSearcherV36::NnueSearcherV36(
    const PhaseQuantizedNnueModel& model,
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
      model_(model),
      move_ordering_weights_(weights) {
    move_ordering_weights_.counter_history_bonus = counter_history_bonus;
}

std::string_view NnueSearcherV36::name() const {
    return "nnue_strict_v36";
}

void NnueSearcherV36::clear_tt() {
    tt_.clear();
}

void NnueSearcherV36::clear_search_heuristics() {
    killer_table_.clear();
    counter_move_table_.clear();
    history_table_.reset();
    counter_history_table_.reset();
}

std::size_t NnueSearcherV36::tt_entry_count() const {
    return tt_.entry_count();
}

void NnueSearcherV36::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& NnueSearcherV36::tt_stats() const {
    return tt_.stats();
}

void NnueSearcherV36::clear_move_ordering_stats() {
    move_ordering_stats_ = MoveOrderingStats{};
}

const NnueSearcherV36::MoveOrderingStats&
NnueSearcherV36::move_ordering_stats() const {
    return move_ordering_stats_;
}

void NnueSearcherV36::set_move_ordering_stats_enabled(bool enabled) {
    move_ordering_stats_enabled_ = enabled;
}

bool NnueSearcherV36::last_search_exact() const {
    return last_search_exact_;
}

const NnueSearcherV36::ExactnessStats&
NnueSearcherV36::exactness_stats() const {
    return exactness_stats_;
}

} // namespace chess
