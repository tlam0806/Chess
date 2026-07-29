#include "nnue_searcher_v37.hpp"

namespace chess {

NnueSearcherV37::NnueSearcherV37(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : NnueSearcherV37(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

NnueSearcherV37::NnueSearcherV37(
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

std::string_view NnueSearcherV37::name() const {
    return "nnue_strict_v37";
}

void NnueSearcherV37::clear_tt() {
    tt_.clear();
}

std::size_t NnueSearcherV37::tt_entry_count() const {
    return tt_.entry_count();
}

void NnueSearcherV37::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& NnueSearcherV37::tt_stats() const {
    return tt_.stats();
}

void NnueSearcherV37::clear_move_ordering_stats() {
    move_ordering_stats_ = MoveOrderingStats{};
}

const NnueSearcherV37::MoveOrderingStats&
NnueSearcherV37::move_ordering_stats() const {
    return move_ordering_stats_;
}

void NnueSearcherV37::set_move_ordering_stats_enabled(bool enabled) {
    move_ordering_stats_enabled_ = enabled;
}

} // namespace chess
