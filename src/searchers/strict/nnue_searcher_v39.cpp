#include "nnue_searcher_v39.hpp"

namespace chess {

NnueSearcherV38::SelectiveConfig NnueSearcherV39::to_v38_config(
    const SelectiveConfig& config
) {
    NnueSearcherV38::SelectiveConfig result;
    result.enable_lmr = config.enable_lmr;
    result.lmr_base = config.lmr_base;
    result.lmr_divisor = config.lmr_divisor;
    result.lmr_min_depth = config.lmr_min_depth;
    result.lmr_min_move_index = config.lmr_min_move_index;
    result.enable_null_move = config.enable_null_move;
    result.null_move_min_depth = config.null_move_min_depth;
    result.null_move_reduction = config.null_move_reduction;
    result.enable_reverse_futility = config.enable_reverse_futility;
    result.reverse_futility_max_depth = config.reverse_futility_max_depth;
    result.reverse_futility_base_margin =
        config.reverse_futility_base_margin;
    result.reverse_futility_margin_per_depth =
        config.reverse_futility_margin_per_depth;
    result.enable_late_move_pruning = config.enable_late_move_pruning;
    result.late_move_pruning_max_depth =
        config.late_move_pruning_max_depth;
    result.late_move_pruning_base = config.late_move_pruning_base;
    result.late_move_pruning_depth_multiplier =
        config.late_move_pruning_depth_multiplier;
    return result;
}

NnueSearcherV39::NnueSearcherV39(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : NnueSearcherV39(
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

NnueSearcherV39::NnueSearcherV39(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : NnueSearcherV39(
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

NnueSearcherV39::NnueSearcherV39(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights,
    SelectiveConfig selective_config
)
    : selective_config_(selective_config),
      searcher_(
          model,
          tt_mb,
          bucket_size,
          history_penalty_divisor_numerator,
          history_penalty_divisor_denominator,
          counter_history_bonus,
          weights,
          to_v38_config(selective_config)) {
}

SearchResult NnueSearcherV39::search_best_move(
    const Position& pos,
    int depth
) {
    return searcher_.search_best_move(pos, depth);
}

SearchResult NnueSearcherV39::search_best_move(
    const Position& pos,
    const SearchLimits& limits
) {
    return searcher_.search_best_move(pos, limits);
}

std::string_view NnueSearcherV39::name() const {
    return "nnue_selective_v39";
}

void NnueSearcherV39::clear_tt() {
    searcher_.clear_tt();
}

std::size_t NnueSearcherV39::tt_entry_count() const {
    return searcher_.tt_entry_count();
}

void NnueSearcherV39::clear_tt_stats() {
    searcher_.clear_tt_stats();
}

const RangeTranspositionTableStats& NnueSearcherV39::tt_stats() const {
    return searcher_.tt_stats();
}

void NnueSearcherV39::clear_move_ordering_stats() {
    searcher_.clear_move_ordering_stats();
}

const NnueSearcherV39::MoveOrderingStats&
NnueSearcherV39::move_ordering_stats() const {
    return searcher_.move_ordering_stats();
}

void NnueSearcherV39::set_move_ordering_stats_enabled(bool enabled) {
    searcher_.set_move_ordering_stats_enabled(enabled);
}

void NnueSearcherV39::clear_selective_stats() {
    searcher_.clear_selective_stats();
}

const NnueSearcherV39::SelectiveStats&
NnueSearcherV39::selective_stats() const {
    return searcher_.selective_stats();
}

void NnueSearcherV39::set_selective_config(SelectiveConfig config) {
    selective_config_ = config;
    searcher_.set_selective_config(to_v38_config(config));
}

const NnueSearcherV39::SelectiveConfig&
NnueSearcherV39::selective_config() const {
    return selective_config_;
}

} // namespace chess
