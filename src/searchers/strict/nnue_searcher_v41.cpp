#include "nnue_searcher_v41.hpp"

namespace chess {

NnueSearcherV41::NnueSearcherV41(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : NnueSearcherV41(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{},
        SelectiveConfig{}) {
}

NnueSearcherV41::NnueSearcherV41(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : NnueSearcherV41(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        weights,
        SelectiveConfig{}) {
}

NnueSearcherV41::NnueSearcherV41(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights,
    SelectiveConfig selective_config
)
    : searcher_(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        weights,
        selective_config) {
}

SearchResult NnueSearcherV41::search_best_move(
    const Position& pos,
    int depth
) {
    return searcher_.search_best_move(pos, depth, {});
}

SearchResult NnueSearcherV41::search_best_move(
    const Position& pos,
    const SearchLimits& limits
) {
    return searcher_.search_best_move(pos, limits, {});
}

SearchResult NnueSearcherV41::search_best_move(
    const Position& pos,
    int depth,
    std::span<const HashKey> game_history
) {
    return searcher_.search_best_move(pos, depth, game_history);
}

SearchResult NnueSearcherV41::search_best_move(
    const Position& pos,
    const SearchLimits& limits,
    std::span<const HashKey> game_history
) {
    return searcher_.search_best_move(pos, limits, game_history);
}

std::string_view NnueSearcherV41::name() const {
    return "nnue_repetition_v41";
}

void NnueSearcherV41::clear_tt() {
    searcher_.clear_tt();
}

void NnueSearcherV41::clear_search_heuristics() {
    searcher_.clear_search_heuristics();
}

std::size_t NnueSearcherV41::tt_entry_count() const {
    return searcher_.tt_entry_count();
}

void NnueSearcherV41::clear_tt_stats() {
    searcher_.clear_tt_stats();
}

const RangeTranspositionTableStats& NnueSearcherV41::tt_stats() const {
    return searcher_.tt_stats();
}

void NnueSearcherV41::clear_move_ordering_stats() {
    searcher_.clear_move_ordering_stats();
}

const NnueSearcherV41::MoveOrderingStats&
NnueSearcherV41::move_ordering_stats() const {
    return searcher_.move_ordering_stats();
}

void NnueSearcherV41::set_move_ordering_stats_enabled(bool enabled) {
    searcher_.set_move_ordering_stats_enabled(enabled);
}

void NnueSearcherV41::clear_selective_stats() {
    searcher_.clear_selective_stats();
}

const NnueSearcherV41::SelectiveStats&
NnueSearcherV41::selective_stats() const {
    return searcher_.selective_stats();
}

const RepetitionStack::Stats& NnueSearcherV41::repetition_stats() const {
    return searcher_.repetition_stats();
}

void NnueSearcherV41::set_twofold_search_draw_enabled(bool enabled) {
    searcher_.set_twofold_search_draw_enabled(enabled);
}

bool NnueSearcherV41::twofold_search_draw_enabled() const {
    return searcher_.twofold_search_draw_enabled();
}

void NnueSearcherV41::set_selective_config(SelectiveConfig config) {
    searcher_.set_selective_config(config);
}

const NnueSearcherV41::SelectiveConfig&
NnueSearcherV41::selective_config() const {
    return searcher_.selective_config();
}

} // namespace chess
