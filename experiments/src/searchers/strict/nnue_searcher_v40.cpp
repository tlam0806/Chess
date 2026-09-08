#include "nnue_searcher_v40.hpp"

namespace chess {

NnueSearcherV40::NnueSearcherV40(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : NnueSearcherV40(
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

NnueSearcherV40::NnueSearcherV40(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : NnueSearcherV40(
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

NnueSearcherV40::NnueSearcherV40(
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
          selective_config) {
}

SearchResult NnueSearcherV40::search_best_move(
    const Position& pos,
    int depth
) {
    return searcher_.search_best_move(pos, depth);
}

SearchResult NnueSearcherV40::search_best_move(
    const Position& pos,
    const SearchLimits& limits
) {
    return searcher_.search_best_move(pos, limits);
}

SearchResult NnueSearcherV40::search_best_move(
    const Position& pos,
    int depth,
    std::span<const HashKey> game_history
) {
    return searcher_.search_best_move(pos, depth, game_history);
}

SearchResult NnueSearcherV40::search_best_move(
    const Position& pos,
    const SearchLimits& limits,
    std::span<const HashKey> game_history
) {
    return searcher_.search_best_move(pos, limits, game_history);
}

std::string_view NnueSearcherV40::name() const {
    return "nnue_selective_v40";
}

void NnueSearcherV40::clear_tt() {
    searcher_.clear_tt();
}

void NnueSearcherV40::clear_search_heuristics() {
    searcher_.clear_search_heuristics();
}

std::size_t NnueSearcherV40::tt_entry_count() const {
    return searcher_.tt_entry_count();
}

void NnueSearcherV40::clear_tt_stats() {
    searcher_.clear_tt_stats();
}

const RangeTranspositionTableStats& NnueSearcherV40::tt_stats() const {
    return searcher_.tt_stats();
}

void NnueSearcherV40::clear_move_ordering_stats() {
    searcher_.clear_move_ordering_stats();
}

const NnueSearcherV40::MoveOrderingStats&
NnueSearcherV40::move_ordering_stats() const {
    return searcher_.move_ordering_stats();
}

void NnueSearcherV40::set_move_ordering_stats_enabled(bool enabled) {
    searcher_.set_move_ordering_stats_enabled(enabled);
}

void NnueSearcherV40::clear_selective_stats() {
    searcher_.clear_selective_stats();
}

const NnueSearcherV40::SelectiveStats&
NnueSearcherV40::selective_stats() const {
    return searcher_.selective_stats();
}

const RepetitionStack::Stats& NnueSearcherV40::repetition_stats() const {
    return searcher_.repetition_stats();
}

void NnueSearcherV40::set_twofold_search_draw_enabled(bool enabled) {
    searcher_.set_twofold_search_draw_enabled(enabled);
}

bool NnueSearcherV40::twofold_search_draw_enabled() const {
    return searcher_.twofold_search_draw_enabled();
}

void NnueSearcherV40::set_selective_config(SelectiveConfig config) {
    selective_config_ = config;
    searcher_.set_selective_config(config);
}

const NnueSearcherV40::SelectiveConfig&
NnueSearcherV40::selective_config() const {
    return selective_config_;
}

void NnueSearcherV40::clear_aspiration_stats() {
    searcher_.clear_aspiration_stats();
}

const NnueSearcherV40::AspirationStats&
NnueSearcherV40::aspiration_stats() const {
    return searcher_.aspiration_stats();
}

void NnueSearcherV40::set_aspiration_config(AspirationConfig config) {
    searcher_.set_aspiration_config(config);
}

const NnueSearcherV40::AspirationConfig&
NnueSearcherV40::aspiration_config() const {
    return searcher_.aspiration_config();
}

} // namespace chess
