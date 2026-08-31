#include "nnue_searcher_v42.hpp"

namespace chess {

namespace {

SearchResult search_fixed_depth_with_legacy_policy(
    NnueSearcherV41& searcher,
    auto&& search
) {
    const NnueSearcherV41::AspirationConfig original =
        searcher.aspiration_config();
    if (!original.enabled) {
        return search();
    }

    auto fixed_depth = original;
    fixed_depth.enabled = false;
    searcher.set_aspiration_config(fixed_depth);
    try {
        SearchResult result = search();
        searcher.set_aspiration_config(original);
        return result;
    } catch (...) {
        searcher.set_aspiration_config(original);
        throw;
    }
}

} // namespace

NnueSearcherV42::NnueSearcherV42(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : NnueSearcherV42(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{},
        SelectiveConfig{}) {
}

NnueSearcherV42::NnueSearcherV42(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : NnueSearcherV42(
        model,
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        weights,
        SelectiveConfig{}) {
}

NnueSearcherV42::NnueSearcherV42(
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
    AspirationConfig aspiration_config;
    aspiration_config.enabled = true;
    searcher_.set_aspiration_config(aspiration_config);
}

SearchResult NnueSearcherV42::search_best_move(
    const Position& pos,
    int depth
) {
    return search_fixed_depth_with_legacy_policy(
        searcher_, [&] { return searcher_.search_best_move(pos, depth, {}); });
}

SearchResult NnueSearcherV42::search_best_move(
    const Position& pos,
    const SearchLimits& limits
) {
    return searcher_.search_best_move(pos, limits, {});
}

SearchResult NnueSearcherV42::search_best_move(
    const Position& pos,
    int depth,
    std::span<const HashKey> game_history
) {
    return search_fixed_depth_with_legacy_policy(searcher_, [&] {
        return searcher_.search_best_move(pos, depth, game_history);
    });
}

SearchResult NnueSearcherV42::search_best_move(
    const Position& pos,
    const SearchLimits& limits,
    std::span<const HashKey> game_history
) {
    return searcher_.search_best_move(pos, limits, game_history);
}

std::string_view NnueSearcherV42::name() const {
    return "nnue_adaptive_aspiration_v42";
}

void NnueSearcherV42::clear_tt() {
    searcher_.clear_tt();
}

void NnueSearcherV42::clear_search_heuristics() {
    searcher_.clear_search_heuristics();
}

std::size_t NnueSearcherV42::tt_entry_count() const {
    return searcher_.tt_entry_count();
}

void NnueSearcherV42::clear_tt_stats() {
    searcher_.clear_tt_stats();
}

const RangeTranspositionTableStats& NnueSearcherV42::tt_stats() const {
    return searcher_.tt_stats();
}

void NnueSearcherV42::clear_move_ordering_stats() {
    searcher_.clear_move_ordering_stats();
}

const NnueSearcherV42::MoveOrderingStats&
NnueSearcherV42::move_ordering_stats() const {
    return searcher_.move_ordering_stats();
}

void NnueSearcherV42::set_move_ordering_stats_enabled(bool enabled) {
    searcher_.set_move_ordering_stats_enabled(enabled);
}

void NnueSearcherV42::clear_selective_stats() {
    searcher_.clear_selective_stats();
}

const NnueSearcherV42::SelectiveStats&
NnueSearcherV42::selective_stats() const {
    return searcher_.selective_stats();
}

const RepetitionStack::Stats& NnueSearcherV42::repetition_stats() const {
    return searcher_.repetition_stats();
}

void NnueSearcherV42::set_twofold_search_draw_enabled(bool enabled) {
    searcher_.set_twofold_search_draw_enabled(enabled);
}

bool NnueSearcherV42::twofold_search_draw_enabled() const {
    return searcher_.twofold_search_draw_enabled();
}

void NnueSearcherV42::set_selective_config(SelectiveConfig config) {
    searcher_.set_selective_config(config);
}

const NnueSearcherV42::SelectiveConfig&
NnueSearcherV42::selective_config() const {
    return searcher_.selective_config();
}

void NnueSearcherV42::clear_aspiration_stats() {
    searcher_.clear_aspiration_stats();
}

const NnueSearcherV42::AspirationStats&
NnueSearcherV42::aspiration_stats() const {
    return searcher_.aspiration_stats();
}

void NnueSearcherV42::set_aspiration_config(AspirationConfig config) {
    searcher_.set_aspiration_config(config);
}

const NnueSearcherV42::AspirationConfig&
NnueSearcherV42::aspiration_config() const {
    return searcher_.aspiration_config();
}

} // namespace chess
