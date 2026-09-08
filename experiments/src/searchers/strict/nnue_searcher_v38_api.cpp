#include "nnue_searcher_v38.hpp"

#include <stdexcept>

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

void NnueSearcherV38::set_twofold_search_draw_enabled(bool enabled) {
    if (twofold_search_draw_enabled_ == enabled) {
        return;
    }
    twofold_search_draw_enabled_ = enabled;
    clear_tt();
}

bool NnueSearcherV38::twofold_search_draw_enabled() const {
    return twofold_search_draw_enabled_;
}

void NnueSearcherV38::set_selective_config(SelectiveConfig config) {
    selective_config_ = config;
}

const NnueSearcherV38::SelectiveConfig&
NnueSearcherV38::selective_config() const {
    return selective_config_;
}

void NnueSearcherV38::clear_aspiration_stats() {
    aspiration_stats_ = AspirationStats{};
}

const NnueSearcherV38::AspirationStats&
NnueSearcherV38::aspiration_stats() const {
    return aspiration_stats_;
}

void NnueSearcherV38::set_aspiration_config(AspirationConfig config) {
    if (config.min_depth < 2) {
        throw std::invalid_argument("aspiration min depth must be >= 2");
    }
    if (config.delta_base_cp < 1) {
        throw std::invalid_argument("aspiration base delta must be >= 1");
    }
    if (config.delta_divisor < 1) {
        throw std::invalid_argument("aspiration delta divisor must be >= 1");
    }
    if (config.expansion_factor_per_mille < 1'000) {
        throw std::invalid_argument(
            "aspiration expansion factor must be >= 1000 per mille");
    }
    if (config.max_fail_high_reductions < 0) {
        throw std::invalid_argument(
            "aspiration max fail-high reductions must be >= 0");
    }
    if (config.mean_score_new_weight_per_mille < 0
        || config.mean_score_new_weight_per_mille > 1'000) {
        throw std::invalid_argument(
            "aspiration mean-score weight must be in [0, 1000]");
    }
    if (config.max_researches < 1) {
        throw std::invalid_argument("aspiration max researches must be >= 1");
    }
    if (config.mean_score_clamp_cp < 1) {
        throw std::invalid_argument(
            "aspiration mean-score clamp must be >= 1");
    }
    aspiration_config_ = config;
}

const NnueSearcherV38::AspirationConfig&
NnueSearcherV38::aspiration_config() const {
    return aspiration_config_;
}

} // namespace chess
