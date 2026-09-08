#include "nnue_searcher_v44.hpp"

#include <stdexcept>

namespace chess {

NnueSearcherV44::NnueSearcherV44(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : NnueSearcherV44(
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

NnueSearcherV44::NnueSearcherV44(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : NnueSearcherV44(
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

NnueSearcherV44::NnueSearcherV44(
    const PhaseQuantizedNnueModel& model,
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights,
    SelectiveConfig selective_config
)
    : tt_(tt_mb),
      history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      counter_history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      model_(model),
      move_ordering_weights_(weights),
      selective_config_(selective_config) {
    if (bucket_size != 4) {
        throw std::invalid_argument("V44 single-bound TT bucket size must be 4");
    }
    move_ordering_weights_.counter_history_bonus = counter_history_bonus;
}

std::string_view NnueSearcherV44::name() const {
    return "nnue_clear_each_search_v44";
}

void NnueSearcherV44::clear_tt() {
    tt_.clear();
}

void NnueSearcherV44::clear_search_heuristics() {
    killer_table_.clear();
    counter_move_table_.clear();
    history_table_.reset();
    counter_history_table_.reset();
}

std::size_t NnueSearcherV44::tt_entry_count() const {
    return tt_.entry_count();
}

std::size_t NnueSearcherV44::tt_occupied_entry_count() const {
    return tt_.occupied_entry_count();
}

void NnueSearcherV44::set_reuse_deeper_tt_scores(bool enabled) {
    reuse_deeper_tt_scores_ = enabled;
}

bool NnueSearcherV44::reuse_deeper_tt_scores() const {
    return reuse_deeper_tt_scores_;
}

void NnueSearcherV44::clear_move_ordering_stats() {
    move_ordering_stats_ = MoveOrderingStats{};
}

const NnueSearcherV44::MoveOrderingStats&
NnueSearcherV44::move_ordering_stats() const {
    return move_ordering_stats_;
}

void NnueSearcherV44::set_move_ordering_stats_enabled(bool enabled) {
    move_ordering_stats_enabled_ = enabled;
}

void NnueSearcherV44::clear_selective_stats() {
    selective_stats_ = SelectiveStats{};
}

const NnueSearcherV44::SelectiveStats&
NnueSearcherV44::selective_stats() const {
    return selective_stats_;
}

const RepetitionStack::Stats& NnueSearcherV44::repetition_stats() const {
    return repetition_stats_;
}

void NnueSearcherV44::set_twofold_search_draw_enabled(bool enabled) {
    if (twofold_search_draw_enabled_ == enabled) {
        return;
    }
    twofold_search_draw_enabled_ = enabled;
    clear_tt();
}

bool NnueSearcherV44::twofold_search_draw_enabled() const {
    return twofold_search_draw_enabled_;
}

void NnueSearcherV44::set_selective_config(SelectiveConfig config) {
    selective_config_ = config;
}

const NnueSearcherV44::SelectiveConfig&
NnueSearcherV44::selective_config() const {
    return selective_config_;
}

void NnueSearcherV44::clear_aspiration_stats() {
    aspiration_stats_ = AspirationStats{};
}

const NnueSearcherV44::AspirationStats&
NnueSearcherV44::aspiration_stats() const {
    return aspiration_stats_;
}

void NnueSearcherV44::set_aspiration_config(AspirationConfig config) {
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

const NnueSearcherV44::AspirationConfig&
NnueSearcherV44::aspiration_config() const {
    return aspiration_config_;
}

} // namespace chess
