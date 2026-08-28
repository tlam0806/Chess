#pragma once

#include "nnue_searcher_v38.hpp"

#include <cstddef>
#include <span>
#include <string_view>

namespace chess {

class NnueSearcherV39 final : public Searcher {
public:
    using CutoffStage = NnueSearcherV38::CutoffStage;
    using MoveOrderingStats = NnueSearcherV38::MoveOrderingStats;
    using MoveOrderingWeights = NnueSearcherV38::MoveOrderingWeights;
    using SelectiveStats = NnueSearcherV38::SelectiveStats;

    struct SelectiveConfig {
        bool enable_lmr = true;
        double lmr_base = 0.45;
        double lmr_divisor = 2.9;
        int lmr_min_depth = 3;
        std::size_t lmr_min_move_index = 6;
        bool enable_null_move = true;
        int null_move_min_depth = 2;
        int null_move_reduction = 3;

        bool enable_reverse_futility = true;
        int reverse_futility_max_depth = 2;
        int reverse_futility_base_margin = 175;
        int reverse_futility_margin_per_depth = 275;

        bool enable_late_move_pruning = true;
        int late_move_pruning_max_depth = 3;
        std::size_t late_move_pruning_base = 4;
        std::size_t late_move_pruning_depth_multiplier = 2;

        // Kept disabled so V39 remains the validated Fast control.
        bool enable_qsearch_see_pruning = false;
        int qsearch_see_threshold = -200;

        // Experimental capture-only pruning. It remains disabled in every
        // production default until the two-dimensional tune is promoted.
        bool enable_main_search_see_pruning = false;
        int main_search_see_max_depth = 5;
        int main_search_see_margin_per_depth = 100;
    };

    explicit NnueSearcherV39(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb = 64,
        std::size_t bucket_size = 4,
        int history_penalty_divisor_numerator = 10,
        int history_penalty_divisor_denominator = 14,
        int counter_history_bonus = 14'000
    );
    NnueSearcherV39(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb,
        std::size_t bucket_size,
        int history_penalty_divisor_numerator,
        int history_penalty_divisor_denominator,
        int counter_history_bonus,
        MoveOrderingWeights weights
    );
    NnueSearcherV39(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb,
        std::size_t bucket_size,
        int history_penalty_divisor_numerator,
        int history_penalty_divisor_denominator,
        int counter_history_bonus,
        MoveOrderingWeights weights,
        SelectiveConfig selective_config
    );

    SearchResult search_best_move(const Position& pos, int depth) override;
    SearchResult search_best_move(
        const Position& pos,
        const SearchLimits& limits
    ) override;
    SearchResult search_best_move(
        const Position& pos,
        int depth,
        std::span<const HashKey> game_history
    );
    SearchResult search_best_move(
        const Position& pos,
        const SearchLimits& limits,
        std::span<const HashKey> game_history
    );
    std::string_view name() const override;

    void clear_tt();
    void clear_search_heuristics();
    std::size_t tt_entry_count() const;
    void clear_tt_stats();
    const RangeTranspositionTableStats& tt_stats() const;
    void clear_move_ordering_stats();
    const MoveOrderingStats& move_ordering_stats() const;
    void set_move_ordering_stats_enabled(bool enabled);
    void clear_selective_stats();
    const SelectiveStats& selective_stats() const;
    const RepetitionStack::Stats& repetition_stats() const;
    void set_selective_config(SelectiveConfig config);
    const SelectiveConfig& selective_config() const;

private:
    static NnueSearcherV38::SelectiveConfig to_v38_config(
        const SelectiveConfig& config
    );

    SelectiveConfig selective_config_{};
    NnueSearcherV38 searcher_;
};

} // namespace chess
