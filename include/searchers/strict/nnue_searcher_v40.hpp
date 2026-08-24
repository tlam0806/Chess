#pragma once

#include "nnue_searcher_v39.hpp"

#include <cstddef>
#include <span>
#include <string_view>

namespace chess {

class NnueSearcherV40 final : public Searcher {
public:
    using CutoffStage = NnueSearcherV39::CutoffStage;
    using MoveOrderingStats = NnueSearcherV39::MoveOrderingStats;
    using MoveOrderingWeights = NnueSearcherV39::MoveOrderingWeights;
    using SelectiveStats = NnueSearcherV39::SelectiveStats;

    struct SelectiveConfig : NnueSearcherV39::SelectiveConfig {
        SelectiveConfig() {
            enable_qsearch_see_pruning = true;
            qsearch_see_threshold = -75;
        }
    };

    explicit NnueSearcherV40(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb = 64,
        std::size_t bucket_size = 4,
        int history_penalty_divisor_numerator = 10,
        int history_penalty_divisor_denominator = 14,
        int counter_history_bonus = 14'000
    );
    NnueSearcherV40(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb,
        std::size_t bucket_size,
        int history_penalty_divisor_numerator,
        int history_penalty_divisor_denominator,
        int counter_history_bonus,
        MoveOrderingWeights weights
    );
    NnueSearcherV40(
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
    SelectiveConfig selective_config_{};
    NnueSearcherV39 searcher_;
};

} // namespace chess
