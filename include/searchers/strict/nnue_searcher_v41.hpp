#pragma once

#include "nnue_searcher_v40.hpp"

#include <cstddef>
#include <span>
#include <string_view>

namespace chess {

// V40 child which evaluates repetition and the 50-move rule inside search.
class NnueSearcherV41 final : public Searcher {
public:
    using CutoffStage = NnueSearcherV40::CutoffStage;
    using MoveOrderingStats = NnueSearcherV40::MoveOrderingStats;
    using MoveOrderingWeights = NnueSearcherV40::MoveOrderingWeights;
    using SelectiveStats = NnueSearcherV40::SelectiveStats;
    using SelectiveConfig = NnueSearcherV40::SelectiveConfig;

    explicit NnueSearcherV41(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb = 64,
        std::size_t bucket_size = 4,
        int history_penalty_divisor_numerator = 10,
        int history_penalty_divisor_denominator = 14,
        int counter_history_bonus = 14'000
    );
    NnueSearcherV41(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb,
        std::size_t bucket_size,
        int history_penalty_divisor_numerator,
        int history_penalty_divisor_denominator,
        int counter_history_bonus,
        MoveOrderingWeights weights
    );
    NnueSearcherV41(
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
    void set_twofold_search_draw_enabled(bool enabled);
    bool twofold_search_draw_enabled() const;
    void set_selective_config(SelectiveConfig config);
    const SelectiveConfig& selective_config() const;

private:
    NnueSearcherV40 searcher_;
};

} // namespace chess
