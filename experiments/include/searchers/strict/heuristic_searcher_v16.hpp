#pragma once

#include "searcher.hpp"
#include "range_transposition_table.hpp"
#include "killer_move_table.hpp"
#include "history_table_v16.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class HeuristicSearcherV16 final : public Searcher {
public:
    enum class MoveOrderingCategory : std::uint8_t {
        TtLower,
        TtUpper,
        Promotion,
        GoodCapture,
        BadCapture,
        Check,
        Killer1,
        Killer2,
        Quiet,
        Count
    };

    static constexpr std::size_t MoveOrderingCategoryCount =
        static_cast<std::size_t>(MoveOrderingCategory::Count);

    struct MoveOrderingStats {
        std::uint64_t searched_nodes = 0;
        std::uint64_t best_index_zero = 0;
        std::uint64_t best_index_sum = 0;
        std::uint64_t cutoff_nodes = 0;
        std::uint64_t cutoff_index_zero = 0;
        std::uint64_t cutoff_index_sum = 0;
        std::uint64_t missed_best_nodes = 0;
        std::array<std::uint64_t, MoveOrderingCategoryCount> missed_predicted_categories{};
        std::array<std::uint64_t, MoveOrderingCategoryCount> missed_best_categories{};
    };

    explicit HeuristicSearcherV16(
        std::size_t tt_mb = 64,
        int history_penalty_divisor_numerator = 6,
        int history_penalty_divisor_denominator = 5
    );

    SearchResult search_best_move(const Position& pos, int depth) override;
    SearchResult search_best_move(const Position& pos, const SearchLimits& limits) override;
    std::string_view name() const override;

    void clear_tt();
    std::size_t tt_entry_count() const;
    void clear_tt_stats();
    const RangeTranspositionTableStats& tt_stats() const;
    void clear_move_ordering_stats();
    const MoveOrderingStats& move_ordering_stats() const;

private:
    struct SearchState;
    enum class ScoringMode;
    struct ScoredMove;
    struct SearchValue;
    struct TTProbeResult;

    SearchResult search_fixed_depth(
        const Position& pos,
        int depth,
        SearchState& state,
        int alpha = -Infinity,
        int beta = Infinity,
        bool allow_root_tt_probe = true
    );
    SearchResult search_root_without_tt_probe(
        const Position& pos,
        int depth,
        SearchState& state
    );

    SearchValue negamax(
        Position pos,
        int depth,
        int ply,
        int alpha,
        int beta,
        SearchState& state
    );

    SearchValue quiescence(
        Position pos,
        int alpha,
        int beta,
        int ply,
        int q_depth,
        SearchState& state
    );

    SearchResult make_fallback_result(const Position& pos) const;

    std::vector<ScoredMove> ordered_moves(const Position& pos, int ply, MoveRange tt_moves = MoveRange{}) const;
    std::vector<ScoredMove> ordered_noisy_moves(const Position& pos) const;

    ScoredMove make_scored_move(
        const Position& pos,
        Move move,
        int ply,
        MoveRange tt_moves,
        ScoringMode stage
    ) const;
    int move_order_score(
        Move move,
        const Position& pos,
        int ply,
        MoveRange tt_moves,
        bool gives_check,
        bool capture,
        bool promotion,
        int see_score,
        ScoringMode stage
    ) const;
    bool should_stop(SearchState& state) const;
    bool is_quiet_move(const ScoredMove& scored_move) const;
    bool better_scored_move(const ScoredMove& lhs, const ScoredMove& rhs) const;
    TTProbeResult probe_tt(
        HashKey key,
        int depth,
        int& alpha,
        int& beta,
        int ply,
        bool allow_probe
    ) const;
    bool should_store_tt(ScoreRange range) const;
    int score_to_tt(int score, int ply) const;

    RangeTranspositionTable tt_;
    KillerMoveTable killer_table_;
    HistoryTableV16 history_table_;
    MoveOrderingStats move_ordering_stats_{};
};

} // namespace chess
