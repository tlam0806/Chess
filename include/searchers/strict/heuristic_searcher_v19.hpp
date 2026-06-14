#pragma once

#include "searcher.hpp"
#include "range_bucket_transposition_table.hpp"
#include "killer_move_table.hpp"
#include "history_table_v16.hpp"
#include "counter_history_table.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class HeuristicSearcherV19 final : public Searcher {
public:
    enum class MoveOrderingCategory : std::uint8_t {
        TtLower,
        TtUpper,
        Promotion,
        GoodCapture,
        BadCapture,
        Check,
        CounterHistory,
        Killer1,
        Killer2,
        Quiet,
        Count
    };

    static constexpr std::size_t MoveOrderingCategoryCount =
        static_cast<std::size_t>(MoveOrderingCategory::Count);

    struct MoveOrderingWeights {
        int tt_lower_bonus = 765'715'000;
        int tt_upper_bonus = 27'262'000;
        int promotion_bonus = 660'961'000;
        int good_capture_bonus = 63'613'000;
        int bad_capture_bonus = 50'000;
        int see_weight = 25;
        int killer1_bonus = 40'000;
        int killer2_bonus = 40'000;
        int check_bonus = 393'980;
        int counter_history_bonus = 16'000;
    };

    struct MoveOrderingStats {
        std::uint64_t searched_nodes = 0;
        std::uint64_t best_index_zero = 0;
        std::uint64_t best_index_sum = 0;
        std::uint64_t cutoff_nodes = 0;
        std::uint64_t cutoff_index_zero = 0;
        std::uint64_t cutoff_index_sum = 0;
        std::uint64_t missed_best_nodes = 0;
        std::array<std::uint64_t, MoveOrderingCategoryCount> predicted_categories{};
        std::array<std::uint64_t, MoveOrderingCategoryCount> best_categories{};
        std::array<std::uint64_t, MoveOrderingCategoryCount> cutoff_categories{};
        std::array<std::uint64_t, MoveOrderingCategoryCount> missed_predicted_categories{};
        std::array<std::uint64_t, MoveOrderingCategoryCount> missed_best_categories{};
        std::array<
            std::array<std::uint64_t, MoveOrderingCategoryCount>,
            MoveOrderingCategoryCount
        > missed_predicted_to_best_categories{};
    };

    explicit HeuristicSearcherV19(
        std::size_t tt_mb = 64,
        std::size_t bucket_size = 4,
        int history_penalty_divisor_numerator = 3,
        int history_penalty_divisor_denominator = 7,
        int counter_history_bonus = 16'000
    );
    HeuristicSearcherV19(
        std::size_t tt_mb,
        std::size_t bucket_size,
        int history_penalty_divisor_numerator,
        int history_penalty_divisor_denominator,
        int counter_history_bonus,
        MoveOrderingWeights weights
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
        Move prev_move,
        PieceType prev_moved_piece,
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

    std::vector<ScoredMove> ordered_moves(
        const Position& pos,
        int ply,
        MoveRange tt_moves = MoveRange{},
        Move prev_move = Move{},
        PieceType prev_moved_piece = PieceType::None
    ) const;
    std::vector<ScoredMove> ordered_noisy_moves(const Position& pos) const;

    ScoredMove make_scored_move(
        const Position& pos,
        Move move,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        ScoringMode stage
    ) const;
    int move_order_score(
        Move move,
        const Position& pos,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
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

    RangeBucketTranspositionTable tt_;
    KillerMoveTable killer_table_;
    HistoryTableV16 history_table_;
    CounterHistoryTable counter_history_table_;
    MoveOrderingWeights move_ordering_weights_{};
    MoveOrderingStats move_ordering_stats_{};
};

} // namespace chess
