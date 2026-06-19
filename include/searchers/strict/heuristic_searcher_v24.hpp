#pragma once

#include "searcher.hpp"
#include "range_bucket_transposition_table.hpp"
#include "killer_move_table.hpp"
#include "history_table_v16.hpp"
#include "counter_history_table.hpp"
#include "king_safety.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class HeuristicSearcherV24 final : public Searcher {
public:
    struct MoveOrderingWeights {
        int tt_lower_bonus = 1'200'000'000;
        int tt_upper_bonus = 0;
        int promotion_bonus = 660'961'000;
        int good_capture_bonus = 63'613'000;
        int bad_capture_bonus = 80'000;
        int see_weight = 280;
        int killer1_bonus = 24'000;
        int killer2_bonus = 8'076;
        int check_bonus = 44'384;
        int counter_history_bonus = 21'000;
        int bad_capture_stage_threshold = 0;
        int qsearch_promotion_bonus = 1'094'272'000;
        int qsearch_good_capture_bonus = 154'962'000;
        int qsearch_bad_capture_bonus = 282'000;
        int qsearch_see_weight = 48;
        int qsearch_captured_value_weight = 486;
    };
    struct MoveTypeStats {
        std::uint64_t total = 0;
        std::uint64_t tt_lower = 0;
        std::uint64_t tt_upper = 0;
        std::uint64_t promotion = 0;
        std::uint64_t capture = 0;
        std::uint64_t quiet = 0;
        std::uint64_t check = 0;
        std::uint64_t killer1 = 0;
        std::uint64_t killer2 = 0;
        std::uint64_t history_positive = 0;
        std::uint64_t counter_history_positive = 0;
    };
    struct MoveCutoffStats {
        std::uint64_t nodes_with_moves = 0;
        std::uint64_t beta_cutoffs = 0;
        std::uint64_t cutoff_index_sum = 0;
        std::array<std::uint64_t, 16> cutoff_index_buckets{};
        MoveTypeStats before_cutoff;
        MoveTypeStats cutoff_move;
        MoveTypeStats after_cutoff;
    };
    struct MoveOrderingStats {
        MoveCutoffStats main;
        MoveCutoffStats qsearch;
        MoveTypeStats noisy_stage;
        MoveTypeStats quiet_stage;
        std::array<MoveTypeStats, 3> priority_stage_top;
        std::uint64_t tt_lower_stage0_attempts = 0;
        std::uint64_t tt_lower_stage0_cutoffs = 0;
        std::uint64_t tt_lower_king_safety_contexts = 0;
        std::uint64_t tt_lower_legal_checks = 0;
        std::uint64_t priority_king_safety_contexts = 0;
        std::uint64_t priority_legal_checks = 0;
        std::uint64_t quiet_king_safety_contexts = 0;
        std::uint64_t quiet_legal_checks = 0;
        std::uint64_t qsearch_king_safety_contexts = 0;
        std::uint64_t qsearch_legal_checks = 0;
    };

    explicit HeuristicSearcherV24(
        std::size_t tt_mb = 64,
        std::size_t bucket_size = 4,
        int history_penalty_divisor_numerator = 10,
        int history_penalty_divisor_denominator = 14,
        int counter_history_bonus = 14'000
    );
    HeuristicSearcherV24(
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
    void set_move_ordering_stats_enabled(bool enabled);

private:
    struct SearchState;
    enum class ScoringMode;
    enum class MoveGenerationStage;
    struct ScoredMove {
        int order_score = 0;
        int see_score = 0;
        Move move{};
        PieceType moved_piece = PieceType::None;
        PieceType captured_piece = PieceType::None;
        bool gives_check = false;
        bool capture = false;
        bool promotion = false;
    };
    static_assert(sizeof(ScoredMove) == 16);
    static_assert(alignof(ScoredMove) == 4);
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
    Move valid_priority_killer(const Position& pos, int ply, int slot) const;

    std::vector<ScoredMove> ordered_moves(
        const Position& pos,
        int ply,
        MoveRange tt_moves = MoveRange{},
        Move prev_move = Move{},
        PieceType prev_moved_piece = PieceType::None,
        Move skip_move = Move{}
    ) const;
    std::vector<ScoredMove> ordered_moves_for_stage(
        const Position& pos,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        Move skip_tt_move,
        Move skip_killer1,
        Move skip_killer2,
        MoveGenerationStage generation_stage
    );

    ScoredMove make_scored_move(
        const Position& pos,
        const KingSafetyContext& king_safety,
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
        PieceType moved_piece,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        bool gives_check,
        bool capture,
        bool promotion,
        PieceType captured_piece,
        int see_score,
        ScoringMode stage
    ) const;
    bool should_stop(SearchState& state) const;
    bool is_quiet_move(const ScoredMove& scored_move) const;
    bool better_scored_move(const ScoredMove& lhs, const ScoredMove& rhs) const;
    void sort_scored_moves(std::vector<ScoredMove>& moves) const;
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
    void update_best_range(
        ScoreRange& node_range,
        Move& best_lower_move,
        Move& best_upper_move,
        int& alpha,
        Move move,
        ScoreRange move_range
    ) const;
    void store_tt_if_needed(
        HashKey key,
        int depth,
        int ply,
        ScoreRange range,
        Move best_lower_move,
        Move best_upper_move,
        Move fallback_best_move
    );
    void reward_quiet_cutoff(
        Color side_to_move,
        int depth,
        int ply,
        const ScoredMove& scored_move,
        Move prev_move,
        PieceType prev_moved_piece
    );
    void penalize_failed_quiets(
        Color side_to_move,
        int depth,
        Move prev_move,
        PieceType prev_moved_piece,
        const std::vector<ScoredMove>& failed_quiet_moves
    );
    void record_node_with_moves(ScoringMode stage);
    void record_beta_cutoff(
        const std::vector<ScoredMove>& moves,
        std::size_t cutoff_index,
        Color side_to_move,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        ScoringMode stage
    );
    void record_generated_stage_move(
        const ScoredMove& move,
        Color side_to_move,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        MoveGenerationStage generation_stage
    );
    void record_priority_stage_top_moves(
        const std::vector<ScoredMove>& moves,
        Color side_to_move,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece
    );
    void add_move_type_stats(
        MoveTypeStats& stats,
        const ScoredMove& move,
        Color side_to_move,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        ScoringMode stage
    ) const;

    RangeBucketTranspositionTable tt_;
    KillerMoveTable killer_table_;
    HistoryTableV16 history_table_;
    CounterHistoryTable counter_history_table_;
    MoveOrderingWeights move_ordering_weights_{};
    MoveOrderingStats move_ordering_stats_{};
    bool move_ordering_stats_enabled_ = false;
};

} // namespace chess
