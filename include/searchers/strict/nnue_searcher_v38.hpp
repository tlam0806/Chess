#pragma once

#include "searcher.hpp"
#include "lower_move_range_bucket_transposition_table.hpp"
#include "counter_move_table.hpp"
#include "killer_move_table.hpp"
#include "history_table_v16.hpp"
#include "counter_history_table.hpp"
#include "king_safety.hpp"
#include "phase_quantized_nnue.hpp"
#include "repetition_stack.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace chess {

class NnueSearcherV38 final : public Searcher {
public:
    enum class CutoffStage : std::size_t {
        TtLower,
        Promotion,
        GoodCapture,
        PriorityQuiet,
        Quiet,
        BadCapture,
        Count
    };
    struct MoveOrderingStats {
        std::uint64_t beta_cutoffs = 0;
        std::array<std::uint64_t, 9> cutoff_move_index{};
        std::array<
            std::uint64_t,
            static_cast<std::size_t>(CutoffStage::Count)
        > cutoff_stage{};
        std::uint64_t cutoff_by_capture = 0;
        std::uint64_t cutoff_by_check = 0;
        std::uint64_t cutoff_by_promotion = 0;
        std::uint64_t cutoff_by_quiet = 0;
    };
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
        int qsearch_capture_metric_weight = 160;
    };
    struct SelectiveConfig {
        bool enable_lmr = true;
        double lmr_base = 0.5;
        double lmr_divisor = 2.6;
        int lmr_min_depth = 4;
        std::size_t lmr_min_move_index = 4;
        bool enable_null_move = true;
        int null_move_min_depth = 3;
        int null_move_reduction = 2;
        bool enable_reverse_futility = false;
        int reverse_futility_max_depth = 3;
        int reverse_futility_base_margin = 100;
        int reverse_futility_margin_per_depth = 100;
        bool enable_late_move_pruning = false;
        int late_move_pruning_max_depth = 3;
        std::size_t late_move_pruning_base = 4;
        std::size_t late_move_pruning_depth_multiplier = 2;
        bool enable_qsearch_see_pruning = false;
        int qsearch_see_threshold = -200;
    };
    struct SelectiveStats {
        std::uint64_t lmr_searches = 0;
        std::uint64_t lmr_researches = 0;
        std::uint64_t null_move_searches = 0;
        std::uint64_t null_move_cutoffs = 0;
        std::uint64_t reverse_futility_evaluations = 0;
        std::uint64_t reverse_futility_cutoffs = 0;
        std::uint64_t late_move_pruned_nodes = 0;
        std::uint64_t late_move_pruned_moves = 0;
        std::uint64_t qsearch_see_evaluations = 0;
        std::uint64_t qsearch_see_pruned_moves = 0;
    };
    explicit NnueSearcherV38(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb = 64,
        std::size_t bucket_size = 4,
        int history_penalty_divisor_numerator = 10,
        int history_penalty_divisor_denominator = 14,
        int counter_history_bonus = 14'000
    );
    NnueSearcherV38(
        const PhaseQuantizedNnueModel& model,
        std::size_t tt_mb,
        std::size_t bucket_size,
        int history_penalty_divisor_numerator,
        int history_penalty_divisor_denominator,
        int counter_history_bonus,
        MoveOrderingWeights weights
    );
    NnueSearcherV38(
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
    SearchResult search_best_move(const Position& pos, const SearchLimits& limits) override;
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
    struct SearchState {
        std::uint64_t nodes = 0;
        std::chrono::steady_clock::time_point deadline{};
        PhaseQuantizedNnueAccumulator accumulator;
        bool has_deadline = false;
        bool stopped = false;
        bool in_null_move = false;
        bool repetition_enabled = false;
        RepetitionStack repetition;
    };
    enum class ScoringMode {
        MainSearch,
        Quiescence
    };
    enum class MoveGenerationStage {
        Promotion,
        GoodCapture,
        Killer,
        QuietNonPromotion,
        BadCapture
    };
    static constexpr std::uint16_t scored_piece_bits(PieceType piece) {
        return static_cast<std::uint16_t>(piece) & 0x7u;
    }
    static constexpr std::uint16_t pack_scored_info(
        PieceType moved_piece,
        PieceType captured_piece,
        bool gives_check
    ) {
        return static_cast<std::uint16_t>(
            scored_piece_bits(moved_piece)
            | (scored_piece_bits(captured_piece) << 3)
            | (static_cast<std::uint16_t>(gives_check) << 6));
    }
    struct ScoredMove {
        int order_score = 0;
        Move move{};
        std::uint16_t info = pack_scored_info(PieceType::None, PieceType::None, false);
    };
    static_assert(sizeof(ScoredMove) == 8);
    static_assert(alignof(ScoredMove) == 4);
    using ScoredMoveList = FixedList<ScoredMove, 256>;
    struct CaptureMoveLists {
        ScoredMoveList good;
        ScoredMoveList bad;
    };
    struct SearchValue {
        ScoreRange range{};
    };
    struct RootSearchResult {
        SearchResult result{};
        ScoreRange range{};
    };
    struct TTProbeResult {
        ScoreRange range{};
        MoveRange moves{};
        bool hit = false;
        bool has_score = false;
    };

    RootSearchResult search_fixed_depth(
        Position pos,
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
        Position& pos,
        int depth,
        int ply,
        Move prev_move,
        PieceType prev_moved_piece,
        int alpha,
        int beta,
        SearchState& state
    );

    SearchValue quiescence(
        Position& pos,
        int alpha,
        int beta,
        int ply,
        int q_depth,
        SearchState& state,
        bool check_current_repetition
    );

    SearchResult make_fallback_result(const Position& pos) const;

    ScoredMoveList ordered_moves(
        const Position& pos,
        int ply,
        MoveRange tt_moves = MoveRange{},
        Move prev_move = Move{},
        PieceType prev_moved_piece = PieceType::None,
        Move skip_move = Move{}
    ) const;
    template <MoveGenerationStage Stage>
    ScoredMoveList ordered_moves_for_stage(
        const Position& pos,
        const KingSafetyContext& king_safety,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        Move skip_tt_move,
        Move skip_priority1 = Move{},
        Move skip_priority2 = Move{},
        Move skip_priority3 = Move{}
    );
    ScoredMoveList ordered_priority_quiet_moves_for_stage(
        const Position& pos,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        Move skip_tt_move
    ) const;

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
    ScoredMove make_tt_lower_scored_move(
        const Position& pos,
        Move move
    ) const;
    template <ScoringMode Mode, bool IsCapture, bool IsPromotion>
    ScoredMove make_scored_legal_move(
        const Position& pos,
        Move move,
        PieceType moved_piece,
        PieceType captured_piece,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece
    ) const;
    template <bool IsCapture, bool IsPromotion>
    ScoredMove make_qsearch_scored_move(
        Move move,
        PieceType moved_piece,
        PieceType captured_piece
    ) const;
    CaptureMoveLists generate_legal_capture_scored_move_lists_for_searcher(
        const Position& pos,
        const KingSafetyContext& king_safety,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        Move skip_tt_move
    );
    template <bool IsCapture, bool IsPromotion>
    int main_order_score(
        Move move,
        const Position& pos,
        PieceType moved_piece,
        int ply,
        MoveRange tt_moves,
        Move prev_move,
        PieceType prev_moved_piece,
        bool gives_check,
        int see_score
    ) const;
    template <bool IsCapture, bool IsPromotion>
    int qsearch_order_score(
        PieceType moved_piece,
        PieceType captured_piece
    ) const;
    bool should_stop(SearchState& state) const;
    KingSafetyContext current_king_safety_context(const Position& pos) const;
    int evaluate_current_position(const Position& pos, const SearchState& state) const;
    bool is_quiet_move(const ScoredMove& scored_move) const;
    static constexpr PieceType scored_moved_piece(const ScoredMove& scored_move) {
        return static_cast<PieceType>(scored_move.info & 0x7u);
    }
    static constexpr PieceType scored_captured_piece(const ScoredMove& scored_move) {
        return static_cast<PieceType>((scored_move.info >> 3) & 0x7u);
    }
    static constexpr bool scored_gives_check(const ScoredMove& scored_move) {
        return (scored_move.info & (1u << 6)) != 0;
    }
    bool better_scored_move(const ScoredMove& lhs, const ScoredMove& rhs) const;
    void sort_scored_moves(ScoredMoveList& moves) const;
    TTProbeResult probe_tt(
        HashKey key,
        int depth,
        int& alpha,
        int& beta,
        int ply,
        bool allow_probe,
        bool allow_score
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
        Move fallback_best_move,
        bool allow_store
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
        const ScoredMoveList& failed_quiet_moves
    );
    void record_beta_cutoff(
        const ScoredMove& scored_move,
        std::size_t move_index,
        CutoffStage stage
    );
    bool can_null_move_prune(
        const Position& pos,
        int depth,
        const SearchState& state
    ) const;
    static void make_null_move(Position& pos);
    static bool castling_rights_changed(
        const PositionStateSnapshot& before,
        const Position& after
    );
    bool history_draw(const Position& pos, SearchState& state) const;
    bool allow_repetition_tt_score(SearchState& state) const;
    void initialize_repetition(
        SearchState& state,
        const Position& pos,
        std::span<const HashKey> game_history
    ) const;
    SearchResult search_best_move_impl(
        const Position& pos,
        int depth,
        std::span<const HashKey> game_history,
        bool enable_repetition
    );
    SearchResult search_best_move_impl(
        const Position& pos,
        const SearchLimits& limits,
        std::span<const HashKey> game_history,
        bool enable_repetition
    );
    LowerMoveRangeBucketTranspositionTable tt_;
    KillerMoveTable killer_table_;
    CounterMoveTable counter_move_table_;
    HistoryTableV16 history_table_;
    CounterHistoryTable counter_history_table_;
    const PhaseQuantizedNnueModel& model_;
    MoveOrderingWeights move_ordering_weights_{};
    MoveOrderingStats move_ordering_stats_{};
    SelectiveConfig selective_config_{};
    SelectiveStats selective_stats_{};
    RepetitionStack::Stats repetition_stats_{};
    bool move_ordering_stats_enabled_ = false;
};

} // namespace chess
