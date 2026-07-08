#pragma once

#include "searcher.hpp"
#include "lower_move_range_bucket_transposition_table.hpp"
#include "counter_move_table.hpp"
#include "killer_move_table.hpp"
#include "history_table_v16.hpp"
#include "counter_history_table.hpp"
#include "king_safety.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class HeuristicSearcherV35 final : public Searcher {
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
        int qsearch_capture_metric_weight = 160;
    };
    explicit HeuristicSearcherV35(
        std::size_t tt_mb = 64,
        std::size_t bucket_size = 4,
        int history_penalty_divisor_numerator = 10,
        int history_penalty_divisor_denominator = 14,
        int counter_history_bonus = 14'000
    );
    HeuristicSearcherV35(
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

private:
    struct SearchState {
        std::uint64_t nodes = 0;
        std::chrono::steady_clock::time_point deadline{};
        bool has_deadline = false;
        bool stopped = false;
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
        SearchState& state
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
    int evaluate_current_position(const Position& pos) const;
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
        const ScoredMoveList& failed_quiet_moves
    );
    LowerMoveRangeBucketTranspositionTable tt_;
    KillerMoveTable killer_table_;
    CounterMoveTable counter_move_table_;
    HistoryTableV16 history_table_;
    CounterHistoryTable counter_history_table_;
    MoveOrderingWeights move_ordering_weights_{};
};

} // namespace chess
