#pragma once

#include "searcher.hpp"
#include "transposition_table.hpp"
#include "killer_move_table.hpp"
#include "history_table.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

namespace chess {

struct V12MoveOrderWeights {
    int tt_bonus = 1'000'000'000;
    int promotion_bonus = 900'000'000;
    int good_capture_bonus = 50'000'000;
    int bad_capture_bonus = 30'000;
    int see_weight = 10;
    int killer1_bonus = 40'000;
    int killer2_bonus = 10'000;
    int check_bonus = 160'000;
};

class HeuristicSearcherV12 final : public Searcher {
public:
    struct MoveOrderingStats {
        std::uint64_t beta_cutoffs = 0;
        std::array<std::uint64_t, 9> cutoff_index_buckets{};
        std::uint64_t cutoff_by_capture = 0;
        std::uint64_t cutoff_by_check = 0;
        std::uint64_t cutoff_by_promotion = 0;
        std::uint64_t cutoff_by_quiet = 0;
        std::uint64_t cutoff_by_killer1 = 0;
        std::uint64_t cutoff_by_killer2 = 0;
        std::uint64_t cutoff_by_lmr = 0;
        std::uint64_t cutoff_by_pvs_scout = 0;
        std::uint64_t pvs_scouts = 0;
        std::uint64_t pvs_researches = 0;
    };

    struct PvsShadowStats {
        std::uint64_t fail_low_samples = 0;
        std::uint64_t scout_nodes = 0;
        std::uint64_t shadow_full_nodes = 0;
        std::array<std::uint64_t, 16> fail_low_by_depth{};
        std::array<std::uint64_t, 16> scout_nodes_by_depth{};
        std::array<std::uint64_t, 16> shadow_full_nodes_by_depth{};
    };

    explicit HeuristicSearcherV12(
        std::size_t tt_mb = 64,
        V12MoveOrderWeights weights = {}
    );

    SearchResult search_best_move(const Position& pos, int depth) override;
    SearchResult search_best_move(const Position& pos, const SearchLimits& limits) override;
    std::string_view name() const override;

    void clear_tt();
    std::size_t tt_entry_count() const;
    void clear_tt_stats();
    const TranspositionTableStats& tt_stats() const;
    void clear_move_ordering_stats();
    const MoveOrderingStats& move_ordering_stats() const;
    void enable_pvs_shadow_measurement(std::uint64_t max_fail_low_samples);
    void clear_pvs_shadow_stats();
    const PvsShadowStats& pvs_shadow_stats() const;

private:
    struct SearchContext {
        std::uint64_t nodes = 0;
        std::chrono::steady_clock::time_point deadline{};
        bool has_deadline = false;
        bool stopped = false;
        bool in_null_move = false;
    };

    enum class ScoringMode {
        MainSearch,
        Quiescence
    };

    struct ScoredMove {
        Move move{};
        int order_score = 0;
        bool gives_check = false;
        bool capture = false;
        bool promotion = false;
    };

    SearchResult search_fixed_depth(const Position& pos, int depth, SearchContext& context);

    int negamax(
        Position pos,
        int depth,
        int ply,
        int alpha,
        int beta,
        SearchContext& context
    );

    int quiescence(
        Position pos,
        int alpha,
        int beta,
        int ply,
        int q_depth,
        SearchContext& context
    );

    SearchResult make_fallback_result(const Position& pos) const;

    std::vector<ScoredMove> ordered_moves(const Position& pos, int ply, Move tt_move = Move{}) const;
    std::vector<ScoredMove> ordered_noisy_moves(const Position& pos) const;

    ScoredMove make_scored_move(
        const Position& pos,
        Move move,
        int ply,
        Move tt_move = Move{},
        ScoringMode stage = ScoringMode::MainSearch
    ) const;
    int move_order_score(
        Move move,
        const Position& pos,
        int ply,
        Move tt_move,
        bool gives_check,
        bool capture,
        bool promotion,
        int see_score,
        ScoringMode stage
    ) const;
    bool should_stop(SearchContext& context) const;
    bool should_reduce_late_move(
        const Position& pos,
        const ScoredMove& scored_move,
        int depth,
        int move_index
    ) const;
    bool can_null_move_prune(const Position& pos, int depth, const SearchContext& context) const;
    void make_null_move(Position& pos) const;

    bool is_quiet_move(const ScoredMove& scored_move) const;
    bool better_scored_move(const ScoredMove& lhs, const ScoredMove& rhs) const;
    void record_beta_cutoff(const ScoredMove& scored_move, int ply, std::size_t move_index, bool lmr, bool pvs_scout);

    TranspositionTable tt_;
    KillerMoveTable killer_table_;
    HistoryTable history_table_;
    V12MoveOrderWeights weights_;
    MoveOrderingStats move_ordering_stats_;
    bool pvs_shadow_enabled_ = false;
    std::uint64_t pvs_shadow_max_fail_low_samples_ = 0;
    PvsShadowStats pvs_shadow_stats_;
};

} // namespace chess
