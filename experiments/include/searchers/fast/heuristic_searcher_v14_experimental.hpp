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

struct V14ExperimentalMoveOrderWeights {
    int tt_bonus = 1'000'000'000;
    int promotion_bonus = 900'000'000;
    int good_capture_bonus = 50'000'000;
    int bad_capture_bonus = 30'000;
    int see_weight = 10;
    int killer1_bonus = 40'000;
    int killer2_bonus = 10'000;
    int check_bonus = 160'000;
};

struct V14ExperimentalSearchOptions {
    bool enable_pvs = false;
    bool enable_pvs_skip_fail_low = false;
    bool enable_aspiration_window = true;
    bool enable_tt = true;
    bool enable_tt_bounds = true;
    bool enable_tt_exact = true;
    bool enable_tt_bound_cutoff = true;
    bool enable_tt_bound_tighten = true;
    bool enable_tt_exact_store = true;
    bool enable_tt_mate_scores = false;
};

class HeuristicSearcherV14Experimental final : public Searcher {
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
        std::uint64_t cutoff_by_pvs_scout = 0;
        std::uint64_t pvs_scouts = 0;
        std::uint64_t pvs_researches = 0;
    };

    explicit HeuristicSearcherV14Experimental(std::size_t tt_mb = 64);
    HeuristicSearcherV14Experimental(
        std::size_t tt_mb,
        V14ExperimentalMoveOrderWeights weights,
        V14ExperimentalSearchOptions options
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

private:
    struct SearchContext {
        std::uint64_t nodes = 0;
        std::chrono::steady_clock::time_point deadline{};
        bool has_deadline = false;
        bool stopped = false;
        bool in_null_move = false;
        bool in_pvs_scout = false;
        TTBound root_bound = TTBound::Exact;
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

    struct SearchValue {
        int score = 0;
        TTBound bound = TTBound::Exact;
    };

    struct TTProbeResult {
        int score = 0;
        TTBound bound = TTBound::Exact;
        Move best_move{};
        bool hit = false;
    };

    SearchResult search_fixed_depth(
        const Position& pos,
        int depth,
        SearchContext& context,
        int alpha = -Infinity,
        int beta = Infinity,
        bool allow_root_tt_probe = true
    );
    SearchResult search_root_without_tt_cutoff(
        const Position& pos,
        int depth,
        SearchContext& context
    );

    SearchValue negamax(
        Position pos,
        int depth,
        int ply,
        int alpha,
        int beta,
        SearchContext& context
    );

    SearchValue quiescence(
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
    bool is_quiet_move(const ScoredMove& scored_move) const;
    bool better_scored_move(const ScoredMove& lhs, const ScoredMove& rhs) const;
    void record_beta_cutoff(
        const ScoredMove& scored_move,
        int ply,
        std::size_t move_index,
        bool pvs_scout
    );
    TTProbeOptions make_tt_probe_options(int ply) const;
    TTProbeResult probe_tt(
        HashKey key,
        int depth,
        int& alpha,
        int& beta,
        int ply,
        bool allow_probe
    ) const;
    bool should_store_tt(TTBound bound) const;
    int score_to_tt(int score, int ply) const;

    TranspositionTable tt_;
    KillerMoveTable killer_table_;
    HistoryTable history_table_;
    V14ExperimentalMoveOrderWeights weights_;
    V14ExperimentalSearchOptions options_;
    MoveOrderingStats move_ordering_stats_;
};

} // namespace chess
