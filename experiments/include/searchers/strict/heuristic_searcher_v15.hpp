#pragma once

#include "searcher.hpp"
#include "range_transposition_table.hpp"
#include "killer_move_table.hpp"
#include "history_table.hpp"

#include <cstddef>
#include <vector>

namespace chess {

class HeuristicSearcherV15 final : public Searcher {
public:
    explicit HeuristicSearcherV15(std::size_t tt_mb = 64);

    SearchResult search_best_move(const Position& pos, int depth) override;
    SearchResult search_best_move(const Position& pos, const SearchLimits& limits) override;
    std::string_view name() const override;

    void clear_tt();
    std::size_t tt_entry_count() const;
    void clear_tt_stats();
    const RangeTranspositionTableStats& tt_stats() const;

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

    std::vector<ScoredMove> ordered_moves(const Position& pos, int ply, Move tt_move = Move{}) const;
    std::vector<ScoredMove> ordered_noisy_moves(const Position& pos) const;

    ScoredMove make_scored_move(
        const Position& pos,
        Move move,
        int ply,
        Move tt_move,
        ScoringMode stage
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
    HistoryTable history_table_;
};

} // namespace chess
