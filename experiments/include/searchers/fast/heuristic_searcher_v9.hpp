#pragma once

#include "searcher.hpp"
#include "transposition_table.hpp"
#include "killer_move_table.hpp"
#include "history_table.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class HeuristicSearcherV9 final : public Searcher {
public:
    explicit HeuristicSearcherV9(std::size_t tt_mb = 64);

    SearchResult search_best_move(const Position& pos, int depth) override;
    SearchResult search_best_move(const Position& pos, const SearchLimits& limits) override;
    std::string_view name() const override;

    void clear_tt();
    std::size_t tt_entry_count() const;
    void clear_tt_stats();
    const TranspositionTableStats& tt_stats() const;

private:
    struct SearchContext {
        std::uint64_t nodes = 0;
        std::chrono::steady_clock::time_point deadline{};
        bool has_deadline = false;
        bool stopped = false;
        bool in_null_move = false;
    };

    struct ScoredMove {
        Move move{};
        int priority = 0;
        int static_score = 0;
        bool gives_check = false;
    };

    enum class ScoringMode {
        MainSearch,
        Quiescence
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
    int move_priority(Move move, const Position& cur, Move tt_move, bool gives_check) const;
    int move_static_score(
        Move move,
        const Position& pos,
        const Position& next,
        int ply,
        bool gives_check,
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

    TranspositionTable tt_;
    KillerMoveTable killer_table_;
    HistoryTable history_table_;
};

} // namespace chess
