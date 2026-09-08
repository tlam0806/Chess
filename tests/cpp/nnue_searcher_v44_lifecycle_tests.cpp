#include "nnue_searcher_v43.hpp"
#include "nnue_searcher_v44.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <iostream>

namespace {

template <typename Searcher>
concept HasStaleTtScoreApi = requires(Searcher& searcher) {
    searcher.set_reuse_stale_tt_scores(true);
    searcher.reuse_stale_tt_scores();
};

void assert_production_config_parity(
    const chess::NnueSearcherV43& v43,
    const chess::NnueSearcherV44& v44
) {
    const auto& old_selective = v43.selective_config();
    const auto& clear_each_search_selective = v44.selective_config();
#define ASSERT_SELECTIVE_FIELD(field) \
    assert(old_selective.field == clear_each_search_selective.field)
    ASSERT_SELECTIVE_FIELD(enable_lmr);
    ASSERT_SELECTIVE_FIELD(lmr_base);
    ASSERT_SELECTIVE_FIELD(lmr_divisor);
    ASSERT_SELECTIVE_FIELD(lmr_min_depth);
    ASSERT_SELECTIVE_FIELD(lmr_min_move_index);
    ASSERT_SELECTIVE_FIELD(enable_null_move);
    ASSERT_SELECTIVE_FIELD(null_move_min_depth);
    ASSERT_SELECTIVE_FIELD(null_move_reduction);
    ASSERT_SELECTIVE_FIELD(enable_reverse_futility);
    ASSERT_SELECTIVE_FIELD(reverse_futility_max_depth);
    ASSERT_SELECTIVE_FIELD(reverse_futility_base_margin);
    ASSERT_SELECTIVE_FIELD(reverse_futility_margin_per_depth);
    ASSERT_SELECTIVE_FIELD(enable_late_move_pruning);
    ASSERT_SELECTIVE_FIELD(late_move_pruning_max_depth);
    ASSERT_SELECTIVE_FIELD(late_move_pruning_base);
    ASSERT_SELECTIVE_FIELD(late_move_pruning_depth_multiplier);
    ASSERT_SELECTIVE_FIELD(enable_qsearch_see_pruning);
    ASSERT_SELECTIVE_FIELD(qsearch_see_threshold);
    ASSERT_SELECTIVE_FIELD(enable_main_search_see_pruning);
    ASSERT_SELECTIVE_FIELD(main_search_see_max_depth);
    ASSERT_SELECTIVE_FIELD(main_search_see_margin_per_depth);
#undef ASSERT_SELECTIVE_FIELD

    const auto& old_aspiration = v43.aspiration_config();
    const auto& clear_each_search_aspiration = v44.aspiration_config();
#define ASSERT_ASPIRATION_FIELD(field) \
    assert(old_aspiration.field == clear_each_search_aspiration.field)
    ASSERT_ASPIRATION_FIELD(enabled);
    ASSERT_ASPIRATION_FIELD(min_depth);
    ASSERT_ASPIRATION_FIELD(delta_base_cp);
    ASSERT_ASPIRATION_FIELD(delta_divisor);
    ASSERT_ASPIRATION_FIELD(expansion_factor_per_mille);
    ASSERT_ASPIRATION_FIELD(max_fail_high_reductions);
    ASSERT_ASPIRATION_FIELD(mean_score_new_weight_per_mille);
    ASSERT_ASPIRATION_FIELD(max_researches);
    ASSERT_ASPIRATION_FIELD(mean_score_clamp_cp);
#undef ASSERT_ASPIRATION_FIELD
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: nnue_searcher_v44_lifecycle_tests <model>\n";
        return 2;
    }

    static_assert(HasStaleTtScoreApi<chess::NnueSearcherV43>);
    static_assert(!HasStaleTtScoreApi<chess::NnueSearcherV44>);

    chess::PhaseQuantizedNnueModel model;
    assert(model.load(argv[1]));
    chess::NnueSearcherV43 v43(model, 1);
    chess::NnueSearcherV44 v44(model, 1);
    assert(v43.name() == "nnue_single_bound_v43");
    assert(v44.name() == "nnue_clear_each_search_v44");
    assert_production_config_parity(v43, v44);
    assert(!v43.reuse_deeper_tt_scores());
    assert(!v44.reuse_deeper_tt_scores());

    chess::Position start;
    start.set_startpos();
    const std::array<chess::HashKey, 1> start_history{start.zobrist_key};

    // A normal fixed-depth search must leave useful entries for the deeper
    // iterative passes within that one top-level call.
    const chess::SearchResult fixed =
        v44.search_best_move(start, 4, start_history);
    assert(!fixed.stopped);
    assert(fixed.depth == 4);
    assert(v44.tt_occupied_entry_count() > 0);

    // Compare the terminal follow-up with the same search on a fresh engine.
    // Equal occupancy proves that nothing retained by the preceding move
    // survived; terminal handling may legitimately store its own root entry.
    chess::Position mate;
    assert(mate.set_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));
    const std::array<chess::HashKey, 1> mate_history{mate.zobrist_key};
    chess::NnueSearcherV44 cold_terminal_v44(model, 1);
    const chess::SearchResult cold_terminal =
        cold_terminal_v44.search_best_move(mate, 4, mate_history);
    const chess::SearchResult terminal =
        v44.search_best_move(mate, 4, mate_history);
    assert(cold_terminal.score == -chess::CheckmateScore);
    assert(terminal.score == -chess::CheckmateScore);
    assert(
        v44.tt_occupied_entry_count()
        == cold_terminal_v44.tt_occupied_entry_count());

    chess::SearchLimits limits;
    limits.max_depth = 4;
    const chess::SearchResult iterative =
        v44.search_best_move(start, limits, start_history);
    assert(!iterative.stopped);
    assert(iterative.depth == 4);
    assert(v44.tt_occupied_entry_count() > 0);

    // The clear happens inside the measured top-level search, before even an
    // already-requested cooperative stop returns from the limits overload.
    std::atomic<bool> stop_requested{true};
    limits.max_depth = 64;
    limits.stop_requested = &stop_requested;
    const chess::SearchResult stopped =
        v44.search_best_move(start, limits, start_history);
    assert(stopped.stopped);
    assert(stopped.nodes == 0);
    assert(v44.tt_occupied_entry_count() == 0);

    std::cout << "nnue searcher v44 lifecycle tests passed\n";
    return 0;
}
