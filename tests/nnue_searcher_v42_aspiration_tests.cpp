#include "attacks.hpp"
#include "move.hpp"
#include "nnue_searcher_v41.hpp"
#include "nnue_searcher_v42.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#if defined(NDEBUG)
#define CHESS_RELEASE_TEST_WITH_ASSERTS 1
#undef NDEBUG
#endif

#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool is_legal(const chess::Position& pos, chess::Move move) {
    const std::vector<chess::Move> legal = chess::generate_legal_moves(pos);
    for (const chess::Move candidate : legal) {
        if (candidate == move) {
            return true;
        }
    }
    return false;
}

chess::SearchLimits depth_limits(int depth) {
    chess::SearchLimits limits;
    limits.max_depth = depth;
    return limits;
}

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "usage: nnue_searcher_v42_aspiration_tests <nnue-model.bin>\n";
        return 2;
    }

    chess::PhaseQuantizedNnueModel model;
    assert(model.load(argv[1]));

    chess::ScoreRange contradictory{100, 99};
    assert(chess::discard_conflicting_score_range(contradictory));
    assert(contradictory.lower == -chess::Infinity);
    assert(contradictory.upper == chess::Infinity);
    chess::ScoreRange exact_equality{100, 100};
    assert(!chess::discard_conflicting_score_range(exact_equality));
    assert(exact_equality.lower == exact_equality.upper);

    chess::NnueSearcherV41 v41(model);
    chess::NnueSearcherV42 v42(model);
    assert(v41.name() == "nnue_repetition_v41");
    assert(v42.name() == "nnue_adaptive_aspiration_v42");
    assert(!v41.aspiration_config().enabled);
    assert(v42.aspiration_config().enabled);

    // The integer-depth API is deliberately one fixed-depth search.  It has
    // no completed previous iteration to form an aspiration window from, so
    // enabling V42 must leave this path bit-for-bit compatible with V41.
    chess::Position start;
    start.set_startpos();
    const std::array<chess::HashKey, 1> start_history{start.zobrist_key};
#ifdef CHESS_RELEASE_TEST_WITH_ASSERTS
    const chess::SearchResult v41_fixed =
        v41.search_best_move(start, 4, start_history);
    const chess::SearchResult v42_fixed =
        v42.search_best_move(start, 4, start_history);
    require(v42_fixed.best_move == v41_fixed.best_move,
        "V42 fixed best move differs from V41");
    require(v42_fixed.score == v41_fixed.score,
        "V42 fixed score differs from V41");
    require(v42_fixed.nodes == v41_fixed.nodes,
        "V42 fixed nodes differ from V41");
    require(v42.aspiration_config().enabled,
        "V42 fixed search did not restore adaptive configuration");
    require(v42.aspiration_stats().completed_iterations == 0,
        "fixed-depth V42 unexpectedly recorded iterations");
    require(v42.aspiration_stats().narrow_attempts == 0,
        "fixed-depth V42 unexpectedly opened an aspiration window");

    v41.clear_tt();
    v41.clear_search_heuristics();
    v42.clear_tt();
    v42.clear_search_heuristics();
    const chess::SearchResult v41_fixed_without_history =
        v41.search_best_move(start, 4);
    const chess::SearchResult v42_fixed_without_history =
        v42.search_best_move(start, 4);
    require(
        v42_fixed_without_history.best_move
            == v41_fixed_without_history.best_move,
        "V42 history-free fixed best move differs from V41");
    require(v42_fixed_without_history.score == v41_fixed_without_history.score,
        "V42 history-free fixed score differs from V41");
    require(v42_fixed_without_history.nodes == v41_fixed_without_history.nodes,
        "V42 history-free fixed nodes differ from V41");
    require(v42.aspiration_config().enabled,
        "history-free V42 fixed search did not restore adaptive config");
#endif

    // SearchLimits selects iterative deepening, which is the production/UCI
    // path and therefore the path where V42's adaptive windows operate.
    v42.clear_tt();
    v42.clear_search_heuristics();
    const chess::SearchResult iterative =
        v42.search_best_move(start, depth_limits(5), start_history);
    const auto iterative_stats = v42.aspiration_stats();
    assert(!iterative.stopped);
    assert(iterative.depth == 5);
    assert(is_legal(start, iterative.best_move));
    assert(iterative_stats.completed_iterations == 5);
    assert(iterative_stats.has_final_mean_score);
    assert(iterative_stats.narrow_attempts >= 3);
    assert(iterative_stats.initial_window_successes
        + iterative_stats.fail_lows
        + iterative_stats.fail_highs > 0);

    // The sealed tuning corpus intentionally contains only non-check roots.
    // Keep root-in-check correctness locked here so that this sampling choice
    // cannot hide an aspiration rollback or unresolved final iteration.
    chess::Position checked_root;
    assert(checked_root.set_fen(
        "4r1k1/8/8/8/8/8/8/4K3 w - - 0 1"));
    assert(chess::in_check(checked_root, checked_root.side_to_move));
    const std::array<chess::HashKey, 1> checked_history{
        checked_root.zobrist_key};
    v42.clear_tt();
    v42.clear_search_heuristics();
    const chess::SearchResult checked_result = v42.search_best_move(
        checked_root, depth_limits(6), checked_history);
    assert(!checked_result.stopped);
    assert(checked_result.depth == 6);
    assert(is_legal(checked_root, checked_result.best_move));
    assert(v42.aspiration_stats().completed_iterations == 6);
    assert(v42.aspiration_stats().unresolved_ranges == 0);

    // The V42 evaluator seeds isolated positions with a one-entry history
    // containing the root, matching the production history-aware API without
    // inventing earlier game occurrences.  This reversible ending has a
    // forced root move and many four-ply cycles, so it proves that the
    // history-aware path and the optional twofold-in-search policy are active
    // rather than merely echoed in evaluator metadata.
    chess::Position cycle_root;
    assert(cycle_root.set_fen(
        "8/8/8/8/8/1k6/8/K7 w - - 0 1"));
    const std::array<chess::HashKey, 1> cycle_history{
        cycle_root.zobrist_key};
    chess::NnueSearcherV42 cycle_v42(model);
    cycle_v42.set_twofold_search_draw_enabled(true);
    const chess::SearchResult cycle_result = cycle_v42.search_best_move(
        cycle_root, depth_limits(8), cycle_history);
    assert(!cycle_result.stopped);
    assert(cycle_result.depth == 8);
    assert(is_legal(cycle_root, cycle_result.best_move));
    assert(cycle_v42.twofold_search_draw_enabled());
    assert(cycle_v42.repetition_stats().search_cycle_draws > 0);

    // Searcher telemetry is per-search, not cumulative.  A following shallow
    // root with no possible cycle must reset the prior nonzero count; this is
    // what lets the evaluator safely sum one snapshot per dataset sample.
    cycle_v42.set_twofold_search_draw_enabled(false);
    (void)cycle_v42.search_best_move(start, depth_limits(1), start_history);
    assert(cycle_v42.repetition_stats().threefold_draws == 0);
    assert(cycle_v42.repetition_stats().search_cycle_draws == 0);
    assert(cycle_v42.repetition_stats().fifty_move_draws == 0);
    assert(cycle_v42.repetition_stats().tt_score_suppressions == 0);

    // Disabling the new policy on V42 must reproduce V41's validated legacy
    // iterative aspiration policy exactly.
    auto disabled = v42.aspiration_config();
    disabled.enabled = false;
    v42.set_aspiration_config(disabled);
#ifdef CHESS_RELEASE_TEST_WITH_ASSERTS
    v41.clear_tt();
    v41.clear_search_heuristics();
    v42.clear_tt();
    v42.clear_search_heuristics();
    const chess::SearchResult legacy =
        v41.search_best_move(start, depth_limits(5), start_history);
    const chess::SearchResult disabled_v42 =
        v42.search_best_move(start, depth_limits(5), start_history);
    require(disabled_v42.best_move == legacy.best_move,
        "disabled V42 start best move differs from V41");
    require(disabled_v42.score == legacy.score,
        "disabled V42 start score differs from V41");
    require(disabled_v42.nodes == legacy.nodes,
        "disabled V42 start nodes differ from V41");
    require(v42.aspiration_stats().completed_iterations == 0,
        "disabled V42 unexpectedly recorded adaptive iterations");

    // Version isolation is part of the experiment: disabling adaptive V42
    // must retain the exact legacy V41 root-range/fallback behavior, including
    // on roots which exercise dense transpositions under the enabled policy.
    const std::array<std::string_view, 3> disabled_parity_fens{
        "8/8/4k3/8/1p2N3/1P1KP3/7b/8 w - - 0 1",
        "8/8/8/1pp5/k7/3K4/BP6/8 w - - 0 1",
        "8/8/2k4p/7P/n7/3B4/8/6K1 w - - 0 1",
    };
    for (const std::string_view fen : disabled_parity_fens) {
        chess::Position parity_position;
        assert(parity_position.set_fen(fen));
        const std::array<chess::HashKey, 1> parity_history{
            parity_position.zobrist_key};
        v41.clear_tt();
        v41.clear_search_heuristics();
        v42.clear_tt();
        v42.clear_search_heuristics();
        const chess::SearchResult expected = v41.search_best_move(
            parity_position, depth_limits(6), parity_history);
        const chess::SearchResult actual = v42.search_best_move(
            parity_position, depth_limits(6), parity_history);
        require(actual.best_move == expected.best_move,
            "disabled V42 best move differs from V41");
        require(actual.ponder_move == expected.ponder_move,
            "disabled V42 ponder move differs from V41");
        require(actual.score == expected.score,
            "disabled V42 score differs from V41");
        require(actual.depth == expected.depth,
            "disabled V42 depth differs from V41");
        require(actual.nodes == expected.nodes,
            "disabled V42 nodes differ from V41");
        require(actual.stopped == expected.stopped,
            "disabled V42 stop state differs from V41");
        require(v42.aspiration_stats().completed_iterations == 0,
            "disabled V42 unexpectedly recorded adaptive iterations");
        require(v42.aspiration_stats().range_conflict_fallbacks == 0,
            "disabled V42 unexpectedly used conflict fallback");
    }
#endif

    // A one-centipawn initial window and one allowed failed attempt should
    // exercise the bounded full-window escape on at least one varied root.
    auto aggressive = disabled;
    aggressive.enabled = true;
    aggressive.min_depth = 2;
    aggressive.delta_base_cp = 1;
    aggressive.delta_divisor = 1'000'000;
    aggressive.expansion_factor_per_mille = 1'000;
    aggressive.max_fail_high_reductions = 0;
    aggressive.max_researches = 1;
    v42.set_aspiration_config(aggressive);
    const std::array<std::string_view, 4> retry_fens{
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1NBPPP/R2QKB1R w KQ - 4 8",
        "4k3/8/8/3p4/2R1R3/8/8/K7 w - - 0 1",
        "r3k2r/ppp2ppp/2n1bn2/3qp3/3p4/1P1P1NP1/PBPNPPBP/R2Q1RK1 w kq - 0 10",
    };
    std::uint64_t retry_fallbacks = 0;
    std::uint64_t retry_failures = 0;
    for (std::size_t retry_index = 0;
         retry_index < retry_fens.size();
         ++retry_index) {
#ifndef CHESS_RELEASE_TEST_WITH_ASSERTS
        // This legacy rook-only fixture trips the independent incremental
        // king-checker debug guard before aspiration logic is reached. Keep
        // it in the Release regression set, and let Debug exercise the other
        // enabled-V42 retry roots.
        if (retry_index == 2) {
            continue;
        }
#endif
        const std::string_view fen = retry_fens[retry_index];
        chess::Position pos;
        assert(pos.set_fen(fen));
        const std::array<chess::HashKey, 1> history{pos.zobrist_key};
        v42.clear_tt();
        v42.clear_search_heuristics();
        const chess::SearchResult result =
            v42.search_best_move(pos, depth_limits(4), history);
        assert(!result.stopped);
        assert(result.depth == 4);
        assert(is_legal(pos, result.best_move));
        retry_fallbacks += v42.aspiration_stats().retry_limit_fallbacks;
        retry_failures += v42.aspiration_stats().fail_lows
            + v42.aspiration_stats().fail_highs;
    }
    assert(retry_failures > 0);
    assert(retry_fallbacks > 0);

    // Narrow retries used to leave incompatible child bounds in the current
    // TT generation here.  The nominal depth-5 iteration then failed its
    // full-window escape and V42 silently returned the completed depth-4
    // result.  A fresh score generation for the fallback must reach depth 6.
    chess::Position fallback_conflict;
    assert(fallback_conflict.set_fen(
        "8/8/4k3/8/1p2N3/1P1KP3/7b/8 w - - 0 1"));
    const std::array<chess::HashKey, 1> fallback_conflict_history{
        fallback_conflict.zobrist_key};
    auto no_fail_high_reduction = v42.aspiration_config();
    no_fail_high_reduction.enabled = true;
    no_fail_high_reduction.min_depth = 3;
    no_fail_high_reduction.delta_base_cp = 30;
    no_fail_high_reduction.delta_divisor = 10'000;
    no_fail_high_reduction.expansion_factor_per_mille = 1'750;
    no_fail_high_reduction.max_fail_high_reductions = 0;
    no_fail_high_reduction.mean_score_new_weight_per_mille = 500;
    no_fail_high_reduction.max_researches = 6;
    no_fail_high_reduction.mean_score_clamp_cp = 1'500;
    v42.set_aspiration_config(no_fail_high_reduction);
    v42.clear_tt();
    v42.clear_search_heuristics();
    const chess::SearchResult conflict_result = v42.search_best_move(
        fallback_conflict,
        depth_limits(6),
        fallback_conflict_history);
    assert(!conflict_result.stopped);
    assert(conflict_result.depth == 6);
    assert(is_legal(fallback_conflict, conflict_result.best_move));
    assert(v42.aspiration_stats().unresolved_ranges == 0);

    // A clean table alone is insufficient for this transposition-rich pawn
    // ending: the emergency search can recreate incompatible score bounds
    // inside its own tree. The fallback therefore has to suppress TT scores
    // for the complete search, while it may still use old moves for ordering.
    chess::Position in_search_conflict;
    assert(in_search_conflict.set_fen(
        "8/8/8/1pp5/k7/3K4/BP6/8 w - - 0 1"));
    const std::array<chess::HashKey, 1> in_search_conflict_history{
        in_search_conflict.zobrist_key};
    v42.clear_tt();
    v42.clear_search_heuristics();
    const chess::SearchResult in_search_conflict_result =
        v42.search_best_move(
            in_search_conflict,
            depth_limits(6),
            in_search_conflict_history);
    assert(!in_search_conflict_result.stopped);
    assert(in_search_conflict_result.depth == 6);
    assert(is_legal(
        in_search_conflict, in_search_conflict_result.best_move));
    assert(v42.aspiration_stats().range_conflict_fallbacks > 0);
    assert(v42.aspiration_stats().full_window_fallbacks > 0);
    assert(v42.aspiration_stats().unresolved_ranges == 0);

    // The same TT-score-free escape is required before min_depth. This root
    // returns a non-exact full-window range at nominal depth 4, before its
    // configured adaptive narrow windows start at depth 5.
    chess::Position pre_narrow_conflict;
    assert(pre_narrow_conflict.set_fen(
        "8/8/2k4p/7P/n7/3B4/8/6K1 w - - 0 1"));
    const std::array<chess::HashKey, 1> pre_narrow_conflict_history{
        pre_narrow_conflict.zobrist_key};
    auto late_aspiration = no_fail_high_reduction;
    late_aspiration.min_depth = 5;
    late_aspiration.delta_base_cp = 90;
    late_aspiration.delta_divisor = 40'000;
    late_aspiration.expansion_factor_per_mille = 3'000;
    late_aspiration.max_fail_high_reductions = 3;
    late_aspiration.mean_score_new_weight_per_mille = 1'000;
    v42.set_aspiration_config(late_aspiration);
    v42.clear_tt();
    v42.clear_search_heuristics();
    const chess::SearchResult pre_narrow_conflict_result =
        v42.search_best_move(
            pre_narrow_conflict,
            depth_limits(6),
            pre_narrow_conflict_history);
    assert(!pre_narrow_conflict_result.stopped);
    assert(pre_narrow_conflict_result.depth == 6);
    assert(is_legal(
        pre_narrow_conflict, pre_narrow_conflict_result.best_move));
    assert(v42.aspiration_stats().full_window_fallbacks > 0);
    assert(v42.aspiration_stats().unresolved_ranges == 0);

    // A stable mate score bypasses narrow windows at later iterations.
    chess::Position mate_in_one;
    assert(mate_in_one.set_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1"));
    const std::array<chess::HashKey, 1> mate_history{
        mate_in_one.zobrist_key};
    v42.clear_tt();
    const chess::SearchResult mating =
        v42.search_best_move(mate_in_one, depth_limits(4), mate_history);
    assert(mating.score >= chess::CheckmateScore / 2);
    assert(v42.aspiration_stats().narrow_attempts == 0);

    // A pre-requested cooperative stop must be observed before any root TT
    // attempt, even when a previous search has warmed the table.
    std::atomic<bool> stop_requested{true};
    chess::SearchLimits stopped_limits = depth_limits(64);
    stopped_limits.stop_requested = &stop_requested;
    const chess::SearchResult stopped =
        v42.search_best_move(start, stopped_limits, start_history);
    assert(stopped.stopped);
    assert(stopped.nodes == 0);
    assert(is_legal(start, stopped.best_move));
    assert(v42.aspiration_stats().completed_iterations == 0);

    bool rejected = false;
    try {
        auto invalid = v42.aspiration_config();
        invalid.delta_divisor = 0;
        v42.set_aspiration_config(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        auto invalid = v42.aspiration_config();
        invalid.min_depth = 1;
        v42.set_aspiration_config(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    std::cout << "nnue v42 adaptive aspiration checks passed\n";
}
