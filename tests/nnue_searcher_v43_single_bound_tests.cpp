#include "attacks.hpp"
#include "nnue_searcher_v41.hpp"
#include "nnue_searcher_v43.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool legal(const chess::Position& pos, chess::Move move) {
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

chess::SearchLimits depth_limits(int depth) {
    chess::SearchLimits limits;
    limits.max_depth = depth;
    return limits;
}

chess::Position forced_king_evasion(int halfmove_clock) {
    chess::Position pos;
    assert(pos.set_fen(
        "8/8/8/8/8/1k6/r7/K7 w - - "
        + std::to_string(halfmove_clock) + " 1"));
    assert(chess::generate_legal_moves(pos).size() == 1);
    return pos;
}

std::vector<chess::HashKey> threefold_child_history(
    const chess::Position& root
) {
    chess::Position child = root;
    child.make_move(chess::generate_legal_moves(root).front());
    return {
        child.zobrist_key,
        0x1111111111111111ULL,
        0x2222222222222222ULL,
        0x3333333333333333ULL,
        child.zobrist_key,
        0x4444444444444444ULL,
        0x5555555555555555ULL,
        root.zobrist_key,
    };
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: nnue_searcher_v43_single_bound_tests <model>\n";
        return 2;
    }

    chess::PhaseQuantizedNnueModel model;
    assert(model.load(argv[1]));
    chess::NnueSearcherV43 v43(model);
    assert(v43.name() == "nnue_single_bound_v43");
    assert(v43.aspiration_config().enabled);
    assert(!v43.reuse_stale_tt_scores());
    assert(!v43.reuse_deeper_tt_scores());
    v43.set_reuse_deeper_tt_scores(true);
    assert(v43.reuse_deeper_tt_scores());
    v43.set_reuse_deeper_tt_scores(false);

    chess::Position start;
    start.set_startpos();
    const std::array<chess::HashKey, 1> start_history{start.zobrist_key};

    const chess::SearchResult fixed =
        v43.search_best_move(start, 4, start_history);
    assert(!fixed.stopped);
    assert(fixed.depth == 4);
    assert(legal(start, fixed.best_move));

    // With every selective shortcut disabled, the scalar conversion must
    // preserve the clean V41 minimax score. TT replacement/cutoff semantics
    // can change node order, so only the score and move legality are fixed.
    chess::NnueSearcherV41 clean_v41(model);
    chess::NnueSearcherV43 clean_v43(model);
    auto clean_v41_config = clean_v41.selective_config();
    auto clean_v43_config = clean_v43.selective_config();
    auto disable_selectivity = [](auto& config) {
        config.enable_lmr = false;
        config.enable_null_move = false;
        config.enable_reverse_futility = false;
        config.enable_late_move_pruning = false;
        config.enable_qsearch_see_pruning = false;
        config.enable_main_search_see_pruning = false;
    };
    disable_selectivity(clean_v41_config);
    disable_selectivity(clean_v43_config);
    clean_v41.set_selective_config(clean_v41_config);
    clean_v43.set_selective_config(clean_v43_config);
    const chess::SearchResult clean_control =
        clean_v41.search_best_move(start, 4, start_history);
    const chess::SearchResult clean_scalar =
        clean_v43.search_best_move(start, 4, start_history);
    assert(clean_scalar.score == clean_control.score);
    assert(legal(start, clean_scalar.best_move));

    // A randomized clean-search audit found this position where reusing a
    // deeper TT score for a shallower fixed-depth request changes the depth-4
    // score. Strict V43 defaults to exact-depth reuse, so its scalar result
    // must retain V41's fixed-depth contract here as well.
    chess::Position exact_depth_regression;
    assert(exact_depth_regression.set_fen(
        "rn4nr/1p1Bk3/2p2pp1/p1Ppp2p/K3N2P/NQ1PP1P1/"
        "P4P2/2R4R b - - 0 27"));
    const std::array<chess::HashKey, 1> exact_depth_history{
        exact_depth_regression.zobrist_key};
    clean_v41.clear_tt();
    clean_v43.clear_tt();
    clean_v41.clear_search_heuristics();
    clean_v43.clear_search_heuristics();
    const chess::SearchResult exact_depth_control =
        clean_v41.search_best_move(
            exact_depth_regression, 4, exact_depth_history);
    const chess::SearchResult exact_depth_scalar =
        clean_v43.search_best_move(
            exact_depth_regression, 4, exact_depth_history);
    assert(exact_depth_scalar.score == exact_depth_control.score);
    assert(legal(exact_depth_regression, exact_depth_scalar.best_move));

    v43.clear_tt();
    const chess::SearchResult iterative =
        v43.search_best_move(start, depth_limits(5), start_history);
    assert(!iterative.stopped);
    assert(iterative.depth == 5);
    assert(legal(start, iterative.best_move));
    assert(v43.aspiration_stats().completed_iterations == 5);
    assert(v43.aspiration_stats().narrow_attempts > 0);
    assert(v43.aspiration_stats().range_conflict_fallbacks == 0);
    assert(v43.aspiration_stats().unresolved_ranges == 0);

    // Preflight the bounded retry escape used by every tuner candidate. A
    // one-centipawn window plus one allowed failed attempt must exercise both
    // fail-high/low detection and the scalar full-window fallback without
    // manufacturing a range conflict.
    chess::NnueSearcherV43 retry_v43(model);
    auto aggressive = retry_v43.aspiration_config();
    aggressive.enabled = true;
    aggressive.min_depth = 2;
    aggressive.delta_base_cp = 1;
    aggressive.delta_divisor = 1'000'000;
    aggressive.expansion_factor_per_mille = 1'000;
    aggressive.max_fail_high_reductions = 0;
    aggressive.max_researches = 1;
    retry_v43.set_aspiration_config(aggressive);
    const std::array<std::string_view, 3> retry_fens{
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1NBPPP/"
        "R2QKB1R w KQ - 4 8",
        "r3k2r/ppp2ppp/2n1bn2/3qp3/3p4/1P1P1NP1/PBPNPPBP/"
        "R2Q1RK1 w kq - 0 10",
    };
    std::uint64_t retry_fail_lows = 0;
    std::uint64_t retry_fail_highs = 0;
    std::uint64_t retry_fallbacks = 0;
    for (std::string_view fen : retry_fens) {
        chess::Position retry_position;
        assert(retry_position.set_fen(fen));
        const std::array<chess::HashKey, 1> retry_history{
            retry_position.zobrist_key};
        retry_v43.clear_tt();
        retry_v43.clear_search_heuristics();
        const chess::SearchResult retry_result = retry_v43.search_best_move(
            retry_position, depth_limits(4), retry_history);
        assert(!retry_result.stopped);
        assert(retry_result.depth == 4);
        assert(legal(retry_position, retry_result.best_move));
        retry_fail_lows += retry_v43.aspiration_stats().fail_lows;
        retry_fail_highs += retry_v43.aspiration_stats().fail_highs;
        retry_fallbacks +=
            retry_v43.aspiration_stats().retry_limit_fallbacks;
        assert(retry_v43.aspiration_stats().range_conflict_fallbacks == 0);
        assert(retry_v43.aspiration_stats().unresolved_ranges == 0);
    }
    assert(retry_fail_lows > 0);
    assert(retry_fail_highs > 0);
    assert(retry_fallbacks > 0);

    // Cooperative stop must be observed at the next root boundary, including
    // with a warm TT, before a retry or a root TT cutoff can consume nodes.
    std::atomic<bool> stop_requested{true};
    chess::SearchLimits stopped_limits = depth_limits(64);
    stopped_limits.stop_requested = &stop_requested;
    const chess::SearchResult stopped =
        v43.search_best_move(start, stopped_limits, start_history);
    assert(stopped.stopped);
    assert(stopped.nodes == 0);
    assert(legal(start, stopped.best_move));
    assert(v43.aspiration_stats().completed_iterations == 0);
    assert(v43.aspiration_stats().narrow_attempts == 0);

    chess::Position checked;
    assert(checked.set_fen("4r1k1/8/8/8/8/8/8/4K3 w - - 0 1"));
    assert(chess::in_check(checked, checked.side_to_move));
    const std::array<chess::HashKey, 1> checked_history{checked.zobrist_key};
    const chess::SearchResult checked_result =
        v43.search_best_move(checked, depth_limits(4), checked_history);
    assert(!checked_result.stopped);
    assert(legal(checked, checked_result.best_move));

    const chess::Position repeat_root = forced_king_evasion(8);
    const std::vector<chess::HashKey> repeat_history =
        threefold_child_history(repeat_root);
    const chess::SearchResult repetition =
        v43.search_best_move(repeat_root, 3, repeat_history);
    assert(repetition.score == 0);
    assert(v43.repetition_stats().threefold_draws > 0);

    chess::Position fifty;
    assert(fifty.set_fen("7k/8/8/8/8/8/8/KR6 w - - 100 1"));
    const chess::SearchResult fifty_result = v43.search_best_move(fifty, 4);
    assert(fifty_result.score == 0);
    assert(legal(fifty, fifty_result.best_move));

    chess::Position mate;
    assert(mate.set_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1"));
    const chess::SearchResult mate_result = v43.search_best_move(mate, 4);
    assert(mate_result.score == -chess::CheckmateScore);

    // This exact root exposed V42's clean-upper/dirty-lower conflict. V43 has
    // no opposite-bound merge, so the adaptive search must finish normally.
    chess::Position conflict_root;
    assert(conflict_root.set_fen(
        "2kr1b2/pp1n2p1/q1p1ppb1/3p3r/3P1B1P/2P1PPN1/"
        "P1P3B1/1R1QK2R w K - 0 16"));
    v43.clear_tt();
    const chess::SearchResult conflict_result =
        v43.search_best_move(conflict_root, depth_limits(5));
    assert(!conflict_result.stopped);
    assert(conflict_result.depth == 5);
    assert(legal(conflict_root, conflict_result.best_move));
    assert(v43.aspiration_stats().range_conflict_fallbacks == 0);
    assert(v43.aspiration_stats().unresolved_ranges == 0);

    v43.set_reuse_stale_tt_scores(true);
    assert(v43.reuse_stale_tt_scores());
    const chess::SearchResult stale_reuse =
        v43.search_best_move(start, depth_limits(3), start_history);
    assert(!stale_reuse.stopped);
    assert(legal(start, stale_reuse.best_move));

    std::cout << "nnue searcher v43 single-bound tests passed\n";
    return 0;
}
