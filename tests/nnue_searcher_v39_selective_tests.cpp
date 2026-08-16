#include "move.hpp"
#include "nnue_searcher_v38.hpp"
#include "nnue_searcher_v39.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

bool contains_move(const std::vector<chess::Move>& moves, chess::Move target) {
    for (chess::Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

void assert_search_returns_legal_move(
    const chess::Position& pos,
    int depth,
    chess::NnueSearcherV39& searcher
) {
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    assert(!moves.empty());

    searcher.clear_tt();
    const chess::SearchResult result = searcher.search_best_move(pos, depth);
    assert(result.nodes > 0);
    assert(contains_move(moves, result.best_move));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "usage: nnue_searcher_v39_selective_tests <nnue-model.bin>\n";
        return 2;
    }

    chess::PhaseQuantizedNnueModel model;
    assert(model.load(argv[1]));

    const chess::NnueSearcherV38::SelectiveConfig v38_defaults;
    assert(!v38_defaults.enable_reverse_futility);
    assert(!v38_defaults.enable_late_move_pruning);

    chess::NnueSearcherV39 searcher(model);
    const chess::NnueSearcherV39::SelectiveConfig& defaults =
        searcher.selective_config();
    assert(defaults.enable_lmr);
    assert(defaults.lmr_base == 0.45);
    assert(defaults.lmr_divisor == 2.9);
    assert(defaults.lmr_min_depth == 3);
    assert(defaults.lmr_min_move_index == 6);
    assert(defaults.enable_null_move);
    assert(defaults.null_move_min_depth == 2);
    assert(defaults.null_move_reduction == 3);
    assert(defaults.enable_reverse_futility);
    assert(defaults.reverse_futility_max_depth == 2);
    assert(defaults.reverse_futility_base_margin == 175);
    assert(defaults.reverse_futility_margin_per_depth == 275);
    assert(defaults.enable_late_move_pruning);
    assert(defaults.late_move_pruning_max_depth == 3);
    assert(defaults.late_move_pruning_base == 4);
    assert(defaults.late_move_pruning_depth_multiplier == 2);
    assert(searcher.name() == "nnue_selective_v39");

    chess::Position start;
    start.set_startpos();
    for (int depth = 1; depth <= 4; ++depth) {
        assert_search_returns_legal_move(start, depth, searcher);
    }

    chess::Position tactical;
    assert(tactical.set_fen(
        "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4"));
    assert_search_returns_legal_move(tactical, 4, searcher);
    const chess::NnueSearcherV39::SelectiveStats default_stats =
        searcher.selective_stats();
    assert(default_stats.reverse_futility_evaluations > 0);
    assert(default_stats.late_move_pruned_moves > 0);

    chess::NnueSearcherV39::SelectiveConfig force_lmp = defaults;
    force_lmp.enable_lmr = false;
    force_lmp.enable_null_move = false;
    force_lmp.enable_reverse_futility = false;
    force_lmp.late_move_pruning_max_depth = 4;
    force_lmp.late_move_pruning_base = 0;
    force_lmp.late_move_pruning_depth_multiplier = 0;
    searcher.set_selective_config(force_lmp);
    searcher.clear_selective_stats();
    assert_search_returns_legal_move(start, 4, searcher);
    assert(searcher.selective_stats().late_move_pruned_nodes > 0);
    assert(searcher.selective_stats().late_move_pruned_moves > 0);

    std::cout << "nnue selective v39 RFP/LMP checks passed\n";
}
