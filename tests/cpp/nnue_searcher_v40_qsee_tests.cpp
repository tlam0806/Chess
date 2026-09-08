#include "move.hpp"
#include "nnue_searcher_v39.hpp"
#include "nnue_searcher_v40.hpp"
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
    chess::NnueSearcherV40& searcher
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
        std::cerr << "usage: nnue_searcher_v40_qsee_tests <nnue-model.bin>\n";
        return 2;
    }

    chess::PhaseQuantizedNnueModel model;
    assert(model.load(argv[1]));

    chess::NnueSearcherV39 v39(model);
    assert(!v39.selective_config().enable_qsearch_see_pruning);

    chess::NnueSearcherV40 v40(model);
    const chess::NnueSearcherV40::SelectiveConfig defaults =
        v40.selective_config();
    assert(defaults.enable_qsearch_see_pruning);
    assert(defaults.qsearch_see_threshold == -75);
    assert(v40.name() == "nnue_selective_v40");

    chess::Position captures;
    assert(captures.set_fen(
        "4k3/8/8/3p4/2R1R3/8/8/K7 w - - 0 1"));

    chess::NnueSearcherV40::SelectiveConfig force_qsee = defaults;
    force_qsee.enable_lmr = false;
    force_qsee.enable_null_move = false;
    force_qsee.enable_reverse_futility = false;
    force_qsee.enable_late_move_pruning = false;
    force_qsee.qsearch_see_threshold = 10'000;
    v40.set_selective_config(force_qsee);
    v40.clear_selective_stats();
    assert_search_returns_legal_move(captures, 1, v40);
    assert(v40.selective_stats().qsearch_see_evaluations > 0);
    assert(v40.selective_stats().qsearch_see_pruned_moves > 0);

    force_qsee.enable_qsearch_see_pruning = false;
    v40.set_selective_config(force_qsee);
    v40.clear_selective_stats();
    assert_search_returns_legal_move(captures, 1, v40);
    assert(v40.selective_stats().qsearch_see_evaluations == 0);
    assert(v40.selective_stats().qsearch_see_pruned_moves == 0);

    std::cout << "nnue selective v40 QSEE checks passed\n";
}
