#include "move.hpp"
#include "nnue_searcher_v10.hpp"
#include "nnue_value.hpp"
#include "position.hpp"

#include <cassert>
#include <chrono>
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
    chess::NnueSearcherV10& searcher
) {
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    assert(!moves.empty());

    const chess::SearchResult result = searcher.search_best_move(pos, depth);
    assert(result.nodes > 0);
    assert(contains_move(moves, result.best_move));
}

void assert_iterative_returns_legal_move(
    const chess::Position& pos,
    int depth,
    chess::NnueSearcherV10& searcher
) {
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    assert(!moves.empty());

    const chess::SearchResult result = searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = depth,
        .move_time = std::chrono::milliseconds{0},
    });
    assert(result.nodes > 0);
    assert(contains_move(moves, result.best_move));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: nnue_searcher_v10_tests <nnue-model.bin>\n";
        return 2;
    }

    chess::NnueValueModel model;
    assert(model.load(argv[1]));

    chess::NnueSearcherV10 searcher(model);

    chess::Position start;
    start.set_startpos();
    for (int depth = 1; depth <= 3; ++depth) {
        assert_search_returns_legal_move(start, depth, searcher);
    }
    assert_iterative_returns_legal_move(start, 3, searcher);

    chess::Position kiwipete;
    assert(kiwipete.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));
    assert_search_returns_legal_move(kiwipete, 3, searcher);

    chess::Position tactical;
    assert(tactical.set_fen("rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4"));
    assert_search_returns_legal_move(tactical, 3, searcher);

    std::cout << "nnue v10 search returned legal moves\n";
}
