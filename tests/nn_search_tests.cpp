#include "move.hpp"
#include "nn_search.hpp"
#include "nn_value.hpp"
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
    const chess::NnValueModel& model
) {
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    assert(!moves.empty());

    const chess::SearchResult result = chess::search_best_move_nn(pos, depth, model);
    assert(result.nodes > 0);
    assert(contains_move(moves, result.best_move));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: nn_search_tests <model.bin>\n";
        return 2;
    }

    chess::NnValueModel model;
    assert(model.load(argv[1]));

    chess::Position start;
    start.set_startpos();
    for (int depth = 1; depth <= 3; ++depth) {
        assert_search_returns_legal_move(start, depth, model);
    }

    chess::Position kiwipete;
    assert(kiwipete.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));
    assert_search_returns_legal_move(kiwipete, 2, model);

    std::cout << "nn search returned legal moves\n";
}
