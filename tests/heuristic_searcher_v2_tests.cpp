#include "heuristic_searcher.hpp"
#include "heuristic_searcher_v2.hpp"

#include <cassert>
#include <chrono>
#include <random>
#include <string_view>
#include <vector>

using namespace chess;

namespace {

void assert_same_result(SearchResult v1_result, SearchResult v2_result) {
    assert(v2_result.score == v1_result.score);
    assert(v2_result.best_move == v1_result.best_move);
    assert(v2_result.nodes > 0);
}

void assert_same_search(Position pos, int max_depth) {
    HeuristicSearcher baseline;
    HeuristicSearcherV2 v2(1);

    for (int depth = 0; depth <= max_depth; ++depth) {
        v2.clear_tt();
        const SearchResult v1_result = baseline.search_best_move(pos, depth);
        const SearchResult v2_cold = v2.search_best_move(pos, depth);
        const SearchResult v2_warm = v2.search_best_move(pos, depth);

        assert_same_result(v1_result, v2_cold);
        assert_same_result(v1_result, v2_warm);
        assert(v2_warm.nodes <= v2_cold.nodes);
    }
}

void assert_same_search_from_fen(std::string_view fen, int max_depth) {
    Position pos;
    assert(pos.set_fen(fen));
    assert_same_search(pos, max_depth);
}

bool contains_move(const std::vector<Move>& moves, Move target) {
    for (Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_startpos();
        assert_same_search(pos, 3);
    }

    assert_same_search_from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        2);

    assert_same_search_from_fen(
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        3);

    assert_same_search_from_fen(
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        3);

    assert_same_search_from_fen(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        2);

    assert_same_search_from_fen(
        "r4rk1/1pp1qppp/p1npbn2/3Np3/2B1P3/2P2N2/PP1BQPPP/R3K2R w KQ - 0 10",
        3);

    {
        std::mt19937 rng(12345);
        Position pos;
        pos.set_startpos();

        for (int sample = 0; sample < 30; ++sample) {
            assert_same_search(pos, 3);

            std::vector<Move> moves = generate_legal_moves(pos);
            if (moves.empty()) {
                pos.set_startpos();
                continue;
            }

            std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
            pos.make_move(moves[dist(rng)]);
        }
    }

    {
        Position pos;
        pos.set_startpos();

        HeuristicSearcherV2 fixed(1);
        HeuristicSearcherV2 iterative(1);

        const SearchResult fixed_result = fixed.search_best_move(pos, 3);
        const SearchResult iterative_result = iterative.search_best_move(pos, SearchLimits{
            .max_depth = 3,
            .move_time = std::chrono::milliseconds{0}
        });

        assert(iterative_result.score == fixed_result.score);
        assert(iterative_result.best_move == fixed_result.best_move);
        assert(iterative_result.depth == 3);
        assert(!iterative_result.stopped);
    }

    {
        Position pos;
        pos.set_startpos();

        HeuristicSearcherV2 timed(1);
        const SearchResult result = timed.search_best_move(pos, SearchLimits{
            .max_depth = 64,
            .move_time = std::chrono::milliseconds{1}
        });

        assert(result.depth >= 0);
        assert(result.depth <= 64);
        assert(contains_move(generate_legal_moves(pos), result.best_move));
    }
}
