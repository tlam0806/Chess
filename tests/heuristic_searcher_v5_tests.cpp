#include "heuristic_searcher_v5.hpp"

#include <cassert>
#include <chrono>
#include <random>
#include <string_view>
#include <vector>

using namespace chess;

namespace {

bool contains_move(const std::vector<Move>& moves, Move target) {
    for (Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

void assert_v5_search_is_stable(Position pos, int max_depth) {
    HeuristicSearcherV5 v5(1);

    for (int depth = 0; depth <= max_depth; ++depth) {
        v5.clear_tt();
        const SearchResult cold = v5.search_best_move(pos, depth);
        const SearchResult warm = v5.search_best_move(pos, depth);

        assert(cold.score == warm.score);
        assert(cold.best_move == warm.best_move);
        assert(warm.nodes <= cold.nodes);

        const std::vector<Move> moves = generate_legal_moves(pos);
        if (!moves.empty() && depth > 0) {
            assert(contains_move(moves, cold.best_move));
        }
    }
}

void assert_v5_search_is_stable_from_fen(std::string_view fen, int max_depth) {
    Position pos;
    assert(pos.set_fen(fen));
    assert_v5_search_is_stable(pos, max_depth);
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_startpos();
        assert_v5_search_is_stable(pos, 3);
    }

    assert_v5_search_is_stable_from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        2);

    assert_v5_search_is_stable_from_fen(
        "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
        3);

    assert_v5_search_is_stable_from_fen(
        "r4rk1/1pp1qppp/p1npbn2/3Np3/2B1P3/2P2N2/PP1BQPPP/R3K2R w KQ - 0 10",
        3);

    {
        std::mt19937 rng(20260614);
        Position pos;
        pos.set_startpos();

        for (int sample = 0; sample < 30; ++sample) {
            assert_v5_search_is_stable(pos, 3);

            const std::vector<Move> moves = generate_legal_moves(pos);
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

        HeuristicSearcherV5 v5(1);
        const SearchResult result = v5.search_best_move(pos, SearchLimits{
            .max_depth = 64,
            .move_time = std::chrono::milliseconds{1}
        });

        assert(result.depth >= 0);
        assert(result.depth <= 64);
        assert(contains_move(generate_legal_moves(pos), result.best_move));
    }
}
