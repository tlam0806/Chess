#include "heuristic_searcher_v2.hpp"
#include "heuristic_searcher_v3.hpp"

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

void assert_v3_matches_v2_score(Position pos, int max_depth) {
    HeuristicSearcherV2 v2(1);
    HeuristicSearcherV3 v3(1);

    for (int depth = 0; depth <= max_depth; ++depth) {
        v2.clear_tt();
        v3.clear_tt();

        const SearchResult v2_result = v2.search_best_move(pos, depth);
        const SearchResult v3_result = v3.search_best_move(pos, depth);

        assert(v3_result.score == v2_result.score);
        assert(v3_result.nodes > 0);

        const std::vector<Move> moves = generate_legal_moves(pos);
        if (!moves.empty() && depth > 0) {
            assert(contains_move(moves, v3_result.best_move));
        }
    }
}

void assert_v3_matches_v2_score_from_fen(std::string_view fen, int max_depth) {
    Position pos;
    assert(pos.set_fen(fen));
    assert_v3_matches_v2_score(pos, max_depth);
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_startpos();
        assert_v3_matches_v2_score(pos, 3);
    }

    assert_v3_matches_v2_score_from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        2);

    assert_v3_matches_v2_score_from_fen(
        "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
        3);

    assert_v3_matches_v2_score_from_fen(
        "r4rk1/1pp1qppp/p1npbn2/3Np3/2B1P3/2P2N2/PP1BQPPP/R3K2R w KQ - 0 10",
        3);

    {
        std::mt19937 rng(20260612);
        Position pos;
        pos.set_startpos();

        for (int sample = 0; sample < 30; ++sample) {
            assert_v3_matches_v2_score(pos, 3);

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

        HeuristicSearcherV3 v3(1);
        const SearchResult result = v3.search_best_move(pos, SearchLimits{
            .max_depth = 64,
            .move_time = std::chrono::milliseconds{1}
        });

        assert(result.depth >= 0);
        assert(result.depth <= 64);
        assert(contains_move(generate_legal_moves(pos), result.best_move));
    }
}
