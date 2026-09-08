#include "heuristic_searcher_v32.hpp"
#include "heuristic_searcher_v34.hpp"
#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

template <typename Searcher>
std::pair<chess::SearchResult, std::uint64_t> run_one(
    Searcher& searcher,
    const chess::Position& pos,
    int depth
) {
    searcher.clear_tt();
    const auto start = std::chrono::steady_clock::now();
    const chess::SearchResult result =
        searcher.search_best_move(pos, chess::SearchLimits{depth, std::chrono::milliseconds{0}});
    const auto end = std::chrono::steady_clock::now();
    return {
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())
    };
}

} // namespace

int main(int argc, char** argv) {
    int depth = 11;
    std::string fen =
        "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4";
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        fen = argv[2];
    }

    chess::Position pos;
    if (!pos.set_fen(fen)) {
        std::cerr << "invalid FEN\n";
        return 1;
    }

    chess::HeuristicSearcherV34 v34;
    chess::HeuristicSearcherV32 v32;
    const auto [r34, v34_us] = run_one(v34, pos, depth);
    const auto [r32, v32_us] = run_one(v32, pos, depth);

    std::cout << "depth=" << depth
              << " v34_score=" << r34.score
              << " v32_score=" << r32.score
              << " v34_best=" << chess::move_to_string(r34.best_move)
              << " v32_best=" << chess::move_to_string(r32.best_move)
              << " v34_nodes=" << r34.nodes
              << " v32_nodes=" << r32.nodes
              << " v34_us=" << v34_us
              << " v32_us=" << v32_us
              << " v34_vs_v32_time=" << (v32_us == 0 ? 0.0 : static_cast<double>(v34_us) / v32_us)
              << " score_mismatch=" << (r34.score != r32.score)
              << " move_mismatch=" << (r34.best_move != r32.best_move)
              << '\n';
}
