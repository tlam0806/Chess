#include "heuristic_searcher_v32.hpp"
#include "heuristic_searcher_v33.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

template <typename Searcher>
chess::SearchResult run(
    Searcher& searcher,
    const chess::Position& pos,
    int depth,
    std::uint64_t& elapsed_us
) {
    searcher.clear_tt();
    const auto start = std::chrono::steady_clock::now();
    chess::SearchResult result =
        searcher.search_best_move(pos, chess::SearchLimits{depth, std::chrono::milliseconds{0}});
    const auto end = std::chrono::steady_clock::now();
    elapsed_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return result;
}

} // namespace

int main(int argc, char** argv) {
    int depth = 11;
    std::string fen = "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4";
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        fen = argv[2];
    }

    chess::Position pos;
    if (!pos.set_fen(fen)) {
        std::cerr << "invalid fen\n";
        return 1;
    }

    chess::HeuristicSearcherV33 v33;
    chess::HeuristicSearcherV32 v32;
    std::uint64_t v33_us = 0;
    std::uint64_t v32_us = 0;
    const chess::SearchResult r33 = run(v33, pos, depth, v33_us);
    const chess::SearchResult r32 = run(v32, pos, depth, v32_us);

    std::cout << "depth=" << depth
              << " v33_score=" << r33.score
              << " v32_score=" << r32.score
              << " v33_best=" << chess::move_to_string(r33.best_move)
              << " v32_best=" << chess::move_to_string(r32.best_move)
              << " v33_nodes=" << r33.nodes
              << " v32_nodes=" << r32.nodes
              << " v33_us=" << v33_us
              << " v32_us=" << v32_us
              << " v33_vs_v32_time=" << (v32_us == 0 ? 0.0 : static_cast<double>(v33_us) / v32_us)
              << " score_mismatch=" << (r33.score != r32.score)
              << " move_mismatch=" << (r33.best_move != r32.best_move)
              << '\n';
}
