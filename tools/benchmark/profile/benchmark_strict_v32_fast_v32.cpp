#include "heuristic_searcher_fast_v32.hpp"
#include "heuristic_searcher_v32.hpp"
#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

struct Row {
    chess::SearchResult result;
    std::uint64_t us = 0;
};

template <typename Searcher>
Row run_one(Searcher& searcher, const chess::Position& pos, int depth) {
    searcher.clear_tt();
    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = std::chrono::steady_clock::now();
    chess::SearchResult result = searcher.search_best_move(pos, limits);
    const auto end = std::chrono::steady_clock::now();
    return Row{
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())
    };
}

bool has_both_kings(const chess::Position& pos) {
    const auto king = static_cast<int>(chess::PieceType::King);
    return chess::popcount(pos.pieces[static_cast<int>(chess::Color::White)][king]) == 1
        && chess::popcount(pos.pieces[static_cast<int>(chess::Color::Black)][king]) == 1;
}

std::vector<chess::Position> make_positions() {
    const std::vector<std::string_view> fens = {
        "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
        "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    };

    std::vector<chess::Position> positions;
    for (std::string_view fen : fens) {
        chess::Position pos;
        if (pos.set_fen(fen) && has_both_kings(pos)) {
            positions.push_back(pos);
        }
    }
    return positions;
}

} // namespace

int main(int argc, char** argv) {
    const int depth = argc >= 2 ? std::stoi(argv[1]) : 6;
    const std::vector<chess::Position> positions = make_positions();
    std::uint64_t strict_nodes = 0;
    std::uint64_t fast_nodes = 0;
    std::uint64_t strict_us = 0;
    std::uint64_t fast_us = 0;
    int score_mismatches = 0;
    int move_mismatches = 0;

    for (std::size_t i = 0; i < positions.size(); ++i) {
        chess::HeuristicSearcherV32 strict_v32(24);
        chess::HeuristicSearcherFastV32 fast_v32(24);
        const Row strict_row = run_one(strict_v32, positions[i], depth);
        const Row fast_row = run_one(fast_v32, positions[i], depth);
        strict_nodes += strict_row.result.nodes;
        fast_nodes += fast_row.result.nodes;
        strict_us += strict_row.us;
        fast_us += fast_row.us;
        score_mismatches += strict_row.result.score != fast_row.result.score ? 1 : 0;
        move_mismatches += strict_row.result.best_move != fast_row.result.best_move ? 1 : 0;
        std::cout << "sample=" << i
                  << " strict_score=" << strict_row.result.score
                  << " fast_score=" << fast_row.result.score
                  << " strict_best=" << chess::move_to_string(strict_row.result.best_move)
                  << " fast_best=" << chess::move_to_string(fast_row.result.best_move)
                  << " strict_nodes=" << strict_row.result.nodes
                  << " fast_nodes=" << fast_row.result.nodes
                  << " strict_us=" << strict_row.us
                  << " fast_us=" << fast_row.us
                  << '\n';
    }

    std::cout << "summary"
              << " depth=" << depth
              << " samples=" << positions.size()
              << " strict_nodes=" << strict_nodes
              << " fast_nodes=" << fast_nodes
              << " strict_us=" << strict_us
              << " fast_us=" << fast_us
              << " fast_vs_strict_nodes=" << (strict_nodes == 0 ? 0.0 : static_cast<double>(fast_nodes) / strict_nodes)
              << " fast_vs_strict_time=" << (strict_us == 0 ? 0.0 : static_cast<double>(fast_us) / strict_us)
              << " score_mismatches=" << score_mismatches
              << " move_mismatches=" << move_mismatches
              << '\n';
}
