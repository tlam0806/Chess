#include "heuristic_searcher_v32.hpp"
#include "heuristic_searcher_v35.hpp"
#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

bool has_both_kings(const chess::Position& pos) {
    const auto king_index = static_cast<int>(chess::PieceType::King);
    return chess::popcount(pos.pieces[static_cast<int>(chess::Color::White)][king_index]) == 1
        && chess::popcount(pos.pieces[static_cast<int>(chess::Color::Black)][king_index]) == 1;
}

struct Row {
    chess::SearchResult result;
    std::uint64_t us = 0;
};

template <typename Searcher>
Row run_one(Searcher& searcher, const chess::Position& pos, int depth) {
    searcher.clear_tt();
    const auto start = std::chrono::steady_clock::now();
    chess::SearchResult result = searcher.search_best_move(pos, depth);
    const auto end = std::chrono::steady_clock::now();
    return Row{
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())
    };
}

template <typename Searcher>
Row run_one(Searcher& searcher, const chess::Position& pos, int depth, bool iterative) {
    searcher.clear_tt();
    const auto start = std::chrono::steady_clock::now();
    chess::SearchResult result = iterative
        ? searcher.search_best_move(pos, chess::SearchLimits{depth, std::chrono::milliseconds{0}})
        : searcher.search_best_move(pos, depth);
    const auto end = std::chrono::steady_clock::now();
    return Row{
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())
    };
}

std::vector<chess::Position> make_positions(int random_positions, int max_random_plies) {
    std::vector<chess::Position> positions;
    auto add_fen = [&](std::string_view fen) {
        chess::Position pos;
        if (pos.set_fen(fen) && has_both_kings(pos)) {
            positions.push_back(pos);
        }
    };

    chess::Position start;
    start.set_startpos();
    positions.push_back(start);
    add_fen("rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4");
    add_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    add_fen("r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9");
    add_fen("2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12");
    add_fen("4r3/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23");

    std::mt19937 rng(20260619);
    chess::Position pos;
    pos.set_startpos();
    for (int i = 0; i < random_positions; ++i) {
        positions.push_back(pos);
        const int plies = 1 + static_cast<int>(rng() % static_cast<unsigned>(max_random_plies));
        for (int ply = 0; ply < plies; ++ply) {
            chess::MoveList moves;
            chess::generate_legal_moves(pos, moves);
            if (moves.empty()) {
                pos.set_startpos();
                break;
            }
            std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
            pos.make_move(moves[dist(rng)]);
        }
    }
    return positions;
}

} // namespace

int main(int argc, char** argv) {
    int depth = 6;
    int random_positions = 20;
    bool iterative = false;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }
    for (int i = 3; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--iterative") {
            iterative = true;
        }
    }

    const std::vector<chess::Position> positions = make_positions(random_positions, 6);
    std::uint64_t v35_us = 0;
    std::uint64_t v32_us = 0;
    std::uint64_t v35_nodes = 0;
    std::uint64_t v32_nodes = 0;
    int score_mismatches = 0;
    int move_mismatches = 0;

    for (std::size_t i = 0; i < positions.size(); ++i) {
        chess::HeuristicSearcherV32 v32;
        chess::HeuristicSearcherV35 v35;
        const Row r34 = run_one(v35, positions[i], depth, iterative);
        const Row r32 = run_one(v32, positions[i], depth, iterative);

        v35_us += r34.us;
        v32_us += r32.us;
        v35_nodes += r34.result.nodes;
        v32_nodes += r32.result.nodes;
        if (r34.result.score != r32.result.score) {
            ++score_mismatches;
        }
        if (r34.result.best_move != r32.result.best_move) {
            ++move_mismatches;
        }

        std::cout << "sample=" << i
                  << " v35_score=" << r34.result.score
                  << " v32_score=" << r32.result.score
                  << " v35_best=" << chess::move_to_string(r34.result.best_move)
                  << " v32_best=" << chess::move_to_string(r32.result.best_move)
                  << " v35_nodes=" << r34.result.nodes
                  << " v32_nodes=" << r32.result.nodes
                  << " v35_us=" << r34.us
                  << " v32_us=" << r32.us
                  << '\n';
    }

    std::cout << "summary"
              << " samples=" << positions.size()
              << " depth=" << depth
              << " iterative=" << (iterative ? 1 : 0)
              << " v35_nodes=" << v35_nodes
              << " v32_nodes=" << v32_nodes
              << " v35_us=" << v35_us
              << " v32_us=" << v32_us
              << " v35_vs_v32_time=" << (v32_us == 0 ? 0.0 : static_cast<double>(v35_us) / v32_us)
              << " v32_vs_v35_time=" << (v35_us == 0 ? 0.0 : static_cast<double>(v32_us) / v35_us)
              << " score_mismatches=" << score_mismatches
              << " move_mismatches=" << move_mismatches
              << '\n';
}
