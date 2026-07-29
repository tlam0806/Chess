#include "heuristic_searcher_v25.hpp"
#include "heuristic_searcher_v26.hpp"
#include "heuristic_searcher_v27.hpp"
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
    const char* name = "";
    chess::SearchResult result;
    std::uint64_t us = 0;
};

template <typename Searcher>
Row run_one(const char* name, Searcher& searcher, const chess::Position& pos, int depth) {
    searcher.clear_tt();
    const auto start = std::chrono::steady_clock::now();
    chess::SearchResult result = searcher.search_best_move(pos, depth);
    const auto end = std::chrono::steady_clock::now();
    return Row{
        name,
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
    add_fen("4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23");

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
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions, 6);
    std::uint64_t v25_us = 0;
    std::uint64_t v26_us = 0;
    std::uint64_t v27_us = 0;
    std::uint64_t v25_nodes = 0;
    std::uint64_t v26_nodes = 0;
    std::uint64_t v27_nodes = 0;
    int score_mismatches = 0;
    int move_mismatches = 0;

    for (std::size_t i = 0; i < positions.size(); ++i) {
        chess::HeuristicSearcherV25 v25;
        chess::HeuristicSearcherV26 v26;
        chess::HeuristicSearcherV27 v27;
        const Row r25 = run_one("v25", v25, positions[i], depth);
        const Row r26 = run_one("v26", v26, positions[i], depth);
        const Row r27 = run_one("v27", v27, positions[i], depth);

        v25_us += r25.us;
        v26_us += r26.us;
        v27_us += r27.us;
        v25_nodes += r25.result.nodes;
        v26_nodes += r26.result.nodes;
        v27_nodes += r27.result.nodes;
        if (r25.result.score != r26.result.score || r25.result.score != r27.result.score) {
            ++score_mismatches;
        }
        if (r25.result.best_move != r26.result.best_move || r25.result.best_move != r27.result.best_move) {
            ++move_mismatches;
        }

        std::cout << "sample=" << i
                  << " v25_score=" << r25.result.score
                  << " v26_score=" << r26.result.score
                  << " v27_score=" << r27.result.score
                  << " v25_best=" << chess::move_to_string(r25.result.best_move)
                  << " v26_best=" << chess::move_to_string(r26.result.best_move)
                  << " v27_best=" << chess::move_to_string(r27.result.best_move)
                  << " v25_nodes=" << r25.result.nodes
                  << " v26_nodes=" << r26.result.nodes
                  << " v27_nodes=" << r27.result.nodes
                  << " v25_us=" << r25.us
                  << " v26_us=" << r26.us
                  << " v27_us=" << r27.us
                  << '\n';
    }

    std::cout << "summary"
              << " samples=" << positions.size()
              << " depth=" << depth
              << " v25_nodes=" << v25_nodes
              << " v26_nodes=" << v26_nodes
              << " v27_nodes=" << v27_nodes
              << " v25_us=" << v25_us
              << " v26_us=" << v26_us
              << " v27_us=" << v27_us
              << " v25_vs_v26_time=" << (v26_us == 0 ? 0.0 : static_cast<double>(v25_us) / v26_us)
              << " v25_vs_v27_time=" << (v27_us == 0 ? 0.0 : static_cast<double>(v25_us) / v27_us)
              << " v26_vs_v27_time=" << (v27_us == 0 ? 0.0 : static_cast<double>(v26_us) / v27_us)
              << " score_mismatches=" << score_mismatches
              << " move_mismatches=" << move_mismatches
              << '\n';
}
