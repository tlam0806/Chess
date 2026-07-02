#include "searchers/strict/heuristic_searcher_v30.hpp"
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
    const auto king = static_cast<int>(chess::PieceType::King);
    return chess::popcount(pos.pieces[static_cast<int>(chess::Color::White)][king]) == 1
        && chess::popcount(pos.pieces[static_cast<int>(chess::Color::Black)][king]) == 1;
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

    std::mt19937 rng(20260701);
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
    int depth = 8;
    int seconds = 30;
    int random_positions = 16;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        seconds = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        random_positions = std::stoi(argv[3]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions, 8);
    chess::HeuristicSearcherV30 searcher;
    std::uint64_t searches = 0;
    std::uint64_t nodes = 0;
    int score_accumulator = 0;

    for (const chess::Position& pos : positions) {
        searcher.clear_tt();
        const chess::SearchResult result = searcher.search_best_move(pos, std::min(depth, 5));
        score_accumulator += result.score;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::seconds(seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        for (const chess::Position& pos : positions) {
            searcher.clear_tt();
            const chess::SearchResult result = searcher.search_best_move(pos, depth);
            nodes += result.nodes;
            score_accumulator += result.score;
            ++searches;
            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "depth=" << depth
              << " seconds=" << seconds
              << " positions=" << positions.size()
              << " searches=" << searches
              << " nodes=" << nodes
              << " elapsed_us=" << elapsed_us
              << " nps=" << (elapsed_us == 0 ? 0.0 : static_cast<double>(nodes) * 1'000'000.0 / elapsed_us)
              << " score_accumulator=" << score_accumulator
              << '\n';
}
