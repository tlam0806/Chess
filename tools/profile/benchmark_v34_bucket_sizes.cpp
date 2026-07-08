#include "heuristic_searcher_v34.hpp"
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
    const int king = static_cast<int>(chess::PieceType::King);
    return chess::popcount(pos.pieces[static_cast<int>(chess::Color::White)][king]) == 1
        && chess::popcount(pos.pieces[static_cast<int>(chess::Color::Black)][king]) == 1;
}

std::vector<chess::Position> make_positions(int random_positions) {
    std::vector<chess::Position> positions;
    auto add_fen = [&](std::string_view fen) {
        chess::Position pos;
        if (pos.set_fen(fen) && has_both_kings(pos)) {
            positions.push_back(pos);
        }
    };

    add_fen("rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4");
    add_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    add_fen("r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9");
    add_fen("2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12");
    add_fen("4r3/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23");

    std::mt19937 rng(20260619);
    chess::Position pos;
    pos.set_startpos();
    for (int i = 0; i < random_positions; ++i) {
        const int plies = 1 + static_cast<int>(rng() % 6u);
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
        positions.push_back(pos);
    }
    return positions;
}

struct Result {
    std::uint64_t nodes = 0;
    std::uint64_t elapsed_us = 0;
    int score_accumulator = 0;
    std::uint64_t move_accumulator = 0;
    std::size_t entry_count = 0;
};

Result run_bucket(std::size_t bucket_size, int depth, const std::vector<chess::Position>& positions) {
    chess::HeuristicSearcherV34 searcher(64, bucket_size);
    Result total;
    total.entry_count = searcher.tt_entry_count();

    const auto start = std::chrono::steady_clock::now();
    for (const chess::Position& pos : positions) {
        searcher.clear_tt();
        const chess::SearchResult result =
            searcher.search_best_move(pos, chess::SearchLimits{depth, std::chrono::milliseconds{0}});
        total.nodes += result.nodes;
        total.score_accumulator += result.score;
        total.move_accumulator += result.best_move.value;
    }
    const auto end = std::chrono::steady_clock::now();
    total.elapsed_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return total;
}

} // namespace

int main(int argc, char** argv) {
    int depth = 10;
    int random_positions = 1;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions);
    const std::vector<std::size_t> bucket_sizes{1, 2, 4, 8, 16};

    std::uint64_t best_us = 0;
    std::size_t best_bucket = 0;
    for (std::size_t bucket_size : bucket_sizes) {
        const Result result = run_bucket(bucket_size, depth, positions);
        if (best_us == 0 || result.elapsed_us < best_us) {
            best_us = result.elapsed_us;
            best_bucket = bucket_size;
        }
        std::cout << "bucket_size=" << bucket_size
                  << " depth=" << depth
                  << " positions=" << positions.size()
                  << " entry_count=" << result.entry_count
                  << " nodes=" << result.nodes
                  << " elapsed_us=" << result.elapsed_us
                  << " nps=" << (result.elapsed_us == 0
                         ? 0.0
                         : static_cast<double>(result.nodes) * 1'000'000.0 / result.elapsed_us)
                  << " score_accumulator=" << result.score_accumulator
                  << " move_accumulator=" << result.move_accumulator
                  << '\n';
    }
    std::cout << "best_bucket_size=" << best_bucket
              << " best_elapsed_us=" << best_us
              << '\n';
}
