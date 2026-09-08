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

struct RunResult {
    std::uint64_t nodes = 0;
    std::uint64_t elapsed_us = 0;
    int score_accumulator = 0;
    std::uint64_t move_accumulator = 0;
    std::size_t entry_count = 0;
};

template <typename Searcher>
RunResult run(Searcher& searcher, int depth, const std::vector<chess::Position>& positions) {
    RunResult result;
    result.entry_count = searcher.tt_entry_count();
    const auto start = std::chrono::steady_clock::now();
    for (const chess::Position& pos : positions) {
        searcher.clear_tt();
        const chess::SearchResult search_result =
            searcher.search_best_move(pos, chess::SearchLimits{depth, std::chrono::milliseconds{0}});
        result.nodes += search_result.nodes;
        result.score_accumulator += search_result.score;
        result.move_accumulator += search_result.best_move.value;
    }
    const auto end = std::chrono::steady_clock::now();
    result.elapsed_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return result;
}

void print_result(
    std::string_view version,
    int tt_mb,
    int depth,
    std::size_t positions,
    const RunResult& result,
    const RunResult* baseline
) {
    std::cout << "version=" << version
              << " tt_mb=" << tt_mb
              << " depth=" << depth
              << " positions=" << positions
              << " entry_count=" << result.entry_count
              << " nodes=" << result.nodes
              << " elapsed_us=" << result.elapsed_us
              << " nps=" << (result.elapsed_us == 0
                     ? 0.0
                     : static_cast<double>(result.nodes) * 1'000'000.0 / result.elapsed_us)
              << " score_accumulator=" << result.score_accumulator
              << " move_accumulator=" << result.move_accumulator;
    if (baseline != nullptr) {
        std::cout << " time_vs_v32_64="
                  << (baseline->elapsed_us == 0
                         ? 0.0
                         : static_cast<double>(result.elapsed_us) / baseline->elapsed_us)
                  << " nodes_vs_v32_64="
                  << (baseline->nodes == 0 ? 0.0 : static_cast<double>(result.nodes) / baseline->nodes)
                  << " score_match=" << (result.score_accumulator == baseline->score_accumulator)
                  << " move_accumulator_match="
                  << (result.move_accumulator == baseline->move_accumulator);
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    int depth = 10;
    int random_positions = 0;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions);
    chess::HeuristicSearcherV32 v32(64, 4);
    const RunResult baseline = run(v32, depth, positions);
    print_result("v32", 64, depth, positions.size(), baseline, nullptr);

    for (int tt_mb : {48, 56, 64, 72}) {
        chess::HeuristicSearcherV35 v35(static_cast<std::size_t>(tt_mb), 4);
        const RunResult result = run(v35, depth, positions);
        print_result("v35", tt_mb, depth, positions.size(), result, &baseline);
    }
}
