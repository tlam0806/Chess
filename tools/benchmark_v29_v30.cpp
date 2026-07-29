#include "heuristic_searcher_v29.hpp"
#include "heuristic_searcher_v30.hpp"
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
    int qsearch_capture_metric_weight = 0;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        qsearch_capture_metric_weight = std::stoi(argv[3]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions, 6);
    std::uint64_t v29_us = 0;
    std::uint64_t v30_us = 0;
    std::uint64_t v29_nodes = 0;
    std::uint64_t v30_nodes = 0;
    int score_mismatches = 0;
    int move_mismatches = 0;
    chess::HeuristicSearcherV30::MoveOrderingWeights weights{};
    weights.qsearch_capture_metric_weight = qsearch_capture_metric_weight;

    for (std::size_t i = 0; i < positions.size(); ++i) {
        chess::HeuristicSearcherV29 v29;
        chess::HeuristicSearcherV30 v30(
            64,
            4,
            10,
            14,
            14'000,
            weights);
        const Row r29 = run_one(v29, positions[i], depth);
        const Row r30 = run_one(v30, positions[i], depth);

        v29_us += r29.us;
        v30_us += r30.us;
        v29_nodes += r29.result.nodes;
        v30_nodes += r30.result.nodes;
        if (r29.result.score != r30.result.score) {
            ++score_mismatches;
        }
        if (r29.result.best_move != r30.result.best_move) {
            ++move_mismatches;
        }

        std::cout << "sample=" << i
                  << " v29_score=" << r29.result.score
                  << " v30_score=" << r30.result.score
                  << " v29_best=" << chess::move_to_string(r29.result.best_move)
                  << " v30_best=" << chess::move_to_string(r30.result.best_move)
                  << " v29_nodes=" << r29.result.nodes
                  << " v30_nodes=" << r30.result.nodes
                  << " v29_us=" << r29.us
                  << " v30_us=" << r30.us
                  << '\n';
    }

    std::cout << "summary"
              << " samples=" << positions.size()
              << " depth=" << depth
              << " qsearch_capture_metric_weight=" << qsearch_capture_metric_weight
              << " v29_nodes=" << v29_nodes
              << " v30_nodes=" << v30_nodes
              << " v29_us=" << v29_us
              << " v30_us=" << v30_us
              << " v29_vs_v30_time=" << (v30_us == 0 ? 0.0 : static_cast<double>(v29_us) / v30_us)
              << " score_mismatches=" << score_mismatches
              << " move_mismatches=" << move_mismatches
              << '\n';
}
