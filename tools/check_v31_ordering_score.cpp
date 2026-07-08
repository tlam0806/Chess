#include "heuristic_searcher_v31.hpp"
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

chess::SearchResult run(
    const chess::Position& pos,
    int depth,
    chess::HeuristicSearcherV31::MoveOrderingWeights weights
) {
    chess::HeuristicSearcherV31 searcher(64, 4, 10, 14, weights.counter_history_bonus, weights);
    return searcher.search_best_move(pos, depth);
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

    chess::HeuristicSearcherV31::MoveOrderingWeights normal{};
    chess::HeuristicSearcherV31::MoveOrderingWeights changed = normal;
    changed.killer1_bonus = 250'000;
    changed.killer2_bonus = 125'000;
    changed.check_bonus = 180'000;
    changed.counter_history_bonus = 80'000;
    changed.good_capture_bonus = 5'000;
    changed.bad_capture_bonus = 1'000;
    changed.see_weight = 20;

    const std::vector<chess::Position> positions = make_positions(random_positions, 6);
    int score_mismatches = 0;
    int move_mismatches = 0;
    std::uint64_t normal_nodes = 0;
    std::uint64_t changed_nodes = 0;

    for (std::size_t i = 0; i < positions.size(); ++i) {
        const chess::SearchResult a = run(positions[i], depth, normal);
        const chess::SearchResult b = run(positions[i], depth, changed);
        normal_nodes += a.nodes;
        changed_nodes += b.nodes;
        if (a.score != b.score) {
            ++score_mismatches;
        }
        if (a.best_move != b.best_move) {
            ++move_mismatches;
        }
        std::cout << "sample=" << i
                  << " normal_score=" << a.score
                  << " changed_score=" << b.score
                  << " normal_best=" << chess::move_to_string(a.best_move)
                  << " changed_best=" << chess::move_to_string(b.best_move)
                  << " normal_nodes=" << a.nodes
                  << " changed_nodes=" << b.nodes
                  << '\n';
    }

    std::cout << "summary"
              << " samples=" << positions.size()
              << " depth=" << depth
              << " normal_nodes=" << normal_nodes
              << " changed_nodes=" << changed_nodes
              << " score_mismatches=" << score_mismatches
              << " move_mismatches=" << move_mismatches
              << '\n';
}
