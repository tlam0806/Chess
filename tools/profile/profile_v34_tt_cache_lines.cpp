#include "heuristic_searcher_v34.hpp"
#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#ifndef CHESS_PROFILE_TT_CACHE_LINES
#error "Build this tool with CHESS_PROFILE_TT_CACHE_LINES"
#endif

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

double avg(std::uint64_t total, std::uint64_t count) {
    return count == 0 ? 0.0 : static_cast<double>(total) / static_cast<double>(count);
}

double pct(std::uint64_t part, std::uint64_t total) {
    return total == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(total);
}

} // namespace

int main(int argc, char** argv) {
    int depth = 9;
    int random_positions = 0;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions);
    chess::HeuristicSearcherV34 searcher;
    searcher.clear_tt_cache_line_stats();

    std::uint64_t nodes = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const chess::Position& pos : positions) {
        searcher.clear_tt();
        const chess::SearchResult result =
            searcher.search_best_move(pos, chess::SearchLimits{depth, std::chrono::milliseconds{0}});
        nodes += result.nodes;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const auto& stats = searcher.tt_cache_line_stats();

    std::cout << "version=v34"
              << " depth=" << depth
              << " positions=" << positions.size()
              << " nodes=" << nodes
              << " elapsed_us=" << elapsed_us
              << " probes=" << stats.probes
              << " cache_line_size=128"
              << '\n';
    std::cout << "total"
              << " avg_total_lines=" << avg(stats.total_lines, stats.probes)
              << " avg_key_lines=" << avg(stats.key_lines, stats.probes)
              << " avg_value_lines=" << avg(stats.value_lines, stats.probes)
              << " avg_key_slots_read=" << avg(stats.key_slots_read, stats.probes)
              << '\n';
    std::cout << "case=empty_miss"
              << " count=" << stats.empty_misses
              << " pct=" << pct(stats.empty_misses, stats.probes)
              << " avg_total_lines=" << avg(stats.empty_total_lines, stats.empty_misses)
              << " avg_key_slots_read=" << avg(stats.empty_key_slots_read, stats.empty_misses)
              << '\n';
    std::cout << "case=index_collision"
              << " count=" << stats.index_collisions
              << " pct=" << pct(stats.index_collisions, stats.probes)
              << " avg_total_lines=" << avg(stats.index_collision_total_lines, stats.index_collisions)
              << " avg_key_slots_read="
              << avg(stats.index_collision_key_slots_read, stats.index_collisions)
              << '\n';
    std::cout << "case=key_hit"
              << " count=" << stats.key_hits
              << " pct=" << pct(stats.key_hits, stats.probes)
              << " avg_total_lines=" << avg(stats.hit_total_lines, stats.key_hits)
              << " avg_key_slots_read=" << avg(stats.hit_key_slots_read, stats.key_hits)
              << " value_read_pct=" << pct(stats.value_reads, stats.key_hits)
              << '\n';
}
