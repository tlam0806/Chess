#include "heuristic_searcher_v32.hpp"
#include "heuristic_searcher_v34.hpp"
#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#ifndef CHESS_PROFILE_TT_PATH_TIMING
#error "Build this tool with CHESS_PROFILE_TT_PATH_TIMING"
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

double ns_per_call(std::uint64_t ns, std::uint64_t calls) {
    return calls == 0 ? 0.0 : static_cast<double>(ns) / static_cast<double>(calls);
}

double pct(std::uint64_t part, std::uint64_t total) {
    return total == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(total);
}

void print_path(
    std::string_view version,
    std::string_view path,
    std::uint64_t calls,
    std::uint64_t ns,
    std::uint64_t total_probe_ns
) {
    std::cout << "tt_probe_path version=" << version
              << " path=" << path
              << " calls=" << calls
              << " total_ns=" << ns
              << " ns_per_call=" << ns_per_call(ns, calls)
              << " pct_probe_time=" << pct(ns, total_probe_ns)
              << '\n';
}

template <typename Searcher>
void run_version(
    std::string_view version,
    int depth,
    const std::vector<chess::Position>& positions
) {
    Searcher searcher;
    searcher.clear_tt_probe_path_timing_stats();
    searcher.clear_tt_stats();

    std::uint64_t nodes = 0;
    int score_accumulator = 0;
    std::uint64_t move_accumulator = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const chess::Position& pos : positions) {
        searcher.clear_tt();
        const chess::SearchResult result =
            searcher.search_best_move(pos, chess::SearchLimits{depth, std::chrono::milliseconds{0}});
        nodes += result.nodes;
        score_accumulator += result.score;
        move_accumulator += result.best_move.value;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    const chess::RangeTranspositionTableStats& stats = searcher.tt_stats();
    const chess::TTProbePathTimingStats& paths = searcher.tt_probe_path_timing_stats();
    const std::uint64_t total_probe_ns =
        paths.empty_miss_ns + paths.index_collision_ns + paths.key_hit_total_ns;
    const std::uint64_t key_scan_dispatch_ns =
        paths.key_hit_total_ns >= paths.probe_hit_body_ns
        ? paths.key_hit_total_ns - paths.probe_hit_body_ns
        : 0;

    std::cout << "version=" << version
              << " depth=" << depth
              << " positions=" << positions.size()
              << " nodes=" << nodes
              << " elapsed_us=" << elapsed_us
              << " nps=" << (elapsed_us == 0 ? 0.0 : static_cast<double>(nodes) * 1'000'000.0 / elapsed_us)
              << " score_accumulator=" << score_accumulator
              << " move_accumulator=" << move_accumulator
              << '\n';
    std::cout << "tt_probe_summary version=" << version
              << " probes=" << stats.probes
              << " empty_misses=" << stats.empty_misses
              << " index_collisions=" << stats.index_collisions
              << " key_hits=" << stats.key_hits
              << " path_calls="
              << (paths.empty_miss_calls + paths.index_collision_calls + paths.key_hit_calls)
              << " total_probe_path_ns=" << total_probe_ns
              << " ns_per_probe=" << ns_per_call(total_probe_ns, stats.probes)
              << '\n';
    print_path(version, "empty_miss", paths.empty_miss_calls, paths.empty_miss_ns, total_probe_ns);
    print_path(
        version,
        "index_collision",
        paths.index_collision_calls,
        paths.index_collision_ns,
        total_probe_ns);
    print_path(version, "key_hit_total", paths.key_hit_calls, paths.key_hit_total_ns, total_probe_ns);
    print_path(
        version,
        "probe_hit_body",
        paths.probe_hit_body_calls,
        paths.probe_hit_body_ns,
        total_probe_ns);
    print_path(version, "key_hit_scan_dispatch", paths.key_hit_calls, key_scan_dispatch_ns, total_probe_ns);
}

} // namespace

int main(int argc, char** argv) {
    int depth = 8;
    int random_positions = 0;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions);
    run_version<chess::HeuristicSearcherV32>("v32", depth, positions);
    run_version<chess::HeuristicSearcherV34>("v34", depth, positions);
}
