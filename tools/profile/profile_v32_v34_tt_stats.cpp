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

    add_fen("rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4");
    add_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    add_fen("r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9");
    add_fen("2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12");
    add_fen("4r3/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23");

    std::mt19937 rng(20260619);
    chess::Position pos;
    pos.set_startpos();
    for (int i = 0; i < random_positions; ++i) {
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
        positions.push_back(pos);
    }
    return positions;
}

double pct(std::uint64_t part, std::uint64_t total) {
    return total == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(total);
}

template <typename Searcher>
void run_version(std::string_view label, int depth, const std::vector<chess::Position>& positions) {
    Searcher searcher;
    std::uint64_t nodes = 0;
    int score_accumulator = 0;
    std::uint64_t move_accumulator = 0;
    searcher.clear_tt_stats();

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
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const chess::RangeTranspositionTableStats& stats = searcher.tt_stats();
    const std::uint64_t return_hits = stats.exact_hits + stats.score_returns;
    const std::uint64_t key_misses = stats.empty_misses + stats.index_collisions;
    const std::uint64_t successful_stores =
        stats.new_stores + stats.same_key_updates + stats.replacement_collisions;

    std::cout << "version=" << label
              << " depth=" << depth
              << " positions=" << positions.size()
              << " nodes=" << nodes
              << " elapsed_us=" << elapsed_us
              << " nps=" << (elapsed_us == 0 ? 0.0 : static_cast<double>(nodes) * 1'000'000.0 / elapsed_us)
              << " score_accumulator=" << score_accumulator
              << " move_accumulator=" << move_accumulator
              << '\n';

    std::cout << "tt version=" << label
              << " probes=" << stats.probes
              << " key_hits=" << stats.key_hits
              << " key_hit_pct=" << pct(stats.key_hits, stats.probes)
              << " key_misses=" << key_misses
              << " key_miss_pct=" << pct(key_misses, stats.probes)
              << " empty_misses=" << stats.empty_misses
              << " empty_miss_pct=" << pct(stats.empty_misses, stats.probes)
              << " index_collisions=" << stats.index_collisions
              << " index_collision_pct=" << pct(stats.index_collisions, stats.probes)
              << " depth_misses=" << stats.depth_misses
              << " depth_miss_pct=" << pct(stats.depth_misses, stats.probes)
              << '\n';

    std::cout << "tt_return version=" << label
              << " return_hits=" << return_hits
              << " return_hit_pct=" << pct(return_hits, stats.probes)
              << " exact_hits=" << stats.exact_hits
              << " exact_hit_pct=" << pct(stats.exact_hits, stats.probes)
              << " score_returns=" << stats.score_returns
              << " score_return_pct=" << pct(stats.score_returns, stats.probes)
              << " move_hint_hits=" << stats.move_hint_hits
              << " move_hint_pct=" << pct(stats.move_hint_hits, stats.probes)
              << " lower_score_hits=" << stats.lower_score_hits
              << " upper_score_hits=" << stats.upper_score_hits
              << " window_narrowings=" << stats.window_narrowings
              << '\n';

    std::cout << "tt_store version=" << label
              << " stores=" << stats.stores
              << " successful_stores=" << successful_stores
              << " successful_store_pct=" << pct(successful_stores, stats.stores)
              << " new_stores=" << stats.new_stores
              << " same_key_updates=" << stats.same_key_updates
              << " replacement_collisions=" << stats.replacement_collisions
              << " replacement_collision_pct=" << pct(stats.replacement_collisions, stats.stores)
              << " skipped_shallow_replacements=" << stats.skipped_shallow_replacements
              << " skipped_shallow_replacement_pct=" << pct(stats.skipped_shallow_replacements, stats.stores)
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    int depth = 9;
    int random_positions = 2;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions, 6);
    run_version<chess::HeuristicSearcherV32>("v32", depth, positions);
    run_version<chess::HeuristicSearcherV34>("v34", depth, positions);
}
