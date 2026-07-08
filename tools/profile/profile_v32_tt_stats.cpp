#include "heuristic_searcher_v32.hpp"
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

double pct(std::uint64_t part, std::uint64_t total) {
    if (total == 0) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(part) / static_cast<double>(total);
}

std::size_t floor_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result <= value / 2) {
        result *= 2;
    }
    return result;
}

std::size_t bucket_count_for(std::size_t tt_mb, std::size_t bucket_size) {
    const std::size_t bytes = tt_mb * 1024 * 1024;
    const std::size_t total_entries = bytes / sizeof(chess::RangeTTEntry);
    const std::size_t bucket_count_input = total_entries / bucket_size;
    return floor_power_of_two(bucket_count_input == 0 ? 1 : bucket_count_input);
}

} // namespace

int main(int argc, char** argv) {
    int depth = 10;
    int random_positions = 2;
    int bucket_size = 4;
    if (argc >= 2) {
        depth = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        random_positions = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        bucket_size = std::stoi(argv[3]);
    }

    const std::vector<chess::Position> positions = make_positions(random_positions, 6);
    chess::HeuristicSearcherV32 searcher(64, static_cast<std::size_t>(bucket_size));
    searcher.clear_tt();
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
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    const chess::RangeTranspositionTableStats& stats = searcher.tt_stats();
    const std::uint64_t return_hits = stats.exact_hits + stats.score_returns;
    const std::uint64_t key_misses = stats.empty_misses + stats.index_collisions;
    const std::uint64_t probe_false = stats.probes - return_hits;
    const std::uint64_t key_hits_without_return =
        stats.key_hits > return_hits ? stats.key_hits - return_hits : 0;
    const std::uint64_t successful_stores =
        stats.new_stores + stats.same_key_updates + stats.replacement_collisions;
    const std::uint64_t bucket_full_store_attempts =
        stats.replacement_collisions + stats.skipped_shallow_replacements;
    constexpr std::size_t tt_mb = 64;
    const std::size_t bucket_count = bucket_count_for(tt_mb, static_cast<std::size_t>(bucket_size));
    const std::size_t entry_count = bucket_count * static_cast<std::size_t>(bucket_size);
    const std::uint64_t total_entry_slots_seen =
        static_cast<std::uint64_t>(entry_count) * static_cast<std::uint64_t>(positions.size());
    const double avg_occupied_per_bucket = bucket_count == 0 || positions.empty()
        ? 0.0
        : static_cast<double>(stats.new_stores)
            / (static_cast<double>(bucket_count) * static_cast<double>(positions.size()));

    std::cout << "version=v32"
              << " depth=" << depth
              << " bucket_size=" << bucket_size
              << " iterative=1"
              << " positions=" << positions.size()
              << " nodes=" << nodes
              << " elapsed_us=" << elapsed_us
              << " nps=" << (elapsed_us == 0 ? 0.0 : static_cast<double>(nodes) * 1'000'000.0 / elapsed_us)
              << " score_accumulator=" << score_accumulator
              << " move_accumulator=" << move_accumulator
              << '\n';

    std::cout << "tt"
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

    std::cout << "tt_return"
              << " return_hits=" << return_hits
              << " return_hit_pct=" << pct(return_hits, stats.probes)
              << " probe_false=" << probe_false
              << " probe_false_pct=" << pct(probe_false, stats.probes)
              << " exact_hits=" << stats.exact_hits
              << " exact_hit_pct=" << pct(stats.exact_hits, stats.probes)
              << " score_returns=" << stats.score_returns
              << " score_return_pct=" << pct(stats.score_returns, stats.probes)
              << " key_hits_without_return=" << key_hits_without_return
              << " key_hits_without_return_pct=" << pct(key_hits_without_return, stats.probes)
              << " move_hint_hits=" << stats.move_hint_hits
              << " move_hint_pct=" << pct(stats.move_hint_hits, stats.probes)
              << " lower_score_hits=" << stats.lower_score_hits
              << " upper_score_hits=" << stats.upper_score_hits
              << " window_narrowings=" << stats.window_narrowings
              << '\n';

    std::cout << "tt_store"
              << " stores=" << stats.stores
              << " bucket_count=" << bucket_count
              << " entry_count=" << entry_count
              << " avg_occupied_per_bucket=" << avg_occupied_per_bucket
              << " occupied_entry_pct=" << pct(stats.new_stores, total_entry_slots_seen)
              << " successful_stores=" << successful_stores
              << " successful_store_pct=" << pct(successful_stores, stats.stores)
              << " failed_bucket_full=" << stats.skipped_shallow_replacements
              << " failed_bucket_full_pct=" << pct(stats.skipped_shallow_replacements, stats.stores)
              << " bucket_full_store_attempts=" << bucket_full_store_attempts
              << " bucket_full_store_attempt_pct=" << pct(bucket_full_store_attempts, stats.stores)
              << " new_stores=" << stats.new_stores
              << " same_key_updates=" << stats.same_key_updates
              << " replacement_collisions=" << stats.replacement_collisions
              << " replacement_collision_pct=" << pct(stats.replacement_collisions, stats.stores)
              << " skipped_shallow_replacements=" << stats.skipped_shallow_replacements
              << '\n';
}
