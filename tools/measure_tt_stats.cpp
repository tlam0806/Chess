#include "attacks.hpp"
#include "heuristic_searcher_v9.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search_types.hpp"
#include "transposition_table.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    int depth = 7;
    int plies = 80;
    int tt_mb = 64;
    std::uint32_t seed = 20260613;
    bool iterative = false;
    bool clear_between_moves = false;
    bool per_ply = false;
};

int parse_int(std::string_view value, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const int result = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + std::string(name) + ": " + std::string(value));
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };

        if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--plies") {
            options.plies = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-mb") {
            options.tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--iterative") {
            options.iterative = true;
        } else if (arg == "--clear-between-moves") {
            options.clear_between_moves = true;
        } else if (arg == "--per-ply") {
            options.per_ply = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: measure_tt_stats [--depth N] [--plies N] [--tt-mb N]\n"
                << "                        [--seed N] [--iterative]\n"
                << "                        [--clear-between-moves] [--per-ply]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.depth < 0) {
        throw std::runtime_error("--depth must be non-negative");
    }
    if (options.plies <= 0) {
        throw std::runtime_error("--plies must be positive");
    }
    if (options.tt_mb <= 0) {
        throw std::runtime_error("--tt-mb must be positive");
    }
    return options;
}

chess::Move random_move(const std::vector<chess::Move>& moves, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

chess::SearchResult search(
    chess::HeuristicSearcherV9& searcher,
    const chess::Position& pos,
    const Options& options
) {
    if (options.iterative) {
        return searcher.search_best_move(pos, chess::SearchLimits{
            .max_depth = options.depth,
            .move_time = std::chrono::milliseconds{0}
        });
    }
    return searcher.search_best_move(pos, options.depth);
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        return 0.0;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

chess::TranspositionTableStats diff_stats(
    const chess::TranspositionTableStats& after,
    const chess::TranspositionTableStats& before
) {
    return chess::TranspositionTableStats{
        .probes = after.probes - before.probes,
        .empty_misses = after.empty_misses - before.empty_misses,
        .index_collisions = after.index_collisions - before.index_collisions,
        .key_hits = after.key_hits - before.key_hits,
        .move_hint_hits = after.move_hint_hits - before.move_hint_hits,
        .depth_misses = after.depth_misses - before.depth_misses,
        .exact_hits = after.exact_hits - before.exact_hits,
        .lower_bound_hits = after.lower_bound_hits - before.lower_bound_hits,
        .upper_bound_hits = after.upper_bound_hits - before.upper_bound_hits,
        .bound_tightens = after.bound_tightens - before.bound_tightens,
        .bound_cutoffs = after.bound_cutoffs - before.bound_cutoffs,
        .stores = after.stores - before.stores,
        .new_stores = after.new_stores - before.new_stores,
        .same_key_updates = after.same_key_updates - before.same_key_updates,
        .replacement_collisions = after.replacement_collisions - before.replacement_collisions,
        .skipped_shallow_replacements =
            after.skipped_shallow_replacements - before.skipped_shallow_replacements
    };
}

std::uint64_t score_hits(const chess::TranspositionTableStats& stats) {
    return stats.exact_hits + stats.bound_cutoffs;
}

void print_stats(std::string_view prefix, const chess::TranspositionTableStats& stats) {
    std::cout << prefix
              << " probes=" << stats.probes
              << " key_hits=" << stats.key_hits
              << " key_hit_rate=" << ratio(stats.key_hits, stats.probes)
              << " score_hits=" << score_hits(stats)
              << " score_hit_rate=" << ratio(score_hits(stats), stats.probes)
              << " move_hint_hits=" << stats.move_hint_hits
              << " move_hint_rate=" << ratio(stats.move_hint_hits, stats.probes)
              << " empty_misses=" << stats.empty_misses
              << " index_collisions=" << stats.index_collisions
              << " probe_collision_rate=" << ratio(stats.index_collisions, stats.probes)
              << " depth_misses=" << stats.depth_misses
              << " exact_hits=" << stats.exact_hits
              << " lower_hits=" << stats.lower_bound_hits
              << " upper_hits=" << stats.upper_bound_hits
              << " bound_tightens=" << stats.bound_tightens
              << " bound_cutoffs=" << stats.bound_cutoffs
              << " stores=" << stats.stores
              << " new_stores=" << stats.new_stores
              << " same_key_updates=" << stats.same_key_updates
              << " replacement_collisions=" << stats.replacement_collisions
              << " store_collision_rate=" << ratio(stats.replacement_collisions, stats.stores)
              << " skipped_shallow_replacements=" << stats.skipped_shallow_replacements
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        chess::Position pos;
        pos.set_startpos();

        chess::HeuristicSearcherV9 searcher(static_cast<std::size_t>(options.tt_mb));
        searcher.clear_tt();
        searcher.clear_tt_stats();

        std::mt19937 rng(options.seed);
        std::uint64_t nodes = 0;
        int searched_plies = 0;

        std::cout << "searcher=heuristic_v9"
                  << " depth=" << options.depth
                  << " plies=" << options.plies
                  << " tt_mb=" << options.tt_mb
                  << " tt_entries=" << searcher.tt_entry_count()
                  << " seed=" << options.seed
                  << " iterative=" << (options.iterative ? 1 : 0)
                  << " clear_between_moves=" << (options.clear_between_moves ? 1 : 0)
                  << '\n';

        for (int ply = 0; ply < options.plies; ++ply) {
            const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
            if (moves.empty()) {
                std::cout << "terminal_at_ply=" << ply
                          << " in_check=" << (chess::in_check(pos, pos.side_to_move) ? 1 : 0)
                          << '\n';
                break;
            }

            if (options.clear_between_moves) {
                searcher.clear_tt();
            }

            const chess::TranspositionTableStats before = searcher.tt_stats();
            const chess::SearchResult result = search(searcher, pos, options);
            const chess::TranspositionTableStats after = searcher.tt_stats();
            nodes += result.nodes;
            ++searched_plies;

            const chess::Move played = random_move(moves, rng);
            if (options.per_ply) {
                const chess::TranspositionTableStats delta = diff_stats(after, before);
                std::cout << "ply=" << (ply + 1)
                          << " legal=" << moves.size()
                          << " random_played=" << chess::move_to_string(played)
                          << " best=" << chess::move_to_string(result.best_move)
                          << " score=" << result.score
                          << " nodes=" << result.nodes;
                print_stats("", delta);
            }

            pos.make_move(played);
        }

        std::cout << "summary"
                  << " searched_plies=" << searched_plies
                  << " nodes=" << nodes;
        print_stats("", searcher.tt_stats());
    } catch (const std::exception& error) {
        std::cerr << "measure_tt_stats: " << error.what() << '\n';
        return 1;
    }
}
