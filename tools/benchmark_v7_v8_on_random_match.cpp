#include "attacks.hpp"
#include "heuristic_searcher_v7.hpp"
#include "heuristic_searcher_v8.hpp"
#include "heuristic_searcher_v9.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search_types.hpp"

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
    int warmup_plies = 0;
    bool iterative = false;
    bool summary_only = false;
    std::uint32_t seed = 20260612;
    std::uint32_t warmup_seed = 20260611;
};

struct Totals {
    std::uint64_t nodes = 0;
    std::uint64_t time_us = 0;
    int moves = 0;
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
        } else if (arg == "--warmup-plies") {
            options.warmup_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--warmup-seed") {
            options.warmup_seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--iterative") {
            options.iterative = true;
        } else if (arg == "--summary-only") {
            options.summary_only = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v7_v8_on_random_match [--depth N] [--plies N]\n"
                << "                                       [--warmup-plies N]\n"
                << "                                       [--seed N] [--warmup-seed N]\n"
                << "                                       [--iterative] [--summary-only]\n";
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
    if (options.warmup_plies < 0) {
        throw std::runtime_error("--warmup-plies must be non-negative");
    }
    return options;
}

chess::Move random_move(const std::vector<chess::Move>& moves, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

template <typename Searcher>
chess::SearchResult timed_search(
    Searcher& searcher,
    const chess::Position& pos,
    int depth,
    std::uint64_t& time_us
) {
    searcher.clear_tt();
    const auto start = std::chrono::steady_clock::now();
    const chess::SearchResult result = searcher.search_best_move(pos, depth);
    const auto end = std::chrono::steady_clock::now();
    time_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return result;
}

template <typename Searcher>
chess::SearchResult timed_iterative_search(
    Searcher& searcher,
    const chess::Position& pos,
    int depth,
    std::uint64_t& time_us
) {
    searcher.clear_tt();
    const auto start = std::chrono::steady_clock::now();
    const chess::SearchResult result = searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = depth,
        .move_time = std::chrono::milliseconds{0}
    });
    const auto end = std::chrono::steady_clock::now();
    time_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return result;
}

template <typename Searcher>
chess::SearchResult timed_search(
    Searcher& searcher,
    const chess::Position& pos,
    int depth,
    bool iterative,
    std::uint64_t& time_us
) {
    if (iterative) {
        return timed_iterative_search(searcher, pos, depth, time_us);
    }
    return timed_search(searcher, pos, depth, time_us);
}

void add_result(Totals& totals, const chess::SearchResult& result, std::uint64_t time_us) {
    ++totals.moves;
    totals.nodes += result.nodes;
    totals.time_us += time_us;
}

void warmup_history(
    chess::HeuristicSearcherV8& v8,
    int depth,
    int plies,
    std::uint32_t seed,
    bool iterative
) {
    chess::Position pos;
    pos.set_startpos();
    std::mt19937 rng(seed);

    for (int ply = 0; ply < plies; ++ply) {
        const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (moves.empty()) {
            break;
        }

        std::uint64_t ignored_us = 0;
        (void) timed_search(v8, pos, depth, iterative, ignored_us);
        pos.make_move(random_move(moves, rng));
    }
}

std::string color_name(chess::Color color) {
    return color == chess::Color::White ? "white" : "black";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        chess::Position pos;
        pos.set_startpos();

        chess::HeuristicSearcherV7 v7;
        chess::HeuristicSearcherV8 v8;
        chess::HeuristicSearcherV9 v9;
        warmup_history(v8, options.depth, options.warmup_plies, options.warmup_seed, options.iterative);

        std::mt19937 rng(options.seed);
        Totals v7_totals;
        Totals v8_totals;
        Totals v9_totals;

        std::cout << "depth=" << options.depth
                  << " plies=" << options.plies
                  << " seed=" << options.seed
                  << " warmup_plies=" << options.warmup_plies
                  << " warmup_seed=" << options.warmup_seed
                  << " iterative=" << (options.iterative ? 1 : 0) << '\n';

        for (int ply = 0; ply < options.plies; ++ply) {
            const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
            if (moves.empty()) {
                std::cout << "terminal_at_ply=" << ply
                          << " side=" << color_name(pos.side_to_move)
                          << " in_check=" << (chess::in_check(pos, pos.side_to_move) ? 1 : 0)
                          << '\n';
                break;
            }

            std::uint64_t v7_us = 0;
            std::uint64_t v8_us = 0;
            std::uint64_t v9_us = 0;
            const chess::SearchResult v7_result =
                timed_search(v7, pos, options.depth, options.iterative, v7_us);
            const chess::SearchResult v8_result =
                timed_search(v8, pos, options.depth, options.iterative, v8_us);
            const chess::SearchResult v9_result =
                timed_search(v9, pos, options.depth, options.iterative, v9_us);
            add_result(v7_totals, v7_result, v7_us);
            add_result(v8_totals, v8_result, v8_us);
            add_result(v9_totals, v9_result, v9_us);

            const chess::Move played = random_move(moves, rng);
            if (!options.summary_only) {
                std::cout << "ply=" << (ply + 1)
                          << " side=" << color_name(pos.side_to_move)
                          << " legal=" << moves.size()
                          << " random_played=" << chess::move_to_string(played)
                          << " v7_best=" << chess::move_to_string(v7_result.best_move)
                          << " v7_score=" << v7_result.score
                          << " v7_nodes=" << v7_result.nodes
                          << " v7_us=" << v7_us
                          << " v8_best=" << chess::move_to_string(v8_result.best_move)
                          << " v8_score=" << v8_result.score
                          << " v8_nodes=" << v8_result.nodes
                          << " v8_us=" << v8_us
                          << " v9_best=" << chess::move_to_string(v9_result.best_move)
                          << " v9_score=" << v9_result.score
                          << " v9_nodes=" << v9_result.nodes
                          << " v9_us=" << v9_us
                          << '\n';
            }

            pos.make_move(played);
        }

        const std::uint64_t v7_avg_nodes = v7_totals.moves == 0
            ? 0
            : v7_totals.nodes / static_cast<std::uint64_t>(v7_totals.moves);
        const std::uint64_t v8_avg_nodes = v8_totals.moves == 0
            ? 0
            : v8_totals.nodes / static_cast<std::uint64_t>(v8_totals.moves);
        const std::uint64_t v7_avg_us = v7_totals.moves == 0
            ? 0
            : v7_totals.time_us / static_cast<std::uint64_t>(v7_totals.moves);
        const std::uint64_t v8_avg_us = v8_totals.moves == 0
            ? 0
            : v8_totals.time_us / static_cast<std::uint64_t>(v8_totals.moves);
        const std::uint64_t v9_avg_nodes = v9_totals.moves == 0
            ? 0
            : v9_totals.nodes / static_cast<std::uint64_t>(v9_totals.moves);
        const std::uint64_t v9_avg_us = v9_totals.moves == 0
            ? 0
            : v9_totals.time_us / static_cast<std::uint64_t>(v9_totals.moves);

        std::cout << "summary"
                  << " moves=" << v7_totals.moves
                  << " v7_nodes=" << v7_totals.nodes
                  << " v7_avg_nodes=" << v7_avg_nodes
                  << " v7_ms=" << (v7_totals.time_us / 1000)
                  << " v7_avg_us=" << v7_avg_us
                  << " v8_nodes=" << v8_totals.nodes
                  << " v8_avg_nodes=" << v8_avg_nodes
                  << " v8_ms=" << (v8_totals.time_us / 1000)
                  << " v8_avg_us=" << v8_avg_us
                  << " v9_nodes=" << v9_totals.nodes
                  << " v9_avg_nodes=" << v9_avg_nodes
                  << " v9_ms=" << (v9_totals.time_us / 1000)
                  << " v9_avg_us=" << v9_avg_us;
        if (v7_totals.nodes != 0) {
            std::cout << " v8_node_ratio=" << static_cast<double>(v8_totals.nodes) / v7_totals.nodes;
            std::cout << " v9_node_ratio=" << static_cast<double>(v9_totals.nodes) / v7_totals.nodes;
        }
        if (v7_totals.time_us != 0) {
            std::cout << " v8_time_ratio=" << static_cast<double>(v8_totals.time_us) / v7_totals.time_us;
            std::cout << " v9_time_ratio=" << static_cast<double>(v9_totals.time_us) / v7_totals.time_us;
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "benchmark_v7_v8_on_random_match: " << error.what() << '\n';
        return 1;
    }
}
