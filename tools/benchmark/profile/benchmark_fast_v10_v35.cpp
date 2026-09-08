#include "heuristic_searcher_fast_v35.hpp"
#include "heuristic_searcher_v10.hpp"
#include "move.hpp"
#include "position.hpp"

#include <algorithm>
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
    int depth = 10;
    int fixed_start = 1;
    int fixed_count = 4;
    int random_positions = 0;
    int random_plies = 12;
    int tt_mb = 64;
    int v10_tt_mb = -1;
    int v35_tt_mb = -1;
    std::uint32_t seed = 20260708;
};

struct Row {
    chess::SearchResult result;
    std::uint64_t us = 0;
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
        } else if (arg == "--fixed-start") {
            options.fixed_start = parse_int(require_value(arg), arg);
        } else if (arg == "--fixed-count") {
            options.fixed_count = parse_int(require_value(arg), arg);
        } else if (arg == "--random-positions") {
            options.random_positions = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-mb") {
            options.tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--v10-tt-mb") {
            options.v10_tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--v35-tt-mb") {
            options.v35_tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_fast_v10_v35 [--depth N] [--tt-mb N]\n"
                << "                                [--v10-tt-mb N] [--v35-tt-mb N]\n"
                << "                                [--fixed-start N] [--fixed-count N]\n"
                << "                                [--random-positions N] [--random-plies N]\n"
                << "                                [--seed N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    return options;
}

bool has_both_kings(const chess::Position& pos) {
    const auto king_index = static_cast<int>(chess::PieceType::King);
    return chess::popcount(pos.pieces[static_cast<int>(chess::Color::White)][king_index]) == 1
        && chess::popcount(pos.pieces[static_cast<int>(chess::Color::Black)][king_index]) == 1;
}

std::vector<chess::Position> make_positions(const Options& options) {
    const std::vector<std::string_view> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
        "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
        "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
    };

    std::vector<chess::Position> positions;
    const int fixed_start = std::max(0, options.fixed_start);
    const int fixed_end = std::min<int>(
        static_cast<int>(fens.size()),
        fixed_start + std::max(0, options.fixed_count));
    for (int index = fixed_start; index < fixed_end; ++index) {
        chess::Position pos;
        if (!pos.set_fen(fens[static_cast<std::size_t>(index)])) {
            throw std::runtime_error("invalid hardcoded FEN");
        }
        if (has_both_kings(pos)) {
            positions.push_back(pos);
        }
    }

    std::mt19937 rng(options.seed);
    for (int sample = 0; sample < options.random_positions; ++sample) {
        chess::Position pos;
        pos.set_startpos();
        std::uniform_int_distribution<int> plies_dist(0, options.random_plies);
        const int plies = plies_dist(rng);
        for (int ply = 0; ply < plies; ++ply) {
            chess::MoveList moves;
            chess::generate_legal_moves(pos, moves);
            if (moves.empty()) {
                break;
            }
            std::uniform_int_distribution<std::size_t> move_dist(0, moves.size() - 1);
            pos.make_move(moves[move_dist(rng)]);
        }
        positions.push_back(pos);
    }
    return positions;
}

template <typename Searcher>
Row run_one(Searcher& searcher, const chess::Position& pos, int depth) {
    searcher.clear_tt();
    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = std::chrono::steady_clock::now();
    chess::SearchResult result = searcher.search_best_move(pos, limits);
    const auto end = std::chrono::steady_clock::now();
    return Row{result, static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())};
}

double nps(std::uint64_t nodes, std::uint64_t us) {
    return us == 0 ? 0.0 : static_cast<double>(nodes) * 1'000'000.0 / static_cast<double>(us);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const int v10_tt_mb = options.v10_tt_mb < 0 ? options.tt_mb : options.v10_tt_mb;
        const int v35_tt_mb = options.v35_tt_mb < 0 ? options.tt_mb : options.v35_tt_mb;
        const std::vector<chess::Position> positions = make_positions(options);

        std::uint64_t v10_us = 0;
        std::uint64_t v35_us = 0;
        std::uint64_t v10_nodes = 0;
        std::uint64_t v35_nodes = 0;
        int score_mismatches = 0;
        int move_mismatches = 0;

        for (std::size_t i = 0; i < positions.size(); ++i) {
            chess::HeuristicSearcherV10 v10(static_cast<std::size_t>(v10_tt_mb));
            chess::HeuristicSearcherFastV35 v35(static_cast<std::size_t>(v35_tt_mb));
            const Row r10 = run_one(v10, positions[i], options.depth);
            const Row r35 = run_one(v35, positions[i], options.depth);

            v10_us += r10.us;
            v35_us += r35.us;
            v10_nodes += r10.result.nodes;
            v35_nodes += r35.result.nodes;
            if (r10.result.score != r35.result.score) {
                ++score_mismatches;
            }
            if (r10.result.best_move != r35.result.best_move) {
                ++move_mismatches;
            }

            std::cout << "sample=" << i
                      << " v10_score=" << r10.result.score
                      << " v35_score=" << r35.result.score
                      << " v10_best=" << chess::move_to_string(r10.result.best_move)
                      << " v35_best=" << chess::move_to_string(r35.result.best_move)
                      << " v10_nodes=" << r10.result.nodes
                      << " v35_nodes=" << r35.result.nodes
                      << " v10_us=" << r10.us
                      << " v35_us=" << r35.us
                      << '\n';
        }

        std::cout << "summary"
                  << " samples=" << positions.size()
                  << " depth=" << options.depth
                  << " v10_tt_mb=" << v10_tt_mb
                  << " v35_tt_mb=" << v35_tt_mb
                  << " v10_nodes=" << v10_nodes
                  << " v35_nodes=" << v35_nodes
                  << " v10_us=" << v10_us
                  << " v35_us=" << v35_us
                  << " v10_nps=" << nps(v10_nodes, v10_us)
                  << " v35_nps=" << nps(v35_nodes, v35_us)
                  << " v35_vs_v10_time=" << (v10_us == 0 ? 0.0 : static_cast<double>(v35_us) / v10_us)
                  << " v35_vs_v10_nodes=" << (v10_nodes == 0 ? 0.0 : static_cast<double>(v35_nodes) / v10_nodes)
                  << " score_mismatches=" << score_mismatches
                  << " move_mismatches=" << move_mismatches
                  << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
