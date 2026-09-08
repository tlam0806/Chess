#include "heuristic_searcher_v35.hpp"
#include "move.hpp"
#include "nnue_searcher_v36.hpp"
#include "phase_quantized_nnue.hpp"
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

enum class RunMode {
    Both,
    Heuristic,
    Nnue
};

struct Options {
    std::string model_path = std::string(chess::DefaultPhaseQuantizedNnueModelPath);
    RunMode mode = RunMode::Both;
    int depth = 10;
    int fixed_start = 0;
    int fixed_count = 10;
    int random_positions = 0;
    int random_plies = 12;
    int repeats = 1;
    int tt_mb = 64;
    int heuristic_tt_mb = -1;
    int nnue_tt_mb = -1;
    bool nnue_neon = true;
    std::uint32_t seed = 20260713;
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

RunMode parse_mode(std::string_view value) {
    if (value == "both") {
        return RunMode::Both;
    }
    if (value == "heuristic") {
        return RunMode::Heuristic;
    }
    if (value == "nnue") {
        return RunMode::Nnue;
    }
    throw std::runtime_error("invalid --only value: " + std::string(value));
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

        if (arg == "--model") {
            options.model_path = std::string(require_value(arg));
        } else if (arg == "--only") {
            options.mode = parse_mode(require_value(arg));
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--fixed-start") {
            options.fixed_start = parse_int(require_value(arg), arg);
        } else if (arg == "--fixed-count") {
            options.fixed_count = parse_int(require_value(arg), arg);
        } else if (arg == "--random-positions") {
            options.random_positions = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--repeats") {
            options.repeats = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-mb") {
            options.tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--heuristic-tt-mb") {
            options.heuristic_tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--nnue-tt-mb") {
            options.nnue_tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--nnue-kernel") {
            const std::string_view kernel = require_value(arg);
            if (kernel == "neon") {
                options.nnue_neon = true;
            } else if (kernel == "scalar") {
                options.nnue_neon = false;
            } else {
                throw std::runtime_error(
                    "invalid --nnue-kernel value: " + std::string(kernel));
            }
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_nnue_v36_vs_heuristic_v35 [--model path]\n"
                << "        [--only both|heuristic|nnue] [--depth N] [--repeats N]\n"
                << "        [--tt-mb N] [--heuristic-tt-mb N] [--nnue-tt-mb N]\n"
                << "        [--nnue-kernel neon|scalar]\n"
                << "        [--fixed-start N] [--fixed-count N]\n"
                << "        [--random-positions N] [--random-plies N] [--seed N]\n";
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
        "8/8/2p5/3p4/3P4/2P5/8/4K1k1 w - - 0 1",
        "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
        "2r3k1/1p1bqppp/p3pn2/3p4/3P4/P1NBPN2/1PQ2PPP/2R2RK1 b - - 0 16",
        "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
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

void print_summary(
    std::string_view version,
    int depth,
    int samples,
    int repeats,
    std::uint64_t nodes,
    std::uint64_t us
) {
    std::cout << "summary"
              << " version=" << version
              << " samples=" << samples
              << " repeats=" << repeats
              << " depth=" << depth
              << " nodes=" << nodes
              << " us=" << us
              << " nps=" << nps(nodes, us)
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        const Options options = parse_args(argc, argv);
        const int heuristic_tt_mb =
            options.heuristic_tt_mb < 0 ? options.tt_mb : options.heuristic_tt_mb;
        const int nnue_tt_mb = options.nnue_tt_mb < 0 ? options.tt_mb : options.nnue_tt_mb;
        const std::vector<chess::Position> positions = make_positions(options);

        chess::PhaseQuantizedNnueModel model;
        if ((options.mode == RunMode::Both || options.mode == RunMode::Nnue)
            && !model.load(options.model_path)) {
            throw std::runtime_error("failed to load NNUE model: " + options.model_path);
        }
        model.set_neon_dotprod_enabled(options.nnue_neon);

        std::uint64_t heuristic_us = 0;
        std::uint64_t nnue_us = 0;
        std::uint64_t heuristic_nodes = 0;
        std::uint64_t nnue_nodes = 0;
        int score_mismatches = 0;
        int move_mismatches = 0;

        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            for (std::size_t i = 0; i < positions.size(); ++i) {
                Row heuristic_row;
                Row nnue_row;

                if (options.mode == RunMode::Both || options.mode == RunMode::Heuristic) {
                    chess::HeuristicSearcherV35 heuristic(
                        static_cast<std::size_t>(heuristic_tt_mb),
                        4);
                    heuristic_row = run_one(heuristic, positions[i], options.depth);
                    heuristic_us += heuristic_row.us;
                    heuristic_nodes += heuristic_row.result.nodes;
                }

                if (options.mode == RunMode::Both || options.mode == RunMode::Nnue) {
                    chess::NnueSearcherV36 nnue(
                        model,
                        static_cast<std::size_t>(nnue_tt_mb),
                        4);
                    nnue_row = run_one(nnue, positions[i], options.depth);
                    nnue_us += nnue_row.us;
                    nnue_nodes += nnue_row.result.nodes;
                }

                if (options.mode == RunMode::Both) {
                    if (heuristic_row.result.score != nnue_row.result.score) {
                        ++score_mismatches;
                    }
                    if (heuristic_row.result.best_move != nnue_row.result.best_move) {
                        ++move_mismatches;
                    }
                }

                std::cout << "sample=" << i
                          << " repeat=" << repeat;
                if (options.mode == RunMode::Both || options.mode == RunMode::Heuristic) {
                    std::cout << " heuristic_score=" << heuristic_row.result.score
                              << " heuristic_best=" << chess::move_to_string(heuristic_row.result.best_move)
                              << " heuristic_nodes=" << heuristic_row.result.nodes
                              << " heuristic_us=" << heuristic_row.us;
                }
                if (options.mode == RunMode::Both || options.mode == RunMode::Nnue) {
                    std::cout << " nnue_score=" << nnue_row.result.score
                              << " nnue_best=" << chess::move_to_string(nnue_row.result.best_move)
                              << " nnue_nodes=" << nnue_row.result.nodes
                              << " nnue_us=" << nnue_row.us;
                }
                std::cout << '\n';
            }
        }

        if (options.mode == RunMode::Both || options.mode == RunMode::Heuristic) {
            print_summary(
                "heuristic_v35",
                options.depth,
                static_cast<int>(positions.size()),
                options.repeats,
                heuristic_nodes,
                heuristic_us);
        }
        if (options.mode == RunMode::Both || options.mode == RunMode::Nnue) {
            print_summary(
                model.uses_neon_dotprod_kernel()
                    ? "nnue_strict_v36_neon"
                    : "nnue_strict_v36_scalar",
                options.depth,
                static_cast<int>(positions.size()),
                options.repeats,
                nnue_nodes,
                nnue_us);
        }
        if (options.mode == RunMode::Both) {
            std::cout << "compare"
                      << " nnue_vs_heuristic_time="
                      << (heuristic_us == 0 ? 0.0 : static_cast<double>(nnue_us) / heuristic_us)
                      << " nnue_vs_heuristic_nodes="
                      << (heuristic_nodes == 0 ? 0.0 : static_cast<double>(nnue_nodes) / heuristic_nodes)
                      << " score_mismatches=" << score_mismatches
                      << " move_mismatches=" << move_mismatches
                      << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
