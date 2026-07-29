#include "move.hpp"
#include "nnue_searcher_v36.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Weights = chess::NnueSearcherV36::MoveOrderingWeights;

struct Options {
    std::string model_path = std::string(chess::DefaultPhaseQuantizedNnueModelPath);
    std::string book_path = "data/stockfish_balanced_openings_10ply_1k_20260723.txt";
    int depth = 6;
    int offset = 0;
    int count = 48;
    int tt_mb = 64;
    std::uint32_t seed = 20260727;
    int counter_history_bonus = 14'000;
    Weights weights{};
};

struct Totals {
    std::uint64_t nodes = 0;
    std::uint64_t microseconds = 0;
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
        throw std::runtime_error(
            "invalid integer for " + std::string(name) + ": "
            + std::string(value));
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        auto require_value = [&]() -> std::string_view {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(arg));
            }
            return argv[++index];
        };

        if (arg == "--model") {
            options.model_path = std::string(require_value());
        } else if (arg == "--book") {
            options.book_path = std::string(require_value());
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(), arg);
        } else if (arg == "--offset") {
            options.offset = parse_int(require_value(), arg);
        } else if (arg == "--count") {
            options.count = parse_int(require_value(), arg);
        } else if (arg == "--tt-mb") {
            options.tt_mb = parse_int(require_value(), arg);
        } else if (arg == "--seed") {
            options.seed =
                static_cast<std::uint32_t>(parse_int(require_value(), arg));
        } else if (arg == "--see-weight") {
            options.weights.see_weight = parse_int(require_value(), arg);
        } else if (arg == "--killer1-bonus") {
            options.weights.killer1_bonus = parse_int(require_value(), arg);
        } else if (arg == "--killer2-bonus") {
            options.weights.killer2_bonus = parse_int(require_value(), arg);
        } else if (arg == "--check-bonus") {
            options.weights.check_bonus = parse_int(require_value(), arg);
        } else if (arg == "--counter-history-bonus") {
            options.counter_history_bonus = parse_int(require_value(), arg);
            options.weights.counter_history_bonus =
                options.counter_history_bonus;
        } else if (arg == "--bad-capture-stage-threshold") {
            options.weights.bad_capture_stage_threshold =
                parse_int(require_value(), arg);
        } else if (arg == "--qsearch-captured-value-weight") {
            options.weights.qsearch_captured_value_weight =
                parse_int(require_value(), arg);
        } else if (arg == "--qsearch-capture-metric-weight") {
            options.weights.qsearch_capture_metric_weight =
                parse_int(require_value(), arg);
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_nnue_v36_ordering_weights [options]\n"
                << "  --model PATH --book PATH --depth N --offset N --count N\n"
                << "  --tt-mb N --seed N\n"
                << "  --see-weight N --killer1-bonus N --killer2-bonus N\n"
                << "  --check-bonus N --counter-history-bonus N\n"
                << "  --bad-capture-stage-threshold N\n"
                << "  --qsearch-captured-value-weight N\n"
                << "  --qsearch-capture-metric-weight N\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.depth <= 0 || options.offset < 0 || options.count <= 0
        || options.tt_mb <= 0) {
        throw std::runtime_error("invalid non-positive benchmark option");
    }
    return options;
}

std::string strip_comment(std::string_view line) {
    const std::size_t hash = line.find('#');
    std::string clean(line.substr(
        0, hash == std::string_view::npos ? line.size() : hash));
    while (!clean.empty()
           && (clean.back() == ' ' || clean.back() == '\t'
               || clean.back() == '\r')) {
        clean.pop_back();
    }
    return clean;
}

chess::Move find_legal_uci_move(
    const chess::Position& position,
    std::string_view uci
) {
    chess::MoveList moves;
    chess::generate_legal_moves(position, moves);
    for (chess::Move move : moves) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    throw std::runtime_error("illegal opening move: " + std::string(uci));
}

std::vector<chess::Position> load_positions(
    const Options& options
) {
    std::ifstream input(options.book_path);
    if (!input) {
        throw std::runtime_error(
            "failed to open opening book: " + options.book_path);
    }

    std::vector<chess::Position> positions;
    std::string line;
    while (std::getline(input, line)) {
        const std::string clean = strip_comment(line);
        if (clean.empty()) {
            continue;
        }
        chess::Position position;
        position.set_startpos();
        std::istringstream moves(clean);
        std::string uci;
        while (moves >> uci) {
            position.make_move(find_legal_uci_move(position, uci));
        }
        positions.push_back(position);
    }

    std::mt19937 generator(options.seed);
    std::shuffle(positions.begin(), positions.end(), generator);
    const std::size_t offset = static_cast<std::size_t>(options.offset);
    const std::size_t count = static_cast<std::size_t>(options.count);
    if (offset + count > positions.size()) {
        throw std::runtime_error("requested position slice exceeds book size");
    }
    return std::vector<chess::Position>(
        positions.begin() + static_cast<std::ptrdiff_t>(offset),
        positions.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

template<typename Searcher>
chess::SearchResult run_one(
    Searcher& searcher,
    const chess::Position& position,
    int depth,
    Totals& totals
) {
    searcher.clear_tt();
    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = Clock::now();
    const chess::SearchResult result =
        searcher.search_best_move(position, limits);
    const auto end = Clock::now();
    totals.nodes += result.nodes;
    totals.microseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count());
    return result;
}

double ratio(std::uint64_t candidate, std::uint64_t baseline) {
    return baseline == 0
        ? 0.0
        : static_cast<double>(candidate) / static_cast<double>(baseline);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model_path)) {
            throw std::runtime_error(
                "failed to load model: " + options.model_path);
        }
        model.set_neon_dotprod_enabled(true);

        const std::vector<chess::Position> positions =
            load_positions(options);
        const Weights baseline_weights{};
        chess::NnueSearcherV36 baseline(
            model,
            static_cast<std::size_t>(options.tt_mb),
            4,
            10,
            14,
            14'000,
            baseline_weights);
        chess::NnueSearcherV36 candidate(
            model,
            static_cast<std::size_t>(options.tt_mb),
            4,
            10,
            14,
            options.counter_history_bonus,
            options.weights);

        Totals baseline_totals;
        Totals candidate_totals;
        int score_mismatches = 0;
        int move_mismatches = 0;
        for (std::size_t index = 0; index < positions.size(); ++index) {
            chess::SearchResult baseline_result;
            chess::SearchResult candidate_result;
            if ((index & 1U) == 0) {
                baseline_result = run_one(
                    baseline, positions[index], options.depth, baseline_totals);
                candidate_result = run_one(
                    candidate, positions[index], options.depth, candidate_totals);
            } else {
                candidate_result = run_one(
                    candidate, positions[index], options.depth, candidate_totals);
                baseline_result = run_one(
                    baseline, positions[index], options.depth, baseline_totals);
            }
            score_mismatches += baseline_result.score != candidate_result.score;
            move_mismatches +=
                baseline_result.best_move != candidate_result.best_move;
        }

        std::cout
            << "summary"
            << " depth=" << options.depth
            << " offset=" << options.offset
            << " positions=" << positions.size()
            << " seed=" << options.seed
            << " baseline_nodes=" << baseline_totals.nodes
            << " candidate_nodes=" << candidate_totals.nodes
            << " node_ratio="
            << ratio(candidate_totals.nodes, baseline_totals.nodes)
            << " baseline_us=" << baseline_totals.microseconds
            << " candidate_us=" << candidate_totals.microseconds
            << " time_ratio="
            << ratio(
                candidate_totals.microseconds,
                baseline_totals.microseconds)
            << " score_mismatches=" << score_mismatches
            << " move_mismatches=" << move_mismatches
            << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
