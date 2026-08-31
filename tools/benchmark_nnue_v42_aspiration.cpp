#include "move.hpp"
#include "nnue_searcher_v41.hpp"
#include "nnue_searcher_v42.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t TtMegabytes = 64;
constexpr std::size_t TtBucketSize = 4;
constexpr int HistoryPenaltyDivisorNumerator = 10;
constexpr int HistoryPenaltyDivisorDenominator = 14;
constexpr int CounterHistoryBonus = 14'000;

// Deliberately fixed so repeated runs compare the same opening, tactical,
// middlegame, and endgame workloads instead of benchmarking position sampling.
constexpr std::array<std::string_view, 12> Fens{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
    "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
    "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
    "2r3k1/1p1bqppp/p3pn2/3p4/3P4/P1NBPN2/1PQ2PPP/2R2RK1 b - - 0 16",
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
    "8/8/2p5/3p4/3P4/2P5/8/4K1k1 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
};

struct Options {
    int depth = 7;
    int repeats = 2;
    std::string model = std::string(chess::DefaultPhaseQuantizedNnueModelPath);
};

struct Measurement {
    chess::SearchResult result{};
    std::uint64_t nanoseconds = 0;
};

struct Totals {
    std::uint64_t nodes = 0;
    std::uint64_t nanoseconds = 0;
};

struct AspirationTotals {
    std::uint64_t searches = 0;
    std::uint64_t completed_iterations = 0;
    std::uint64_t narrow_attempts = 0;
    std::uint64_t initial_window_successes = 0;
    std::uint64_t fail_lows = 0;
    std::uint64_t fail_highs = 0;
    std::uint64_t reduced_depth_attempts = 0;
    std::uint64_t full_window_fallbacks = 0;
    std::uint64_t unresolved_ranges = 0;
    std::uint64_t retry_limit_fallbacks = 0;
    std::int64_t last_initial_delta_sum_cp = 0;
    std::uint64_t searches_with_narrow_attempt = 0;
    std::int64_t final_mean_sum_cp = 0;
    std::uint64_t searches_with_final_mean = 0;

    void add(const chess::NnueSearcherV42::AspirationStats& stats) {
        ++searches;
        completed_iterations += stats.completed_iterations;
        narrow_attempts += stats.narrow_attempts;
        initial_window_successes += stats.initial_window_successes;
        fail_lows += stats.fail_lows;
        fail_highs += stats.fail_highs;
        reduced_depth_attempts += stats.reduced_depth_attempts;
        full_window_fallbacks += stats.full_window_fallbacks;
        unresolved_ranges += stats.unresolved_ranges;
        retry_limit_fallbacks += stats.retry_limit_fallbacks;
        if (stats.narrow_attempts != 0) {
            last_initial_delta_sum_cp += stats.last_initial_delta_cp;
            ++searches_with_narrow_attempt;
        }
        if (stats.has_final_mean_score) {
            final_mean_sum_cp += stats.final_mean_score_cp;
            ++searches_with_final_mean;
        }
    }
};

int parse_positive(const char* text, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const int value = std::stoi(text, &parsed);
        if (parsed != std::string_view(text).size() || value <= 0) {
            throw std::invalid_argument("not a positive integer");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(
            std::string(name) + " must be a positive integer: " + text);
    }
}

Options parse_options(int argc, char** argv) {
    if (argc > 4) {
        throw std::runtime_error(
            "usage: benchmark_nnue_v42_aspiration [depth [repeats [model]]]"
        );
    }
    Options options;
    if (argc >= 2) {
        options.depth = parse_positive(argv[1], "depth");
    }
    if (argc >= 3) {
        options.repeats = parse_positive(argv[2], "repeats");
    }
    if (argc >= 4) {
        options.model = argv[3];
    }
    return options;
}

std::array<chess::Position, Fens.size()> load_positions() {
    std::array<chess::Position, Fens.size()> positions;
    for (std::size_t index = 0; index < Fens.size(); ++index) {
        if (!positions[index].set_fen(Fens[index])) {
            throw std::runtime_error(
                "invalid hardcoded FEN at index " + std::to_string(index));
        }
        if (chess::generate_legal_moves(positions[index]).empty()) {
            throw std::runtime_error(
                "terminal hardcoded FEN at index " + std::to_string(index));
        }
    }
    return positions;
}

bool contains_move(const std::vector<chess::Move>& moves, chess::Move move) {
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

void validate_result(
    const chess::Position& position,
    const chess::SearchResult& result,
    int expected_depth,
    std::string_view version,
    std::size_t position_index,
    int repeat
) {
    if (result.stopped) {
        throw std::runtime_error(
            std::string(version) + " unexpectedly stopped at position "
            + std::to_string(position_index) + ", repeat "
            + std::to_string(repeat));
    }
    if (result.depth != expected_depth) {
        throw std::runtime_error(
            std::string(version) + " returned depth "
            + std::to_string(result.depth) + " instead of "
            + std::to_string(expected_depth) + " at position "
            + std::to_string(position_index));
    }
    const std::vector<chess::Move> legal_moves =
        chess::generate_legal_moves(position);
    if (!contains_move(legal_moves, result.best_move)) {
        throw std::runtime_error(
            std::string(version) + " returned illegal move "
            + chess::move_to_string(result.best_move) + " at position "
            + std::to_string(position_index));
    }
}

template<typename Searcher>
Measurement measure(
    Searcher& searcher,
    const chess::Position& position,
    int depth,
    std::string_view version,
    std::size_t position_index,
    int repeat
) {
    // Each timed search starts without TT or history-table carry-over. This
    // keeps the only intended search difference the aspiration policy.
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    searcher.clear_aspiration_stats();

    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = Clock::now();
    const chess::SearchResult result =
        searcher.search_best_move(position, limits);
    const auto stop = Clock::now();

    validate_result(
        position, result, depth, version, position_index, repeat);
    return Measurement{
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stop - start).count())};
}

void add(Totals& totals, const Measurement& measurement) {
    totals.nodes += measurement.result.nodes;
    totals.nanoseconds += measurement.nanoseconds;
}

double milliseconds(std::uint64_t nanoseconds) {
    return static_cast<double>(nanoseconds) / 1'000'000.0;
}

double nps(const Totals& totals) {
    return totals.nanoseconds == 0
        ? 0.0
        : static_cast<double>(totals.nodes) * 1'000'000'000.0
            / static_cast<double>(totals.nanoseconds);
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
        ? 0.0
        : static_cast<double>(numerator) / static_cast<double>(denominator);
}

double average(std::int64_t sum, std::uint64_t count) {
    return count == 0
        ? 0.0
        : static_cast<double>(sum) / static_cast<double>(count);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model: " + options.model);
        }
        model.set_neon_dotprod_enabled(true);
        const auto positions = load_positions();

        // Pass identical construction inputs to both versions. V41 keeps its
        // legacy fixed 50-cp aspiration; V42 enables adaptive aspiration.
        // Reported depth is the nominal iterative-deepening depth: after a
        // fail-high, V42 deliberately permits a shallower retry.
        const chess::NnueSearcherV41::MoveOrderingWeights ordering_weights{};
        const chess::NnueSearcherV41::SelectiveConfig selective_config{};
        chess::NnueSearcherV41 legacy(
            model,
            TtMegabytes,
            TtBucketSize,
            HistoryPenaltyDivisorNumerator,
            HistoryPenaltyDivisorDenominator,
            CounterHistoryBonus,
            ordering_weights,
            selective_config);
        chess::NnueSearcherV42 adaptive(
            model,
            TtMegabytes,
            TtBucketSize,
            HistoryPenaltyDivisorNumerator,
            HistoryPenaltyDivisorDenominator,
            CounterHistoryBonus,
            ordering_weights,
            selective_config);

        auto legacy_aspiration = legacy.aspiration_config();
        legacy_aspiration.enabled = false;
        legacy.set_aspiration_config(legacy_aspiration);
        auto adaptive_aspiration = adaptive.aspiration_config();
        adaptive_aspiration.enabled = true;
        adaptive.set_aspiration_config(adaptive_aspiration);

        Totals legacy_totals;
        Totals adaptive_totals;
        AspirationTotals aspiration_totals;
        std::uint64_t score_mismatches = 0;
        std::uint64_t move_mismatches = 0;

        std::cout << std::fixed << std::setprecision(3);
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            for (std::size_t index = 0; index < positions.size(); ++index) {
                Measurement legacy_measurement;
                Measurement adaptive_measurement;
                const bool legacy_first =
                    ((static_cast<std::size_t>(repeat) + index) & 1U) == 0;
                if (legacy_first) {
                    legacy_measurement = measure(
                        legacy,
                        positions[index],
                        options.depth,
                        "v41",
                        index,
                        repeat);
                    adaptive_measurement = measure(
                        adaptive,
                        positions[index],
                        options.depth,
                        "v42",
                        index,
                        repeat);
                } else {
                    adaptive_measurement = measure(
                        adaptive,
                        positions[index],
                        options.depth,
                        "v42",
                        index,
                        repeat);
                    legacy_measurement = measure(
                        legacy,
                        positions[index],
                        options.depth,
                        "v41",
                        index,
                        repeat);
                }

                add(legacy_totals, legacy_measurement);
                add(adaptive_totals, adaptive_measurement);
                aspiration_totals.add(adaptive.aspiration_stats());
                const bool score_mismatch =
                    legacy_measurement.result.score
                    != adaptive_measurement.result.score;
                const bool move_mismatch =
                    legacy_measurement.result.best_move
                    != adaptive_measurement.result.best_move;
                score_mismatches += score_mismatch;
                move_mismatches += move_mismatch;

                std::cout
                    << "sample repeat=" << repeat
                    << " position=" << index
                    << " order=" << (legacy_first ? "AB" : "BA")
                    << " v41_move="
                    << chess::move_to_string(
                        legacy_measurement.result.best_move)
                    << " v42_move="
                    << chess::move_to_string(
                        adaptive_measurement.result.best_move)
                    << " v41_score=" << legacy_measurement.result.score
                    << " v42_score=" << adaptive_measurement.result.score
                    << " v41_nodes=" << legacy_measurement.result.nodes
                    << " v42_nodes=" << adaptive_measurement.result.nodes
                    << " node_ratio="
                    << ratio(
                        adaptive_measurement.result.nodes,
                        legacy_measurement.result.nodes)
                    << " v41_ms="
                    << milliseconds(legacy_measurement.nanoseconds)
                    << " v42_ms="
                    << milliseconds(adaptive_measurement.nanoseconds)
                    << " score_mismatch=" << score_mismatch
                    << " move_mismatch=" << move_mismatch
                    << '\n';
            }
        }

        const double legacy_nps = nps(legacy_totals);
        const double adaptive_nps = nps(adaptive_totals);
        std::cout
            << "summary depth=" << options.depth
            << " positions=" << positions.size()
            << " repeats=" << options.repeats
            << " searches_per_version="
            << positions.size() * static_cast<std::size_t>(options.repeats)
            << " tt_mb=" << TtMegabytes
            << " bucket_size=" << TtBucketSize
            << " v41_nodes=" << legacy_totals.nodes
            << " v42_nodes=" << adaptive_totals.nodes
            << " node_ratio="
            << ratio(adaptive_totals.nodes, legacy_totals.nodes)
            << " v41_wall_ms=" << milliseconds(legacy_totals.nanoseconds)
            << " v42_wall_ms=" << milliseconds(adaptive_totals.nanoseconds)
            << " wall_ratio="
            << ratio(
                adaptive_totals.nanoseconds, legacy_totals.nanoseconds)
            << " v41_nps=" << legacy_nps
            << " v42_nps=" << adaptive_nps
            << " nps_ratio="
            << (legacy_nps == 0.0 ? 0.0 : adaptive_nps / legacy_nps)
            << " score_mismatches=" << score_mismatches
            << " move_mismatches=" << move_mismatches
            << '\n';

        std::cout
            << "aspiration_summary searches=" << aspiration_totals.searches
            << " completed_iterations="
            << aspiration_totals.completed_iterations
            << " narrow_attempts=" << aspiration_totals.narrow_attempts
            << " initial_window_successes="
            << aspiration_totals.initial_window_successes
            << " fail_lows=" << aspiration_totals.fail_lows
            << " fail_highs=" << aspiration_totals.fail_highs
            << " reduced_depth_attempts="
            << aspiration_totals.reduced_depth_attempts
            << " full_window_fallbacks="
            << aspiration_totals.full_window_fallbacks
            << " unresolved_ranges="
            << aspiration_totals.unresolved_ranges
            << " retry_limit_fallbacks="
            << aspiration_totals.retry_limit_fallbacks
            << " avg_last_initial_delta_cp="
            << average(
                aspiration_totals.last_initial_delta_sum_cp,
                aspiration_totals.searches_with_narrow_attempt)
            << " avg_final_mean_cp="
            << average(
                aspiration_totals.final_mean_sum_cp,
                aspiration_totals.searches_with_final_mean)
            << '\n';
    } catch (const std::exception& error) {
        std::cerr << "benchmark_nnue_v42_aspiration: "
                  << error.what() << '\n';
        return 1;
    }
}
