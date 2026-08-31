#include "move.hpp"
#include "nnue_searcher_v41.hpp"
#include "nnue_searcher_v42.hpp"
#include "nnue_searcher_v43.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t TtMegabytes = 64;

// Keep this corpus identical to benchmark_nnue_v42_aspiration so old V41/V42
// measurements remain directly comparable with the V43 experiment.
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

struct ConflictCase {
    std::string_view name;
    std::string_view fen;
    bool starts_aspiration_late = false;
};

// These are the three V42 roots that required score-range conflict recovery.
// They are deliberately outside the timed corpus: correctness diagnostics
// must not silently change the historical performance workload.
constexpr std::array<ConflictCase, 3> ConflictCases{{
    {
        "fallback_generation_conflict",
        "8/8/4k3/8/1p2N3/1P1KP3/7b/8 w - - 0 1",
        false,
    },
    {
        "in_search_conflict",
        "8/8/8/1pp5/k7/3K4/BP6/8 w - - 0 1",
        false,
    },
    {
        "pre_narrow_conflict",
        "8/8/2k4p/7P/n7/3B4/8/6K1 w - - 0 1",
        true,
    },
}};

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

struct MismatchTotals {
    std::uint64_t score = 0;
    std::uint64_t move = 0;
};

struct AspirationTotals {
    std::uint64_t searches = 0;
    std::uint64_t completed_iterations = 0;
    std::uint64_t narrow_iterations = 0;
    std::uint64_t narrow_attempts = 0;
    std::uint64_t initial_window_successes = 0;
    std::uint64_t fail_lows = 0;
    std::uint64_t fail_highs = 0;
    std::uint64_t reduced_depth_attempts = 0;
    std::uint64_t accepted_reduced_depth_iterations = 0;
    std::uint64_t accepted_narrow_nominal_depth_sum = 0;
    std::uint64_t accepted_narrow_search_depth_sum = 0;
    int max_accepted_depth_reduction = 0;
    std::uint64_t full_window_fallbacks = 0;
    std::uint64_t range_conflict_fallbacks = 0;
    std::uint64_t unresolved_ranges = 0;
    std::uint64_t retry_limit_fallbacks = 0;
    std::int64_t initial_delta_sum_cp = 0;
    int initial_delta_min_cp = std::numeric_limits<int>::max();
    int initial_delta_max_cp = std::numeric_limits<int>::min();
    std::uint64_t searches_with_delta = 0;
    std::int64_t final_mean_sum_cp = 0;
    std::uint64_t searches_with_final_mean = 0;

    template<typename Stats>
    void add(const Stats& stats) {
        ++searches;
        completed_iterations += stats.completed_iterations;
        narrow_iterations += stats.narrow_iterations;
        narrow_attempts += stats.narrow_attempts;
        initial_window_successes += stats.initial_window_successes;
        fail_lows += stats.fail_lows;
        fail_highs += stats.fail_highs;
        reduced_depth_attempts += stats.reduced_depth_attempts;
        accepted_reduced_depth_iterations +=
            stats.accepted_reduced_depth_iterations;
        accepted_narrow_nominal_depth_sum +=
            stats.accepted_narrow_nominal_depth_sum;
        accepted_narrow_search_depth_sum +=
            stats.accepted_narrow_search_depth_sum;
        max_accepted_depth_reduction = std::max(
            max_accepted_depth_reduction,
            stats.max_accepted_depth_reduction);
        full_window_fallbacks += stats.full_window_fallbacks;
        range_conflict_fallbacks += stats.range_conflict_fallbacks;
        unresolved_ranges += stats.unresolved_ranges;
        retry_limit_fallbacks += stats.retry_limit_fallbacks;
        if (stats.narrow_attempts != 0) {
            initial_delta_sum_cp += stats.initial_delta_sum_cp;
            initial_delta_min_cp = std::min(
                initial_delta_min_cp, stats.initial_delta_min_cp);
            initial_delta_max_cp = std::max(
                initial_delta_max_cp, stats.initial_delta_max_cp);
            ++searches_with_delta;
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
            throw std::invalid_argument("not positive");
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
            "usage: benchmark_nnue_v43_single_bound "
            "[depth [repeats [model]]]"
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
                "invalid benchmark FEN at index " + std::to_string(index));
        }
        if (chess::generate_legal_moves(positions[index]).empty()) {
            throw std::runtime_error(
                "terminal benchmark FEN at index "
                + std::to_string(index));
        }
    }
    return positions;
}

bool is_legal(const chess::Position& position, chess::Move move) {
    const std::vector<chess::Move> legal =
        chess::generate_legal_moves(position);
    return std::find(legal.begin(), legal.end(), move) != legal.end();
}

void validate_result(
    const chess::Position& position,
    const chess::SearchResult& result,
    int expected_depth,
    std::string_view version,
    std::string_view sample
) {
    if (result.stopped) {
        throw std::runtime_error(
            std::string(version) + " unexpectedly stopped on "
            + std::string(sample));
    }
    if (result.depth != expected_depth) {
        throw std::runtime_error(
            std::string(version) + " returned depth "
            + std::to_string(result.depth) + " instead of "
            + std::to_string(expected_depth) + " on "
            + std::string(sample));
    }
    if (!is_legal(position, result.best_move)) {
        throw std::runtime_error(
            std::string(version) + " returned illegal move "
            + chess::move_to_string(result.best_move) + " on "
            + std::string(sample));
    }
}

template<typename Searcher>
Measurement measure(
    Searcher& searcher,
    const chess::Position& position,
    int depth,
    std::string_view version,
    std::string_view sample
) {
    // Cold per-search state isolates the TT/search contract. Cross-root move
    // reuse is a separate experiment and must not contaminate node parity.
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    searcher.clear_aspiration_stats();

    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = Clock::now();
    const chess::SearchResult result =
        searcher.search_best_move(position, limits);
    const auto stop = Clock::now();
    validate_result(position, result, depth, version, sample);
    return Measurement{
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stop - start).count()),
    };
}

template<typename Searcher>
Measurement measure_with_history(
    Searcher& searcher,
    const chess::Position& position,
    int depth,
    std::string_view version,
    std::string_view sample
) {
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    searcher.clear_aspiration_stats();
    const std::array<chess::HashKey, 1> history{position.zobrist_key};
    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = Clock::now();
    const chess::SearchResult result =
        searcher.search_best_move(position, limits, history);
    const auto stop = Clock::now();
    validate_result(position, result, depth, version, sample);
    return Measurement{
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stop - start).count()),
    };
}

template<typename Searcher>
Measurement measure_fixed_with_history(
    Searcher& searcher,
    const chess::Position& position,
    int depth,
    std::string_view version,
    std::string_view sample
) {
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    searcher.clear_aspiration_stats();
    const std::array<chess::HashKey, 1> history{position.zobrist_key};
    const auto start = Clock::now();
    const chess::SearchResult result =
        searcher.search_best_move(position, depth, history);
    const auto stop = Clock::now();
    validate_result(position, result, depth, version, sample);
    return Measurement{
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stop - start).count()),
    };
}

void add(Totals& totals, const Measurement& measurement) {
    totals.nodes += measurement.result.nodes;
    totals.nanoseconds += measurement.nanoseconds;
}

void compare(
    MismatchTotals& mismatches,
    const Measurement& lhs,
    const Measurement& rhs
) {
    mismatches.score += lhs.result.score != rhs.result.score;
    mismatches.move += lhs.result.best_move != rhs.result.best_move;
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
        ? 0.0
        : static_cast<double>(numerator)
            / static_cast<double>(denominator);
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

double average(std::int64_t sum, std::uint64_t count) {
    return count == 0
        ? 0.0
        : static_cast<double>(sum) / static_cast<double>(count);
}

template<typename Config>
Config conflict_config(bool starts_late) {
    Config config{};
    config.enabled = true;
    config.min_depth = starts_late ? 5 : 3;
    config.delta_base_cp = starts_late ? 90 : 30;
    config.delta_divisor = starts_late ? 40'000 : 10'000;
    config.expansion_factor_per_mille = starts_late ? 3'000 : 1'750;
    config.max_fail_high_reductions = starts_late ? 3 : 0;
    config.mean_score_new_weight_per_mille = starts_late ? 1'000 : 500;
    config.max_researches = 6;
    config.mean_score_clamp_cp = 1'500;
    return config;
}

template<typename Config>
Config clean_selective_config() {
    Config config{};
    config.enable_lmr = false;
    config.enable_null_move = false;
    config.enable_reverse_futility = false;
    config.enable_late_move_pruning = false;
    config.enable_qsearch_see_pruning = false;
    config.enable_main_search_see_pruning = false;
    return config;
}

void print_aspiration_summary(
    std::string_view version,
    const AspirationTotals& totals
) {
    const int delta_min = totals.searches_with_delta == 0
        ? 0
        : totals.initial_delta_min_cp;
    const int delta_max = totals.searches_with_delta == 0
        ? 0
        : totals.initial_delta_max_cp;
    std::cout
        << "aspiration_summary version=" << version
        << " searches=" << totals.searches
        << " completed_iterations=" << totals.completed_iterations
        << " narrow_iterations=" << totals.narrow_iterations
        << " narrow_attempts=" << totals.narrow_attempts
        << " initial_window_successes="
        << totals.initial_window_successes
        << " fail_lows=" << totals.fail_lows
        << " fail_highs=" << totals.fail_highs
        << " reduced_depth_attempts=" << totals.reduced_depth_attempts
        << " accepted_reduced_depth_iterations="
        << totals.accepted_reduced_depth_iterations
        << " accepted_nominal_depth_sum="
        << totals.accepted_narrow_nominal_depth_sum
        << " accepted_search_depth_sum="
        << totals.accepted_narrow_search_depth_sum
        << " max_accepted_depth_reduction="
        << totals.max_accepted_depth_reduction
        << " full_window_fallbacks=" << totals.full_window_fallbacks
        << " range_conflict_fallbacks="
        << totals.range_conflict_fallbacks
        << " unresolved_ranges=" << totals.unresolved_ranges
        << " retry_limit_fallbacks=" << totals.retry_limit_fallbacks
        << " avg_initial_delta_cp="
        << average(
            totals.initial_delta_sum_cp, totals.narrow_iterations)
        << " initial_delta_min_cp=" << delta_min
        << " initial_delta_max_cp=" << delta_max
        << " avg_final_mean_cp="
        << average(
            totals.final_mean_sum_cp,
            totals.searches_with_final_mean)
        << '\n';
}

void run_conflict_repros(
    const chess::PhaseQuantizedNnueModel& model
) {
    for (const ConflictCase& conflict : ConflictCases) {
        chess::Position position;
        if (!position.set_fen(conflict.fen)) {
            throw std::runtime_error(
                "invalid conflict FEN: " + std::string(conflict.name));
        }

        chess::NnueSearcherV42 v42(model, TtMegabytes);
        chess::NnueSearcherV43 v43(model, TtMegabytes);
        v42.set_aspiration_config(
            conflict_config<chess::NnueSearcherV42::AspirationConfig>(
                conflict.starts_aspiration_late));
        v43.set_aspiration_config(
            conflict_config<chess::NnueSearcherV43::AspirationConfig>(
                conflict.starts_aspiration_late));

        constexpr int ConflictDepth = 6;
        const Measurement range = measure_with_history(
            v42, position, ConflictDepth, "v42", conflict.name);
        const Measurement scalar = measure_with_history(
            v43, position, ConflictDepth, "v43", conflict.name);
        const auto& v42_stats = v42.aspiration_stats();
        const auto& v43_stats = v43.aspiration_stats();
        if (v43_stats.range_conflict_fallbacks != 0
            || v43_stats.unresolved_ranges != 0) {
            throw std::runtime_error(
                "scalar V43 reported a range conflict on "
                + std::string(conflict.name));
        }

        std::cout
            << "conflict_repro name=" << conflict.name
            << " v42_move=" << chess::move_to_string(range.result.best_move)
            << " v43_move=" << chess::move_to_string(scalar.result.best_move)
            << " v42_score=" << range.result.score
            << " v43_score=" << scalar.result.score
            << " v42_nodes=" << range.result.nodes
            << " v43_nodes=" << scalar.result.nodes
            << " v43_v42_node_ratio="
            << ratio(scalar.result.nodes, range.result.nodes)
            << " v42_wall_ms=" << milliseconds(range.nanoseconds)
            << " v43_wall_ms=" << milliseconds(scalar.nanoseconds)
            << " v42_range_conflicts="
            << v42_stats.range_conflict_fallbacks
            << " v43_range_conflicts="
            << v43_stats.range_conflict_fallbacks
            << " v42_full_window_fallbacks="
            << v42_stats.full_window_fallbacks
            << " v43_full_window_fallbacks="
            << v43_stats.full_window_fallbacks
            << " v42_unresolved=" << v42_stats.unresolved_ranges
            << " v43_unresolved=" << v43_stats.unresolved_ranges
            << '\n';
    }
}

void run_clean_fixed_depth_audit(
    const chess::PhaseQuantizedNnueModel& model,
    const std::array<chess::Position, Fens.size()>& positions
) {
    chess::NnueSearcherV41 v41(model, TtMegabytes);
    chess::NnueSearcherV43 v43(model, TtMegabytes);
    v41.set_selective_config(
        clean_selective_config<chess::NnueSearcherV41::SelectiveConfig>());
    v43.set_selective_config(
        clean_selective_config<chess::NnueSearcherV43::SelectiveConfig>());

    Totals v41_totals;
    Totals v43_totals;
    std::uint64_t score_mismatches = 0;
    std::uint64_t move_mismatches = 0;
    constexpr std::array<int, 3> Depths{4, 5, 6};
    for (const int depth : Depths) {
        for (std::size_t index = 0; index < positions.size(); ++index) {
            const std::string sample =
                "clean_depth=" + std::to_string(depth)
                + ",position=" + std::to_string(index);
            const Measurement range = measure_fixed_with_history(
                v41, positions[index], depth, "v41_clean", sample);
            const Measurement scalar = measure_fixed_with_history(
                v43, positions[index], depth, "v43_clean", sample);
            add(v41_totals, range);
            add(v43_totals, scalar);
            const bool score_mismatch =
                range.result.score != scalar.result.score;
            const bool move_mismatch =
                range.result.best_move != scalar.result.best_move;
            score_mismatches += score_mismatch;
            move_mismatches += move_mismatch;
            std::cout
                << "clean_audit depth=" << depth
                << " position=" << index
                << " v41_move="
                << chess::move_to_string(range.result.best_move)
                << " v43_move="
                << chess::move_to_string(scalar.result.best_move)
                << " v41_score=" << range.result.score
                << " v43_score=" << scalar.result.score
                << " v41_nodes=" << range.result.nodes
                << " v43_nodes=" << scalar.result.nodes
                << " node_ratio="
                << ratio(scalar.result.nodes, range.result.nodes)
                << " score_mismatch=" << score_mismatch
                << " move_mismatch=" << move_mismatch
                << '\n';
        }
    }

    std::cout
        << "clean_audit_summary positions=" << positions.size()
        << " depths=" << Depths.size()
        << " searches_per_version=" << positions.size() * Depths.size()
        << " v41_nodes=" << v41_totals.nodes
        << " v43_nodes=" << v43_totals.nodes
        << " node_ratio=" << ratio(v43_totals.nodes, v41_totals.nodes)
        << " v41_wall_ms=" << milliseconds(v41_totals.nanoseconds)
        << " v43_wall_ms=" << milliseconds(v43_totals.nanoseconds)
        << " wall_ratio="
        << ratio(v43_totals.nanoseconds, v41_totals.nanoseconds)
        << " v41_nps=" << nps(v41_totals)
        << " v43_nps=" << nps(v43_totals)
        << " score_mismatches=" << score_mismatches
        << " move_mismatches=" << move_mismatches
        << '\n';
    if (score_mismatches != 0) {
        throw std::runtime_error(
            "V43 clean fixed-depth score differs from V41");
    }
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

        chess::NnueSearcherV41 v41(model, TtMegabytes);
        chess::NnueSearcherV42 v42(model, TtMegabytes);
        chess::NnueSearcherV43 v43(model, TtMegabytes);
        auto legacy_config = v41.aspiration_config();
        legacy_config.enabled = false;
        v41.set_aspiration_config(legacy_config);
        auto range_config = v42.aspiration_config();
        range_config.enabled = true;
        v42.set_aspiration_config(range_config);
        auto scalar_config = v43.aspiration_config();
        scalar_config.enabled = true;
        v43.set_aspiration_config(scalar_config);

        Totals v41_totals;
        Totals v42_totals;
        Totals v43_totals;
        AspirationTotals v42_aspiration;
        AspirationTotals v43_aspiration;
        MismatchTotals v42_v41;
        MismatchTotals v43_v41;
        MismatchTotals v43_v42;

        std::cout << std::fixed << std::setprecision(3);
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            for (std::size_t index = 0; index < positions.size(); ++index) {
                std::array<Measurement, 3> measurements;
                const std::size_t rotation =
                    (static_cast<std::size_t>(repeat) + index) % 3;
                for (std::size_t slot = 0; slot < 3; ++slot) {
                    const std::size_t version = (rotation + slot) % 3;
                    const std::string sample =
                        "repeat=" + std::to_string(repeat)
                        + ",position=" + std::to_string(index);
                    if (version == 0) {
                        measurements[0] = measure(
                            v41, positions[index], options.depth, "v41", sample);
                    } else if (version == 1) {
                        measurements[1] = measure(
                            v42, positions[index], options.depth, "v42", sample);
                    } else {
                        measurements[2] = measure(
                            v43, positions[index], options.depth, "v43", sample);
                    }
                }

                add(v41_totals, measurements[0]);
                add(v42_totals, measurements[1]);
                add(v43_totals, measurements[2]);
                v42_aspiration.add(v42.aspiration_stats());
                v43_aspiration.add(v43.aspiration_stats());
                compare(v42_v41, measurements[1], measurements[0]);
                compare(v43_v41, measurements[2], measurements[0]);
                compare(v43_v42, measurements[2], measurements[1]);

                const auto& old = measurements[0];
                const auto& range = measurements[1];
                const auto& scalar = measurements[2];
                std::cout
                    << "sample repeat=" << repeat
                    << " position=" << index
                    << " rotation=" << rotation
                    << " v41_move="
                    << chess::move_to_string(old.result.best_move)
                    << " v42_move="
                    << chess::move_to_string(range.result.best_move)
                    << " v43_move="
                    << chess::move_to_string(scalar.result.best_move)
                    << " v41_score=" << old.result.score
                    << " v42_score=" << range.result.score
                    << " v43_score=" << scalar.result.score
                    << " v41_nodes=" << old.result.nodes
                    << " v42_nodes=" << range.result.nodes
                    << " v43_nodes=" << scalar.result.nodes
                    << " v43_v41_node_ratio="
                    << ratio(scalar.result.nodes, old.result.nodes)
                    << " v43_v42_node_ratio="
                    << ratio(scalar.result.nodes, range.result.nodes)
                    << " v41_ms=" << milliseconds(old.nanoseconds)
                    << " v42_ms=" << milliseconds(range.nanoseconds)
                    << " v43_ms=" << milliseconds(scalar.nanoseconds)
                    << '\n';
            }
        }

        const double v41_nps = nps(v41_totals);
        const double v42_nps = nps(v42_totals);
        const double v43_nps = nps(v43_totals);
        std::cout
            << "summary depth=" << options.depth
            << " positions=" << positions.size()
            << " repeats=" << options.repeats
            << " searches_per_version="
            << positions.size() * static_cast<std::size_t>(options.repeats)
            << " tt_mb=" << TtMegabytes
            << " v41_nodes=" << v41_totals.nodes
            << " v42_nodes=" << v42_totals.nodes
            << " v43_nodes=" << v43_totals.nodes
            << " v43_v41_node_ratio="
            << ratio(v43_totals.nodes, v41_totals.nodes)
            << " v43_v42_node_ratio="
            << ratio(v43_totals.nodes, v42_totals.nodes)
            << " v41_wall_ms=" << milliseconds(v41_totals.nanoseconds)
            << " v42_wall_ms=" << milliseconds(v42_totals.nanoseconds)
            << " v43_wall_ms=" << milliseconds(v43_totals.nanoseconds)
            << " v43_v41_wall_ratio="
            << ratio(v43_totals.nanoseconds, v41_totals.nanoseconds)
            << " v43_v42_wall_ratio="
            << ratio(v43_totals.nanoseconds, v42_totals.nanoseconds)
            << " v41_nps=" << v41_nps
            << " v42_nps=" << v42_nps
            << " v43_nps=" << v43_nps
            << " v43_v41_nps_ratio="
            << (v41_nps == 0.0 ? 0.0 : v43_nps / v41_nps)
            << " v43_v42_nps_ratio="
            << (v42_nps == 0.0 ? 0.0 : v43_nps / v42_nps)
            << " v42_v41_score_mismatches=" << v42_v41.score
            << " v42_v41_move_mismatches=" << v42_v41.move
            << " v43_v41_score_mismatches=" << v43_v41.score
            << " v43_v41_move_mismatches=" << v43_v41.move
            << " v43_v42_score_mismatches=" << v43_v42.score
            << " v43_v42_move_mismatches=" << v43_v42.move
            << '\n';

        print_aspiration_summary("v42", v42_aspiration);
        print_aspiration_summary("v43", v43_aspiration);
        if (v43_aspiration.range_conflict_fallbacks != 0
            || v43_aspiration.unresolved_ranges != 0) {
            throw std::runtime_error(
                "scalar V43 reported a range conflict in timed corpus");
        }

        run_conflict_repros(model);
        run_clean_fixed_depth_audit(model, positions);
    } catch (const std::exception& error) {
        std::cerr << "benchmark_nnue_v43_single_bound: "
                  << error.what() << '\n';
        return 1;
    }
}
