#include "move.hpp"
#include "nnue_searcher_v43.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"
#include "v43_single_bound_transposition_table.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Searcher = chess::NnueSearcherV43;

constexpr std::size_t BytesPerMegabyte = 1024 * 1024;

// Keep this corpus in sync with the V42/V43 representation benchmarks.  A
// capacity result from a different position mix is not directly comparable.
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
    int warmup_depth = 4;
    std::string model = std::string(chess::DefaultPhaseQuantizedNnueModelPath);
    std::vector<std::size_t> tt_sizes_mb{64, 128, 256, 512};
    bool help = false;
};

struct Measurement {
    chess::SearchResult result{};
    std::uint64_t nanoseconds = 0;
    std::uint64_t clear_nanoseconds = 0;
};

struct Totals {
    std::uint64_t nodes = 0;
    std::uint64_t nanoseconds = 0;
};

struct ResultFingerprint {
    int score = 0;
    chess::Move move{};
};

struct SizeSummary {
    std::size_t tt_mb = 0;
    Totals search{};
    std::uint64_t clear_nanoseconds = 0;
    std::uint64_t construct_nanoseconds = 0;
    std::uint64_t score_mismatches = 0;
    std::uint64_t move_mismatches = 0;
    std::vector<ResultFingerprint> results;
};

struct BlockMeasurement {
    Totals search{};
    std::uint64_t clear_nanoseconds = 0;
    std::uint64_t construct_nanoseconds = 0;
    std::vector<ResultFingerprint> results;
};

void print_usage(std::ostream& output) {
    output
        << "usage: benchmark_nnue_v43_tt_sizes [options]\n"
        << "  --depth N          fixed maximum depth (default: 7)\n"
        << "  --repeats N        corpus repetitions (default: 2)\n"
        << "  --warmup-depth N   untimed warmup depth (default: 4)\n"
        << "  --tt-sizes CSV     power-of-two MiB sizes; first is baseline\n"
        << "                     (default: 64,128,256,512)\n"
        << "  --model PATH       phase-quantized NNUE model\n"
        << "  --help             show this message\n";
}

std::uint64_t parse_positive_u64(
    std::string_view text,
    std::string_view name
) {
    try {
        std::size_t parsed = 0;
        const unsigned long long value = std::stoull(std::string(text), &parsed);
        if (parsed != text.size() || value == 0) {
            throw std::invalid_argument("not positive");
        }
        return static_cast<std::uint64_t>(value);
    } catch (const std::exception&) {
        throw std::runtime_error(
            std::string(name) + " must be a positive integer: "
            + std::string(text));
    }
}

int parse_positive_int(std::string_view text, std::string_view name) {
    const std::uint64_t value = parse_positive_u64(text, name);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " is too large");
    }
    return static_cast<int>(value);
}

bool is_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::vector<std::size_t> parse_tt_sizes(std::string_view csv) {
    std::vector<std::size_t> result;
    std::size_t begin = 0;
    while (begin <= csv.size()) {
        const std::size_t comma = csv.find(',', begin);
        const std::size_t end = comma == std::string_view::npos
            ? csv.size()
            : comma;
        const std::string_view field = csv.substr(begin, end - begin);
        if (field.empty()) {
            throw std::runtime_error("--tt-sizes contains an empty field");
        }
        const std::uint64_t parsed = parse_positive_u64(field, "TT size");
        if (parsed > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("TT size is too large");
        }
        const std::size_t megabytes = static_cast<std::size_t>(parsed);
        // The V43 table rounds its bucket count down to a power of two.  Reject
        // aliases such as 96 MiB -> 64 MiB so the benchmark label cannot lie.
        if (!is_power_of_two(megabytes)) {
            throw std::runtime_error(
                "TT sizes must be powers of two in MiB: "
                + std::to_string(megabytes));
        }
        if (std::find(result.begin(), result.end(), megabytes) != result.end()) {
            throw std::runtime_error(
                "duplicate TT size: " + std::to_string(megabytes));
        }
        result.push_back(megabytes);
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1;
    }
    if (result.empty()) {
        throw std::runtime_error("--tt-sizes must contain at least one size");
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") {
            options.help = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::runtime_error(
                "missing value after " + std::string(argument));
        }
        const std::string_view value = argv[++index];
        if (argument == "--depth") {
            options.depth = parse_positive_int(value, "depth");
        } else if (argument == "--repeats") {
            options.repeats = parse_positive_int(value, "repeats");
        } else if (argument == "--warmup-depth") {
            options.warmup_depth = parse_positive_int(value, "warmup depth");
        } else if (argument == "--tt-sizes") {
            options.tt_sizes_mb = parse_tt_sizes(value);
        } else if (argument == "--model") {
            options.model = std::string(value);
        } else {
            throw std::runtime_error("unknown option: " + std::string(argument));
        }
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
    std::string_view label,
    std::string_view sample
) {
    if (result.stopped || result.depth != expected_depth
        || !is_legal(position, result.best_move)) {
        throw std::runtime_error(
            std::string(label) + " returned an invalid result on "
            + std::string(sample));
    }
}

Measurement measure(
    Searcher& searcher,
    const chess::Position& position,
    int depth,
    std::string_view label,
    std::string_view sample
) {
    // Cold-root mode isolates capacity/collision effects.  Clearing the table
    // is deliberately outside search wall time and reported independently;
    // otherwise a large table would lose mainly because memset is larger.
    searcher.clear_search_heuristics();
    searcher.clear_aspiration_stats();
    const auto clear_start = Clock::now();
    searcher.clear_tt();
    const auto clear_stop = Clock::now();
    const std::array<chess::HashKey, 1> history{position.zobrist_key};
    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = Clock::now();
    const chess::SearchResult result =
        searcher.search_best_move(position, limits, history);
    const auto stop = Clock::now();
    validate_result(position, result, depth, label, sample);
    return Measurement{
        result,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stop - start).count()),
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                clear_stop - clear_start).count()),
    };
}

void reset(Searcher& searcher) {
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    searcher.clear_aspiration_stats();
}

void warm_up(
    Searcher& searcher,
    const chess::Position& position,
    int depth,
    std::string_view label
) {
    reset(searcher);
    (void)measure(searcher, position, depth, label, "warmup");
}

std::size_t actual_tt_mb(const Searcher& searcher) {
    const std::size_t bytes = searcher.tt_entry_count()
        * sizeof(chess::V43SingleBoundTranspositionTable::Entry);
    if (bytes % BytesPerMegabyte != 0) {
        throw std::runtime_error("V43 TT byte count is not an integral MiB");
    }
    return bytes / BytesPerMegabyte;
}

void add(Totals& totals, const Measurement& measurement) {
    totals.nodes += measurement.result.nodes;
    totals.nanoseconds += measurement.nanoseconds;
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

BlockMeasurement run_size_block(
    const chess::PhaseQuantizedNnueModel& model,
    const std::array<chess::Position, Fens.size()>& positions,
    const Options& options,
    std::size_t tt_mb,
    int repeat,
    std::size_t order_slot
) {
    // Only one table exists at a time.  This avoids memory pressure becoming a
    // hidden disadvantage for the larger variants.
    const auto construct_start = Clock::now();
    auto searcher = std::make_unique<Searcher>(model, tt_mb);
    const auto construct_stop = Clock::now();
    if (actual_tt_mb(*searcher) != tt_mb) {
        throw std::runtime_error("requested and allocated TT sizes differ");
    }

    const int warmup_depth = std::min(options.depth, options.warmup_depth);
    warm_up(*searcher, positions[0], warmup_depth, "size_warmup");

    BlockMeasurement block;
    block.construct_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            construct_stop - construct_start).count());
    block.results.reserve(positions.size());
    for (std::size_t position_index = 0;
         position_index < positions.size();
         ++position_index) {
        const std::string sample =
            "tt_mb=" + std::to_string(tt_mb)
            + ",repeat=" + std::to_string(repeat)
            + ",position=" + std::to_string(position_index);
        const Measurement measurement = measure(
            *searcher, positions[position_index], options.depth,
            "v43", sample);
        add(block.search, measurement);
        block.clear_nanoseconds += measurement.clear_nanoseconds;
        block.results.push_back(ResultFingerprint{
            measurement.result.score,
            measurement.result.best_move,
        });
        std::cout
            << "sample tt_mb=" << tt_mb
            << " repeat=" << repeat
            << " order_slot=" << order_slot
            << " position=" << position_index
            << " nodes=" << measurement.result.nodes
            << " wall_ms=" << milliseconds(measurement.nanoseconds)
            << " move=" << chess::move_to_string(measurement.result.best_move)
            << " score=" << measurement.result.score
            << '\n';
    }
    return block;
}

void print_size_summary(
    const SizeSummary& summary,
    const SizeSummary& baseline,
    const Options& options
) {
    const double baseline_nps = nps(baseline.search);
    const double candidate_nps = nps(summary.search);
    std::cout
        << "size_summary baseline_mb=" << baseline.tt_mb
        << " candidate_mb=" << summary.tt_mb
        << " control=" << (baseline.tt_mb == summary.tt_mb)
        << " depth=" << options.depth
        << " repeats=" << options.repeats
        << " searches="
        << Fens.size() * static_cast<std::size_t>(options.repeats)
        << " baseline_nodes=" << baseline.search.nodes
        << " candidate_nodes=" << summary.search.nodes
        << " node_ratio="
        << ratio(summary.search.nodes, baseline.search.nodes)
        << " baseline_wall_ms=" << milliseconds(baseline.search.nanoseconds)
        << " candidate_wall_ms=" << milliseconds(summary.search.nanoseconds)
        << " wall_ratio="
        << ratio(summary.search.nanoseconds, baseline.search.nanoseconds)
        << " baseline_nps=" << baseline_nps
        << " candidate_nps=" << candidate_nps
        << " nps_ratio="
        << (baseline_nps == 0.0 ? 0.0 : candidate_nps / baseline_nps)
        << " clear_total_ms=" << milliseconds(summary.clear_nanoseconds)
        << " clear_avg_per_search_ms="
        << milliseconds(summary.clear_nanoseconds)
            / static_cast<double>(
                Fens.size() * static_cast<std::size_t>(options.repeats))
        << " construct_total_ms="
        << milliseconds(summary.construct_nanoseconds)
        << " score_mismatches=" << summary.score_mismatches
        << " move_mismatches=" << summary.move_mismatches
        << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_usage(std::cout);
            return 0;
        }

        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model: " + options.model);
        }
        model.set_neon_dotprod_enabled(true);
        const auto positions = load_positions();
        const std::size_t baseline_mb = options.tt_sizes_mb.front();

        std::cout << std::fixed << std::setprecision(3);
        std::cout
            << "config depth=" << options.depth
            << " repeats=" << options.repeats
            << " warmup_depth=" << std::min(options.depth, options.warmup_depth)
            << " positions=" << positions.size()
            << " baseline_mb=" << baseline_mb
            << " mode=one_size_block_cold_per_root"
            << " size_order=cyclic_rotation"
            << " history_heuristics=reset_per_search"
            << " clear_in_search_wall=false"
            << '\n';

        std::vector<SizeSummary> summaries(options.tt_sizes_mb.size());
        for (std::size_t index = 0; index < summaries.size(); ++index) {
            summaries[index].tt_mb = options.tt_sizes_mb[index];
            summaries[index].results.reserve(
                positions.size() * static_cast<std::size_t>(options.repeats));
        }
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            const std::size_t rotation =
                static_cast<std::size_t>(repeat) % summaries.size();
            for (std::size_t slot = 0; slot < summaries.size(); ++slot) {
                const std::size_t index = (rotation + slot) % summaries.size();
                BlockMeasurement block = run_size_block(
                    model, positions, options, summaries[index].tt_mb,
                    repeat, slot);
                summaries[index].search.nodes += block.search.nodes;
                summaries[index].search.nanoseconds += block.search.nanoseconds;
                summaries[index].clear_nanoseconds += block.clear_nanoseconds;
                summaries[index].construct_nanoseconds +=
                    block.construct_nanoseconds;
                summaries[index].results.insert(
                    summaries[index].results.end(),
                    block.results.begin(), block.results.end());
            }
        }

        const SizeSummary& baseline = summaries.front();
        for (SizeSummary& summary : summaries) {
            if (summary.results.size() != baseline.results.size()) {
                throw std::runtime_error("incomplete TT-size result matrix");
            }
            for (std::size_t index = 0; index < summary.results.size(); ++index) {
                summary.score_mismatches +=
                    summary.results[index].score != baseline.results[index].score;
                summary.move_mismatches +=
                    summary.results[index].move != baseline.results[index].move;
            }
            print_size_summary(summary, baseline, options);
        }

        std::vector<const SizeSummary*> ranking;
        ranking.reserve(summaries.size());
        for (const SizeSummary& summary : summaries) {
            ranking.push_back(&summary);
        }
        std::stable_sort(
            ranking.begin(), ranking.end(),
            [&baseline](const SizeSummary* lhs, const SizeSummary* rhs) {
                return ratio(
                           lhs->search.nanoseconds,
                           baseline.search.nanoseconds)
                    < ratio(
                           rhs->search.nanoseconds,
                           baseline.search.nanoseconds);
            });
        for (std::size_t index = 0; index < ranking.size(); ++index) {
            const SizeSummary& summary = *ranking[index];
            std::cout
                << "ranking rank=" << index + 1
                << " candidate_mb=" << summary.tt_mb
                << " wall_ratio_vs_" << baseline_mb << "mb="
                << ratio(
                    summary.search.nanoseconds,
                    baseline.search.nanoseconds)
                << " node_ratio="
                << ratio(summary.search.nodes, baseline.search.nodes)
                << " control=" << (summary.tt_mb == baseline_mb)
                << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark_nnue_v43_tt_sizes: "
                  << error.what() << '\n';
        return 1;
    }
}
