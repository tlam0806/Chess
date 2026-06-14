#include "board_encoder.hpp"
#include "heuristic_searcher_v14_experimental.hpp"
#include "position.hpp"
#include "search_types.hpp"
#include "zobrist.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string input = "data/nnue_mix_train_600k.jsonl";
    int samples = 12;
    int depth = 8;
    std::uint32_t seed = 20260613;
    std::uint64_t line = 0;
    std::string variant;
    bool fixed = false;
};

struct Sample {
    chess::Position pos;
    std::uint64_t source_index = 0;
};

struct Totals {
    std::uint64_t nodes = 0;
    std::uint64_t pvs_scouts = 0;
    std::uint64_t pvs_researches = 0;
    std::uint64_t pvs_scout_cutoffs = 0;
    double millis = 0.0;
    int searched = 0;
};

struct Variant {
    const char* name = "";
    chess::V14ExperimentalSearchOptions options{};
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

        if (arg == "--input") {
            options.input = std::string(require_value(arg));
        } else if (arg == "--samples") {
            options.samples = parse_int(require_value(arg), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--line") {
            const int parsed = parse_int(require_value(arg), arg);
            if (parsed < 0) {
                throw std::runtime_error("line must be non-negative");
            }
            options.line = static_cast<std::uint64_t>(parsed);
        } else if (arg == "--variant") {
            options.variant = std::string(require_value(arg));
        } else if (arg == "--fixed") {
            options.fixed = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v14_pvs [--input path] [--samples N] [--depth D] [--seed N] [--line N] [--variant name] [--fixed]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.samples <= 0 || options.depth < 0) {
        throw std::runtime_error("samples must be positive and depth must be non-negative");
    }
    return options;
}

std::vector<int> extract_int_array(const std::string& line, std::string_view key) {
    const std::size_t key_pos = line.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("sample is missing " + std::string(key));
    }

    const std::size_t open = line.find('[', key_pos);
    const std::size_t close = line.find(']', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("invalid array for " + std::string(key));
    }

    std::vector<int> values;
    std::size_t pos = open + 1;
    while (pos < close) {
        while (pos < close && (line[pos] == ' ' || line[pos] == ',')) {
            ++pos;
        }
        if (pos >= close) {
            break;
        }
        std::size_t end = pos;
        while (end < close && line[end] != ',') {
            ++end;
        }
        values.push_back(parse_int(std::string_view(line).substr(pos, end - pos), std::string(key)));
        pos = end + 1;
    }
    return values;
}

bool decode_sample(const std::string& line, Sample& sample) {
    const std::vector<int> aux_values = extract_int_array(line, "\"aux\"");
    if (aux_values.size() != chess::AuxFeatureCount) {
        throw std::runtime_error("unexpected aux feature count");
    }
    if (aux_values[chess::HasEnPassant] != 0) {
        return false;
    }

    chess::Position pos;
    pos.clear();
    pos.side_to_move = chess::Color::White;
    pos.white_can_castle_kingside = aux_values[chess::FriendlyCanCastleKingside] != 0;
    pos.white_can_castle_queenside = aux_values[chess::FriendlyCanCastleQueenside] != 0;
    pos.black_can_castle_kingside = aux_values[chess::EnemyCanCastleKingside] != 0;
    pos.black_can_castle_queenside = aux_values[chess::EnemyCanCastleQueenside] != 0;

    std::array<bool, chess::EncodedFeatureCount> seen{};
    const std::vector<int> features = extract_int_array(line, "\"features\"");
    for (int raw_feature : features) {
        if (raw_feature < 0 || raw_feature >= chess::EncodedFeatureCount) {
            throw std::runtime_error("feature index out of range");
        }

        chess::FeatureIndex index = static_cast<chess::FeatureIndex>(raw_feature);
        const chess::Square piece_square = static_cast<chess::Square>(index % chess::EncoderSquares);
        index /= chess::EncoderSquares;
        index /= chess::EncoderSquares;
        const auto king_context = static_cast<chess::EncodedKingContext>(index % chess::EncoderKingContexts);
        index /= chess::EncoderKingContexts;
        const auto piece_side = static_cast<chess::EncodedPieceSide>(index % chess::EncoderPieceSides);
        index /= chess::EncoderPieceSides;
        const auto piece = static_cast<chess::PieceType>(index);

        if (king_context != chess::EncodedKingContext::FriendlyKing || seen[raw_feature]) {
            continue;
        }
        seen[raw_feature] = true;

        const chess::Color color = piece_side == chess::EncodedPieceSide::Friendly
            ? chess::Color::White
            : chess::Color::Black;
        if (!pos.is_empty(piece_square)) {
            throw std::runtime_error("decoded duplicate piece square");
        }
        pos.set_piece(color, piece, piece_square);
    }

    if (chess::popcount(pos.occupancy(chess::Color::White, chess::PieceType::King)) != 1
        || chess::popcount(pos.occupancy(chess::Color::Black, chess::PieceType::King)) != 1) {
        return false;
    }

    pos.zobrist_key = chess::zobrist::compute_hash(pos);
    sample.pos = pos;
    return true;
}

std::vector<Sample> load_random_samples(const Options& options) {
    std::ifstream input(options.input);
    if (!input) {
        throw std::runtime_error("failed to open input: " + options.input);
    }

    std::mt19937 rng(options.seed);
    std::vector<Sample> reservoir;
    reservoir.reserve(static_cast<std::size_t>(options.samples));

    std::uint64_t valid_seen = 0;
    std::uint64_t line_index = 0;
    std::string line;
    while (std::getline(input, line)) {
        ++line_index;
        if (options.line != 0 && line_index != options.line) {
            continue;
        }

        Sample sample;
        if (!decode_sample(line, sample)) {
            continue;
        }
        sample.source_index = line_index;
        ++valid_seen;

        if (static_cast<int>(reservoir.size()) < options.samples) {
            reservoir.push_back(sample);
            if (options.line != 0) {
                break;
            }
            continue;
        }

        std::uniform_int_distribution<std::uint64_t> dist(0, valid_seen - 1);
        const std::uint64_t selected = dist(rng);
        if (selected < static_cast<std::uint64_t>(options.samples)) {
            reservoir[static_cast<std::size_t>(selected)] = sample;
        }
    }

    if (reservoir.empty()) {
        throw std::runtime_error("no decodable samples found");
    }
    return reservoir;
}

chess::SearchResult iterative_search(chess::HeuristicSearcherV14Experimental& searcher, const chess::Position& pos, int depth) {
    return searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = depth,
        .move_time = std::chrono::milliseconds{0},
    });
}

chess::SearchResult run_search(
    chess::HeuristicSearcherV14Experimental& searcher,
    const chess::Position& pos,
    int depth,
    bool fixed
) {
    if (fixed) {
        return searcher.search_best_move(pos, depth);
    }
    return iterative_search(searcher, pos, depth);
}

Totals run_variant(
    const std::vector<Sample>& samples,
    int depth,
    bool fixed,
    const chess::V14ExperimentalSearchOptions& options,
    std::vector<chess::SearchResult>* out
) {
    Totals totals;

    for (const Sample& sample : samples) {
        chess::HeuristicSearcherV14Experimental searcher(64, {}, options);
        searcher.clear_move_ordering_stats();

        const auto start = std::chrono::steady_clock::now();
        const chess::SearchResult result = run_search(searcher, sample.pos, depth, fixed);
        const auto stop = std::chrono::steady_clock::now();

        const auto stats = searcher.move_ordering_stats();
        totals.nodes += result.nodes;
        totals.pvs_scouts += stats.pvs_scouts;
        totals.pvs_researches += stats.pvs_researches;
        totals.pvs_scout_cutoffs += stats.cutoff_by_pvs_scout;
        totals.millis += std::chrono::duration<double, std::milli>(stop - start).count();
        ++totals.searched;

        if (out != nullptr) {
            out->push_back(result);
        }
    }

    return totals;
}

double ratio(double numerator, double denominator) {
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

double percent(std::uint64_t value, std::uint64_t total) {
    return total == 0 ? 0.0 : 100.0 * static_cast<double>(value) / static_cast<double>(total);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_random_samples(options);

        std::cout
            << "benchmark_v14_pvs"
            << " input=" << options.input
            << " samples=" << samples.size()
            << " depth=" << options.depth
            << " seed=" << options.seed
            << " fixed=" << (options.fixed ? 1 : 0)
            << '\n';

        chess::V14ExperimentalSearchOptions pvs_safe;
        pvs_safe.enable_pvs = true;

        chess::V14ExperimentalSearchOptions pvs_tt_off;
        pvs_tt_off.enable_pvs = true;
        pvs_tt_off.enable_pvs_skip_fail_low = true;
        pvs_tt_off.enable_tt = false;

        chess::V14ExperimentalSearchOptions pvs_exact_only;
        pvs_exact_only.enable_pvs = true;
        pvs_exact_only.enable_pvs_skip_fail_low = true;
        pvs_exact_only.enable_tt = true;
        pvs_exact_only.enable_tt_bounds = false;

        chess::V14ExperimentalSearchOptions pvs_bounds_no_tighten;
        pvs_bounds_no_tighten.enable_pvs = true;
        pvs_bounds_no_tighten.enable_pvs_skip_fail_low = true;
        pvs_bounds_no_tighten.enable_tt = true;
        pvs_bounds_no_tighten.enable_tt_bounds = true;
        pvs_bounds_no_tighten.enable_tt_bound_cutoff = false;
        pvs_bounds_no_tighten.enable_tt_bound_tighten = false;

        chess::V14ExperimentalSearchOptions pvs_bounds_tighten_only = pvs_bounds_no_tighten;
        pvs_bounds_tighten_only.enable_tt_bound_tighten = true;

        chess::V14ExperimentalSearchOptions pvs_bounds_cutoff_only = pvs_bounds_no_tighten;
        pvs_bounds_cutoff_only.enable_tt_bound_cutoff = true;

        chess::V14ExperimentalSearchOptions pvs_full;
        pvs_full.enable_pvs = true;
        pvs_full.enable_pvs_skip_fail_low = true;

        const Variant variants[] = {
            {"pvs_safe_always_research", pvs_safe},
            {"pvs_tt_off", pvs_tt_off},
            {"pvs_tt_exact_only", pvs_exact_only},
            {"pvs_tt_bounds_no_tighten", pvs_bounds_no_tighten},
            {"pvs_tt_bounds_tighten_only", pvs_bounds_tighten_only},
            {"pvs_tt_bounds_cutoff_only", pvs_bounds_cutoff_only},
            {"pvs_tt_full", pvs_full},
        };

        for (const Variant& variant : variants) {
            if (!options.variant.empty() && options.variant != variant.name) {
                continue;
            }

            chess::V14ExperimentalSearchOptions control_options = variant.options;
            control_options.enable_pvs = false;
            control_options.enable_pvs_skip_fail_low = false;

            std::vector<chess::SearchResult> baseline_results;
            baseline_results.reserve(samples.size());
            const Totals baseline = run_variant(samples, options.depth, options.fixed, control_options, &baseline_results);

            std::vector<chess::SearchResult> results;
            results.reserve(samples.size());
            const Totals totals = run_variant(samples, options.depth, options.fixed, variant.options, &results);

            int score_mismatches = 0;
            int move_mismatches = 0;
            for (std::size_t i = 0; i < samples.size(); ++i) {
                if (results[i].score != baseline_results[i].score) {
                    ++score_mismatches;
                    if (score_mismatches <= 5) {
                        std::cout
                            << "score_mismatch"
                            << " variant=" << variant.name
                            << " sample=" << i
                            << " source_line=" << samples[i].source_index
                            << " baseline_score=" << baseline_results[i].score
                            << " score=" << results[i].score
                            << " baseline_move=" << baseline_results[i].best_move.value
                            << " move=" << results[i].best_move.value
                            << '\n';
                    }
                }
                if (results[i].best_move != baseline_results[i].best_move) {
                    ++move_mismatches;
                }
            }

            std::cout
                << "variant=" << variant.name
                << " baseline_nodes=" << baseline.nodes
                << " baseline_ms=" << baseline.millis
                << " nodes=" << totals.nodes
                << " avg_nodes=" << (totals.nodes / static_cast<std::uint64_t>(totals.searched))
                << " ms=" << totals.millis
                << " avg_ms=" << (totals.millis / totals.searched)
                << " scouts=" << totals.pvs_scouts
                << " researches=" << totals.pvs_researches
                << " research_rate_pct=" << percent(totals.pvs_researches, totals.pvs_scouts)
                << " scout_cutoffs=" << totals.pvs_scout_cutoffs
                << " node_ratio=" << ratio(static_cast<double>(totals.nodes), static_cast<double>(baseline.nodes))
                << " time_ratio=" << ratio(totals.millis, baseline.millis)
                << " score_mismatches=" << score_mismatches
                << " move_mismatches=" << move_mismatches
                ;
            if (samples.size() == 1) {
                std::cout
                    << " baseline_score=" << baseline_results.front().score
                    << " score=" << results.front().score
                    << " baseline_move=" << baseline_results.front().best_move.value
                    << " move=" << results.front().best_move.value;
            }
            std::cout << '\n';
        }

        if (!options.variant.empty()) {
            bool found = false;
            for (const Variant& variant : variants) {
                if (options.variant == variant.name) {
                    found = true;
                }
            }
            if (!found) {
                throw std::runtime_error("unknown variant: " + options.variant);
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
