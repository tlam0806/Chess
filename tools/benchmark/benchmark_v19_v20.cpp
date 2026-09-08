#include "board_encoder.hpp"
#include "heuristic_searcher_v19.hpp"
#include "heuristic_searcher_v20.hpp"
#include "move.hpp"
#include "position.hpp"
#include "range_transposition_table.hpp"
#include "search_types.hpp"

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
    std::string input = "data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl";
    int samples = 100;
    int depth = 7;
    int tt_mb = 64;
    int bucket_size = 4;
    int qsearch_promotion_bonus = chess::HeuristicSearcherV20::MoveOrderingWeights{}.qsearch_promotion_bonus;
    int qsearch_good_capture_bonus = chess::HeuristicSearcherV20::MoveOrderingWeights{}.qsearch_good_capture_bonus;
    int qsearch_bad_capture_bonus = chess::HeuristicSearcherV20::MoveOrderingWeights{}.qsearch_bad_capture_bonus;
    int qsearch_see_weight = chess::HeuristicSearcherV20::MoveOrderingWeights{}.qsearch_see_weight;
    int qsearch_captured_value_weight =
        chess::HeuristicSearcherV20::MoveOrderingWeights{}.qsearch_captured_value_weight;
    bool iterative = true;
    bool first_samples = false;
    bool only_v20 = false;
    std::uint32_t seed = 20260614;
};

struct Sample {
    chess::Position pos;
    int target = 0;
    std::uint64_t source_index = 0;
};

struct Totals {
    std::uint64_t nodes = 0;
    std::uint64_t time_us = 0;
    int searched = 0;
    int score_mismatches = 0;
    int move_mismatches = 0;
};

struct TtTotals {
    std::uint64_t probes = 0;
    std::uint64_t empty_misses = 0;
    std::uint64_t index_collisions = 0;
    std::uint64_t key_hits = 0;
    std::uint64_t move_hint_hits = 0;
    std::uint64_t depth_misses = 0;
    std::uint64_t exact_hits = 0;
    std::uint64_t lower_score_hits = 0;
    std::uint64_t upper_score_hits = 0;
    std::uint64_t window_narrowings = 0;
    std::uint64_t score_returns = 0;
    std::uint64_t stores = 0;
    std::uint64_t new_stores = 0;
    std::uint64_t same_key_updates = 0;
    std::uint64_t replacement_collisions = 0;
    std::uint64_t skipped_shallow_replacements = 0;
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
        } else if (arg == "--tt-mb") {
            options.tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--bucket-size") {
            options.bucket_size = parse_int(require_value(arg), arg);
        } else if (arg == "--qsearch-promotion-bonus") {
            options.qsearch_promotion_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--qsearch-good-capture-bonus") {
            options.qsearch_good_capture_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--qsearch-bad-capture-bonus") {
            options.qsearch_bad_capture_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--qsearch-see-weight") {
            options.qsearch_see_weight = parse_int(require_value(arg), arg);
        } else if (arg == "--qsearch-captured-value-weight") {
            options.qsearch_captured_value_weight = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--fixed-depth") {
            options.iterative = false;
        } else if (arg == "--iterative") {
            options.iterative = true;
        } else if (arg == "--first") {
            options.first_samples = true;
        } else if (arg == "--only-v20") {
            options.only_v20 = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v19_v20_bucket_tt [--input path]\n"
                << "                                     [--samples N]\n"
                << "                                     [--depth D]\n"
                << "                                     [--tt-mb MB]\n"
                << "                                     [--bucket-size N]\n"
                << "                                     [--qsearch-promotion-bonus N]\n"
                << "                                     [--qsearch-good-capture-bonus N]\n"
                << "                                     [--qsearch-bad-capture-bonus N]\n"
                << "                                     [--qsearch-see-weight N]\n"
                << "                                     [--qsearch-captured-value-weight N]\n"
                << "                                     [--seed N]\n"
                << "                                     [--iterative|--fixed-depth]\n"
                << "                                     [--first]\n"
                << "                                     [--only-v20]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.samples <= 0 || options.depth < 0 || options.tt_mb <= 0 || options.bucket_size <= 0) {
        throw std::runtime_error("samples, tt-mb, and bucket-size must be positive; depth must be non-negative");
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

int extract_target(const std::string& line) {
    constexpr std::string_view key = "\"target\":";
    const std::size_t key_pos = line.find(key);
    if (key_pos == std::string::npos) {
        return 0;
    }
    std::size_t start = key_pos + key.size();
    std::size_t end = start;
    while (end < line.size() && line[end] != ',' && line[end] != '}') {
        ++end;
    }
    return parse_int(std::string_view(line).substr(start, end - start), "target");
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

    sample.pos = pos;
    sample.target = extract_target(line);
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
        Sample sample;
        if (!decode_sample(line, sample)) {
            continue;
        }
        sample.source_index = line_index;
        ++valid_seen;

        if (options.first_samples) {
            reservoir.push_back(sample);
            if (static_cast<int>(reservoir.size()) >= options.samples) {
                break;
            }
            continue;
        }

        if (static_cast<int>(reservoir.size()) < options.samples) {
            reservoir.push_back(sample);
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

template <typename Searcher>
chess::SearchResult search(Searcher& searcher, const chess::Position& pos, int depth, bool iterative) {
    if (iterative) {
        return searcher.search_best_move(pos, chess::SearchLimits{
            .max_depth = depth,
            .move_time = std::chrono::milliseconds{0},
        });
    }
    return searcher.search_best_move(pos, depth);
}

void add_result(Totals& totals, const chess::SearchResult& result, std::uint64_t time_us) {
    ++totals.searched;
    totals.nodes += result.nodes;
    totals.time_us += time_us;
}

void add_tt_stats(TtTotals& totals, const chess::RangeTranspositionTableStats& stats) {
    totals.probes += stats.probes;
    totals.empty_misses += stats.empty_misses;
    totals.index_collisions += stats.index_collisions;
    totals.key_hits += stats.key_hits;
    totals.move_hint_hits += stats.move_hint_hits;
    totals.depth_misses += stats.depth_misses;
    totals.exact_hits += stats.exact_hits;
    totals.lower_score_hits += stats.lower_score_hits;
    totals.upper_score_hits += stats.upper_score_hits;
    totals.window_narrowings += stats.window_narrowings;
    totals.score_returns += stats.score_returns;
    totals.stores += stats.stores;
    totals.new_stores += stats.new_stores;
    totals.same_key_updates += stats.same_key_updates;
    totals.replacement_collisions += stats.replacement_collisions;
    totals.skipped_shallow_replacements += stats.skipped_shallow_replacements;
}

std::uint64_t avg_u64(std::uint64_t value, int count) {
    return count == 0 ? 0 : value / static_cast<std::uint64_t>(count);
}

double ratio(std::uint64_t value, std::uint64_t total) {
    return total == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(total);
}

void print_tt_summary(std::string_view prefix, const TtTotals& stats) {
    std::cout << prefix
              << "_tt probes=" << stats.probes
              << " key_hits=" << stats.key_hits
              << " key_hit_rate=" << ratio(stats.key_hits, stats.probes)
              << " move_hint_hits=" << stats.move_hint_hits
              << " move_hint_hit_rate=" << ratio(stats.move_hint_hits, stats.probes)
              << " index_collisions=" << stats.index_collisions
              << " index_collision_rate=" << ratio(stats.index_collisions, stats.probes)
              << " empty_misses=" << stats.empty_misses
              << " empty_miss_rate=" << ratio(stats.empty_misses, stats.probes)
              << " depth_misses=" << stats.depth_misses
              << " exact_hits=" << stats.exact_hits
              << " lower_score_hits=" << stats.lower_score_hits
              << " upper_score_hits=" << stats.upper_score_hits
              << " window_narrowings=" << stats.window_narrowings
              << " score_returns=" << stats.score_returns
              << " stores=" << stats.stores
              << " new_stores=" << stats.new_stores
              << " same_key_updates=" << stats.same_key_updates
              << " replacement_collisions=" << stats.replacement_collisions
              << " replacement_collision_rate=" << ratio(stats.replacement_collisions, stats.stores)
              << " skipped_shallow_replacements=" << stats.skipped_shallow_replacements
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_random_samples(options);

        Totals v19_totals;
        Totals v20_totals;
        TtTotals v19_tt_totals;
        TtTotals v20_tt_totals;

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " depth=" << options.depth
                  << " tt_mb=" << options.tt_mb
                  << " bucket_size=" << options.bucket_size
                  << " qsearch_promotion_bonus=" << options.qsearch_promotion_bonus
                  << " qsearch_good_capture_bonus=" << options.qsearch_good_capture_bonus
                  << " qsearch_bad_capture_bonus=" << options.qsearch_bad_capture_bonus
                  << " qsearch_see_weight=" << options.qsearch_see_weight
                  << " qsearch_captured_value_weight=" << options.qsearch_captured_value_weight
                  << " iterative=" << (options.iterative ? 1 : 0)
                  << " only_v20=" << (options.only_v20 ? 1 : 0)
                  << " first=" << (options.first_samples ? 1 : 0)
                  << " seed=" << options.seed << '\n' << std::flush;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const Sample& sample = samples[i];
            chess::HeuristicSearcherV19 v19(
                static_cast<std::size_t>(options.tt_mb),
                static_cast<std::size_t>(options.bucket_size)
            );
            chess::HeuristicSearcherV20::MoveOrderingWeights weights;
            weights.qsearch_promotion_bonus = options.qsearch_promotion_bonus;
            weights.qsearch_good_capture_bonus = options.qsearch_good_capture_bonus;
            weights.qsearch_bad_capture_bonus = options.qsearch_bad_capture_bonus;
            weights.qsearch_see_weight = options.qsearch_see_weight;
            weights.qsearch_captured_value_weight = options.qsearch_captured_value_weight;
            chess::HeuristicSearcherV20 v20(
                static_cast<std::size_t>(options.tt_mb),
                static_cast<std::size_t>(options.bucket_size),
                3,
                7,
                weights.counter_history_bonus,
                weights
            );

            chess::SearchResult v19_result;
            std::uint64_t v19_us = 0;
            if (!options.only_v20) {
                const auto v19_start = std::chrono::steady_clock::now();
                v19_result = search(v19, sample.pos, options.depth, options.iterative);
                const auto v19_end = std::chrono::steady_clock::now();
                v19_us = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(v19_end - v19_start).count()
                );
            }

            const auto v20_start = std::chrono::steady_clock::now();
            const chess::SearchResult v20_result = search(v20, sample.pos, options.depth, options.iterative);
            const auto v20_end = std::chrono::steady_clock::now();

            const auto v20_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(v20_end - v20_start).count()
            );

            if (!options.only_v20) {
                add_result(v19_totals, v19_result, v19_us);
                add_tt_stats(v19_tt_totals, v19.tt_stats());
            }
            add_result(v20_totals, v20_result, v20_us);
            add_tt_stats(v20_tt_totals, v20.tt_stats());
            if (!options.only_v20 && v19_result.score != v20_result.score) {
                ++v19_totals.score_mismatches;
                ++v20_totals.score_mismatches;
            }
            if (!options.only_v20 && v19_result.best_move != v20_result.best_move) {
                ++v19_totals.move_mismatches;
                ++v20_totals.move_mismatches;
            }

            std::cout << "sample=" << i
                      << " line=" << sample.source_index
                      << " target=" << sample.target
                      << " legal=" << chess::generate_legal_moves(sample.pos).size();
            if (!options.only_v20) {
                std::cout << " v19_nodes=" << v19_result.nodes
                          << " v19_us=" << v19_us
                          << " v19_score=" << v19_result.score
                          << " v19_best=" << chess::move_to_string(v19_result.best_move);
            }
            std::cout
                      << " v20_nodes=" << v20_result.nodes
                      << " v20_us=" << v20_us
                      << " v20_score=" << v20_result.score
                      << " v20_best=" << chess::move_to_string(v20_result.best_move);
            if (!options.only_v20 && v19_result.nodes != 0) {
                std::cout << " v20_vs_v19_nodes="
                          << (static_cast<double>(v20_result.nodes) / static_cast<double>(v19_result.nodes));
            }
            if (!options.only_v20 && v19_us != 0) {
                std::cout << " v20_vs_v19_time="
                          << (static_cast<double>(v20_us) / static_cast<double>(v19_us));
            }
            std::cout << '\n' << std::flush;
        }

        std::cout << "summary"
                  << " v19_nodes=" << v19_totals.nodes
                  << " v19_avg_nodes=" << avg_u64(v19_totals.nodes, v19_totals.searched)
                  << " v19_us=" << v19_totals.time_us
                  << " v19_avg_us=" << avg_u64(v19_totals.time_us, v19_totals.searched)
                  << " v20_nodes=" << v20_totals.nodes
                  << " v20_avg_nodes=" << avg_u64(v20_totals.nodes, v20_totals.searched)
                  << " v20_us=" << v20_totals.time_us
                  << " v20_avg_us=" << avg_u64(v20_totals.time_us, v20_totals.searched);
        if (v19_totals.nodes != 0) {
            std::cout << " v20_vs_v19_nodes="
                      << (static_cast<double>(v20_totals.nodes) / static_cast<double>(v19_totals.nodes));
        }
        if (v19_totals.time_us != 0) {
            std::cout << " v20_vs_v19_time="
                      << (static_cast<double>(v20_totals.time_us) / static_cast<double>(v19_totals.time_us));
        }
        std::cout << " score_mismatches=" << v19_totals.score_mismatches
                  << " move_mismatches=" << v19_totals.move_mismatches
                  << '\n';
        if (!options.only_v20) {
            print_tt_summary("v19", v19_tt_totals);
        }
        print_tt_summary("v20", v20_tt_totals);
    } catch (const std::exception& error) {
        std::cerr << "benchmark_v19_v20_bucket_tt: " << error.what() << '\n';
        return 1;
    }
}
