#include "board_encoder.hpp"
#include "heuristic_searcher_v23.hpp"
#include "heuristic_searcher_v24.hpp"
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
    int history_penalty_num = 9;
    int history_penalty_den = 14;
    int tt_lower_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.tt_lower_bonus;
    int tt_upper_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.tt_upper_bonus;
    int promotion_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.promotion_bonus;
    int good_capture_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.good_capture_bonus;
    int bad_capture_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.bad_capture_bonus;
    int see_weight = chess::HeuristicSearcherV24::MoveOrderingWeights{}.see_weight;
    int killer1_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.killer1_bonus;
    int killer2_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.killer2_bonus;
    int check_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.check_bonus;
    int counter_history_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.counter_history_bonus;
    int bad_capture_stage_threshold =
        chess::HeuristicSearcherV24::MoveOrderingWeights{}.bad_capture_stage_threshold;
    int qsearch_promotion_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.qsearch_promotion_bonus;
    int qsearch_good_capture_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.qsearch_good_capture_bonus;
    int qsearch_bad_capture_bonus = chess::HeuristicSearcherV24::MoveOrderingWeights{}.qsearch_bad_capture_bonus;
    int qsearch_see_weight = chess::HeuristicSearcherV24::MoveOrderingWeights{}.qsearch_see_weight;
    int qsearch_captured_value_weight =
        chess::HeuristicSearcherV24::MoveOrderingWeights{}.qsearch_captured_value_weight;
    bool iterative = true;
    bool first_samples = false;
    bool only_v24 = false;
    bool instrument_ordering = false;
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

void add_move_type_stats(
    chess::HeuristicSearcherV24::MoveTypeStats& totals,
    const chess::HeuristicSearcherV24::MoveTypeStats& stats
) {
    totals.total += stats.total;
    totals.tt_lower += stats.tt_lower;
    totals.tt_upper += stats.tt_upper;
    totals.promotion += stats.promotion;
    totals.capture += stats.capture;
    totals.quiet += stats.quiet;
    totals.check += stats.check;
    totals.killer1 += stats.killer1;
    totals.killer2 += stats.killer2;
    totals.history_positive += stats.history_positive;
    totals.counter_history_positive += stats.counter_history_positive;
}

void add_cutoff_stats(
    chess::HeuristicSearcherV24::MoveCutoffStats& totals,
    const chess::HeuristicSearcherV24::MoveCutoffStats& stats
) {
    totals.nodes_with_moves += stats.nodes_with_moves;
    totals.beta_cutoffs += stats.beta_cutoffs;
    totals.cutoff_index_sum += stats.cutoff_index_sum;
    for (std::size_t i = 0; i < totals.cutoff_index_buckets.size(); ++i) {
        totals.cutoff_index_buckets[i] += stats.cutoff_index_buckets[i];
    }
    add_move_type_stats(totals.before_cutoff, stats.before_cutoff);
    add_move_type_stats(totals.cutoff_move, stats.cutoff_move);
    add_move_type_stats(totals.after_cutoff, stats.after_cutoff);
}

void add_ordering_stats(
    chess::HeuristicSearcherV24::MoveOrderingStats& totals,
    const chess::HeuristicSearcherV24::MoveOrderingStats& stats
) {
    add_cutoff_stats(totals.main, stats.main);
    add_cutoff_stats(totals.qsearch, stats.qsearch);
    add_move_type_stats(totals.noisy_stage, stats.noisy_stage);
    add_move_type_stats(totals.quiet_stage, stats.quiet_stage);
    for (std::size_t i = 0; i < totals.priority_stage_top.size(); ++i) {
        add_move_type_stats(totals.priority_stage_top[i], stats.priority_stage_top[i]);
    }
    totals.tt_lower_stage0_attempts += stats.tt_lower_stage0_attempts;
    totals.tt_lower_stage0_cutoffs += stats.tt_lower_stage0_cutoffs;
    totals.tt_lower_king_safety_contexts += stats.tt_lower_king_safety_contexts;
    totals.tt_lower_legal_checks += stats.tt_lower_legal_checks;
    totals.priority_king_safety_contexts += stats.priority_king_safety_contexts;
    totals.priority_legal_checks += stats.priority_legal_checks;
    totals.quiet_king_safety_contexts += stats.quiet_king_safety_contexts;
    totals.quiet_legal_checks += stats.quiet_legal_checks;
    totals.qsearch_king_safety_contexts += stats.qsearch_king_safety_contexts;
    totals.qsearch_legal_checks += stats.qsearch_legal_checks;
}

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
        } else if (arg == "--history-penalty-num") {
            options.history_penalty_num = parse_int(require_value(arg), arg);
        } else if (arg == "--history-penalty-den") {
            options.history_penalty_den = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-lower-bonus") {
            options.tt_lower_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-upper-bonus") {
            options.tt_upper_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--promotion-bonus") {
            options.promotion_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--good-capture-bonus") {
            options.good_capture_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--bad-capture-bonus") {
            options.bad_capture_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--see-weight") {
            options.see_weight = parse_int(require_value(arg), arg);
        } else if (arg == "--killer1-bonus") {
            options.killer1_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--killer2-bonus") {
            options.killer2_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--check-bonus") {
            options.check_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--counter-history-bonus") {
            options.counter_history_bonus = parse_int(require_value(arg), arg);
        } else if (arg == "--bad-capture-stage-threshold") {
            options.bad_capture_stage_threshold = parse_int(require_value(arg), arg);
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
        } else if (arg == "--only-v24") {
            options.only_v24 = true;
        } else if (arg == "--instrument-ordering") {
            options.instrument_ordering = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v23_v24_bucket_tt [--input path]\n"
                << "                                     [--samples N]\n"
                << "                                     [--depth D]\n"
                << "                                     [--tt-mb MB]\n"
                << "                                     [--bucket-size N]\n"
                << "                                     [--history-penalty-num N]\n"
                << "                                     [--history-penalty-den N]\n"
                << "                                     [--tt-lower-bonus N]\n"
                << "                                     [--tt-upper-bonus N]\n"
                << "                                     [--promotion-bonus N]\n"
                << "                                     [--good-capture-bonus N]\n"
                << "                                     [--bad-capture-bonus N]\n"
                << "                                     [--see-weight N]\n"
                << "                                     [--killer1-bonus N]\n"
                << "                                     [--killer2-bonus N]\n"
                << "                                     [--check-bonus N]\n"
                << "                                     [--counter-history-bonus N]\n"
                << "                                     [--bad-capture-stage-threshold N]\n"
                << "                                     [--qsearch-promotion-bonus N]\n"
                << "                                     [--qsearch-good-capture-bonus N]\n"
                << "                                     [--qsearch-bad-capture-bonus N]\n"
                << "                                     [--qsearch-see-weight N]\n"
                << "                                     [--qsearch-captured-value-weight N]\n"
                << "                                     [--seed N]\n"
                << "                                     [--iterative|--fixed-depth]\n"
                << "                                     [--first]\n"
                << "                                     [--only-v24]\n"
                << "                                     [--instrument-ordering]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.samples <= 0 || options.depth < 0 || options.tt_mb <= 0 || options.bucket_size <= 0
        || options.history_penalty_num <= 0 || options.history_penalty_den <= 0) {
        throw std::runtime_error("samples, tt-mb, bucket-size, and history penalty values must be positive; depth must be non-negative");
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

chess::HeuristicSearcherV24::MoveTypeStats total_appeared_move_types(
    const chess::HeuristicSearcherV24::MoveCutoffStats& stats
) {
    chess::HeuristicSearcherV24::MoveTypeStats totals;
    add_move_type_stats(totals, stats.before_cutoff);
    add_move_type_stats(totals, stats.cutoff_move);
    add_move_type_stats(totals, stats.after_cutoff);
    return totals;
}

void print_type_rate(
    std::string_view prefix,
    std::string_view name,
    std::uint64_t cutoff,
    std::uint64_t appeared
) {
    const std::uint64_t non_cutoff = appeared >= cutoff ? appeared - cutoff : 0;
    std::cout << prefix
              << " type=" << name
              << " cutoff=" << cutoff
              << " appeared=" << appeared
              << " cutoff_rate=" << ratio(cutoff, appeared)
              << " non_cutoff=" << non_cutoff
              << " non_cutoff_rate=" << ratio(non_cutoff, appeared)
              << '\n';
}

void print_cutoff_profile(
    std::string_view prefix,
    const chess::HeuristicSearcherV24::MoveCutoffStats& stats
) {
    const chess::HeuristicSearcherV24::MoveTypeStats appeared = total_appeared_move_types(stats);
    const chess::HeuristicSearcherV24::MoveTypeStats& cutoff = stats.cutoff_move;
    std::cout << prefix
              << " nodes_with_moves=" << stats.nodes_with_moves
              << " beta_cutoffs=" << stats.beta_cutoffs
              << " cutoff_rate=" << ratio(stats.beta_cutoffs, stats.nodes_with_moves)
              << " avg_cutoff_index=" << ratio(stats.cutoff_index_sum, stats.beta_cutoffs)
              << " cutoff_index_buckets=";
    for (std::size_t i = 0; i < stats.cutoff_index_buckets.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        if (i + 1 == stats.cutoff_index_buckets.size()) {
            std::cout << "15+=" << stats.cutoff_index_buckets[i];
        } else {
            std::cout << i << '=' << stats.cutoff_index_buckets[i];
        }
    }
    std::cout << '\n';
    const std::string type_prefix = std::string(prefix) + "_type_rate";
    print_type_rate(type_prefix, "tt_lower_move", cutoff.tt_lower, appeared.tt_lower);
    print_type_rate(type_prefix, "tt_upper_move", cutoff.tt_upper, appeared.tt_upper);
    print_type_rate(type_prefix, "promotion", cutoff.promotion, appeared.promotion);
    print_type_rate(type_prefix, "capture", cutoff.capture, appeared.capture);
    print_type_rate(type_prefix, "quiet", cutoff.quiet, appeared.quiet);
    print_type_rate(type_prefix, "check", cutoff.check, appeared.check);
    print_type_rate(type_prefix, "killer1", cutoff.killer1, appeared.killer1);
    print_type_rate(type_prefix, "killer2", cutoff.killer2, appeared.killer2);
    print_type_rate(type_prefix, "history_positive", cutoff.history_positive, appeared.history_positive);
    print_type_rate(type_prefix, "counter_history_positive", cutoff.counter_history_positive, appeared.counter_history_positive);
}

void print_stage_type_stats(std::string_view prefix, const chess::HeuristicSearcherV24::MoveTypeStats& stats) {
    std::cout << prefix
              << " total=" << stats.total
              << " tt_lower_move=" << stats.tt_lower
              << " tt_upper_move=" << stats.tt_upper
              << " promotion=" << stats.promotion
              << " capture=" << stats.capture
              << " quiet=" << stats.quiet
              << " check=" << stats.check
              << " killer1=" << stats.killer1
              << " killer2=" << stats.killer2
              << " history_positive=" << stats.history_positive
              << " counter_history_positive=" << stats.counter_history_positive
              << '\n';
}

void print_tt_lower_stage0_stats(
    std::string_view prefix,
    const chess::HeuristicSearcherV24::MoveOrderingStats& stats
) {
    std::cout << prefix
              << " attempts=" << stats.tt_lower_stage0_attempts
              << " cutoffs=" << stats.tt_lower_stage0_cutoffs
              << " cutoff_rate=" << ratio(stats.tt_lower_stage0_cutoffs, stats.tt_lower_stage0_attempts)
              << '\n';
}

void print_king_safety_stats(
    std::string_view prefix,
    std::uint64_t contexts,
    std::uint64_t legal_checks,
    std::uint64_t nodes
) {
    std::cout << prefix
              << " contexts=" << contexts
              << " legal_checks=" << legal_checks
              << " legal_checks_per_context=" << ratio(legal_checks, contexts)
              << " contexts_per_node=" << ratio(contexts, nodes)
              << " legal_checks_per_node=" << ratio(legal_checks, nodes)
              << '\n';
}

void print_cutoff_side_type_stats(
    std::string_view prefix,
    const chess::HeuristicSearcherV24::MoveCutoffStats& stats
) {
    print_stage_type_stats(std::string(prefix) + "_before", stats.before_cutoff);
    print_stage_type_stats(std::string(prefix) + "_cutoff", stats.cutoff_move);
    print_stage_type_stats(std::string(prefix) + "_after", stats.after_cutoff);
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

        Totals v23_totals;
        Totals v24_totals;
        TtTotals v23_tt_totals;
        TtTotals v24_tt_totals;
        chess::HeuristicSearcherV24::MoveOrderingStats v24_ordering_totals;

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " depth=" << options.depth
                  << " tt_mb=" << options.tt_mb
                  << " bucket_size=" << options.bucket_size
                  << " history_penalty_num=" << options.history_penalty_num
                  << " history_penalty_den=" << options.history_penalty_den
                  << " tt_lower_bonus=" << options.tt_lower_bonus
                  << " tt_upper_bonus=" << options.tt_upper_bonus
                  << " promotion_bonus=" << options.promotion_bonus
                  << " good_capture_bonus=" << options.good_capture_bonus
                  << " bad_capture_bonus=" << options.bad_capture_bonus
                  << " see_weight=" << options.see_weight
                  << " killer1_bonus=" << options.killer1_bonus
                  << " killer2_bonus=" << options.killer2_bonus
                  << " check_bonus=" << options.check_bonus
                  << " counter_history_bonus=" << options.counter_history_bonus
                  << " bad_capture_stage_threshold=" << options.bad_capture_stage_threshold
                  << " qsearch_promotion_bonus=" << options.qsearch_promotion_bonus
                  << " qsearch_good_capture_bonus=" << options.qsearch_good_capture_bonus
                  << " qsearch_bad_capture_bonus=" << options.qsearch_bad_capture_bonus
                  << " qsearch_see_weight=" << options.qsearch_see_weight
                  << " qsearch_captured_value_weight=" << options.qsearch_captured_value_weight
                  << " iterative=" << (options.iterative ? 1 : 0)
                  << " only_v24=" << (options.only_v24 ? 1 : 0)
                  << " instrument_ordering=" << (options.instrument_ordering ? 1 : 0)
                  << " first=" << (options.first_samples ? 1 : 0)
                  << " seed=" << options.seed << '\n' << std::flush;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const Sample& sample = samples[i];
            chess::HeuristicSearcherV23 v23(
                static_cast<std::size_t>(options.tt_mb),
                static_cast<std::size_t>(options.bucket_size)
            );
            chess::HeuristicSearcherV24::MoveOrderingWeights weights;
            weights.tt_lower_bonus = options.tt_lower_bonus;
            weights.tt_upper_bonus = options.tt_upper_bonus;
            weights.promotion_bonus = options.promotion_bonus;
            weights.good_capture_bonus = options.good_capture_bonus;
            weights.bad_capture_bonus = options.bad_capture_bonus;
            weights.see_weight = options.see_weight;
            weights.killer1_bonus = options.killer1_bonus;
            weights.killer2_bonus = options.killer2_bonus;
            weights.check_bonus = options.check_bonus;
            weights.counter_history_bonus = options.counter_history_bonus;
            weights.bad_capture_stage_threshold = options.bad_capture_stage_threshold;
            weights.qsearch_promotion_bonus = options.qsearch_promotion_bonus;
            weights.qsearch_good_capture_bonus = options.qsearch_good_capture_bonus;
            weights.qsearch_bad_capture_bonus = options.qsearch_bad_capture_bonus;
            weights.qsearch_see_weight = options.qsearch_see_weight;
            weights.qsearch_captured_value_weight = options.qsearch_captured_value_weight;
            chess::HeuristicSearcherV24 v24(
                static_cast<std::size_t>(options.tt_mb),
                static_cast<std::size_t>(options.bucket_size),
                options.history_penalty_num,
                options.history_penalty_den,
                weights.counter_history_bonus,
                weights
            );
            v24.set_move_ordering_stats_enabled(options.instrument_ordering);

            chess::SearchResult v23_result;
            std::uint64_t v23_us = 0;
            if (!options.only_v24) {
                const auto v23_start = std::chrono::steady_clock::now();
                v23_result = search(v23, sample.pos, options.depth, options.iterative);
                const auto v23_end = std::chrono::steady_clock::now();
                v23_us = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(v23_end - v23_start).count()
                );
            }

            const auto v24_start = std::chrono::steady_clock::now();
            const chess::SearchResult v24_result = search(v24, sample.pos, options.depth, options.iterative);
            const auto v24_end = std::chrono::steady_clock::now();

            const auto v24_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(v24_end - v24_start).count()
            );

            if (!options.only_v24) {
                add_result(v23_totals, v23_result, v23_us);
                add_tt_stats(v23_tt_totals, v23.tt_stats());
            }
            add_result(v24_totals, v24_result, v24_us);
            add_tt_stats(v24_tt_totals, v24.tt_stats());
            if (options.instrument_ordering) {
                add_ordering_stats(v24_ordering_totals, v24.move_ordering_stats());
            }
            if (!options.only_v24 && v23_result.score != v24_result.score) {
                ++v23_totals.score_mismatches;
                ++v24_totals.score_mismatches;
            }
            if (!options.only_v24 && v23_result.best_move != v24_result.best_move) {
                ++v23_totals.move_mismatches;
                ++v24_totals.move_mismatches;
            }

            std::cout << "sample=" << i
                      << " line=" << sample.source_index
                      << " target=" << sample.target
                      << " legal=" << chess::generate_legal_moves(sample.pos).size();
            if (!options.only_v24) {
                std::cout << " v23_nodes=" << v23_result.nodes
                          << " v23_us=" << v23_us
                          << " v23_score=" << v23_result.score
                          << " v23_best=" << chess::move_to_string(v23_result.best_move);
            }
            std::cout
                      << " v24_nodes=" << v24_result.nodes
                      << " v24_us=" << v24_us
                      << " v24_score=" << v24_result.score
                      << " v24_best=" << chess::move_to_string(v24_result.best_move);
            if (!options.only_v24 && v23_result.nodes != 0) {
                std::cout << " v24_vs_v23_nodes="
                          << (static_cast<double>(v24_result.nodes) / static_cast<double>(v23_result.nodes));
            }
            if (!options.only_v24 && v23_us != 0) {
                std::cout << " v24_vs_v23_time="
                          << (static_cast<double>(v24_us) / static_cast<double>(v23_us));
            }
            std::cout << '\n' << std::flush;
        }

        std::cout << "summary"
                  << " v23_nodes=" << v23_totals.nodes
                  << " v23_avg_nodes=" << avg_u64(v23_totals.nodes, v23_totals.searched)
                  << " v23_us=" << v23_totals.time_us
                  << " v23_avg_us=" << avg_u64(v23_totals.time_us, v23_totals.searched)
                  << " v24_nodes=" << v24_totals.nodes
                  << " v24_avg_nodes=" << avg_u64(v24_totals.nodes, v24_totals.searched)
                  << " v24_us=" << v24_totals.time_us
                  << " v24_avg_us=" << avg_u64(v24_totals.time_us, v24_totals.searched);
        if (v23_totals.nodes != 0) {
            std::cout << " v24_vs_v23_nodes="
                      << (static_cast<double>(v24_totals.nodes) / static_cast<double>(v23_totals.nodes));
        }
        if (v23_totals.time_us != 0) {
            std::cout << " v24_vs_v23_time="
                      << (static_cast<double>(v24_totals.time_us) / static_cast<double>(v23_totals.time_us));
        }
        std::cout << " score_mismatches=" << v23_totals.score_mismatches
                  << " move_mismatches=" << v23_totals.move_mismatches
                  << '\n';
        if (!options.only_v24) {
            print_tt_summary("v23", v23_tt_totals);
        }
        print_tt_summary("v24", v24_tt_totals);
        if (options.instrument_ordering) {
            print_cutoff_profile("v24_order_main", v24_ordering_totals.main);
            print_cutoff_side_type_stats("v24_order_main", v24_ordering_totals.main);
            print_cutoff_profile("v24_order_qsearch", v24_ordering_totals.qsearch);
            print_cutoff_side_type_stats("v24_order_qsearch", v24_ordering_totals.qsearch);
            print_stage_type_stats("v24_stage_priority", v24_ordering_totals.noisy_stage);
            print_stage_type_stats("v24_stage_quiet", v24_ordering_totals.quiet_stage);
            print_tt_lower_stage0_stats("v24_tt_lower_stage0", v24_ordering_totals);
            const std::uint64_t searched_nodes = v24_totals.nodes;
            print_king_safety_stats(
                "v24_king_safety_tt_lower",
                v24_ordering_totals.tt_lower_king_safety_contexts,
                v24_ordering_totals.tt_lower_legal_checks,
                searched_nodes);
            print_king_safety_stats(
                "v24_king_safety_priority",
                v24_ordering_totals.priority_king_safety_contexts,
                v24_ordering_totals.priority_legal_checks,
                searched_nodes);
            print_king_safety_stats(
                "v24_king_safety_quiet",
                v24_ordering_totals.quiet_king_safety_contexts,
                v24_ordering_totals.quiet_legal_checks,
                searched_nodes);
            print_king_safety_stats(
                "v24_king_safety_qsearch",
                v24_ordering_totals.qsearch_king_safety_contexts,
                v24_ordering_totals.qsearch_legal_checks,
                searched_nodes);
            print_king_safety_stats(
                "v24_king_safety_total",
                v24_ordering_totals.tt_lower_king_safety_contexts
                    + v24_ordering_totals.priority_king_safety_contexts
                    + v24_ordering_totals.quiet_king_safety_contexts
                    + v24_ordering_totals.qsearch_king_safety_contexts,
                v24_ordering_totals.tt_lower_legal_checks
                    + v24_ordering_totals.priority_legal_checks
                    + v24_ordering_totals.quiet_legal_checks
                    + v24_ordering_totals.qsearch_legal_checks,
                searched_nodes);
            for (std::size_t i = 0; i < v24_ordering_totals.priority_stage_top.size(); ++i) {
                print_stage_type_stats(
                    "v24_stage_priority_top" + std::to_string(i + 1),
                    v24_ordering_totals.priority_stage_top[i]);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark_v23_v24_bucket_tt: " << error.what() << '\n';
        return 1;
    }
}
