#include "board_encoder.hpp"
#include "heuristic_searcher_v17.hpp"
#include "heuristic_searcher_v18.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search_types.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string input = "data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl";
    int samples = 100;
    int depth = 7;
    int history_penalty_divisor_numerator = 6;
    int history_penalty_divisor_denominator = 5;
    int tt_lower_bonus = 1'000'000'000;
    int tt_upper_bonus = 20'000'000;
    int promotion_bonus = 900'000'000;
    int good_capture_bonus = 50'000'000;
    int bad_capture_bonus = 30'000;
    int see_weight = 10;
    int killer1_bonus = 40'000;
    int killer2_bonus = 10'000;
    int check_bonus = 160'000;
    int counter_history_bonus = 0;
    bool iterative = true;
    bool first_samples = false;
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
    int move_mismatch_same_score = 0;
    int move_mismatch_different_score = 0;
};

struct OrderingTotals {
    std::uint64_t searched_nodes = 0;
    std::uint64_t best_index_zero = 0;
    std::uint64_t best_index_sum = 0;
    std::uint64_t cutoff_nodes = 0;
    std::uint64_t cutoff_index_zero = 0;
    std::uint64_t cutoff_index_sum = 0;
    std::uint64_t missed_best_nodes = 0;
    std::array<std::uint64_t, chess::HeuristicSearcherV18::MoveOrderingCategoryCount> predicted_categories{};
    std::array<std::uint64_t, chess::HeuristicSearcherV18::MoveOrderingCategoryCount> best_categories{};
    std::array<std::uint64_t, chess::HeuristicSearcherV18::MoveOrderingCategoryCount> cutoff_categories{};
    std::array<std::uint64_t, chess::HeuristicSearcherV18::MoveOrderingCategoryCount> missed_predicted_categories{};
    std::array<std::uint64_t, chess::HeuristicSearcherV18::MoveOrderingCategoryCount> missed_best_categories{};
    std::array<
        std::array<std::uint64_t, chess::HeuristicSearcherV18::MoveOrderingCategoryCount>,
        chess::HeuristicSearcherV18::MoveOrderingCategoryCount
    > missed_predicted_to_best_categories{};
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

std::pair<int, int> parse_positive_ratio(std::string_view value, std::string_view name) {
    const std::size_t slash = value.find('/');
    if (slash == std::string_view::npos) {
        const int numerator = parse_int(value, name);
        if (numerator <= 0) {
            throw std::runtime_error(std::string(name) + " must be positive");
        }
        return {numerator, 1};
    }

    const int numerator = parse_int(value.substr(0, slash), name);
    const int denominator = parse_int(value.substr(slash + 1), name);
    if (numerator <= 0 || denominator <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return {numerator, denominator};
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
        } else if (arg == "--penalty-divisor") {
            const auto [numerator, denominator] = parse_positive_ratio(require_value(arg), arg);
            options.history_penalty_divisor_numerator = numerator;
            options.history_penalty_divisor_denominator = denominator;
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
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--fixed-depth") {
            options.iterative = false;
        } else if (arg == "--iterative") {
            options.iterative = true;
        } else if (arg == "--first") {
            options.first_samples = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v17_v18 [--input path]\n"
                << "                                  [--samples N]\n"
                << "                                  [--depth D]\n"
                << "                                  [--penalty-divisor N or N/D]\n"
                << "                                  [--tt-lower-bonus N]\n"
                << "                                  [--tt-upper-bonus N]\n"
                << "                                  [--promotion-bonus N]\n"
                << "                                  [--good-capture-bonus N]\n"
                << "                                  [--bad-capture-bonus N]\n"
                << "                                  [--see-weight N]\n"
                << "                                  [--killer1-bonus N]\n"
                << "                                  [--killer2-bonus N]\n"
                << "                                  [--check-bonus N]\n"
                << "                                  [--counter-history-bonus N]\n"
                << "                                  [--seed N]\n"
                << "                                  [--iterative|--fixed-depth]\n"
                << "                                  [--first]\n";
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

template <typename Stats>
void add_ordering_stats(OrderingTotals& totals, const Stats& stats) {
    totals.searched_nodes += stats.searched_nodes;
    totals.best_index_zero += stats.best_index_zero;
    totals.best_index_sum += stats.best_index_sum;
    totals.cutoff_nodes += stats.cutoff_nodes;
    totals.cutoff_index_zero += stats.cutoff_index_zero;
    totals.cutoff_index_sum += stats.cutoff_index_sum;
    totals.missed_best_nodes += stats.missed_best_nodes;
    for (std::size_t i = 0; i < chess::HeuristicSearcherV18::MoveOrderingCategoryCount; ++i) {
        totals.predicted_categories[i] += stats.predicted_categories[i];
        totals.best_categories[i] += stats.best_categories[i];
        totals.cutoff_categories[i] += stats.cutoff_categories[i];
        totals.missed_predicted_categories[i] += stats.missed_predicted_categories[i];
        totals.missed_best_categories[i] += stats.missed_best_categories[i];
        for (std::size_t j = 0; j < chess::HeuristicSearcherV18::MoveOrderingCategoryCount; ++j) {
            totals.missed_predicted_to_best_categories[i][j] +=
                stats.missed_predicted_to_best_categories[i][j];
        }
    }
}

std::uint64_t avg_u64(std::uint64_t value, int count) {
    return count == 0 ? 0 : value / static_cast<std::uint64_t>(count);
}

double ratio(std::uint64_t value, std::uint64_t total) {
    return total == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(total);
}

const char* category_name(std::size_t index) {
    using Category = chess::HeuristicSearcherV18::MoveOrderingCategory;
    switch (static_cast<Category>(index)) {
    case Category::TtLower:
        return "tt_lower";
    case Category::TtUpper:
        return "tt_upper";
    case Category::Promotion:
        return "promotion";
    case Category::GoodCapture:
        return "good_capture";
    case Category::BadCapture:
        return "bad_capture";
    case Category::Check:
        return "check";
    case Category::CounterHistory:
        return "counter_history";
    case Category::Killer1:
        return "killer1";
    case Category::Killer2:
        return "killer2";
    case Category::Quiet:
        return "quiet";
    case Category::Count:
        break;
    }
    return "unknown";
}

void print_category_counts(
    std::string_view prefix,
    std::string_view label,
    const std::array<std::uint64_t, chess::HeuristicSearcherV18::MoveOrderingCategoryCount>& counts,
    std::uint64_t denominator
) {
    for (std::size_t i = 0; i < chess::HeuristicSearcherV18::MoveOrderingCategoryCount; ++i) {
        std::cout << prefix << '_' << label << '_' << category_name(i) << '=' << counts[i]
                  << ' ' << prefix << '_' << label << '_' << category_name(i)
                  << "_rate=" << ratio(counts[i], denominator) << '\n';
    }
}

void print_missed_matrix(std::string_view prefix, const OrderingTotals& totals) {
    for (std::size_t predicted = 0; predicted < chess::HeuristicSearcherV18::MoveOrderingCategoryCount; ++predicted) {
        for (std::size_t best = 0; best < chess::HeuristicSearcherV18::MoveOrderingCategoryCount; ++best) {
            const std::uint64_t count = totals.missed_predicted_to_best_categories[predicted][best];
            if (count == 0) {
                continue;
            }
            std::cout << prefix << "_missed_pair predicted=" << category_name(predicted)
                      << " best=" << category_name(best)
                      << " count=" << count
                      << " missed_rate=" << ratio(count, totals.missed_best_nodes)
                      << " predicted_bucket_rate=" << ratio(count, totals.missed_predicted_categories[predicted])
                      << " best_bucket_rate=" << ratio(count, totals.missed_best_categories[best])
                      << '\n';
        }
    }
}

void print_ordering_summary(std::string_view prefix, const OrderingTotals& totals) {
    std::cout << prefix
              << "_ordering searched_nodes=" << totals.searched_nodes
              << " best_index0=" << totals.best_index_zero
              << " best_index0_rate=" << ratio(totals.best_index_zero, totals.searched_nodes)
              << " avg_best_index=" << ratio(totals.best_index_sum, totals.searched_nodes)
              << " missed_best_nodes=" << totals.missed_best_nodes
              << " missed_best_rate=" << ratio(totals.missed_best_nodes, totals.searched_nodes)
              << " cutoff_nodes=" << totals.cutoff_nodes
              << " cutoff_index0=" << totals.cutoff_index_zero
              << " cutoff_index0_rate=" << ratio(totals.cutoff_index_zero, totals.cutoff_nodes)
              << " avg_cutoff_index=" << ratio(totals.cutoff_index_sum, totals.cutoff_nodes)
              << '\n';

    print_category_counts(prefix, "predicted", totals.predicted_categories, totals.searched_nodes);
    print_category_counts(prefix, "best", totals.best_categories, totals.searched_nodes);
    print_category_counts(prefix, "cutoff", totals.cutoff_categories, totals.cutoff_nodes);
    print_category_counts(prefix, "missed_predicted", totals.missed_predicted_categories, totals.missed_best_nodes);
    print_category_counts(prefix, "missed_best", totals.missed_best_categories, totals.missed_best_nodes);
    print_missed_matrix(prefix, totals);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_random_samples(options);

        Totals v17_totals;
        Totals v18_totals;
        OrderingTotals v17_ordering;
        OrderingTotals v18_ordering;

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " depth=" << options.depth
                  << " penalty_divisor=" << options.history_penalty_divisor_numerator
                  << "/" << options.history_penalty_divisor_denominator
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
                  << " iterative=" << (options.iterative ? 1 : 0)
                  << " first=" << (options.first_samples ? 1 : 0)
                  << " seed=" << options.seed << '\n' << std::flush;
        std::cout << std::fixed << std::setprecision(6);

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const Sample& sample = samples[i];
            chess::HeuristicSearcherV17 v17(64);
            chess::HeuristicSearcherV18 v18(64);

            const auto v17_start = std::chrono::steady_clock::now();
            const chess::SearchResult v17_result = search(v17, sample.pos, options.depth, options.iterative);
            const auto v17_end = std::chrono::steady_clock::now();

            const auto v18_start = std::chrono::steady_clock::now();
            const chess::SearchResult v18_result = search(v18, sample.pos, options.depth, options.iterative);
            const auto v18_end = std::chrono::steady_clock::now();

            const auto v17_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(v17_end - v17_start).count()
            );
            const auto v18_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(v18_end - v18_start).count()
            );

            add_result(v17_totals, v17_result, v17_us);
            add_result(v18_totals, v18_result, v18_us);
            add_ordering_stats(v17_ordering, v17.move_ordering_stats());
            add_ordering_stats(v18_ordering, v18.move_ordering_stats());
            if (v17_result.score != v18_result.score) {
                ++v17_totals.score_mismatches;
                ++v18_totals.score_mismatches;
            }
            if (v17_result.best_move != v18_result.best_move) {
                if (v17_result.score == v18_result.score) {
                    ++v17_totals.move_mismatch_same_score;
                    ++v18_totals.move_mismatch_same_score;
                } else {
                    ++v17_totals.move_mismatch_different_score;
                    ++v18_totals.move_mismatch_different_score;
                }
            }

            std::cout << "sample=" << i
                      << " line=" << sample.source_index
                      << " target=" << sample.target
                      << " legal=" << chess::generate_legal_moves(sample.pos).size()
                      << " v17_nodes=" << v17_result.nodes
                      << " v17_us=" << v17_us
                      << " v17_score=" << v17_result.score
                      << " v17_best=" << chess::move_to_string(v17_result.best_move)
                      << " v18_nodes=" << v18_result.nodes
                      << " v18_us=" << v18_us
                      << " v18_score=" << v18_result.score
                      << " v18_best=" << chess::move_to_string(v18_result.best_move);
            if (v17_result.nodes != 0) {
                std::cout << " v18_vs_v17_nodes="
                          << (static_cast<double>(v18_result.nodes) / static_cast<double>(v17_result.nodes));
            }
            if (v17_us != 0) {
                std::cout << " v18_vs_v17_time="
                          << (static_cast<double>(v18_us) / static_cast<double>(v17_us));
            }
            std::cout << '\n' << std::flush;
        }

        std::cout << "summary"
                  << " v17_nodes=" << v17_totals.nodes
                  << " v17_avg_nodes=" << avg_u64(v17_totals.nodes, v17_totals.searched)
                  << " v17_us=" << v17_totals.time_us
                  << " v17_avg_us=" << avg_u64(v17_totals.time_us, v17_totals.searched)
                  << " v18_nodes=" << v18_totals.nodes
                  << " v18_avg_nodes=" << avg_u64(v18_totals.nodes, v18_totals.searched)
                  << " v18_us=" << v18_totals.time_us
                  << " v18_avg_us=" << avg_u64(v18_totals.time_us, v18_totals.searched);
        if (v17_totals.nodes != 0) {
            std::cout << " v18_vs_v17_nodes="
                      << (static_cast<double>(v18_totals.nodes) / static_cast<double>(v17_totals.nodes));
        }
        if (v17_totals.time_us != 0) {
            std::cout << " v18_vs_v17_time="
                      << (static_cast<double>(v18_totals.time_us) / static_cast<double>(v17_totals.time_us));
        }
        std::cout << " score_mismatches=" << v17_totals.score_mismatches
                  << " move_mismatch_same_score=" << v17_totals.move_mismatch_same_score
                  << " move_mismatch_different_score=" << v17_totals.move_mismatch_different_score
                  << '\n' << std::flush;

        print_ordering_summary("v17", v17_ordering);
        print_ordering_summary("v18", v18_ordering);
    } catch (const std::exception& error) {
        std::cerr << "analyze_v17_v18_move_ordering: " << error.what() << '\n';
        return 1;
    }
}
