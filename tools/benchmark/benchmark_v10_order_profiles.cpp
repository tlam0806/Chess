#include "board_encoder.hpp"
#include "heuristic_searcher_v10.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search_types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string input = "data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl";
    int samples = 100;
    int passes = 5;
    int depth = 7;
    int tt_mb = 64;
    bool iterative = true;
    std::uint32_t seed = 20260613;
};

struct Sample {
    chess::Position pos;
    int target = 0;
    std::uint64_t source_index = 0;
};

struct ProfileSpec {
    chess::V10MoveOrderingProfile profile;
    std::string_view name;
};

struct Totals {
    std::uint64_t time_us = 0;
    std::uint64_t nodes = 0;
    std::uint64_t searches = 0;
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
        } else if (arg == "--passes") {
            options.passes = parse_int(require_value(arg), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-mb") {
            options.tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--fixed-depth") {
            options.iterative = false;
        } else if (arg == "--iterative") {
            options.iterative = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v10_order_profiles [--input path]\n"
                << "                                      [--samples N] [--passes N]\n"
                << "                                      [--depth N] [--tt-mb N]\n"
                << "                                      [--seed N] [--iterative|--fixed-depth]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.samples <= 0) {
        throw std::runtime_error("--samples must be positive");
    }
    if (options.passes <= 0) {
        throw std::runtime_error("--passes must be positive");
    }
    if (options.depth < 0) {
        throw std::runtime_error("--depth must be non-negative");
    }
    if (options.tt_mb <= 0) {
        throw std::runtime_error("--tt-mb must be positive");
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

chess::SearchResult timed_search(
    const Sample& sample,
    const Options& options,
    chess::V10MoveOrderingProfile profile,
    std::uint64_t& time_us
) {
    chess::HeuristicSearcherV10 searcher(static_cast<std::size_t>(options.tt_mb), profile);
    const auto start = std::chrono::steady_clock::now();
    chess::SearchResult result;
    if (options.iterative) {
        result = searcher.search_best_move(sample.pos, chess::SearchLimits{
            .max_depth = options.depth,
            .move_time = std::chrono::milliseconds{0}
        });
    } else {
        result = searcher.search_best_move(sample.pos, options.depth);
    }
    const auto end = std::chrono::steady_clock::now();
    time_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return result;
}

void add_result(Totals& totals, const chess::SearchResult& result, std::uint64_t time_us) {
    ++totals.searches;
    totals.time_us += time_us;
    totals.nodes += result.nodes;
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        return 0.0;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_random_samples(options);

        const std::array<ProfileSpec, 5> profiles{{
            {chess::V10MoveOrderingProfile::Baseline, "baseline"},
            {chess::V10MoveOrderingProfile::CheckBeforeGoodCapture, "check_before_good_capture"},
            {chess::V10MoveOrderingProfile::KillerBeforeCheck, "killer_before_check"},
            {chess::V10MoveOrderingProfile::HistoryBeforeKiller, "history_before_killer"},
            {chess::V10MoveOrderingProfile::BadCaptureBeforeQuiet, "bad_capture_before_quiet"}
        }};
        std::array<Totals, profiles.size()> totals{};

        std::mt19937 rng(options.seed ^ 0x9e3779b9U);
        std::vector<std::size_t> sample_order(samples.size());
        std::iota(sample_order.begin(), sample_order.end(), 0);
        std::vector<std::size_t> profile_order(profiles.size());
        std::iota(profile_order.begin(), profile_order.end(), 0);

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " passes=" << options.passes
                  << " depth=" << options.depth
                  << " tt_mb=" << options.tt_mb
                  << " seed=" << options.seed
                  << " iterative=" << (options.iterative ? 1 : 0) << '\n';

        for (int pass = 0; pass < options.passes; ++pass) {
            std::shuffle(sample_order.begin(), sample_order.end(), rng);
            for (std::size_t sample_index : sample_order) {
                std::shuffle(profile_order.begin(), profile_order.end(), rng);
                for (std::size_t profile_index : profile_order) {
                    std::uint64_t time_us = 0;
                    const chess::SearchResult result = timed_search(
                        samples[sample_index],
                        options,
                        profiles[profile_index].profile,
                        time_us);
                    add_result(totals[profile_index], result, time_us);
                }
            }
        }

        const Totals& baseline = totals[0];
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            const Totals& total = totals[i];
            const std::uint64_t avg_us = total.searches == 0 ? 0 : total.time_us / total.searches;
            const std::uint64_t avg_nodes = total.searches == 0 ? 0 : total.nodes / total.searches;
            std::cout << "summary"
                      << " profile=" << profiles[i].name
                      << " searches=" << total.searches
                      << " total_ms=" << (total.time_us / 1000)
                      << " avg_us=" << avg_us
                      << " nodes=" << total.nodes
                      << " avg_nodes=" << avg_nodes
                      << " time_ratio_vs_baseline=" << ratio(total.time_us, baseline.time_us)
                      << " node_ratio_vs_baseline=" << ratio(total.nodes, baseline.nodes)
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark_v10_order_profiles: " << error.what() << '\n';
        return 1;
    }
}
