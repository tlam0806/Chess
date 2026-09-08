#include "board_encoder.hpp"
#include "heuristic_searcher_v6.hpp"
#include "heuristic_searcher_v7.hpp"
#include "move.hpp"
#include "position.hpp"

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
    std::string input = "data/train_mix_100k_important50_capture30_material20_depth6.jsonl";
    int samples = 10;
    int depth = 8;
    std::uint32_t seed = 20260612;
};

struct Sample {
    chess::Position pos;
    int target = 0;
    std::uint64_t source_index = 0;
};

struct EngineTotals {
    std::uint64_t time_us = 0;
    std::uint64_t nodes = 0;
    int searched = 0;
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
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_searchers_on_dataset [--input path]\n"
                << "                                      [--samples N] [--depth D] [--seed N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.samples <= 0) {
        throw std::runtime_error("--samples must be positive");
    }
    if (options.depth < 0) {
        throw std::runtime_error("--depth must be non-negative");
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
        index /= chess::EncoderSquares; // king square is only context for NN encoding.
        const auto king_context = static_cast<chess::EncodedKingContext>(index % chess::EncoderKingContexts);
        index /= chess::EncoderKingContexts;
        const auto piece_side = static_cast<chess::EncodedPieceSide>(index % chess::EncoderPieceSides);
        index /= chess::EncoderPieceSides;
        const auto piece = static_cast<chess::PieceType>(index);

        if (king_context != chess::EncodedKingContext::FriendlyKing) {
            continue;
        }
        if (seen[raw_feature]) {
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

template <typename Searcher>
chess::SearchResult timed_search(
    Searcher& searcher,
    const chess::Position& pos,
    int depth,
    std::uint64_t& time_us
) {
    searcher.clear_tt();
    const auto start = std::chrono::steady_clock::now();
    chess::SearchResult result = searcher.search_best_move(pos, depth);
    const auto end = std::chrono::steady_clock::now();
    time_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return result;
}

void add_result(EngineTotals& totals, const chess::SearchResult& result, std::uint64_t time_us) {
    ++totals.searched;
    totals.time_us += time_us;
    totals.nodes += result.nodes;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_random_samples(options);

        chess::HeuristicSearcherV6 v6;
        chess::HeuristicSearcherV7 v7;
        EngineTotals v6_totals;
        EngineTotals v7_totals;

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " depth=" << options.depth
                  << " seed=" << options.seed << '\n';

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const Sample& sample = samples[i];
            std::uint64_t v6_us = 0;
            std::uint64_t v7_us = 0;
            chess::SearchResult v6_result;
            chess::SearchResult v7_result;

            if ((i & 1U) == 0) {
                v6_result = timed_search(v6, sample.pos, options.depth, v6_us);
                v7_result = timed_search(v7, sample.pos, options.depth, v7_us);
            } else {
                v7_result = timed_search(v7, sample.pos, options.depth, v7_us);
                v6_result = timed_search(v6, sample.pos, options.depth, v6_us);
            }

            add_result(v6_totals, v6_result, v6_us);
            add_result(v7_totals, v7_result, v7_us);

            std::cout << "sample=" << i
                      << " line=" << sample.source_index
                      << " target=" << sample.target
                      << " legal_moves=" << chess::generate_legal_moves(sample.pos).size()
                      << " v6_us=" << v6_us
                      << " v6_nodes=" << v6_result.nodes
                      << " v6_score=" << v6_result.score
                      << " v6_best=" << chess::move_to_string(v6_result.best_move)
                      << " v7_us=" << v7_us
                      << " v7_nodes=" << v7_result.nodes
                      << " v7_score=" << v7_result.score
                      << " v7_best=" << chess::move_to_string(v7_result.best_move)
                      << '\n';
        }

        const std::uint64_t v6_avg_us = v6_totals.searched == 0
            ? 0
            : v6_totals.time_us / static_cast<std::uint64_t>(v6_totals.searched);
        const std::uint64_t v7_avg_us = v7_totals.searched == 0
            ? 0
            : v7_totals.time_us / static_cast<std::uint64_t>(v7_totals.searched);
        const std::uint64_t v6_avg_nodes = v6_totals.searched == 0
            ? 0
            : v6_totals.nodes / static_cast<std::uint64_t>(v6_totals.searched);
        const std::uint64_t v7_avg_nodes = v7_totals.searched == 0
            ? 0
            : v7_totals.nodes / static_cast<std::uint64_t>(v7_totals.searched);

        std::cout << "summary"
                  << " v6_total_ms=" << (v6_totals.time_us / 1000)
                  << " v6_avg_us=" << v6_avg_us
                  << " v6_nodes=" << v6_totals.nodes
                  << " v6_avg_nodes=" << v6_avg_nodes
                  << " v7_total_ms=" << (v7_totals.time_us / 1000)
                  << " v7_avg_us=" << v7_avg_us
                  << " v7_nodes=" << v7_totals.nodes
                  << " v7_avg_nodes=" << v7_avg_nodes;
        if (v6_totals.time_us != 0) {
            std::cout << " v7_time_ratio=" << (static_cast<double>(v7_totals.time_us) / v6_totals.time_us);
        }
        if (v6_totals.nodes != 0) {
            std::cout << " v7_node_ratio=" << (static_cast<double>(v7_totals.nodes) / v6_totals.nodes);
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "benchmark_searchers_on_dataset: " << error.what() << '\n';
        return 1;
    }
}
