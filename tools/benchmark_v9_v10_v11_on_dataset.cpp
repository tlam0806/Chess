#include "board_encoder.hpp"
#include "heuristic_searcher_v9.hpp"
#include "heuristic_searcher_v10.hpp"
#include "heuristic_searcher_v11.hpp"
#include "heuristic_searcher_v13.hpp"
#include "move.hpp"
#include "position.hpp"
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
    std::string input = "data/nnue_mix_train_600k.jsonl";
    int samples = 8;
    int depth = 7;
    bool iterative = false;
    std::uint32_t seed = 20260613;
};

struct Sample {
    chess::Position pos;
    int target = 0;
    std::uint64_t source_index = 0;
};

struct Totals {
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
        } else if (arg == "--iterative") {
            options.iterative = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v9_v10_v11_on_dataset [--input path]\n"
                << "                                           [--samples N]\n"
                << "                                           [--depth D]\n"
                << "                                           [--seed N]\n"
                << "                                           [--iterative]\n";
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

void add_result(Totals& totals, const chess::SearchResult& result) {
    ++totals.searched;
    totals.nodes += result.nodes;
}

std::uint64_t avg_nodes(const Totals& totals) {
    return totals.searched == 0 ? 0 : totals.nodes / static_cast<std::uint64_t>(totals.searched);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_random_samples(options);

        Totals v9_totals;
        Totals v10_totals;
        Totals v11_totals;
        Totals v13_totals;

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " depth=" << options.depth
                  << " iterative=" << (options.iterative ? 1 : 0)
                  << " seed=" << options.seed << '\n';

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const Sample& sample = samples[i];
            chess::HeuristicSearcherV9 v9;
            chess::HeuristicSearcherV10 v10;
            chess::HeuristicSearcherV11 v11;
            chess::HeuristicSearcherV13 v13;

            const chess::SearchResult v9_result = search(v9, sample.pos, options.depth, options.iterative);
            const chess::SearchResult v10_result = search(v10, sample.pos, options.depth, options.iterative);
            const chess::SearchResult v11_result = search(v11, sample.pos, options.depth, options.iterative);
            const chess::SearchResult v13_result = search(v13, sample.pos, options.depth, options.iterative);

            add_result(v9_totals, v9_result);
            add_result(v10_totals, v10_result);
            add_result(v11_totals, v11_result);
            add_result(v13_totals, v13_result);

            std::cout << "sample=" << i
                      << " line=" << sample.source_index
                      << " target=" << sample.target
                      << " legal=" << chess::generate_legal_moves(sample.pos).size()
                      << " v9_nodes=" << v9_result.nodes
                      << " v9_score=" << v9_result.score
                      << " v9_best=" << chess::move_to_string(v9_result.best_move)
                      << " v10_nodes=" << v10_result.nodes
                      << " v10_score=" << v10_result.score
                      << " v10_best=" << chess::move_to_string(v10_result.best_move)
                      << " v11_nodes=" << v11_result.nodes
                      << " v11_score=" << v11_result.score
                      << " v11_best=" << chess::move_to_string(v11_result.best_move);
            if (v10_result.nodes != 0) {
                std::cout << " v11_vs_v10_nodes="
                          << (static_cast<double>(v11_result.nodes) / static_cast<double>(v10_result.nodes));
            }
            std::cout << " v13_nodes=" << v13_result.nodes
                      << " v13_score=" << v13_result.score
                      << " v13_best=" << chess::move_to_string(v13_result.best_move);
            if (v9_result.nodes != 0) {
                std::cout << " v13_vs_v9_nodes="
                          << (static_cast<double>(v13_result.nodes) / static_cast<double>(v9_result.nodes));
            }
            std::cout << '\n';
        }

        std::cout << "summary"
                  << " v9_nodes=" << v9_totals.nodes
                  << " v9_avg_nodes=" << avg_nodes(v9_totals)
                  << " v10_nodes=" << v10_totals.nodes
                  << " v10_avg_nodes=" << avg_nodes(v10_totals)
                  << " v11_nodes=" << v11_totals.nodes
                  << " v11_avg_nodes=" << avg_nodes(v11_totals)
                  << " v13_nodes=" << v13_totals.nodes
                  << " v13_avg_nodes=" << avg_nodes(v13_totals);
        if (v10_totals.nodes != 0) {
            std::cout << " v11_vs_v10_nodes="
                      << (static_cast<double>(v11_totals.nodes) / static_cast<double>(v10_totals.nodes));
        }
        if (v9_totals.nodes != 0) {
            std::cout << " v11_vs_v9_nodes="
                      << (static_cast<double>(v11_totals.nodes) / static_cast<double>(v9_totals.nodes))
                      << " v13_vs_v9_nodes="
                      << (static_cast<double>(v13_totals.nodes) / static_cast<double>(v9_totals.nodes));
        }
        if (v10_totals.nodes != 0) {
            std::cout << " v13_vs_v10_nodes="
                      << (static_cast<double>(v13_totals.nodes) / static_cast<double>(v10_totals.nodes));
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "benchmark_v9_v10_v11_on_dataset: " << error.what() << '\n';
        return 1;
    }
}
