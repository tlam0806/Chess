#include "board_encoder.hpp"
#include "heuristic_searcher_v12.hpp"
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
    std::string input = "data/nnue_mix_train_600k.jsonl";
    int samples = 30;
    int depth = 7;
    int max_configs = 60;
    int top = 10;
    bool iterative = true;
    bool local = false;
    bool has_single_candidate = false;
    chess::V12MoveOrderWeights single_candidate{};
    std::uint32_t seed = 20260613;
};

struct Sample {
    chess::Position pos;
    int target = 0;
    std::uint64_t source_index = 0;
};

struct CandidateResult {
    chess::V12MoveOrderWeights weights;
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
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-configs") {
            options.max_configs = parse_int(require_value(arg), arg);
        } else if (arg == "--top") {
            options.top = parse_int(require_value(arg), arg);
        } else if (arg == "--good") {
            options.single_candidate.good_capture_bonus = parse_int(require_value(arg), arg);
            options.has_single_candidate = true;
        } else if (arg == "--bad") {
            options.single_candidate.bad_capture_bonus = parse_int(require_value(arg), arg);
            options.has_single_candidate = true;
        } else if (arg == "--see-weight") {
            options.single_candidate.see_weight = parse_int(require_value(arg), arg);
            options.has_single_candidate = true;
        } else if (arg == "--killer1") {
            options.single_candidate.killer1_bonus = parse_int(require_value(arg), arg);
            options.has_single_candidate = true;
        } else if (arg == "--killer2") {
            options.single_candidate.killer2_bonus = parse_int(require_value(arg), arg);
            options.has_single_candidate = true;
        } else if (arg == "--check") {
            options.single_candidate.check_bonus = parse_int(require_value(arg), arg);
            options.has_single_candidate = true;
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--fixed-depth") {
            options.iterative = false;
        } else if (arg == "--iterative") {
            options.iterative = true;
        } else if (arg == "--local") {
            options.local = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: tune_v12_move_order_weights [--input path]\n"
                << "                                  [--samples N] [--depth D]\n"
                << "                                  [--max-configs N] [--top N]\n"
                << "                                  [--good N] [--bad N]\n"
                << "                                  [--see-weight N]\n"
                << "                                  [--killer1 N] [--killer2 N]\n"
                << "                                  [--check N]\n"
                << "                                  [--seed N]\n"
                << "                                  [--iterative|--fixed-depth]\n"
                << "                                  [--local]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.samples <= 0 || options.depth < 0 || options.max_configs <= 0 || options.top <= 0) {
        throw std::runtime_error("samples/max-configs/top must be positive and depth must be non-negative");
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

std::vector<chess::V12MoveOrderWeights> generate_candidates(const Options& options) {
    if (options.has_single_candidate) {
        return {options.single_candidate};
    }

    std::vector<chess::V12MoveOrderWeights> candidates;
    candidates.push_back(chess::V12MoveOrderWeights{});

    if (options.local) {
        const chess::V12MoveOrderWeights base{};
        const std::array<int, 5> good_capture{
            base.good_capture_bonus / 5,
            base.good_capture_bonus / 2,
            base.good_capture_bonus,
            base.good_capture_bonus * 2,
            base.good_capture_bonus * 5
        };
        const std::array<int, 5> bad_capture{-300'000, -100'000, 0, 30'000, 100'000};
        const std::array<int, 6> see_weight{10, 50, 100, 200, 500, 1000};
        const std::array<int, 5> killer1{
            std::max(0, base.killer1_bonus / 4),
            std::max(0, base.killer1_bonus / 2),
            base.killer1_bonus,
            base.killer1_bonus * 2,
            base.killer1_bonus * 4
        };
        const std::array<int, 5> killer2{
            std::max(0, base.killer2_bonus / 4),
            std::max(0, base.killer2_bonus / 2),
            base.killer2_bonus,
            base.killer2_bonus * 2,
            base.killer2_bonus * 4
        };
        const std::array<int, 6> check{0, base.check_bonus / 2, base.check_bonus, base.check_bonus * 2, base.check_bonus * 4, base.check_bonus * 8};

        for (int good : good_capture) {
            for (int bad : bad_capture) {
                for (int see : see_weight) {
                    for (int k1 : killer1) {
                        for (int k2 : killer2) {
                            if (k2 > k1) {
                                continue;
                            }
                            for (int check_bonus : check) {
                                chess::V12MoveOrderWeights weights = base;
                                weights.good_capture_bonus = good;
                                weights.bad_capture_bonus = bad;
                                weights.see_weight = see;
                                weights.killer1_bonus = k1;
                                weights.killer2_bonus = k2;
                                weights.check_bonus = check_bonus;
                                candidates.push_back(weights);
                            }
                        }
                    }
                }
            }
        }

        std::mt19937 rng(options.seed ^ 0x6eed0e9dU);
        std::shuffle(candidates.begin() + 1, candidates.end(), rng);
        if (static_cast<int>(candidates.size()) > options.max_configs) {
            candidates.resize(static_cast<std::size_t>(options.max_configs));
        }
        return candidates;
    }

    const std::array<int, 4> good_capture{20'000'000, 50'000'000, 100'000'000, 200'000'000};
    const std::array<int, 4> bad_capture{-300'000, -100'000, -30'000, 0};
    const std::array<int, 5> see_weight{1, 5, 10, 50, 100};
    const std::array<int, 4> killer1{10'000, 30'000, 60'000, 120'000};
    const std::array<int, 4> killer2{5'000, 15'000, 30'000, 60'000};
    const std::array<int, 5> check{0, 5'000, 10'000, 20'000, 40'000};

    for (int good : good_capture) {
        for (int bad : bad_capture) {
            for (int see : see_weight) {
                for (int k1 : killer1) {
                    for (int k2 : killer2) {
                        if (k2 > k1) {
                            continue;
                        }
                        for (int check_bonus : check) {
                            chess::V12MoveOrderWeights weights;
                            weights.good_capture_bonus = good;
                            weights.bad_capture_bonus = bad;
                            weights.see_weight = see;
                            weights.killer1_bonus = k1;
                            weights.killer2_bonus = k2;
                            weights.check_bonus = check_bonus;
                            candidates.push_back(weights);
                        }
                    }
                }
            }
        }
    }

    std::mt19937 rng(options.seed ^ 0x517cc1b7U);
    std::shuffle(candidates.begin() + 1, candidates.end(), rng);
    if (static_cast<int>(candidates.size()) > options.max_configs) {
        candidates.resize(static_cast<std::size_t>(options.max_configs));
    }
    return candidates;
}

chess::SearchResult search(
    const chess::Position& pos,
    int depth,
    bool iterative,
    const chess::V12MoveOrderWeights& weights
) {
    chess::HeuristicSearcherV12 searcher(64, weights);
    if (iterative) {
        return searcher.search_best_move(pos, chess::SearchLimits{
            .max_depth = depth,
            .move_time = std::chrono::milliseconds{0},
        });
    }
    return searcher.search_best_move(pos, depth);
}

std::uint64_t avg_nodes(const CandidateResult& result) {
    return result.searches == 0 ? 0 : result.nodes / result.searches;
}

void print_weights(const chess::V12MoveOrderWeights& weights) {
    std::cout << " good=" << weights.good_capture_bonus
              << " bad=" << weights.bad_capture_bonus
              << " see_weight=" << weights.see_weight
              << " killer1=" << weights.killer1_bonus
              << " killer2=" << weights.killer2_bonus
              << " check=" << weights.check_bonus;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_random_samples(options);
        std::vector<chess::V12MoveOrderWeights> candidates =
            generate_candidates(options);

        std::vector<CandidateResult> results;
        results.reserve(candidates.size());

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " depth=" << options.depth
                  << " iterative=" << (options.iterative ? 1 : 0)
                  << " seed=" << options.seed
                  << " configs=" << candidates.size() << '\n';

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            CandidateResult result;
            result.weights = candidates[i];
            for (const Sample& sample : samples) {
                const chess::SearchResult search_result =
                    search(sample.pos, options.depth, options.iterative, candidates[i]);
                result.nodes += search_result.nodes;
                ++result.searches;
            }
            results.push_back(result);

            std::cout << "config=" << i
                      << " nodes=" << result.nodes
                      << " avg_nodes=" << avg_nodes(result);
            print_weights(result.weights);
            std::cout << '\n';
        }

        std::sort(results.begin(), results.end(), [](const CandidateResult& lhs, const CandidateResult& rhs) {
            return lhs.nodes < rhs.nodes;
        });

        const int top = std::min(options.top, static_cast<int>(results.size()));
        for (int i = 0; i < top; ++i) {
            const CandidateResult& result = results[static_cast<std::size_t>(i)];
            std::cout << "top rank=" << (i + 1)
                      << " nodes=" << result.nodes
                      << " avg_nodes=" << avg_nodes(result);
            print_weights(result.weights);
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "tune_v12_move_order_weights: " << error.what() << '\n';
        return 1;
    }
}
