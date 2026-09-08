#include "board_encoder.hpp"
#include "heuristic_searcher_v12.hpp"
#include "heuristic_searcher_v13.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search_types.hpp"

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
    int samples = 100;
    int depth = 7;
    bool iterative = true;
    std::uint32_t seed = 20260613;
    int max_print = 20;
    std::uint64_t line = 0;
    bool full_tt_only = false;
};

struct Sample {
    chess::Position pos;
    int target = 0;
    std::uint64_t source_index = 0;
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
        } else if (arg == "--fixed") {
            options.iterative = false;
        } else if (arg == "--max-print") {
            options.max_print = parse_int(require_value(arg), arg);
        } else if (arg == "--line") {
            const int parsed = parse_int(require_value(arg), arg);
            if (parsed < 0) {
                throw std::runtime_error("line must be non-negative");
            }
            options.line = static_cast<std::uint64_t>(parsed);
        } else if (arg == "--full-tt-only") {
            options.full_tt_only = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: compare_v12_v13_logic [--input path]\n"
                << "                              [--samples N]\n"
                << "                              [--depth D]\n"
                << "                              [--seed N]\n"
                << "                              [--fixed]\n"
                << "                              [--max-print N]\n"
                << "                              [--line N]\n"
                << "                              [--full-tt-only]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.samples <= 0 || options.depth < 0 || options.max_print < 0) {
        throw std::runtime_error("samples must be positive; depth/max-print must be non-negative");
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

    std::vector<int> features = extract_int_array(line, "\"features\"");
    std::vector<bool> seen(chess::EncodedFeatureCount, false);
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

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_random_samples(options);

        if (options.full_tt_only) {
            constexpr chess::V13SearchOptions NoSelectiveNoTtNoAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = false,
                .enable_tt = false,
            };
            constexpr chess::V13SearchOptions FullTtNoAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = false,
                .enable_tt = true,
                .enable_tt_bounds = false,
                .enable_tt_exact = true,
                .enable_tt_bound_cutoff = true,
                .enable_tt_bound_tighten = true,
                .enable_tt_exact_store = true,
                .enable_tt_mate_scores = false,
            };
            constexpr chess::V13SearchOptions FullTtAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = true,
                .enable_tt = true,
                .enable_tt_bounds = false,
                .enable_tt_exact = true,
                .enable_tt_bound_cutoff = true,
                .enable_tt_bound_tighten = true,
                .enable_tt_exact_store = true,
                .enable_tt_mate_scores = false,
            };

            int full_fixed_score_mismatches = 0;
            int full_iter_score_mismatches = 0;
            int full_asp_score_mismatches = 0;
            int full_fixed_move_mismatches = 0;
            int full_iter_move_mismatches = 0;
            int full_asp_move_mismatches = 0;
            std::uint64_t baseline_nodes = 0;
            std::uint64_t full_fixed_nodes = 0;
            std::uint64_t full_iter_nodes = 0;
            std::uint64_t full_asp_nodes = 0;

            std::cout << "full_tt_only"
                      << " input=" << options.input
                      << " samples=" << samples.size()
                      << " depth=" << options.depth
                      << " seed=" << options.seed << '\n';

            for (std::size_t i = 0; i < samples.size(); ++i) {
                chess::HeuristicSearcherV13 baseline(64, {}, NoSelectiveNoTtNoAsp);
                chess::HeuristicSearcherV13 full_fixed(64, {}, FullTtNoAsp);
                chess::HeuristicSearcherV13 full_iter(64, {}, FullTtNoAsp);
                chess::HeuristicSearcherV13 full_asp(64, {}, FullTtAsp);

                const chess::SearchResult baseline_result =
                    baseline.search_best_move(samples[i].pos, options.depth);
                const chess::SearchResult full_fixed_result =
                    full_fixed.search_best_move(samples[i].pos, options.depth);
                const chess::SearchResult full_iter_result =
                    full_iter.search_best_move(samples[i].pos, chess::SearchLimits{
                        .max_depth = options.depth,
                        .move_time = std::chrono::milliseconds{0},
                    });
                const chess::SearchResult full_asp_result =
                    full_asp.search_best_move(samples[i].pos, chess::SearchLimits{
                        .max_depth = options.depth,
                        .move_time = std::chrono::milliseconds{0},
                    });

                baseline_nodes += baseline_result.nodes;
                full_fixed_nodes += full_fixed_result.nodes;
                full_iter_nodes += full_iter_result.nodes;
                full_asp_nodes += full_asp_result.nodes;

                const bool fixed_score_diff = baseline_result.score != full_fixed_result.score;
                const bool iter_score_diff = baseline_result.score != full_iter_result.score;
                const bool asp_score_diff = baseline_result.score != full_asp_result.score;
                const bool fixed_move_diff = baseline_result.best_move != full_fixed_result.best_move;
                const bool iter_move_diff = baseline_result.best_move != full_iter_result.best_move;
                const bool asp_move_diff = baseline_result.best_move != full_asp_result.best_move;

                full_fixed_score_mismatches += fixed_score_diff ? 1 : 0;
                full_iter_score_mismatches += iter_score_diff ? 1 : 0;
                full_asp_score_mismatches += asp_score_diff ? 1 : 0;
                full_fixed_move_mismatches += fixed_move_diff ? 1 : 0;
                full_iter_move_mismatches += iter_move_diff ? 1 : 0;
                full_asp_move_mismatches += asp_move_diff ? 1 : 0;

                if ((fixed_score_diff || iter_score_diff || asp_score_diff)
                    && full_fixed_score_mismatches
                        + full_iter_score_mismatches
                        + full_asp_score_mismatches <= options.max_print * 3) {
                    std::cout << "full_tt_score_mismatch"
                              << " sample=" << i
                              << " line=" << samples[i].source_index
                              << " baseline_score=" << baseline_result.score
                              << " baseline_best=" << chess::move_to_string(baseline_result.best_move)
                              << " full_fixed_score=" << full_fixed_result.score
                              << " full_fixed_best=" << chess::move_to_string(full_fixed_result.best_move)
                              << " full_iter_score=" << full_iter_result.score
                              << " full_iter_best=" << chess::move_to_string(full_iter_result.best_move)
                              << " full_asp_score=" << full_asp_result.score
                              << " full_asp_best=" << chess::move_to_string(full_asp_result.best_move)
                              << '\n';
                }
            }

            std::cout << "full_tt_summary"
                      << " samples=" << samples.size()
                      << " fixed_score_mismatches=" << full_fixed_score_mismatches
                      << " iter_score_mismatches=" << full_iter_score_mismatches
                      << " asp_score_mismatches=" << full_asp_score_mismatches
                      << " fixed_move_mismatches=" << full_fixed_move_mismatches
                      << " iter_move_mismatches=" << full_iter_move_mismatches
                      << " asp_move_mismatches=" << full_asp_move_mismatches
                      << " baseline_nodes=" << baseline_nodes
                      << " full_fixed_nodes=" << full_fixed_nodes
                      << " full_iter_nodes=" << full_iter_nodes
                      << " full_asp_nodes=" << full_asp_nodes;
            if (baseline_nodes != 0) {
                std::cout << " full_fixed_vs_baseline_nodes="
                          << static_cast<double>(full_fixed_nodes)
                              / static_cast<double>(baseline_nodes)
                          << " full_iter_vs_baseline_nodes="
                          << static_cast<double>(full_iter_nodes)
                              / static_cast<double>(baseline_nodes)
                          << " full_asp_vs_baseline_nodes="
                          << static_cast<double>(full_asp_nodes)
                              / static_cast<double>(baseline_nodes);
            }
            std::cout << '\n';
            return 0;
        }

        int score_mismatches = 0;
        int move_mismatches = 0;
        int v13_no_prune_iter_score_mismatches = 0;
        int v13_no_prune_asp_score_mismatches = 0;
        int v13_no_prune_iter_move_mismatches = 0;
        int v13_no_prune_asp_move_mismatches = 0;
        int v13_no_prune_no_tt_iter_score_mismatches = 0;
        int v13_no_prune_no_tt_asp_score_mismatches = 0;
        int v13_no_prune_no_tt_iter_move_mismatches = 0;
        int v13_no_prune_no_tt_asp_move_mismatches = 0;
        int v13_no_prune_no_bounds_iter_score_mismatches = 0;
        int v13_no_prune_no_bounds_asp_score_mismatches = 0;
        int v13_no_prune_no_bounds_iter_move_mismatches = 0;
        int v13_no_prune_no_bounds_asp_move_mismatches = 0;
        int v13_no_prune_no_exact_iter_score_mismatches = 0;
        int v13_no_prune_no_exact_asp_score_mismatches = 0;
        int v13_no_prune_no_exact_store_iter_score_mismatches = 0;
        int v13_no_prune_no_exact_store_asp_score_mismatches = 0;
        int v13_no_prune_full_tt_iter_score_mismatches = 0;
        int v13_no_prune_full_tt_asp_score_mismatches = 0;
        int v13_no_prune_full_tt_iter_move_mismatches = 0;
        int v13_no_prune_full_tt_asp_move_mismatches = 0;
        std::uint64_t v12_nodes = 0;
        std::uint64_t v13_nodes = 0;
        std::uint64_t v13_no_prune_fixed_nodes = 0;
        std::uint64_t v13_no_prune_iter_nodes = 0;
        std::uint64_t v13_no_prune_asp_nodes = 0;
        std::uint64_t v13_no_prune_no_tt_fixed_nodes = 0;
        std::uint64_t v13_no_prune_no_tt_iter_nodes = 0;
        std::uint64_t v13_no_prune_no_tt_asp_nodes = 0;
        std::uint64_t v13_no_prune_no_bounds_fixed_nodes = 0;
        std::uint64_t v13_no_prune_no_bounds_iter_nodes = 0;
        std::uint64_t v13_no_prune_no_bounds_asp_nodes = 0;
        std::uint64_t v13_no_prune_no_exact_fixed_nodes = 0;
        std::uint64_t v13_no_prune_no_exact_iter_nodes = 0;
        std::uint64_t v13_no_prune_no_exact_asp_nodes = 0;
        std::uint64_t v13_no_prune_no_exact_store_fixed_nodes = 0;
        std::uint64_t v13_no_prune_no_exact_store_iter_nodes = 0;
        std::uint64_t v13_no_prune_no_exact_store_asp_nodes = 0;
        std::uint64_t v13_no_prune_full_tt_fixed_nodes = 0;
        std::uint64_t v13_no_prune_full_tt_iter_nodes = 0;
        std::uint64_t v13_no_prune_full_tt_asp_nodes = 0;

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " depth=" << options.depth
                  << " iterative=" << (options.iterative ? 1 : 0)
                  << " seed=" << options.seed << '\n';

        for (std::size_t i = 0; i < samples.size(); ++i) {
            chess::HeuristicSearcherV12 v12;
            chess::HeuristicSearcherV13 v13;
            constexpr chess::V13SearchOptions NoSelectiveNoAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = true,
            };
            constexpr chess::V13SearchOptions NoSelectiveNoTtNoAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = false,
                .enable_tt = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveNoTtAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = true,
                .enable_tt = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveNoBoundsNoAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = false,
                .enable_tt = true,
                .enable_tt_bounds = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveNoBoundsAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = true,
                .enable_tt = true,
                .enable_tt_bounds = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveNoExactNoAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = false,
                .enable_tt = true,
                .enable_tt_bounds = true,
                .enable_tt_exact = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveNoExactAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = true,
                .enable_tt = true,
                .enable_tt_bounds = true,
                .enable_tt_exact = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveNoExactStoreNoAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = false,
                .enable_tt = true,
                .enable_tt_bounds = true,
                .enable_tt_exact_store = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveNoExactStoreAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = true,
                .enable_tt = true,
                .enable_tt_bounds = true,
                .enable_tt_exact_store = false,
            };
            constexpr chess::V13SearchOptions NoSelectiveFullTtNoAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = false,
                .enable_tt = true,
                .enable_tt_bounds = true,
                .enable_tt_exact = true,
                .enable_tt_bound_cutoff = true,
                .enable_tt_bound_tighten = true,
                .enable_tt_exact_store = true,
            };
            constexpr chess::V13SearchOptions NoSelectiveFullTtAsp{
                .enable_lmr = false,
                .enable_null_move_pruning = false,
                .enable_aspiration_window = true,
                .enable_tt = true,
                .enable_tt_bounds = true,
                .enable_tt_exact = true,
                .enable_tt_bound_cutoff = true,
                .enable_tt_bound_tighten = true,
                .enable_tt_exact_store = true,
            };
            chess::HeuristicSearcherV13 v13_no_prune_fixed(64, {}, NoSelectiveNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_iter(64, {}, NoSelectiveNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_asp(64, {}, NoSelectiveAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_tt_fixed(64, {}, NoSelectiveNoTtNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_tt_iter(64, {}, NoSelectiveNoTtNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_tt_asp(64, {}, NoSelectiveNoTtAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_bounds_fixed(64, {}, NoSelectiveNoBoundsNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_bounds_iter(64, {}, NoSelectiveNoBoundsNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_bounds_asp(64, {}, NoSelectiveNoBoundsAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_exact_fixed(64, {}, NoSelectiveNoExactNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_exact_iter(64, {}, NoSelectiveNoExactNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_exact_asp(64, {}, NoSelectiveNoExactAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_exact_store_fixed(64, {}, NoSelectiveNoExactStoreNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_exact_store_iter(64, {}, NoSelectiveNoExactStoreNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_no_exact_store_asp(64, {}, NoSelectiveNoExactStoreAsp);
            chess::HeuristicSearcherV13 v13_no_prune_full_tt_fixed(64, {}, NoSelectiveFullTtNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_full_tt_iter(64, {}, NoSelectiveFullTtNoAsp);
            chess::HeuristicSearcherV13 v13_no_prune_full_tt_asp(64, {}, NoSelectiveFullTtAsp);

            const chess::SearchResult v12_result = search(v12, samples[i].pos, options.depth, options.iterative);
            const chess::SearchResult v13_result = search(v13, samples[i].pos, options.depth, options.iterative);
            const chess::SearchResult v13_no_prune_fixed_result =
                v13_no_prune_fixed.search_best_move(samples[i].pos, options.depth);
            const chess::SearchResult v13_no_prune_iter_result =
                v13_no_prune_iter.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_asp_result =
                v13_no_prune_asp.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_no_tt_fixed_result =
                v13_no_prune_no_tt_fixed.search_best_move(samples[i].pos, options.depth);
            const chess::SearchResult v13_no_prune_no_tt_iter_result =
                v13_no_prune_no_tt_iter.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_no_tt_asp_result =
                v13_no_prune_no_tt_asp.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_no_bounds_fixed_result =
                v13_no_prune_no_bounds_fixed.search_best_move(samples[i].pos, options.depth);
            const chess::SearchResult v13_no_prune_no_bounds_iter_result =
                v13_no_prune_no_bounds_iter.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_no_bounds_asp_result =
                v13_no_prune_no_bounds_asp.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_no_exact_fixed_result =
                v13_no_prune_no_exact_fixed.search_best_move(samples[i].pos, options.depth);
            const chess::SearchResult v13_no_prune_no_exact_iter_result =
                v13_no_prune_no_exact_iter.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_no_exact_asp_result =
                v13_no_prune_no_exact_asp.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_no_exact_store_fixed_result =
                v13_no_prune_no_exact_store_fixed.search_best_move(samples[i].pos, options.depth);
            const chess::SearchResult v13_no_prune_no_exact_store_iter_result =
                v13_no_prune_no_exact_store_iter.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_no_exact_store_asp_result =
                v13_no_prune_no_exact_store_asp.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_full_tt_fixed_result =
                v13_no_prune_full_tt_fixed.search_best_move(samples[i].pos, options.depth);
            const chess::SearchResult v13_no_prune_full_tt_iter_result =
                v13_no_prune_full_tt_iter.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });
            const chess::SearchResult v13_no_prune_full_tt_asp_result =
                v13_no_prune_full_tt_asp.search_best_move(samples[i].pos, chess::SearchLimits{
                    .max_depth = options.depth,
                    .move_time = std::chrono::milliseconds{0},
                });

            v12_nodes += v12_result.nodes;
            v13_nodes += v13_result.nodes;
            v13_no_prune_fixed_nodes += v13_no_prune_fixed_result.nodes;
            v13_no_prune_iter_nodes += v13_no_prune_iter_result.nodes;
            v13_no_prune_asp_nodes += v13_no_prune_asp_result.nodes;
            v13_no_prune_no_tt_fixed_nodes += v13_no_prune_no_tt_fixed_result.nodes;
            v13_no_prune_no_tt_iter_nodes += v13_no_prune_no_tt_iter_result.nodes;
            v13_no_prune_no_tt_asp_nodes += v13_no_prune_no_tt_asp_result.nodes;
            v13_no_prune_no_bounds_fixed_nodes += v13_no_prune_no_bounds_fixed_result.nodes;
            v13_no_prune_no_bounds_iter_nodes += v13_no_prune_no_bounds_iter_result.nodes;
            v13_no_prune_no_bounds_asp_nodes += v13_no_prune_no_bounds_asp_result.nodes;
            v13_no_prune_no_exact_fixed_nodes += v13_no_prune_no_exact_fixed_result.nodes;
            v13_no_prune_no_exact_iter_nodes += v13_no_prune_no_exact_iter_result.nodes;
            v13_no_prune_no_exact_asp_nodes += v13_no_prune_no_exact_asp_result.nodes;
            v13_no_prune_no_exact_store_fixed_nodes += v13_no_prune_no_exact_store_fixed_result.nodes;
            v13_no_prune_no_exact_store_iter_nodes += v13_no_prune_no_exact_store_iter_result.nodes;
            v13_no_prune_no_exact_store_asp_nodes += v13_no_prune_no_exact_store_asp_result.nodes;
            v13_no_prune_full_tt_fixed_nodes += v13_no_prune_full_tt_fixed_result.nodes;
            v13_no_prune_full_tt_iter_nodes += v13_no_prune_full_tt_iter_result.nodes;
            v13_no_prune_full_tt_asp_nodes += v13_no_prune_full_tt_asp_result.nodes;

            const bool score_diff = v12_result.score != v13_result.score;
            const bool move_diff = v12_result.best_move != v13_result.best_move;
            score_mismatches += score_diff ? 1 : 0;
            move_mismatches += move_diff ? 1 : 0;
            const bool no_prune_iter_score_diff =
                v13_no_prune_fixed_result.score != v13_no_prune_iter_result.score;
            const bool no_prune_asp_score_diff =
                v13_no_prune_fixed_result.score != v13_no_prune_asp_result.score;
            const bool no_prune_iter_move_diff =
                v13_no_prune_fixed_result.best_move != v13_no_prune_iter_result.best_move;
            const bool no_prune_asp_move_diff =
                v13_no_prune_fixed_result.best_move != v13_no_prune_asp_result.best_move;
            v13_no_prune_iter_score_mismatches += no_prune_iter_score_diff ? 1 : 0;
            v13_no_prune_asp_score_mismatches += no_prune_asp_score_diff ? 1 : 0;
            v13_no_prune_iter_move_mismatches += no_prune_iter_move_diff ? 1 : 0;
            v13_no_prune_asp_move_mismatches += no_prune_asp_move_diff ? 1 : 0;
            const bool no_prune_no_tt_iter_score_diff =
                v13_no_prune_no_tt_fixed_result.score != v13_no_prune_no_tt_iter_result.score;
            const bool no_prune_no_tt_asp_score_diff =
                v13_no_prune_no_tt_fixed_result.score != v13_no_prune_no_tt_asp_result.score;
            const bool no_prune_no_tt_iter_move_diff =
                v13_no_prune_no_tt_fixed_result.best_move != v13_no_prune_no_tt_iter_result.best_move;
            const bool no_prune_no_tt_asp_move_diff =
                v13_no_prune_no_tt_fixed_result.best_move != v13_no_prune_no_tt_asp_result.best_move;
            v13_no_prune_no_tt_iter_score_mismatches += no_prune_no_tt_iter_score_diff ? 1 : 0;
            v13_no_prune_no_tt_asp_score_mismatches += no_prune_no_tt_asp_score_diff ? 1 : 0;
            v13_no_prune_no_tt_iter_move_mismatches += no_prune_no_tt_iter_move_diff ? 1 : 0;
            v13_no_prune_no_tt_asp_move_mismatches += no_prune_no_tt_asp_move_diff ? 1 : 0;
            const bool no_prune_no_bounds_iter_score_diff =
                v13_no_prune_no_bounds_fixed_result.score != v13_no_prune_no_bounds_iter_result.score;
            const bool no_prune_no_bounds_asp_score_diff =
                v13_no_prune_no_bounds_fixed_result.score != v13_no_prune_no_bounds_asp_result.score;
            const bool no_prune_no_bounds_iter_move_diff =
                v13_no_prune_no_bounds_fixed_result.best_move != v13_no_prune_no_bounds_iter_result.best_move;
            const bool no_prune_no_bounds_asp_move_diff =
                v13_no_prune_no_bounds_fixed_result.best_move != v13_no_prune_no_bounds_asp_result.best_move;
            v13_no_prune_no_bounds_iter_score_mismatches += no_prune_no_bounds_iter_score_diff ? 1 : 0;
            v13_no_prune_no_bounds_asp_score_mismatches += no_prune_no_bounds_asp_score_diff ? 1 : 0;
            v13_no_prune_no_bounds_iter_move_mismatches += no_prune_no_bounds_iter_move_diff ? 1 : 0;
            v13_no_prune_no_bounds_asp_move_mismatches += no_prune_no_bounds_asp_move_diff ? 1 : 0;
            const bool no_prune_no_exact_iter_score_diff =
                v13_no_prune_no_exact_fixed_result.score != v13_no_prune_no_exact_iter_result.score;
            const bool no_prune_no_exact_asp_score_diff =
                v13_no_prune_no_exact_fixed_result.score != v13_no_prune_no_exact_asp_result.score;
            v13_no_prune_no_exact_iter_score_mismatches += no_prune_no_exact_iter_score_diff ? 1 : 0;
            v13_no_prune_no_exact_asp_score_mismatches += no_prune_no_exact_asp_score_diff ? 1 : 0;
            const bool no_prune_no_exact_store_iter_score_diff =
                v13_no_prune_no_exact_store_fixed_result.score != v13_no_prune_no_exact_store_iter_result.score;
            const bool no_prune_no_exact_store_asp_score_diff =
                v13_no_prune_no_exact_store_fixed_result.score != v13_no_prune_no_exact_store_asp_result.score;
            v13_no_prune_no_exact_store_iter_score_mismatches += no_prune_no_exact_store_iter_score_diff ? 1 : 0;
            v13_no_prune_no_exact_store_asp_score_mismatches += no_prune_no_exact_store_asp_score_diff ? 1 : 0;
            const bool no_prune_full_tt_iter_score_diff =
                v13_no_prune_full_tt_fixed_result.score != v13_no_prune_full_tt_iter_result.score;
            const bool no_prune_full_tt_asp_score_diff =
                v13_no_prune_full_tt_fixed_result.score != v13_no_prune_full_tt_asp_result.score;
            const bool no_prune_full_tt_iter_move_diff =
                v13_no_prune_full_tt_fixed_result.best_move != v13_no_prune_full_tt_iter_result.best_move;
            const bool no_prune_full_tt_asp_move_diff =
                v13_no_prune_full_tt_fixed_result.best_move != v13_no_prune_full_tt_asp_result.best_move;
            v13_no_prune_full_tt_iter_score_mismatches += no_prune_full_tt_iter_score_diff ? 1 : 0;
            v13_no_prune_full_tt_asp_score_mismatches += no_prune_full_tt_asp_score_diff ? 1 : 0;
            v13_no_prune_full_tt_iter_move_mismatches += no_prune_full_tt_iter_move_diff ? 1 : 0;
            v13_no_prune_full_tt_asp_move_mismatches += no_prune_full_tt_asp_move_diff ? 1 : 0;

            if ((score_diff || move_diff) && score_mismatches + move_mismatches <= options.max_print * 2) {
                std::cout << "mismatch"
                          << " sample=" << i
                          << " line=" << samples[i].source_index
                          << " target=" << samples[i].target
                          << " legal=" << chess::generate_legal_moves(samples[i].pos).size()
                          << " v12_score=" << v12_result.score
                          << " v12_best=" << chess::move_to_string(v12_result.best_move)
                          << " v12_nodes=" << v12_result.nodes
                          << " v13_score=" << v13_result.score
                          << " v13_best=" << chess::move_to_string(v13_result.best_move)
                          << " v13_nodes=" << v13_result.nodes
                          << '\n';
            }
            if ((no_prune_iter_score_diff || no_prune_asp_score_diff)
                && v13_no_prune_iter_score_mismatches + v13_no_prune_asp_score_mismatches <= options.max_print * 2) {
                std::cout << "v13_no_prune_score_mismatch"
                          << " sample=" << i
                          << " line=" << samples[i].source_index
                          << " fixed_score=" << v13_no_prune_fixed_result.score
                          << " fixed_best=" << chess::move_to_string(v13_no_prune_fixed_result.best_move)
                          << " iter_score=" << v13_no_prune_iter_result.score
                          << " iter_best=" << chess::move_to_string(v13_no_prune_iter_result.best_move)
                          << " asp_score=" << v13_no_prune_asp_result.score
                          << " asp_best=" << chess::move_to_string(v13_no_prune_asp_result.best_move)
                          << '\n';
            }
            if ((no_prune_no_tt_iter_score_diff || no_prune_no_tt_asp_score_diff)
                && v13_no_prune_no_tt_iter_score_mismatches
                    + v13_no_prune_no_tt_asp_score_mismatches <= options.max_print * 2) {
                std::cout << "v13_no_prune_no_tt_score_mismatch"
                          << " sample=" << i
                          << " line=" << samples[i].source_index
                          << " fixed_score=" << v13_no_prune_no_tt_fixed_result.score
                          << " fixed_best=" << chess::move_to_string(v13_no_prune_no_tt_fixed_result.best_move)
                          << " iter_score=" << v13_no_prune_no_tt_iter_result.score
                          << " iter_best=" << chess::move_to_string(v13_no_prune_no_tt_iter_result.best_move)
                          << " asp_score=" << v13_no_prune_no_tt_asp_result.score
                          << " asp_best=" << chess::move_to_string(v13_no_prune_no_tt_asp_result.best_move)
                          << '\n';
            }
            if ((no_prune_no_bounds_iter_score_diff || no_prune_no_bounds_asp_score_diff)
                && v13_no_prune_no_bounds_iter_score_mismatches
                    + v13_no_prune_no_bounds_asp_score_mismatches <= options.max_print * 2) {
                std::cout << "v13_no_prune_no_bounds_score_mismatch"
                          << " sample=" << i
                          << " line=" << samples[i].source_index
                          << " fixed_score=" << v13_no_prune_no_bounds_fixed_result.score
                          << " fixed_best=" << chess::move_to_string(v13_no_prune_no_bounds_fixed_result.best_move)
                          << " iter_score=" << v13_no_prune_no_bounds_iter_result.score
                          << " iter_best=" << chess::move_to_string(v13_no_prune_no_bounds_iter_result.best_move)
                          << " asp_score=" << v13_no_prune_no_bounds_asp_result.score
                          << " asp_best=" << chess::move_to_string(v13_no_prune_no_bounds_asp_result.best_move)
                          << '\n';
            }
            if ((no_prune_no_exact_iter_score_diff || no_prune_no_exact_asp_score_diff)
                && v13_no_prune_no_exact_iter_score_mismatches
                    + v13_no_prune_no_exact_asp_score_mismatches <= options.max_print * 2) {
                std::cout << "v13_no_prune_no_exact_score_mismatch"
                          << " sample=" << i
                          << " line=" << samples[i].source_index
                          << " fixed_score=" << v13_no_prune_no_exact_fixed_result.score
                          << " iter_score=" << v13_no_prune_no_exact_iter_result.score
                          << " asp_score=" << v13_no_prune_no_exact_asp_result.score
                          << '\n';
            }
            if ((no_prune_no_exact_store_iter_score_diff || no_prune_no_exact_store_asp_score_diff)
                && v13_no_prune_no_exact_store_iter_score_mismatches
                    + v13_no_prune_no_exact_store_asp_score_mismatches <= options.max_print * 2) {
                std::cout << "v13_no_prune_no_exact_store_score_mismatch"
                          << " sample=" << i
                          << " line=" << samples[i].source_index
                          << " fixed_score=" << v13_no_prune_no_exact_store_fixed_result.score
                          << " iter_score=" << v13_no_prune_no_exact_store_iter_result.score
                          << " asp_score=" << v13_no_prune_no_exact_store_asp_result.score
                          << '\n';
            }
            if ((no_prune_full_tt_iter_score_diff || no_prune_full_tt_asp_score_diff)
                && v13_no_prune_full_tt_iter_score_mismatches
                    + v13_no_prune_full_tt_asp_score_mismatches <= options.max_print * 2) {
                std::cout << "v13_no_prune_full_tt_score_mismatch"
                          << " sample=" << i
                          << " line=" << samples[i].source_index
                          << " fixed_score=" << v13_no_prune_full_tt_fixed_result.score
                          << " fixed_best=" << chess::move_to_string(v13_no_prune_full_tt_fixed_result.best_move)
                          << " iter_score=" << v13_no_prune_full_tt_iter_result.score
                          << " iter_best=" << chess::move_to_string(v13_no_prune_full_tt_iter_result.best_move)
                          << " asp_score=" << v13_no_prune_full_tt_asp_result.score
                          << " asp_best=" << chess::move_to_string(v13_no_prune_full_tt_asp_result.best_move)
                          << '\n';
            }
        }

        std::cout << "summary"
                  << " samples=" << samples.size()
                  << " score_mismatches=" << score_mismatches
                  << " move_mismatches=" << move_mismatches
                  << " v12_nodes=" << v12_nodes
                  << " v13_nodes=" << v13_nodes;
        if (v12_nodes != 0) {
            std::cout << " v13_vs_v12_nodes="
                      << static_cast<double>(v13_nodes) / static_cast<double>(v12_nodes);
        }
        std::cout << '\n';

        std::cout << "v13_no_prune_summary"
                  << " samples=" << samples.size()
                  << " iter_score_mismatches_vs_fixed=" << v13_no_prune_iter_score_mismatches
                  << " asp_score_mismatches_vs_fixed=" << v13_no_prune_asp_score_mismatches
                  << " iter_move_mismatches_vs_fixed=" << v13_no_prune_iter_move_mismatches
                  << " asp_move_mismatches_vs_fixed=" << v13_no_prune_asp_move_mismatches
                  << " fixed_nodes=" << v13_no_prune_fixed_nodes
                  << " iter_nodes=" << v13_no_prune_iter_nodes
                  << " asp_nodes=" << v13_no_prune_asp_nodes;
        if (v13_no_prune_fixed_nodes != 0) {
            std::cout << " iter_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_iter_nodes)
                          / static_cast<double>(v13_no_prune_fixed_nodes)
                      << " asp_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_asp_nodes)
                          / static_cast<double>(v13_no_prune_fixed_nodes);
        }
        if (v13_no_prune_iter_nodes != 0) {
            std::cout << " asp_vs_iter_nodes="
                      << static_cast<double>(v13_no_prune_asp_nodes)
                          / static_cast<double>(v13_no_prune_iter_nodes);
        }
        std::cout << '\n';

        std::cout << "v13_no_prune_no_tt_summary"
                  << " samples=" << samples.size()
                  << " iter_score_mismatches_vs_fixed=" << v13_no_prune_no_tt_iter_score_mismatches
                  << " asp_score_mismatches_vs_fixed=" << v13_no_prune_no_tt_asp_score_mismatches
                  << " iter_move_mismatches_vs_fixed=" << v13_no_prune_no_tt_iter_move_mismatches
                  << " asp_move_mismatches_vs_fixed=" << v13_no_prune_no_tt_asp_move_mismatches
                  << " fixed_nodes=" << v13_no_prune_no_tt_fixed_nodes
                  << " iter_nodes=" << v13_no_prune_no_tt_iter_nodes
                  << " asp_nodes=" << v13_no_prune_no_tt_asp_nodes;
        if (v13_no_prune_no_tt_fixed_nodes != 0) {
            std::cout << " iter_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_no_tt_iter_nodes)
                          / static_cast<double>(v13_no_prune_no_tt_fixed_nodes)
                      << " asp_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_no_tt_asp_nodes)
                          / static_cast<double>(v13_no_prune_no_tt_fixed_nodes);
        }
        if (v13_no_prune_no_tt_iter_nodes != 0) {
            std::cout << " asp_vs_iter_nodes="
                      << static_cast<double>(v13_no_prune_no_tt_asp_nodes)
                          / static_cast<double>(v13_no_prune_no_tt_iter_nodes);
        }
        std::cout << '\n';

        std::cout << "v13_no_prune_no_bounds_summary"
                  << " samples=" << samples.size()
                  << " iter_score_mismatches_vs_fixed=" << v13_no_prune_no_bounds_iter_score_mismatches
                  << " asp_score_mismatches_vs_fixed=" << v13_no_prune_no_bounds_asp_score_mismatches
                  << " iter_move_mismatches_vs_fixed=" << v13_no_prune_no_bounds_iter_move_mismatches
                  << " asp_move_mismatches_vs_fixed=" << v13_no_prune_no_bounds_asp_move_mismatches
                  << " fixed_nodes=" << v13_no_prune_no_bounds_fixed_nodes
                  << " iter_nodes=" << v13_no_prune_no_bounds_iter_nodes
                  << " asp_nodes=" << v13_no_prune_no_bounds_asp_nodes;
        if (v13_no_prune_no_bounds_fixed_nodes != 0) {
            std::cout << " iter_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_no_bounds_iter_nodes)
                          / static_cast<double>(v13_no_prune_no_bounds_fixed_nodes)
                      << " asp_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_no_bounds_asp_nodes)
                          / static_cast<double>(v13_no_prune_no_bounds_fixed_nodes);
        }
        if (v13_no_prune_no_bounds_iter_nodes != 0) {
            std::cout << " asp_vs_iter_nodes="
                      << static_cast<double>(v13_no_prune_no_bounds_asp_nodes)
                          / static_cast<double>(v13_no_prune_no_bounds_iter_nodes);
        }
        std::cout << '\n';

        std::cout << "v13_no_prune_no_exact_summary"
                  << " samples=" << samples.size()
                  << " iter_score_mismatches_vs_fixed=" << v13_no_prune_no_exact_iter_score_mismatches
                  << " asp_score_mismatches_vs_fixed=" << v13_no_prune_no_exact_asp_score_mismatches
                  << " fixed_nodes=" << v13_no_prune_no_exact_fixed_nodes
                  << " iter_nodes=" << v13_no_prune_no_exact_iter_nodes
                  << " asp_nodes=" << v13_no_prune_no_exact_asp_nodes;
        if (v13_no_prune_no_exact_fixed_nodes != 0) {
            std::cout << " iter_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_no_exact_iter_nodes)
                          / static_cast<double>(v13_no_prune_no_exact_fixed_nodes)
                      << " asp_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_no_exact_asp_nodes)
                          / static_cast<double>(v13_no_prune_no_exact_fixed_nodes);
        }
        std::cout << '\n';

        std::cout << "v13_no_prune_no_exact_store_summary"
                  << " samples=" << samples.size()
                  << " iter_score_mismatches_vs_fixed=" << v13_no_prune_no_exact_store_iter_score_mismatches
                  << " asp_score_mismatches_vs_fixed=" << v13_no_prune_no_exact_store_asp_score_mismatches
                  << " fixed_nodes=" << v13_no_prune_no_exact_store_fixed_nodes
                  << " iter_nodes=" << v13_no_prune_no_exact_store_iter_nodes
                  << " asp_nodes=" << v13_no_prune_no_exact_store_asp_nodes;
        if (v13_no_prune_no_exact_store_fixed_nodes != 0) {
            std::cout << " iter_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_no_exact_store_iter_nodes)
                          / static_cast<double>(v13_no_prune_no_exact_store_fixed_nodes)
                      << " asp_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_no_exact_store_asp_nodes)
                          / static_cast<double>(v13_no_prune_no_exact_store_fixed_nodes);
        }
        std::cout << '\n';

        std::cout << "v13_no_prune_full_tt_summary"
                  << " samples=" << samples.size()
                  << " iter_score_mismatches_vs_fixed=" << v13_no_prune_full_tt_iter_score_mismatches
                  << " asp_score_mismatches_vs_fixed=" << v13_no_prune_full_tt_asp_score_mismatches
                  << " iter_move_mismatches_vs_fixed=" << v13_no_prune_full_tt_iter_move_mismatches
                  << " asp_move_mismatches_vs_fixed=" << v13_no_prune_full_tt_asp_move_mismatches
                  << " fixed_nodes=" << v13_no_prune_full_tt_fixed_nodes
                  << " iter_nodes=" << v13_no_prune_full_tt_iter_nodes
                  << " asp_nodes=" << v13_no_prune_full_tt_asp_nodes;
        if (v13_no_prune_full_tt_fixed_nodes != 0) {
            std::cout << " iter_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_full_tt_iter_nodes)
                          / static_cast<double>(v13_no_prune_full_tt_fixed_nodes)
                      << " asp_vs_fixed_nodes="
                      << static_cast<double>(v13_no_prune_full_tt_asp_nodes)
                          / static_cast<double>(v13_no_prune_full_tt_fixed_nodes);
        }
        if (v13_no_prune_full_tt_iter_nodes != 0) {
            std::cout << " asp_vs_iter_nodes="
                      << static_cast<double>(v13_no_prune_full_tt_asp_nodes)
                          / static_cast<double>(v13_no_prune_full_tt_iter_nodes);
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "compare_v12_v13_logic: " << error.what() << '\n';
        return 1;
    }
}
