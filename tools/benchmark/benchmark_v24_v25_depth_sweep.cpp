#include "board_encoder.hpp"
#include "heuristic_searcher_v25.hpp"
#include "heuristic_searcher_v24.hpp"
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
    std::string input = "data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl";
    int samples = 100;
    int min_depth = 7;
    int max_depth = 7;
    bool first_samples = true;
    bool instrument_ordering = false;
    bool compare_v24 = true;
    std::uint32_t seed = 20260614;
    int see_weight = chess::HeuristicSearcherV25::MoveOrderingWeights{}.see_weight;
    int qsearch_see_weight = chess::HeuristicSearcherV25::MoveOrderingWeights{}.qsearch_see_weight;
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
    std::uint64_t key_hits = 0;
    std::uint64_t move_hint_hits = 0;
    std::uint64_t depth_misses = 0;
    std::uint64_t score_hits = 0;
    std::uint64_t score_returns = 0;
    std::uint64_t index_collisions = 0;
    std::uint64_t stores = 0;
    std::uint64_t replacement_collisions = 0;
};

struct StageCutoffTotals {
    std::uint64_t tt_lower_stage0_attempts = 0;
    std::uint64_t tt_lower_stage0_cutoffs = 0;
    std::uint64_t priority_stage_cutoffs = 0;
    std::uint64_t quiet_stage_cutoffs = 0;
    std::uint64_t qsearch_cutoffs = 0;
    std::uint64_t priority_stage_calls = 0;
    std::uint64_t priority_legal_checks = 0;
    std::uint64_t priority_generated_moves = 0;
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
        } else if (arg == "--min-depth") {
            options.min_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-depth") {
            options.max_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--first") {
            options.first_samples = true;
        } else if (arg == "--random") {
            options.first_samples = false;
        } else if (arg == "--instrument-ordering") {
            options.instrument_ordering = true;
        } else if (arg == "--skip-v24") {
            options.compare_v24 = false;
        } else if (arg == "--see-weight") {
            options.see_weight = parse_int(require_value(arg), arg);
        } else if (arg == "--qsearch-see-weight") {
            options.qsearch_see_weight = parse_int(require_value(arg), arg);
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v25_v24_depth_sweep [--input path]\n"
                << "                                      [--samples N]\n"
                << "                                      [--min-depth D]\n"
                << "                                      [--max-depth D]\n"
                << "                                      [--seed N]\n"
                << "                                      [--first|--random]\n"
                << "                                      [--skip-v24]\n"
                << "                                      [--see-weight N]\n"
                << "                                      [--qsearch-see-weight N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.samples <= 0 || options.min_depth < 0 || options.max_depth < options.min_depth) {
        throw std::runtime_error("samples/depth arguments are invalid");
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

std::vector<Sample> load_samples(const Options& options) {
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
chess::SearchResult search_iterative(Searcher& searcher, const chess::Position& pos, int depth) {
    return searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = depth,
        .move_time = std::chrono::milliseconds{0},
    });
}

void add_result(Totals& totals, const chess::SearchResult& result, std::uint64_t time_us) {
    ++totals.searched;
    totals.nodes += result.nodes;
    totals.time_us += time_us;
}

void add_tt_stats(TtTotals& totals, const chess::RangeTranspositionTableStats& stats) {
    totals.probes += stats.probes;
    totals.key_hits += stats.key_hits;
    totals.move_hint_hits += stats.move_hint_hits;
    totals.depth_misses += stats.depth_misses;
    totals.score_hits += stats.exact_hits + stats.score_returns;
    totals.score_returns += stats.score_returns;
    totals.index_collisions += stats.index_collisions;
    totals.stores += stats.stores;
    totals.replacement_collisions += stats.replacement_collisions;
}

void add_stage_cutoff_stats(
    StageCutoffTotals& totals,
    const chess::HeuristicSearcherV25::MoveOrderingStats& stats
) {
    totals.tt_lower_stage0_attempts += stats.tt_lower_stage0_attempts;
    totals.tt_lower_stage0_cutoffs += stats.tt_lower_stage0_cutoffs;
    totals.priority_stage_cutoffs += stats.priority_stage_cutoffs;
    totals.quiet_stage_cutoffs += stats.quiet_stage_cutoffs;
    totals.qsearch_cutoffs += stats.qsearch.beta_cutoffs;
    totals.priority_stage_calls += stats.priority_king_safety_contexts;
    totals.priority_legal_checks += stats.priority_legal_checks;
    totals.priority_generated_moves += stats.noisy_stage.total;
}

std::uint64_t avg_u64(std::uint64_t value, int count) {
    return count == 0 ? 0 : value / static_cast<std::uint64_t>(count);
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Sample> samples = load_samples(options);

        std::cout << "input=" << options.input
                  << " samples=" << samples.size()
                  << " min_depth=" << options.min_depth
                  << " max_depth=" << options.max_depth
                  << " first=" << (options.first_samples ? 1 : 0)
                  << " compare_v24=" << (options.compare_v24 ? 1 : 0)
                  << " seed=" << options.seed
                  << " see_weight=" << options.see_weight
                  << " qsearch_see_weight=" << options.qsearch_see_weight
                  << '\n' << std::flush;

        for (int depth = options.min_depth; depth <= options.max_depth; ++depth) {
            Totals v25_totals;
            Totals v24_totals;
            TtTotals v25_tt;
            TtTotals v24_tt;
            StageCutoffTotals v25_stage_cutoffs;

            for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
                const Sample& sample = samples[sample_index];
                chess::HeuristicSearcherV25::MoveOrderingWeights weights;
                weights.see_weight = options.see_weight;
                weights.qsearch_see_weight = options.qsearch_see_weight;
                chess::HeuristicSearcherV25 v25(64, 4, 10, 14, 14'000, weights);
                v25.clear_tt_stats();
                if (options.instrument_ordering) {
                    v25.clear_move_ordering_stats();
                    v25.set_move_ordering_stats_enabled(true);
                }

                const auto v25_start = std::chrono::steady_clock::now();
                const chess::SearchResult v25_result = search_iterative(v25, sample.pos, depth);
                const auto v25_end = std::chrono::steady_clock::now();

                chess::SearchResult v24_result{};
                std::uint64_t v24_us = 0;
                chess::HeuristicSearcherV24 v24;
                if (options.compare_v24) {
                    v24.clear_tt_stats();
                    const auto v24_start = std::chrono::steady_clock::now();
                    v24_result = search_iterative(v24, sample.pos, depth);
                    const auto v24_end = std::chrono::steady_clock::now();
                    v24_us = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(v24_end - v24_start).count()
                    );
                }

                const auto v25_us = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(v25_end - v25_start).count()
                );
                add_result(v25_totals, v25_result, v25_us);
                if (options.compare_v24) {
                    add_result(v24_totals, v24_result, v24_us);
                }
                add_tt_stats(v25_tt, v25.tt_stats());
                if (options.compare_v24) {
                    add_tt_stats(v24_tt, v24.tt_stats());
                }
                if (options.instrument_ordering) {
                    add_stage_cutoff_stats(v25_stage_cutoffs, v25.move_ordering_stats());
                }
                if (options.compare_v24 && v25_result.score != v24_result.score) {
                    ++v25_totals.score_mismatches;
                    ++v24_totals.score_mismatches;
                }
                if (options.compare_v24 && v25_result.best_move != v24_result.best_move) {
                    ++v25_totals.move_mismatches;
                    ++v24_totals.move_mismatches;
                }

                std::cout << "sample=" << sample_index
                          << " line=" << sample.source_index
                          << " depth=" << depth
                          << " v25_score=" << v25_result.score
                          << " v24_score=" << (options.compare_v24 ? v24_result.score : 0)
                          << " score_diff=" << (options.compare_v24 ? v24_result.score - v25_result.score : 0)
                          << " v25_best=" << chess::move_to_string(v25_result.best_move)
                          << " v24_best=" << (options.compare_v24 ? chess::move_to_string(v24_result.best_move) : "-")
                          << " v25_nodes=" << v25_result.nodes
                          << " v24_nodes=" << v24_result.nodes
                          << " v25_us=" << v25_us
                          << " v24_us=" << v24_us
                          << '\n' << std::flush;
            }

            std::cout << "depth=" << depth
                      << " v25_nodes=" << v25_totals.nodes
                      << " v25_avg_nodes=" << avg_u64(v25_totals.nodes, v25_totals.searched)
                      << " v25_us=" << v25_totals.time_us
                      << " v25_avg_us=" << avg_u64(v25_totals.time_us, v25_totals.searched)
                      << " v24_nodes=" << v24_totals.nodes
                      << " v24_avg_nodes=" << avg_u64(v24_totals.nodes, v24_totals.searched)
                      << " v24_us=" << v24_totals.time_us
                      << " v24_avg_us=" << avg_u64(v24_totals.time_us, v24_totals.searched);
            if (v25_totals.nodes != 0) {
                std::cout << " v24_vs_v25_nodes="
                          << (static_cast<double>(v24_totals.nodes) / static_cast<double>(v25_totals.nodes));
            }
            if (v25_totals.time_us != 0) {
                std::cout << " v24_vs_v25_time="
                          << (static_cast<double>(v24_totals.time_us) / static_cast<double>(v25_totals.time_us));
            }
            const std::uint64_t v25_main_stage_cutoffs =
                v25_stage_cutoffs.tt_lower_stage0_cutoffs
                + v25_stage_cutoffs.priority_stage_cutoffs
                + v25_stage_cutoffs.quiet_stage_cutoffs;
            std::cout << " score_mismatches=" << v25_totals.score_mismatches
                      << " move_mismatches=" << v25_totals.move_mismatches;
            if (options.instrument_ordering) {
                std::cout << " v25_stage0_attempts=" << v25_stage_cutoffs.tt_lower_stage0_attempts
                      << " v25_stage0_cutoffs=" << v25_stage_cutoffs.tt_lower_stage0_cutoffs
                      << " v25_stage0_cutoff_rate="
                      << ratio(v25_stage_cutoffs.tt_lower_stage0_cutoffs, v25_stage_cutoffs.tt_lower_stage0_attempts)
                      << " v25_priority_cutoffs=" << v25_stage_cutoffs.priority_stage_cutoffs
                      << " v25_priority_cutoff_share="
                      << ratio(v25_stage_cutoffs.priority_stage_cutoffs, v25_main_stage_cutoffs)
                      << " v25_priority_stage_calls=" << v25_stage_cutoffs.priority_stage_calls
                      << " v25_priority_legal_checks=" << v25_stage_cutoffs.priority_legal_checks
                      << " v25_priority_legal_checks_per_call="
                      << ratio(v25_stage_cutoffs.priority_legal_checks, v25_stage_cutoffs.priority_stage_calls)
                      << " v25_priority_generated_moves=" << v25_stage_cutoffs.priority_generated_moves
                      << " v25_priority_generated_moves_per_call="
                      << ratio(v25_stage_cutoffs.priority_generated_moves, v25_stage_cutoffs.priority_stage_calls)
                      << " v25_quiet_cutoffs=" << v25_stage_cutoffs.quiet_stage_cutoffs
                      << " v25_quiet_cutoff_share="
                      << ratio(v25_stage_cutoffs.quiet_stage_cutoffs, v25_main_stage_cutoffs)
                      << " v25_stage0_cutoff_share="
                      << ratio(v25_stage_cutoffs.tt_lower_stage0_cutoffs, v25_main_stage_cutoffs)
                      << " v25_main_stage_cutoffs=" << v25_main_stage_cutoffs
                      << " v25_qsearch_cutoffs=" << v25_stage_cutoffs.qsearch_cutoffs;
            }
            std::cout << " v25_tt_probes=" << v25_tt.probes
                      << " v25_tt_key_hits=" << v25_tt.key_hits
                      << " v25_tt_key_hit_rate=" << ratio(v25_tt.key_hits, v25_tt.probes)
                      << " v25_tt_score_hits=" << v25_tt.score_hits
                      << " v25_tt_score_hit_rate=" << ratio(v25_tt.score_hits, v25_tt.probes)
                      << " v25_tt_score_returns=" << v25_tt.score_returns
                      << " v25_tt_move_hint_hits=" << v25_tt.move_hint_hits
                      << " v25_tt_move_hint_rate=" << ratio(v25_tt.move_hint_hits, v25_tt.probes)
                      << " v25_tt_index_collisions=" << v25_tt.index_collisions
                      << " v25_tt_probe_collision_rate=" << ratio(v25_tt.index_collisions, v25_tt.probes)
                      << " v25_tt_depth_misses=" << v25_tt.depth_misses
                      << " v25_tt_stores=" << v25_tt.stores
                      << " v25_tt_replacement_collisions=" << v25_tt.replacement_collisions
                      << " v25_tt_store_collision_rate=" << ratio(v25_tt.replacement_collisions, v25_tt.stores)
                      << " v24_tt_probes=" << v24_tt.probes
                      << " v24_tt_key_hits=" << v24_tt.key_hits
                      << " v24_tt_key_hit_rate=" << ratio(v24_tt.key_hits, v24_tt.probes)
                      << " v24_tt_score_hits=" << v24_tt.score_hits
                      << " v24_tt_score_hit_rate=" << ratio(v24_tt.score_hits, v24_tt.probes)
                      << " v24_tt_score_returns=" << v24_tt.score_returns
                      << " v24_tt_move_hint_hits=" << v24_tt.move_hint_hits
                      << " v24_tt_move_hint_rate=" << ratio(v24_tt.move_hint_hits, v24_tt.probes)
                      << " v24_tt_index_collisions=" << v24_tt.index_collisions
                      << " v24_tt_probe_collision_rate=" << ratio(v24_tt.index_collisions, v24_tt.probes)
                      << " v24_tt_depth_misses=" << v24_tt.depth_misses
                      << " v24_tt_stores=" << v24_tt.stores
                      << " v24_tt_replacement_collisions=" << v24_tt.replacement_collisions
                      << " v24_tt_store_collision_rate=" << ratio(v24_tt.replacement_collisions, v24_tt.stores);
            std::cout << '\n' << std::flush;
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark_v25_v24_depth_sweep: " << error.what() << '\n';
        return 1;
    }
}
