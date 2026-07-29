#include "attacks.hpp"
#include "move.hpp"
#include "nnue_searcher_v36.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model =
        std::string(chess::DefaultPhaseQuantizedNnueModelPath);
    std::string book =
        "data/stockfish_balanced_openings_10ply_1k_20260723.txt";
    int samples = 10;
    int depth = 5;
    std::uint32_t seed = 20260724;
    std::size_t tt_mb = 64;
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
        throw std::runtime_error(
            "invalid integer for " + std::string(name) + ": "
            + std::string(value));
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        auto require_value = [&]() -> std::string_view {
            if (index + 1 >= argc) {
                throw std::runtime_error(
                    "missing value for " + std::string(arg));
            }
            return argv[++index];
        };
        if (arg == "--model") {
            options.model = std::string(require_value());
        } else if (arg == "--book") {
            options.book = std::string(require_value());
        } else if (arg == "--samples") {
            options.samples = parse_int(require_value(), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(
                parse_int(require_value(), arg));
        } else if (arg == "--tt-mb") {
            const int value = parse_int(require_value(), arg);
            if (value <= 0) {
                throw std::runtime_error("--tt-mb must be positive");
            }
            options.tt_mb = static_cast<std::size_t>(value);
        } else if (arg == "--help") {
            std::cout
                << "Usage: measure_nnue_v36_cutoff_stats"
                << " [--model path] [--book path]"
                << " [--samples 10] [--depth 5]"
                << " [--seed N] [--tt-mb 64]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.samples <= 0 || options.depth <= 0) {
        throw std::runtime_error("samples and depth must be positive");
    }
    return options;
}

std::string strip_comment(std::string line) {
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) {
        line.resize(comment);
    }
    return line;
}

std::vector<std::string> split_words(const std::string& line) {
    std::istringstream input(line);
    std::vector<std::string> words;
    std::string word;
    while (input >> word) {
        words.push_back(word);
    }
    return words;
}

chess::Move find_legal_uci_move(
    const chess::Position& position,
    std::string_view uci
) {
    for (chess::Move move : chess::generate_legal_moves(position)) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    throw std::runtime_error("illegal book move: " + std::string(uci));
}

struct Sample {
    int line_number = 0;
    std::string moves;
    chess::Position position;
};

std::vector<Sample> load_samples(const Options& options) {
    std::ifstream input(options.book);
    if (!input) {
        throw std::runtime_error("failed to open book: " + options.book);
    }
    std::vector<Sample> all;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string clean = strip_comment(line);
        const std::vector<std::string> moves = split_words(clean);
        if (moves.empty()) {
            continue;
        }
        chess::Position position;
        position.set_startpos();
        for (const std::string& move : moves) {
            position.make_move(find_legal_uci_move(position, move));
        }
        all.push_back(Sample{line_number, clean, position});
    }
    if (static_cast<int>(all.size()) < options.samples) {
        throw std::runtime_error("book has too few positions");
    }
    std::mt19937 rng(options.seed);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(static_cast<std::size_t>(options.samples));
    return all;
}

void add_stats(
    chess::NnueSearcherV36::MoveOrderingStats& total,
    const chess::NnueSearcherV36::MoveOrderingStats& current
) {
    total.beta_cutoffs += current.beta_cutoffs;
    for (std::size_t index = 0;
         index < total.cutoff_move_index.size();
         ++index) {
        total.cutoff_move_index[index] += current.cutoff_move_index[index];
    }
    for (std::size_t stage = 0;
         stage < total.cutoff_stage.size();
         ++stage) {
        total.cutoff_stage[stage] += current.cutoff_stage[stage];
    }
    total.cutoff_by_capture += current.cutoff_by_capture;
    total.cutoff_by_check += current.cutoff_by_check;
    total.cutoff_by_promotion += current.cutoff_by_promotion;
    total.cutoff_by_quiet += current.cutoff_by_quiet;
}

double percent(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
        ? 0.0
        : 100.0 * static_cast<double>(numerator)
            / static_cast<double>(denominator);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model: " + options.model);
        }
        const std::vector<Sample> samples = load_samples(options);
        chess::NnueSearcherV36::MoveOrderingStats total_stats;
        std::uint64_t total_nodes = 0;

        std::cout << "settings model=" << options.model
                  << " book=" << options.book
                  << " samples=" << samples.size()
                  << " depth=" << options.depth
                  << " fixed_depth=1"
                  << " seed=" << options.seed
                  << " includes_qsearch_cutoffs=0"
                  << " includes_tt_score_cutoffs=0\n";

        for (std::size_t index = 0; index < samples.size(); ++index) {
            chess::NnueSearcherV36 searcher(model, options.tt_mb, 4);
            searcher.set_move_ordering_stats_enabled(true);
            const chess::SearchResult result =
                searcher.search_best_move(samples[index].position, options.depth);
            const auto& stats = searcher.move_ordering_stats();
            add_stats(total_stats, stats);
            total_nodes += result.nodes;
            std::cout << "position index=" << index
                      << " book_line=" << samples[index].line_number
                      << " best_move=" << chess::move_to_string(result.best_move)
                      << " score=" << result.score
                      << " nodes=" << result.nodes
                      << " beta_cutoffs=" << stats.beta_cutoffs
                      << " first_move_cutoff_pct="
                      << std::fixed << std::setprecision(4)
                      << percent(
                          stats.cutoff_move_index[0],
                          stats.beta_cutoffs)
                      << " opening=\"" << samples[index].moves << "\"\n";
        }

        std::cout << "summary nodes=" << total_nodes
                  << " avg_nodes=" << total_nodes / samples.size()
                  << " beta_cutoffs=" << total_stats.beta_cutoffs << '\n';

        for (std::size_t index = 0;
             index < total_stats.cutoff_move_index.size();
             ++index) {
            std::cout << "cutoff_move"
                      << " move_number="
                      << (index + 1 == total_stats.cutoff_move_index.size()
                          ? std::string("9+")
                          : std::to_string(index + 1))
                      << " count=" << total_stats.cutoff_move_index[index]
                      << " pct=" << std::fixed << std::setprecision(6)
                      << percent(
                          total_stats.cutoff_move_index[index],
                          total_stats.beta_cutoffs)
                      << '\n';
        }

        constexpr std::array<const char*, 6> stage_names{
            "tt_lower",
            "promotion",
            "good_capture",
            "priority_quiet",
            "quiet",
            "bad_capture",
        };
        for (std::size_t stage = 0; stage < stage_names.size(); ++stage) {
            std::cout << "cutoff_stage"
                      << " stage=" << stage_names[stage]
                      << " count=" << total_stats.cutoff_stage[stage]
                      << " pct=" << std::fixed << std::setprecision(6)
                      << percent(
                          total_stats.cutoff_stage[stage],
                          total_stats.beta_cutoffs)
                      << '\n';
        }

        std::cout << "cutoff_property"
                  << " capture=" << total_stats.cutoff_by_capture
                  << " capture_pct="
                  << percent(
                      total_stats.cutoff_by_capture,
                      total_stats.beta_cutoffs)
                  << " check=" << total_stats.cutoff_by_check
                  << " check_pct="
                  << percent(
                      total_stats.cutoff_by_check,
                      total_stats.beta_cutoffs)
                  << " promotion=" << total_stats.cutoff_by_promotion
                  << " promotion_pct="
                  << percent(
                      total_stats.cutoff_by_promotion,
                      total_stats.beta_cutoffs)
                  << " quiet=" << total_stats.cutoff_by_quiet
                  << " quiet_pct="
                  << percent(
                      total_stats.cutoff_by_quiet,
                      total_stats.beta_cutoffs)
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "measure_nnue_v36_cutoff_stats: "
                  << error.what() << '\n';
        return 1;
    }
}
