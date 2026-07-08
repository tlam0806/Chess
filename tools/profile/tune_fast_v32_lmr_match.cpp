#include "attacks.hpp"
#include "heuristic_searcher_fast_v32.hpp"
#include "move.hpp"
#include "position.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string book_path = "data/opening_book_6plies.txt";
    int games_per_variant = 20;
    int movetime_ms = 1000;
    int max_depth = 64;
    int max_plies = 120;
    int time_limit_seconds = 3600;
    int tt_mb = 24;
    std::uint32_t seed = 20260708;
    std::vector<std::string> variant_specs;
};

struct BookLine {
    int line_number = 0;
    std::string text;
    std::vector<std::string> moves;
};

struct Variant {
    chess::HeuristicSearcherFastV32::LmrConfig config{};
    std::string name;
};

struct EngineStats {
    std::uint64_t nodes = 0;
    std::uint64_t time_us = 0;
    std::uint64_t moves = 0;
};

struct MatchScore {
    int wins = 0;
    int draws = 0;
    int losses = 0;
};

struct VariantResult {
    Variant variant;
    MatchScore score;
    EngineStats candidate_stats;
    EngineStats baseline_stats;
    int invalid_candidate_moves = 0;
    int invalid_baseline_moves = 0;
    int games = 0;
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

        if (arg == "--book") {
            options.book_path = std::string(require_value(arg));
        } else if (arg == "--games-per-variant") {
            options.games_per_variant = parse_int(require_value(arg), arg);
        } else if (arg == "--movetime-ms") {
            options.movetime_ms = parse_int(require_value(arg), arg);
        } else if (arg == "--max-depth") {
            options.max_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--time-limit-seconds") {
            options.time_limit_seconds = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-mb") {
            options.tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--variant") {
            options.variant_specs.push_back(std::string(require_value(arg)));
        } else if (arg == "--help") {
            std::cout
                << "Usage: tune_fast_v32_lmr_match [--book path]\n"
                << "                               [--games-per-variant N]\n"
                << "                               [--movetime-ms N] [--max-depth N]\n"
                << "                               [--max-plies N] [--time-limit-seconds N]\n"
                << "                               [--tt-mb N] [--seed N]\n"
                << "                               [--variant base,div,min_depth,min_idx]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.games_per_variant <= 0 || options.movetime_ms < 0 || options.max_depth <= 0
        || options.max_plies <= 0 || options.time_limit_seconds <= 0 || options.tt_mb <= 0) {
        throw std::runtime_error("invalid non-positive option");
    }
    return options;
}

std::string strip_comment(std::string_view line) {
    const std::size_t hash = line.find('#');
    std::string clean(line.substr(0, hash == std::string_view::npos ? line.size() : hash));
    while (!clean.empty() && (clean.back() == ' ' || clean.back() == '\t' || clean.back() == '\r')) {
        clean.pop_back();
    }
    return clean;
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

chess::Move find_legal_uci_move(const chess::Position& pos, const std::string& uci) {
    chess::MoveList moves;
    chess::generate_legal_moves(pos, moves);
    for (chess::Move move : moves) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    throw std::runtime_error("illegal book move: " + uci);
}

std::vector<BookLine> load_book(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open book: " + path);
    }

    std::vector<BookLine> book;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string clean = strip_comment(line);
        std::vector<std::string> moves = split_words(clean);
        if (moves.empty()) {
            continue;
        }
        if (moves.size() != 6) {
            throw std::runtime_error("book line " + std::to_string(line_number) + " must have 6 plies");
        }
        book.push_back(BookLine{line_number, clean, std::move(moves)});
    }
    if (book.empty()) {
        throw std::runtime_error("book is empty");
    }
    return book;
}

chess::Position position_after_book(const BookLine& line) {
    chess::Position pos;
    pos.set_startpos();
    for (const std::string& uci : line.moves) {
        pos.make_move(find_legal_uci_move(pos, uci));
    }
    return pos;
}

bool contains_move(const chess::MoveList& moves, chess::Move target) {
    for (chess::Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

chess::Move choose_move(
    chess::HeuristicSearcherFastV32& searcher,
    const chess::Position& pos,
    const Options& options,
    EngineStats& stats,
    int& invalid_moves
) {
    const chess::SearchLimits limits{
        .max_depth = options.max_depth,
        .move_time = std::chrono::milliseconds{options.movetime_ms},
    };
    const auto start = Clock::now();
    const chess::SearchResult result = searcher.search_best_move(pos, limits);
    const auto end = Clock::now();
    stats.nodes += result.nodes;
    stats.time_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    ++stats.moves;

    chess::MoveList legal_moves;
    chess::generate_legal_moves(pos, legal_moves);
    if (contains_move(legal_moves, result.best_move)) {
        return result.best_move;
    }
    ++invalid_moves;
    return legal_moves.empty() ? chess::Move{} : legal_moves[0];
}

enum class GameOutcome {
    CandidateWin,
    BaselineWin,
    Draw,
};

GameOutcome play_game(
    const Variant& variant,
    const BookLine& line,
    bool candidate_is_white,
    const Options& options,
    VariantResult& result
) {
    chess::Position pos = position_after_book(line);
    chess::HeuristicSearcherFastV32 candidate(static_cast<std::size_t>(options.tt_mb));
    chess::HeuristicSearcherFastV32 baseline(static_cast<std::size_t>(options.tt_mb));
    candidate.set_lmr_config(variant.config);

    for (int ply = 0; ply < options.max_plies; ++ply) {
        chess::MoveList legal_moves;
        chess::generate_legal_moves(pos, legal_moves);
        if (legal_moves.empty()) {
            if (!chess::in_check(pos, pos.side_to_move)) {
                return GameOutcome::Draw;
            }
            const bool white_lost = pos.side_to_move == chess::Color::White;
            const bool candidate_lost = white_lost == candidate_is_white;
            return candidate_lost ? GameOutcome::BaselineWin : GameOutcome::CandidateWin;
        }

        const bool candidate_to_move =
            (pos.side_to_move == chess::Color::White) == candidate_is_white;
        chess::Move move = candidate_to_move
            ? choose_move(candidate, pos, options, result.candidate_stats, result.invalid_candidate_moves)
            : choose_move(baseline, pos, options, result.baseline_stats, result.invalid_baseline_moves);
        if (move.value == 0) {
            return GameOutcome::Draw;
        }
        pos.make_move(move);
    }
    return GameOutcome::Draw;
}

std::vector<Variant> make_variants(std::uint32_t seed) {
    std::vector<Variant> variants;
    const std::vector<double> bases{0.25, 0.5, 0.75, 1.0};
    const std::vector<double> divisors{2.2, 2.6, 3.0, 3.4, 3.8, 4.4};
    const std::vector<int> min_depths{3, 4};
    const std::vector<std::size_t> min_indices{2, 3, 4, 5};

    for (double base : bases) {
        for (double divisor : divisors) {
            for (int min_depth : min_depths) {
                for (std::size_t min_index : min_indices) {
                    chess::HeuristicSearcherFastV32::LmrConfig config;
                    config.enabled = true;
                    config.base = base;
                    config.divisor = divisor;
                    config.min_depth = min_depth;
                    config.min_move_index = min_index;
                    std::ostringstream name;
                    name << "base=" << base
                         << ",div=" << divisor
                         << ",min_depth=" << min_depth
                         << ",min_idx=" << min_index;
                    variants.push_back(Variant{config, name.str()});
                }
            }
        }
    }

    std::mt19937 rng(seed);
    std::shuffle(variants.begin(), variants.end(), rng);
    return variants;
}

Variant parse_variant_spec(const std::string& spec) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= spec.size()) {
        const std::size_t comma = spec.find(',', start);
        const std::size_t end = comma == std::string::npos ? spec.size() : comma;
        parts.push_back(spec.substr(start, end - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (parts.size() != 4) {
        throw std::runtime_error("variant must be base,div,min_depth,min_idx: " + spec);
    }

    chess::HeuristicSearcherFastV32::LmrConfig config;
    config.enabled = true;
    config.base = std::stod(parts[0]);
    config.divisor = std::stod(parts[1]);
    config.min_depth = parse_int(parts[2], "variant min_depth");
    config.min_move_index = static_cast<std::size_t>(parse_int(parts[3], "variant min_idx"));
    std::ostringstream name;
    name << "base=" << config.base
         << ",div=" << config.divisor
         << ",min_depth=" << config.min_depth
         << ",min_idx=" << config.min_move_index;
    return Variant{config, name.str()};
}

std::vector<Variant> selected_variants(const Options& options) {
    if (options.variant_specs.empty()) {
        return make_variants(options.seed);
    }

    std::vector<Variant> variants;
    variants.reserve(options.variant_specs.size());
    for (const std::string& spec : options.variant_specs) {
        variants.push_back(parse_variant_spec(spec));
    }
    return variants;
}

double score_points(const MatchScore& score) {
    return static_cast<double>(score.wins) + 0.5 * static_cast<double>(score.draws);
}

double nps(const EngineStats& stats) {
    return stats.time_us == 0 ? 0.0
        : static_cast<double>(stats.nodes) * 1'000'000.0 / static_cast<double>(stats.time_us);
}

void print_result(const VariantResult& result) {
    const double points = score_points(result.score);
    std::cout << "variant \"" << result.variant.name << "\""
              << " games=" << result.games
              << " score=" << points << "/" << result.games
              << " wdl=" << result.score.wins << "/" << result.score.draws << "/" << result.score.losses
              << " cand_nodes=" << result.candidate_stats.nodes
              << " base_nodes=" << result.baseline_stats.nodes
              << " cand_nps=" << nps(result.candidate_stats)
              << " base_nps=" << nps(result.baseline_stats)
              << " cand_invalid=" << result.invalid_candidate_moves
              << " base_invalid=" << result.invalid_baseline_moves
              << '\n'
              << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<BookLine> book = load_book(options.book_path);
        const std::vector<Variant> variants = selected_variants(options);
        const Clock::time_point start = Clock::now();
        const Clock::time_point deadline = start + std::chrono::seconds{options.time_limit_seconds};

        std::cout << "book_lines=" << book.size()
                  << " variants=" << variants.size()
                  << " games_per_variant=" << options.games_per_variant
                  << " movetime_ms=" << options.movetime_ms
                  << " max_depth=" << options.max_depth
                  << " max_plies=" << options.max_plies
                  << " tt_mb=" << options.tt_mb
                  << " time_limit_seconds=" << options.time_limit_seconds
                  << '\n'
                  << std::flush;

        std::vector<VariantResult> results;
        for (std::size_t variant_index = 0; variant_index < variants.size(); ++variant_index) {
            if (Clock::now() >= deadline) {
                break;
            }
            VariantResult result;
            result.variant = variants[variant_index];

            for (int game = 0; game < options.games_per_variant; ++game) {
                if (Clock::now() >= deadline) {
                    break;
                }
                const BookLine& line =
                    book[(variant_index * static_cast<std::size_t>(options.games_per_variant)
                        + static_cast<std::size_t>(game)) % book.size()];
                const bool candidate_is_white = game % 2 == 0;
                const GameOutcome outcome =
                    play_game(result.variant, line, candidate_is_white, options, result);
                ++result.games;
                if (outcome == GameOutcome::CandidateWin) {
                    ++result.score.wins;
                } else if (outcome == GameOutcome::BaselineWin) {
                    ++result.score.losses;
                } else {
                    ++result.score.draws;
                }
            }

            if (result.games == 0) {
                break;
            }
            print_result(result);
            results.push_back(result);
        }

        std::sort(results.begin(), results.end(), [](const VariantResult& lhs, const VariantResult& rhs) {
            const double lhs_points = score_points(lhs.score);
            const double rhs_points = score_points(rhs.score);
            if (lhs_points != rhs_points) {
                return lhs_points > rhs_points;
            }
            return lhs.candidate_stats.nodes < rhs.candidate_stats.nodes;
        });

        std::cout << "top_results\n";
        const std::size_t top_count = std::min<std::size_t>(10, results.size());
        for (std::size_t i = 0; i < top_count; ++i) {
            std::cout << "rank=" << i + 1 << ' ';
            print_result(results[i]);
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
