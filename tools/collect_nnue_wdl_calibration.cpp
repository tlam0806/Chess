#include "attacks.hpp"
#include "game_state.hpp"
#include "move.hpp"
#include "nnue_searcher_v36.hpp"
#include "nnue_searcher_v38.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Opening {
    int source_line = 0;
    std::vector<std::string> moves;
};

struct Options {
    std::string book;
    std::string model;
    std::string output;
    int games = 600;
    int base_ms = 10'000;
    int increment_ms = 100;
    int overhead_ms = 20;
    int max_plies = 200;
    int sample_every = 4;
    int samples_per_game = 8;
    int teacher_depth = 8;
    std::size_t tt_mb = 64;
    std::uint32_t seed = 20260803;
    bool include_max_plies = false;
};

struct Sample {
    chess::Position position;
    int game_ply = 0;
};

struct GameResult {
    std::string result;
    std::string reason;
    int plies = 0;
    std::vector<Sample> samples;
};

int parse_int(std::string_view value, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const int result = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) throw std::invalid_argument("trailing");
        return result;
    } catch (...) {
        throw std::runtime_error("invalid integer for " + std::string(name));
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(
                    "missing value for " + std::string(arg));
            }
            return argv[index];
        };
        if (arg == "--book") options.book = next();
        else if (arg == "--model") options.model = next();
        else if (arg == "--output") options.output = next();
        else if (arg == "--games") options.games = parse_int(next(), arg);
        else if (arg == "--base-ms") options.base_ms = parse_int(next(), arg);
        else if (arg == "--increment-ms") {
            options.increment_ms = parse_int(next(), arg);
        } else if (arg == "--overhead-ms") {
            options.overhead_ms = parse_int(next(), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(next(), arg);
        } else if (arg == "--sample-every") {
            options.sample_every = parse_int(next(), arg);
        } else if (arg == "--samples-per-game") {
            options.samples_per_game = parse_int(next(), arg);
        } else if (arg == "--teacher-depth") {
            options.teacher_depth = parse_int(next(), arg);
        } else if (arg == "--tt-mb") {
            options.tt_mb = static_cast<std::size_t>(parse_int(next(), arg));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(next(), arg));
        } else if (arg == "--include-max-plies") {
            options.include_max_plies = true;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.book.empty() || options.model.empty()
        || options.output.empty()) {
        throw std::runtime_error("--book, --model and --output are required");
    }
    if (options.games <= 0 || options.base_ms <= 0
        || options.increment_ms < 0 || options.overhead_ms < 0
        || options.max_plies <= 0 || options.sample_every <= 0
        || options.samples_per_game <= 0 || options.teacher_depth <= 0
        || options.tt_mb == 0) {
        throw std::runtime_error("invalid non-positive option");
    }
    return options;
}

std::vector<std::string> words(std::string_view text) {
    std::istringstream input{std::string(text)};
    std::vector<std::string> result;
    std::string word;
    while (input >> word) result.push_back(word);
    return result;
}

chess::Move legal_uci(const chess::Position& position, std::string_view uci) {
    for (const chess::Move move : chess::generate_legal_moves(position)) {
        if (chess::move_to_string(move) == uci) return move;
    }
    throw std::runtime_error("illegal opening move: " + std::string(uci));
}

chess::Position opening_position(const Opening& opening) {
    chess::Position position;
    position.set_startpos();
    for (const std::string& uci : opening.moves) {
        position.make_move(legal_uci(position, uci));
    }
    return position;
}

std::vector<Opening> load_openings(const Options& options) {
    std::ifstream input(options.book);
    if (!input) throw std::runtime_error("failed to open book");
    std::vector<Opening> openings;
    std::string line;
    for (int line_number = 1; std::getline(input, line); ++line_number) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        const std::vector<std::string> moves = words(line);
        if (moves.empty()) continue;
        Opening opening{line_number, moves};
        static_cast<void>(opening_position(opening));
        openings.push_back(std::move(opening));
    }
    if (static_cast<int>(openings.size()) < options.games) {
        throw std::runtime_error("book has fewer openings than --games");
    }
    std::mt19937 rng(options.seed);
    std::shuffle(openings.begin(), openings.end(), rng);
    openings.resize(static_cast<std::size_t>(options.games));
    return openings;
}

int move_budget_ms(int remaining_ms, const Options& options) {
    const int reserve = std::min(
        remaining_ms / 2,
        std::max(options.overhead_ms, remaining_ms / 50));
    const int usable = std::max(1, remaining_ms - reserve);
    int budget = usable / 20 + options.increment_ms / 2;
    budget = std::max(5, budget);
    return std::min(budget, std::max(1, usable / 5));
}

std::string terminal_result(const chess::Position& position) {
    if (!chess::in_check(position, position.side_to_move)) return "1/2-1/2";
    return position.side_to_move == chess::Color::White ? "0-1" : "1-0";
}

chess::NnueSearcherV38::SelectiveConfig balanced_old_config() {
    chess::NnueSearcherV38::SelectiveConfig config;
    config.lmr_base = 0.45;
    config.lmr_divisor = 2.45;
    config.lmr_min_depth = 5;
    config.lmr_min_move_index = 5;
    config.null_move_min_depth = 3;
    config.null_move_reduction = 2;
    config.enable_reverse_futility = false;
    config.enable_late_move_pruning = false;
    return config;
}

std::vector<Sample> thin_samples(
    const std::vector<Sample>& samples,
    int limit
) {
    if (static_cast<int>(samples.size()) <= limit) return samples;
    std::vector<Sample> result;
    result.reserve(static_cast<std::size_t>(limit));
    if (limit == 1) {
        result.push_back(samples[samples.size() / 2]);
        return result;
    }
    std::set<std::size_t> indices;
    for (int index = 0; index < limit; ++index) {
        indices.insert(static_cast<std::size_t>(
            (static_cast<std::uint64_t>(index) * (samples.size() - 1))
            / static_cast<std::uint64_t>(limit - 1)));
    }
    for (const std::size_t index : indices) result.push_back(samples[index]);
    return result;
}

GameResult play_game(
    const Opening& opening,
    const Options& options,
    chess::NnueSearcherV38& white,
    chess::NnueSearcherV38& black
) {
    chess::Position position;
    position.set_startpos();
    std::vector<chess::HashKey> history{position.zobrist_key};
    for (const std::string& uci : opening.moves) {
        position.make_move(legal_uci(position, uci));
        history.push_back(position.zobrist_key);
    }
    std::array<int, 2> remaining{options.base_ms, options.base_ms};
    std::vector<Sample> samples;

    for (int ply = 0; ply < options.max_plies; ++ply) {
        if (ply % options.sample_every == 0) {
            samples.push_back(Sample{position, ply});
        }
        if (chess::is_threefold_repetition(position.zobrist_key, history)
            || position.halfmove_clock >= 100) {
            return {"1/2-1/2", "rule_draw", ply,
                    thin_samples(samples, options.samples_per_game)};
        }
        const std::vector<chess::Move> legal =
            chess::generate_legal_moves(position);
        if (legal.empty()) {
            return {
                terminal_result(position),
                chess::in_check(position, position.side_to_move)
                    ? "checkmate" : "stalemate",
                ply,
                thin_samples(samples, options.samples_per_game),
            };
        }
        const bool white_to_move =
            position.side_to_move == chess::Color::White;
        const int clock_index = white_to_move ? 0 : 1;
        const int budget = move_budget_ms(remaining[clock_index], options);
        const chess::SearchLimits limits{
            .max_depth = 64,
            .move_time = std::chrono::milliseconds{budget},
        };
        const auto start = std::chrono::steady_clock::now();
        const chess::SearchResult result = white_to_move
            ? white.search_best_move(position, limits)
            : black.search_best_move(position, limits);
        const int elapsed = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count());
        remaining[clock_index] -= elapsed;
        if (remaining[clock_index] < 0) {
            return {
                white_to_move ? "0-1" : "1-0",
                "time",
                ply,
                thin_samples(samples, options.samples_per_game),
            };
        }
        remaining[clock_index] += options.increment_ms;
        if (std::find(legal.begin(), legal.end(), result.best_move)
            == legal.end()) {
            return {
                white_to_move ? "0-1" : "1-0",
                "illegal",
                ply,
                thin_samples(samples, options.samples_per_game),
            };
        }
        position.make_move(result.best_move);
        history.push_back(position.zobrist_key);
    }
    return {
        "1/2-1/2",
        "max_plies",
        options.max_plies,
        thin_samples(samples, options.samples_per_game),
    };
}

int absolute_ply(const chess::Position& position) {
    return 2 * (position.fullmove_number - 1)
        + (position.side_to_move == chess::Color::Black ? 1 : 0);
}

int outcome_for_side_to_move(
    std::string_view result,
    chess::Color side_to_move
) {
    if (result == "1/2-1/2") return 0;
    const bool white_won = result == "1-0";
    const bool stm_white = side_to_move == chess::Color::White;
    return white_won == stm_white ? 1 : -1;
}

std::size_t piece_count(const chess::Position& position) {
    return static_cast<std::size_t>(chess::popcount(position.occupancy()));
}

std::set<int> completed_games(const std::string& path) {
    std::ifstream input(path);
    std::set<int> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.find("\"kind\":\"game\"") == std::string::npos) continue;
        const std::string marker = "\"game_id\":";
        const std::size_t start = line.find(marker);
        if (start == std::string::npos) continue;
        result.insert(std::stoi(line.substr(start + marker.size())));
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Opening> openings = load_openings(options);
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model");
        }
        model.set_neon_dotprod_enabled(true);
        const auto config = balanced_old_config();
        chess::NnueSearcherV38 white(
            model, options.tt_mb, 4, 10, 14, 14'000,
            chess::NnueSearcherV38::MoveOrderingWeights{}, config);
        chess::NnueSearcherV38 black(
            model, options.tt_mb, 4, 10, 14, 14'000,
            chess::NnueSearcherV38::MoveOrderingWeights{}, config);
        chess::NnueSearcherV36 teacher(model, options.tt_mb);
        const std::set<int> done = completed_games(options.output);
        std::ofstream output(options.output, std::ios::app);
        if (!output) throw std::runtime_error("failed to open output");

        for (int game_id = 0; game_id < options.games; ++game_id) {
            if (done.contains(game_id)) continue;
            white.clear_tt();
            black.clear_tt();
            const Opening& opening =
                openings[static_cast<std::size_t>(game_id)];
            const GameResult game =
                play_game(opening, options, white, black);
            const bool usable = (
                game.reason == "checkmate"
                || game.reason == "rule_draw"
                || game.reason == "stalemate"
                || (options.include_max_plies
                    && game.reason == "max_plies"));
            int emitted = 0;
            if (usable) {
                for (const Sample& sample : game.samples) {
                    teacher.clear_tt();
                    const chess::SearchResult label =
                        teacher.search_best_move(
                            sample.position, options.teacher_depth);
                    const std::size_t pieces = piece_count(sample.position);
                    const std::size_t phase_index = std::min<std::size_t>(
                        (pieces - 1) / 4,
                        chess::PhaseQuantizedNnueModel::PhaseCount - 1);
                    output
                        << "{\"kind\":\"position\""
                        << ",\"game_id\":" << game_id
                        << ",\"opening_line\":" << opening.source_line
                        << ",\"game_ply\":" << sample.game_ply
                        << ",\"ply\":" << absolute_ply(sample.position)
                        << ",\"piece_count\":" << pieces
                        << ",\"phase_index\":" << phase_index
                        << ",\"teacher_depth\":" << options.teacher_depth
                        << ",\"teacher_cp\":" << label.score
                        << ",\"teacher_nodes\":" << label.nodes
                        << ",\"static_cp\":"
                        << model.evaluate_cp_rounded(sample.position)
                        << ",\"outcome\":"
                        << outcome_for_side_to_move(
                            game.result, sample.position.side_to_move)
                        << "}\n";
                    ++emitted;
                }
            }
            output
                << "{\"kind\":\"game\""
                << ",\"game_id\":" << game_id
                << ",\"opening_line\":" << opening.source_line
                << ",\"result\":\"" << game.result << "\""
                << ",\"reason\":\"" << game.reason << "\""
                << ",\"plies\":" << game.plies
                << ",\"positions\":" << emitted
                << "}\n";
            output.flush();
            std::cerr
                << "calibration_progress"
                << " game=" << (game_id + 1) << "/" << options.games
                << " opening_line=" << opening.source_line
                << " result=" << game.result
                << " reason=" << game.reason
                << " positions=" << emitted
                << '\n';
        }
        std::cerr << "calibration_collection_complete\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
