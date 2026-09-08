#include "attacks.hpp"
#include "game_state.hpp"
#include "move.hpp"
#include "nn_searcher_v6.hpp"
#include "nn_value.hpp"
#include "position.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string current_model = "models/value_net_legal_moves_depth6_v6_r12_5k.bin";
    std::string previous_model = "models/value_net_depth8_v6_r12_600_plus_stress50k.bin";
    int depth = 6;
    int move_time_ms = 0;
    int max_plies = 300;
    int progress_interval = 10;
    std::string output_dir = "data/matches_nn_models";
};

struct PlyRecord {
    int ply = 0;
    chess::Color side = chess::Color::White;
    std::string engine;
    std::string move;
    int score = 0;
    std::uint64_t nodes = 0;
    int depth = 0;
    bool stopped = false;
};

struct GameResult {
    int depth = 0;
    std::string white;
    std::string black;
    std::string result;
    std::string reason;
    std::vector<PlyRecord> plies;
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
        throw std::runtime_error("invalid integer for " + std::string(name));
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

        if (arg == "--current-model") {
            options.current_model = std::string(require_value(arg));
        } else if (arg == "--previous-model") {
            options.previous_model = std::string(require_value(arg));
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--move-time-ms") {
            options.move_time_ms = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_int(require_value(arg), arg);
        } else if (arg == "--output-dir") {
            options.output_dir = std::string(require_value(arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: match_nn_models [--current-model path] [--previous-model path]\n"
                << "                       [--depth N] [--max-plies N]\n"
                << "                       [--move-time-ms N]\n"
                << "                       [--progress-interval N] [--output-dir path]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.depth <= 0) {
        throw std::runtime_error("--depth must be positive");
    }
    if (options.move_time_ms < 0) {
        throw std::runtime_error("--move-time-ms must be non-negative");
    }
    if (options.max_plies <= 0) {
        throw std::runtime_error("--max-plies must be positive");
    }
    if (options.progress_interval <= 0) {
        throw std::runtime_error("--progress-interval must be positive");
    }
    return options;
}

std::string color_name(chess::Color color) {
    return color == chess::Color::White ? "white" : "black";
}

bool contains_move(const std::vector<chess::Move>& moves, chess::Move target) {
    for (chess::Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

std::string terminal_result(const chess::Position& pos, std::string& reason) {
    if (chess::in_check(pos, pos.side_to_move)) {
        reason = "checkmate";
        return pos.side_to_move == chess::Color::White ? "0-1" : "1-0";
    }
    reason = "stalemate";
    return "1/2-1/2";
}

GameResult play_game(
    int depth,
    int move_time_ms,
    int max_plies,
    int progress_interval,
    const std::string& white_name,
    chess::NnSearcherV6& white_searcher,
    const std::string& black_name,
    chess::NnSearcherV6& black_searcher
) {
    GameResult game;
    game.depth = depth;
    game.white = white_name;
    game.black = black_name;

    chess::Position pos;
    pos.set_startpos();
    std::vector<chess::HashKey> position_hashes{pos.zobrist_key};

    for (int ply = 0; ply < max_plies; ++ply) {
        if (chess::is_threefold_repetition(pos.zobrist_key, position_hashes)) {
            game.result = "1/2-1/2";
            game.reason = "threefold repetition";
            return game;
        }

        const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (moves.empty()) {
            game.result = terminal_result(pos, game.reason);
            return game;
        }

        const bool white_to_move = pos.side_to_move == chess::Color::White;
        chess::NnSearcherV6& searcher = white_to_move ? white_searcher : black_searcher;
        const std::string& engine_name = white_to_move ? white_name : black_name;
        chess::SearchResult result;
        if (move_time_ms > 0) {
            chess::SearchLimits limits;
            limits.max_depth = depth;
            limits.move_time = std::chrono::milliseconds(move_time_ms);
            result = searcher.search_best_move(pos, limits);
        } else {
            result = searcher.search_best_move(pos, depth);
        }
        if (!contains_move(moves, result.best_move)) {
            throw std::runtime_error(engine_name + " returned illegal move");
        }

        if ((ply + 1) % progress_interval == 0) {
            std::cerr << white_name << " vs " << black_name
                      << " ply " << (ply + 1)
                      << " move " << chess::move_to_string(result.best_move)
                      << " score " << result.score
                      << " depth " << result.depth
                      << " nodes " << result.nodes
                      << " stopped " << (result.stopped ? 1 : 0) << '\n';
        }

        game.plies.push_back(PlyRecord{
            ply,
            pos.side_to_move,
            engine_name,
            chess::move_to_string(result.best_move),
            result.score,
            result.nodes,
            result.depth,
            result.stopped
        });
        pos.make_move(result.best_move);
        position_hashes.push_back(pos.zobrist_key);
    }

    game.result = "1/2-1/2";
    game.reason = "max plies";
    return game;
}

void write_replay(const GameResult& game, const std::filesystem::path& output_dir) {
    std::filesystem::create_directories(output_dir);
    const std::filesystem::path path = output_dir / (
        "depth" + std::to_string(game.depth) + "_" + game.white + "_white_" + game.black + "_black.txt");
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write " + path.string());
    }

    out << "depth " << game.depth << '\n';
    out << "white " << game.white << '\n';
    out << "black " << game.black << '\n';
    out << "result " << game.result << '\n';
    out << "reason " << game.reason << "\n\n";

    for (const PlyRecord& ply : game.plies) {
        out << (ply.ply + 1) << ' '
            << color_name(ply.side) << ' '
            << ply.engine << ' '
            << ply.move << " score " << ply.score
            << " depth " << ply.depth
            << " nodes " << ply.nodes
            << " stopped " << (ply.stopped ? 1 : 0) << '\n';
    }
}

void print_summary(const GameResult& game) {
    std::cout << "depth " << game.depth
              << " | white=" << game.white
              << " black=" << game.black
              << " | result=" << game.result
              << " (" << game.reason << ")"
              << " | plies=" << game.plies.size() << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        chess::NnValueModel current_model;
        if (!current_model.load(options.current_model)) {
            throw std::runtime_error("failed to load current model: " + options.current_model);
        }

        chess::NnValueModel previous_model;
        if (!previous_model.load(options.previous_model)) {
            throw std::runtime_error("failed to load previous model: " + options.previous_model);
        }

        chess::NnSearcherV6 current_searcher(current_model);
        chess::NnSearcherV6 previous_searcher(previous_model);

        const auto start = std::chrono::steady_clock::now();

        GameResult current_white = play_game(
            options.depth,
            options.move_time_ms,
            options.max_plies,
            options.progress_interval,
            "current_nn",
            current_searcher,
            "previous_nn",
            previous_searcher);
        write_replay(current_white, options.output_dir);
        print_summary(current_white);

        current_searcher.clear_tt();
        previous_searcher.clear_tt();

        GameResult previous_white = play_game(
            options.depth,
            options.move_time_ms,
            options.max_plies,
            options.progress_interval,
            "previous_nn",
            previous_searcher,
            "current_nn",
            current_searcher);
        write_replay(previous_white, options.output_dir);
        print_summary(previous_white);

        const auto end = std::chrono::steady_clock::now();
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::cout << "wrote replays to " << options.output_dir << '\n';
        std::cout << "elapsed " << seconds << "s\n";
    } catch (const std::exception& error) {
        std::cerr << "match_nn_models: " << error.what() << '\n';
        return 1;
    }
}
