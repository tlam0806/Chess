#include "attacks.hpp"
#include "evaluate.hpp"
#include "move.hpp"
#include "nn_search.hpp"
#include "nn_value.hpp"
#include "position.hpp"
#include "search.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class EngineKind {
    Heuristic,
    Nn
};

struct Options {
    std::string model_path = "models/value_net_stream_100k.bin";
    int min_depth = 2;
    int max_depth = 4;
    int max_plies = 120;
    std::string output_dir = "data/matches";
};

struct PlyRecord {
    int ply = 0;
    chess::Color side = chess::Color::White;
    EngineKind engine = EngineKind::Heuristic;
    std::string move;
    int score = 0;
    std::uint64_t nodes = 0;
};

struct GameResult {
    int depth = 0;
    EngineKind white = EngineKind::Heuristic;
    EngineKind black = EngineKind::Heuristic;
    std::string result;
    std::string reason;
    std::vector<PlyRecord> plies;
};

std::string engine_name(EngineKind engine) {
    return engine == EngineKind::Nn ? "nn" : "heuristic";
}

std::string color_name(chess::Color color) {
    return color == chess::Color::White ? "white" : "black";
}

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

        if (arg == "--model") {
            options.model_path = std::string(require_value(arg));
        } else if (arg == "--min-depth") {
            options.min_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-depth") {
            options.max_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--output-dir") {
            options.output_dir = std::string(require_value(arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: match_engines [--model path] [--min-depth N] [--max-depth N]\n"
                << "                     [--max-plies N] [--output-dir path]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.min_depth <= 0 || options.max_depth < options.min_depth) {
        throw std::runtime_error("invalid depth range");
    }
    if (options.max_plies <= 0) {
        throw std::runtime_error("--max-plies must be positive");
    }
    return options;
}

bool contains_move(const std::vector<chess::Move>& moves, chess::Move target) {
    for (chess::Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

chess::SearchResult search(
    const chess::Position& pos,
    int depth,
    EngineKind engine,
    const chess::NnValueModel& model
) {
    return engine == EngineKind::Nn
        ? chess::search_best_move_nn(pos, depth, model)
        : chess::search_best_move(pos, depth);
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
    EngineKind white,
    EngineKind black,
    int max_plies,
    const chess::NnValueModel& model
) {
    GameResult game;
    game.depth = depth;
    game.white = white;
    game.black = black;

    chess::Position pos;
    pos.set_startpos();

    for (int ply = 0; ply < max_plies; ++ply) {
        const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (moves.empty()) {
            game.result = terminal_result(pos, game.reason);
            return game;
        }

        const EngineKind engine = pos.side_to_move == chess::Color::White ? white : black;
        const chess::SearchResult result = search(pos, depth, engine, model);
        if (!contains_move(moves, result.best_move)) {
            throw std::runtime_error("engine returned illegal move");
        }

        game.plies.push_back(PlyRecord{
            ply,
            pos.side_to_move,
            engine,
            chess::move_to_string(result.best_move),
            result.score,
            result.nodes
        });
        pos.make_move(result.best_move);
    }

    game.result = "1/2-1/2";
    game.reason = "max plies";
    return game;
}

std::string game_file_stem(const GameResult& game) {
    return "depth" + std::to_string(game.depth)
        + "_" + engine_name(game.white) + "_white_"
        + engine_name(game.black) + "_black";
}

void write_replay(const GameResult& game, const std::filesystem::path& output_dir) {
    std::filesystem::create_directories(output_dir);

    const std::filesystem::path path = output_dir / (game_file_stem(game) + ".txt");
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write " + path.string());
    }

    out << "depth " << game.depth << '\n';
    out << "white " << engine_name(game.white) << '\n';
    out << "black " << engine_name(game.black) << '\n';
    out << "result " << game.result << '\n';
    out << "reason " << game.reason << "\n\n";

    for (const PlyRecord& ply : game.plies) {
        out << (ply.ply + 1) << ' '
            << color_name(ply.side) << ' '
            << engine_name(ply.engine) << ' '
            << ply.move << " score " << ply.score
            << " nodes " << ply.nodes << '\n';
    }
}

void print_summary(const GameResult& game) {
    std::uint64_t nn_nodes = 0;
    std::uint64_t heuristic_nodes = 0;
    for (const PlyRecord& ply : game.plies) {
        if (ply.engine == EngineKind::Nn) {
            nn_nodes += ply.nodes;
        } else {
            heuristic_nodes += ply.nodes;
        }
    }

    std::cout << "depth " << game.depth
              << " | white=" << engine_name(game.white)
              << " black=" << engine_name(game.black)
              << " | result=" << game.result
              << " (" << game.reason << ")"
              << " | plies=" << game.plies.size()
              << " | nn_nodes=" << nn_nodes
              << " heuristic_nodes=" << heuristic_nodes << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        chess::NnValueModel model;
        if (!model.load(options.model_path)) {
            throw std::runtime_error("failed to load model: " + options.model_path);
        }

        const auto start = std::chrono::steady_clock::now();
        for (int depth = options.min_depth; depth <= options.max_depth; ++depth) {
            for (const auto [white, black] : {
                     std::pair{EngineKind::Nn, EngineKind::Heuristic},
                     std::pair{EngineKind::Heuristic, EngineKind::Nn}
                 }) {
                const GameResult game = play_game(depth, white, black, options.max_plies, model);
                write_replay(game, options.output_dir);
                print_summary(game);
            }
        }

        const auto end = std::chrono::steady_clock::now();
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::cout << "wrote replays to " << options.output_dir << '\n';
        std::cout << "elapsed " << seconds << "s\n";
    } catch (const std::exception& error) {
        std::cerr << "match_engines: " << error.what() << '\n';
        return 1;
    }
}
