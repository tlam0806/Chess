#include "attacks.hpp"
#include "game_state.hpp"
#include "move.hpp"
#include "nn_search.hpp"
#include "nn_value.hpp"
#include "position.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model_path = "models/value_net_stream_100k.bin";
    std::string fen;
    int depth = 3;
    bool go_once = false;
};

std::string color_name(chess::Color color) {
    return color == chess::Color::White ? "White" : "Black";
}

std::string normalized_input(std::string input) {
    input.erase(std::remove_if(input.begin(), input.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }), input.end());

    for (char& ch : input) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    return input;
}

std::optional<chess::Move> find_legal_move(const chess::Position& pos, const std::string& input) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == input) {
            return move;
        }
    }
    return std::nullopt;
}

void print_position(const chess::Position& pos) {
    std::cout << '\n';
    pos.print(std::cout);
    std::cout << "Side: " << color_name(pos.side_to_move) << '\n';
}

bool print_game_over_if_needed(
    const chess::Position& pos,
    const std::vector<chess::Position>& history
) {
    if (chess::is_threefold_repetition(pos, history)) {
        std::cout << "Draw by threefold repetition.\n";
        return true;
    }

    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    if (!moves.empty()) {
        return false;
    }

    if (chess::in_check(pos, pos.side_to_move)) {
        std::cout << "Checkmate. " << color_name(chess::opposite(pos.side_to_move)) << " wins.\n";
    } else {
        std::cout << "Stalemate.\n";
    }
    return true;
}

void print_legal_moves(const chess::Position& pos) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        std::cout << chess::move_to_string(move) << ' ';
    }
    std::cout << '\n';
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };

        if (arg == "--model") {
            options.model_path = require_value(arg);
        } else if (arg == "--fen") {
            options.fen = require_value(arg);
        } else if (arg == "--depth") {
            options.depth = std::stoi(require_value(arg));
        } else if (arg == "--go-once") {
            options.go_once = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: nn_engine [--model path] [--depth N] [--fen fen] [--go-once]\n"
                << "Without --go-once, starts an interactive CLI where the NN bot plays Black.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.depth <= 0) {
        throw std::runtime_error("--depth must be positive");
    }
    return options;
}

chess::SearchResult think(
    const chess::Position& pos,
    int depth,
    const chess::NnValueModel& model
) {
    return chess::search_best_move_nn(pos, depth, model);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        chess::NnValueModel model;
        if (!model.load(options.model_path)) {
            std::cerr << "failed to load model: " << options.model_path << '\n';
            return 1;
        }

        chess::Position pos;
        if (options.fen.empty()) {
            pos.set_startpos();
        } else if (!pos.set_fen(options.fen)) {
            std::cerr << "invalid FEN\n";
            return 1;
        }
        std::vector<chess::Position> history;

        if (options.go_once) {
            if (print_game_over_if_needed(pos, history)) {
                return 0;
            }
            const chess::SearchResult result = think(pos, options.depth, model);
            std::cout << "bestmove " << chess::move_to_string(result.best_move)
                      << " score " << result.score
                      << " nodes " << result.nodes << '\n';
            return 0;
        }

        std::cout << "NN chess engine CLI\n";
        std::cout << "Model: " << options.model_path << '\n';
        std::cout << "You are White. Enter UCI moves like e2e4, g1f3, e7e8q.\n";
        std::cout << "Commands: moves, bot, quit\n";

        while (true) {
            print_position(pos);
            if (print_game_over_if_needed(pos, history)) {
                break;
            }

            if (pos.side_to_move == chess::Color::White) {
                std::string input;
                std::cout << "Your move: ";
                if (!std::getline(std::cin, input)) {
                    break;
                }

                input = normalized_input(input);
                if (input == "quit" || input == "exit") {
                    break;
                }
                if (input == "moves") {
                    print_legal_moves(pos);
                    continue;
                }
                if (input == "bot") {
                    const chess::SearchResult result = think(pos, options.depth, model);
                    std::cout << "NN plays " << chess::move_to_string(result.best_move)
                              << " score " << result.score
                              << " nodes " << result.nodes << '\n';
                    history.push_back(pos);
                    pos.make_move(result.best_move);
                    continue;
                }

                const std::optional<chess::Move> move = find_legal_move(pos, input);
                if (!move.has_value()) {
                    std::cout << "Illegal move. Type 'moves' to list legal moves.\n";
                    continue;
                }

                history.push_back(pos);
                pos.make_move(*move);
            } else {
                std::cout << "NN bot thinking at depth " << options.depth << "...\n";
                const chess::SearchResult result = think(pos, options.depth, model);
                std::cout << "NN plays " << chess::move_to_string(result.best_move)
                          << " score " << result.score
                          << " nodes " << result.nodes << '\n';
                history.push_back(pos);
                pos.make_move(result.best_move);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "nn_engine: " << error.what() << '\n';
        return 1;
    }
}
