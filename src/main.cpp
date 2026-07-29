#include "attacks.hpp"
#include "game_state.hpp"
#include "move.hpp"
#include "nnue_searcher_v36.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr int BotDepth = 4;

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
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);

    for (chess::Move move : moves) {
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
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    for (chess::Move move : moves) {
        std::cout << chess::move_to_string(move) << ' ';
    }
    std::cout << '\n';
}

} // namespace

int main() {
    chess::Position pos;
    pos.set_startpos();
    chess::PhaseQuantizedNnueModel model;
    if (!model.load(chess::DefaultPhaseQuantizedNnueModelPath)) {
        std::cerr << "Failed to load default NNUE model: "
                  << chess::DefaultPhaseQuantizedNnueModelPath << '\n';
        return 1;
    }
    chess::NnueSearcherV36 bot(model);
    std::vector<chess::Position> history;

    std::cout << "Mini chess engine CLI\n";
    std::cout << "You are White. Enter UCI moves like e2e4, g1f3, e7e8q.\n";
    std::cout << "Commands: moves, quit\n";

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

            const std::optional<chess::Move> move = find_legal_move(pos, input);
            if (!move.has_value()) {
                std::cout << "Illegal move. Type 'moves' to list legal moves.\n";
                continue;
            }

            history.push_back(pos);
            pos.make_move(*move);
        } else {
            std::cout << "Bot thinking at depth " << BotDepth << "...\n";
            const chess::SearchResult result = bot.search_best_move(pos, BotDepth);
            std::cout << "Bot plays " << chess::move_to_string(result.best_move)
                      << " score " << result.score
                      << " nodes " << result.nodes << '\n';
            history.push_back(pos);
            pos.make_move(result.best_move);
        }
    }
}
