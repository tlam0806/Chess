#include "heuristic_searcher_v7.hpp"
#include "move.hpp"
#include "position.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int DefaultDepth = 8;

bool apply_uci_move(chess::Position& pos, const std::string& uci) {
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    for (chess::Move move : moves) {
        if (chess::move_to_string(move) == uci) {
            pos.make_move(move);
            return true;
        }
    }
    return false;
}

void set_position(chess::Position& pos, chess::HeuristicSearcherV7& searcher, std::istringstream& input) {
    std::string token;
    input >> token;

    if (token == "startpos") {
        pos.set_startpos();
        if (input >> token && token != "moves") {
            return;
        }
    } else if (token == "fen") {
        std::vector<std::string> fen_parts;
        while (input >> token && token != "moves") {
            fen_parts.push_back(token);
        }

        std::string fen;
        for (std::size_t i = 0; i < fen_parts.size(); ++i) {
            if (i != 0) {
                fen += ' ';
            }
            fen += fen_parts[i];
        }
        if (!pos.set_fen(fen)) {
            pos.set_startpos();
            return;
        }
    } else {
        return;
    }

    searcher.clear_tt();
    if (token == "moves") {
        while (input >> token) {
            if (!apply_uci_move(pos, token)) {
                break;
            }
        }
    }
}

chess::SearchLimits parse_go_limits(std::istringstream& input, chess::Color side_to_move) {
    chess::SearchLimits limits;
    limits.max_depth = DefaultDepth;
    int white_time_ms = -1;
    int black_time_ms = -1;
    int white_increment_ms = 0;
    int black_increment_ms = 0;
    bool explicit_depth = false;
    bool explicit_movetime = false;

    std::string token;
    while (input >> token) {
        if (token == "depth") {
            input >> limits.max_depth;
            explicit_depth = true;
        } else if (token == "movetime") {
            int movetime_ms = 0;
            input >> movetime_ms;
            limits.move_time = std::chrono::milliseconds{std::max(0, movetime_ms)};
            limits.max_depth = 64;
            explicit_movetime = true;
        } else if (token == "wtime" || token == "btime") {
            int value = 0;
            input >> value;
            if (token == "wtime") {
                white_time_ms = value;
            } else {
                black_time_ms = value;
            }
        } else if (token == "winc" || token == "binc") {
            int value = 0;
            input >> value;
            if (token == "winc") {
                white_increment_ms = value;
            } else {
                black_increment_ms = value;
            }
        }
    }

    if (!explicit_depth && !explicit_movetime) {
        const int remaining_ms = side_to_move == chess::Color::White ? white_time_ms : black_time_ms;
        const int increment_ms = side_to_move == chess::Color::White ? white_increment_ms : black_increment_ms;
        if (remaining_ms > 0) {
            int budget_ms = remaining_ms / 40 + increment_ms / 2;
            budget_ms = std::clamp(budget_ms, 50, 1200);
            budget_ms = std::min(budget_ms, std::max(1, remaining_ms / 5));
            limits.max_depth = 64;
            limits.move_time = std::chrono::milliseconds{budget_ms};
        }
    }

    return limits;
}

void search_and_print(chess::HeuristicSearcherV7& searcher, const chess::Position& pos, const chess::SearchLimits& limits) {
    chess::SearchResult result = searcher.search_best_move(pos, limits);
    if (result.best_move.value == 0) {
        const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (!moves.empty()) {
            result.best_move = moves.front();
        }
    }

    std::cout << "info depth " << result.depth
              << " score cp " << result.score
              << " nodes " << result.nodes << '\n';
    std::cout << "bestmove " << chess::move_to_string(result.best_move) << std::endl;
}

} // namespace

int main() {
    chess::Position pos;
    pos.set_startpos();
    chess::HeuristicSearcherV7 searcher;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;

        if (command == "uci") {
            std::cout << "id name ChessEngineV7\n";
            std::cout << "id author TungLamNguyen\n";
            std::cout << "uciok" << std::endl;
        } else if (command == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (command == "ucinewgame") {
            pos.set_startpos();
            searcher.clear_tt();
        } else if (command == "position") {
            set_position(pos, searcher, input);
        } else if (command == "go") {
            const chess::SearchLimits limits = parse_go_limits(input, pos.side_to_move);
            search_and_print(searcher, pos, limits);
        } else if (command == "stop") {
            std::cout << "bestmove 0000" << std::endl;
        } else if (command == "quit") {
            break;
        }
    }
}
