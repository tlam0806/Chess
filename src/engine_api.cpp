#include "attacks.hpp"
#include "evaluate.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int DefaultBotDepth = 4;

char piece_char(chess::Color color, chess::PieceType piece) {
    constexpr char WhitePieces[] = {'P', 'N', 'B', 'R', 'Q', 'K'};
    constexpr char BlackPieces[] = {'p', 'n', 'b', 'r', 'q', 'k'};
    return color == chess::Color::White ? WhitePieces[static_cast<int>(piece)]
                                        : BlackPieces[static_cast<int>(piece)];
}

char piece_on(const chess::Position& pos, chess::Square square) {
    for (chess::Color color : {chess::Color::White, chess::Color::Black}) {
        for (chess::PieceType piece :
             {chess::PieceType::Pawn, chess::PieceType::Knight, chess::PieceType::Bishop,
              chess::PieceType::Rook, chess::PieceType::Queen, chess::PieceType::King}) {
            const chess::Bitboard bb = pos.pieces[static_cast<int>(color)][static_cast<int>(piece)];
            if (bb & chess::bit(square)) {
                return piece_char(color, piece);
            }
        }
    }
    return '.';
}

std::string json_escape(const std::string& text) {
    std::string out;
    for (char ch : text) {
        if (ch == '\\' || ch == '"') {
            out += '\\';
        }
        out += ch;
    }
    return out;
}

std::vector<std::string> legal_move_strings(const chess::Position& pos) {
    std::vector<std::string> moves;
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        moves.push_back(chess::move_to_string(move));
    }
    std::sort(moves.begin(), moves.end());
    return moves;
}

bool is_game_over(const chess::Position& pos) {
    return chess::generate_legal_moves(pos).empty();
}

std::string game_status(const chess::Position& pos) {
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    if (!moves.empty()) {
        return chess::in_check(pos, pos.side_to_move) ? "check" : "playing";
    }
    return chess::in_check(pos, pos.side_to_move) ? "checkmate" : "stalemate";
}

void write_state(const chess::Position& pos, bool ok, const std::string& message = "",
                 const std::string& last_move = "") {
    std::cout << "{\"ok\":" << (ok ? "true" : "false");
    std::cout << ",\"message\":\"" << json_escape(message) << "\"";
    std::cout << ",\"lastMove\":\"" << json_escape(last_move) << "\"";
    std::cout << ",\"side\":\"" << (pos.side_to_move == chess::Color::White ? "w" : "b") << "\"";
    std::cout << ",\"status\":\"" << game_status(pos) << "\"";
    std::cout << ",\"eval\":" << chess::evaluate(pos);
    std::cout << ",\"check\":" << (chess::in_check(pos, pos.side_to_move) ? "true" : "false");
    std::cout << ",\"gameOver\":" << (is_game_over(pos) ? "true" : "false");

    std::cout << ",\"board\":[";
    for (int rank = 7; rank >= 0; --rank) {
        if (rank != 7) {
            std::cout << ',';
        }
        std::string row;
        for (int file = 0; file < 8; ++file) {
            row += piece_on(pos, chess::make_square(file, rank));
        }
        std::cout << '"' << row << '"';
    }
    std::cout << ']';

    const std::vector<std::string> moves = legal_move_strings(pos);
    std::cout << ",\"legal\":[";
    for (std::size_t i = 0; i < moves.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << '"' << moves[i] << '"';
    }
    std::cout << "]}";
    std::cout << std::endl;
}

bool make_uci_move(chess::Position& pos, std::vector<chess::Position>& history,
                   const std::string& uci) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == uci) {
            history.push_back(pos);
            pos.make_move(move);
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    chess::Position pos;
    pos.set_startpos();
    std::vector<chess::Position> history;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string command;
        in >> command;

        if (command == "state") {
            write_state(pos, true);
        } else if (command == "reset") {
            pos.set_startpos();
            history.clear();
            write_state(pos, true, "reset");
        } else if (command == "move") {
            std::string uci;
            in >> uci;
            if (make_uci_move(pos, history, uci)) {
                write_state(pos, true, "", uci);
            } else {
                write_state(pos, false, "illegal move");
            }
        } else if (command == "bot") {
            int depth = DefaultBotDepth;
            in >> depth;
            if (is_game_over(pos)) {
                write_state(pos, false, "game is over");
                continue;
            }
            const chess::SearchResult result = chess::search_best_move(pos, depth);
            history.push_back(pos);
            pos.make_move(result.best_move);
            write_state(pos, true, "score " + std::to_string(result.score)
                                   + ", nodes " + std::to_string(result.nodes),
                        chess::move_to_string(result.best_move));
        } else if (command == "undo") {
            if (!history.empty()) {
                pos = history.back();
                history.pop_back();
                write_state(pos, true, "undo");
            } else {
                write_state(pos, false, "nothing to undo");
            }
        } else if (command == "undo_turn") {
            int undone = 0;
            while (!history.empty() && undone < 2) {
                pos = history.back();
                history.pop_back();
                ++undone;
            }
            write_state(pos, undone > 0, undone > 0 ? "undo turn" : "nothing to undo");
        } else if (command == "quit") {
            break;
        } else {
            write_state(pos, false, "unknown command");
        }
    }
}
