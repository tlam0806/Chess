#include "move.hpp"
#include "position.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

using namespace chess;

namespace {

bool contains_move(const std::vector<Move>& moves, Move move) {
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

}

int main() {
    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 4));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.side_to_move = Color::White;
        pos.en_passant_square = make_square(3, 5);

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 4), make_square(3, 5), MoveFlag::EnPassant)));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(4, 3));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(5, 3));
        pos.side_to_move = Color::Black;
        pos.en_passant_square = make_square(5, 2);

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 3), make_square(5, 2), MoveFlag::EnPassant)));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 4));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.side_to_move = Color::White;
        pos.en_passant_square = NoSquare;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(!contains_move(moves, make_move(make_square(4, 4), make_square(3, 5), MoveFlag::EnPassant)));
    }
}
