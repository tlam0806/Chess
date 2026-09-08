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
        pos.set_startpos();

        assert(generate_bishop_moves(pos).empty());
        assert(generate_rook_moves(pos).empty());
        assert(generate_queen_moves(pos).empty());
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Bishop, make_square(3, 3));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(7, 7));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_bishop_moves(pos);

        assert(moves.size() == 13);
        assert(contains_move(moves, make_move(make_square(3, 3), make_square(7, 7), MoveFlag::Capture)));
        assert(contains_move(moves, make_move(make_square(3, 3), make_square(0, 0))));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(1, 0));
        pos.set_piece(Color::White, PieceType::Rook, make_square(0, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Rook, make_square(3, 3));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(3, 5));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(5, 3));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_rook_moves(pos);

        assert(!contains_move(moves, make_move(make_square(3, 3), make_square(3, 5))));
        assert(contains_move(moves, make_move(make_square(3, 3), make_square(5, 3), MoveFlag::Capture)));
        assert(!contains_move(moves, make_move(make_square(3, 3), make_square(6, 3))));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(1, 0));
        pos.set_piece(Color::White, PieceType::Rook, make_square(0, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Queen, make_square(3, 3));
        pos.side_to_move = Color::Black;

        const std::vector<Move> moves = generate_queen_moves(pos);

        assert(contains_move(moves, make_move(make_square(3, 3), make_square(3, 0))));
        assert(contains_move(moves, make_move(make_square(3, 3), make_square(6, 6))));
        assert(contains_move(moves, make_move(make_square(3, 3), make_square(0, 0), MoveFlag::Capture)));
    }
}
