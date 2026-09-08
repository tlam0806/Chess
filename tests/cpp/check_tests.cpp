#include "attacks.hpp"
#include "position.hpp"

#include <cassert>

using namespace chess;

int main() {
    {
        Position pos;
        const Square e1 = make_square(4, 0);
        const Square e8 = make_square(4, 7);

        pos.set_piece(Color::White, PieceType::King, e1);
        pos.set_piece(Color::Black, PieceType::King, e8);

        assert(king_square(pos, Color::White) == e1);
        assert(king_square(pos, Color::Black) == e8);
        assert(!in_check(pos, Color::White));
        assert(!in_check(pos, Color::Black));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));

        assert(in_check(pos, Color::White));
        assert(!in_check(pos, Color::Black));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 3));

        assert(!in_check(pos, Color::White));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Bishop, make_square(1, 3));

        assert(in_check(pos, Color::White));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(5, 2));

        assert(in_check(pos, Color::White));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 1));

        assert(in_check(pos, Color::White));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 1));

        assert(in_check(pos, Color::White));
        assert(in_check(pos, Color::Black));
    }
}
