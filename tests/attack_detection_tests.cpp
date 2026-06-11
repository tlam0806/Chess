#include "attacks.hpp"
#include "position.hpp"

#include <cassert>

using namespace chess;

int main() {
    {
        Position pos;
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        assert(is_square_attacked(pos, make_square(3, 2), Color::White));
        assert(is_square_attacked(pos, make_square(5, 2), Color::White));
        assert(!is_square_attacked(pos, make_square(4, 2), Color::White));
        assert(!is_square_attacked(pos, make_square(3, 0), Color::White));
    }

    {
        Position pos;
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(4, 6));
        assert(is_square_attacked(pos, make_square(3, 5), Color::Black));
        assert(is_square_attacked(pos, make_square(5, 5), Color::Black));
        assert(!is_square_attacked(pos, make_square(4, 5), Color::Black));
        assert(!is_square_attacked(pos, make_square(3, 7), Color::Black));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::Knight, make_square(5, 2));
        assert(is_square_attacked(pos, make_square(4, 4), Color::White));
        assert(is_square_attacked(pos, make_square(3, 3), Color::White));
        assert(!is_square_attacked(pos, make_square(5, 3), Color::White));
    }

    {
        Position pos;
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 4));
        assert(is_square_attacked(pos, make_square(3, 3), Color::Black));
        assert(is_square_attacked(pos, make_square(4, 3), Color::Black));
        assert(!is_square_attacked(pos, make_square(2, 2), Color::Black));
    }

    {
        Position pos;
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));
        assert(is_square_attacked(pos, make_square(4, 0), Color::Black));

        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 3));
        assert(!is_square_attacked(pos, make_square(4, 0), Color::Black));
        assert(is_square_attacked(pos, make_square(4, 3), Color::Black));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::Bishop, make_square(2, 0));
        assert(is_square_attacked(pos, make_square(5, 3), Color::White));

        pos.set_piece(Color::Black, PieceType::Pawn, make_square(4, 2));
        assert(!is_square_attacked(pos, make_square(5, 3), Color::White));
        assert(is_square_attacked(pos, make_square(4, 2), Color::White));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::Queen, make_square(3, 3));
        assert(is_square_attacked(pos, make_square(3, 7), Color::White));
        assert(is_square_attacked(pos, make_square(7, 7), Color::White));
        assert(!is_square_attacked(pos, make_square(0, 7), Color::White));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::Rook, make_square(0, 0));
        assert(!is_square_attacked(pos, make_square(0, 7), Color::Black));
    }
}
