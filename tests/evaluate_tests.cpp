#include "evaluate.hpp"

#include <cassert>

using namespace chess;

int main() {
    {
        assert(relative_square(Color::White, make_square(4, 1)) == make_square(4, 1));
        assert(relative_square(Color::Black, make_square(4, 1)) == make_square(4, 6));
        assert(relative_square(Color::Black, make_square(0, 0)) == make_square(0, 7));
        assert(relative_square(Color::Black, make_square(7, 7)) == make_square(7, 0));
    }

    {
        Position pos;
        pos.set_startpos();

        assert(evaluate(pos) == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Queen, make_square(3, 0));

        assert(evaluate(pos) > 800);
        pos.side_to_move = Color::White;
        assert(evaluate_for_side_to_move(pos) == evaluate(pos));
        pos.side_to_move = Color::Black;
        assert(evaluate_for_side_to_move(pos) == -evaluate(pos));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(0, 7));

        assert(evaluate(pos) < -400);
    }

    {
        Position corner_knight;
        corner_knight.set_piece(Color::White, PieceType::King, make_square(4, 0));
        corner_knight.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        corner_knight.set_piece(Color::White, PieceType::Knight, make_square(0, 0));

        Position center_knight;
        center_knight.set_piece(Color::White, PieceType::King, make_square(4, 0));
        center_knight.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        center_knight.set_piece(Color::White, PieceType::Knight, make_square(3, 3));

        assert(evaluate(center_knight) > evaluate(corner_knight));
    }

    {
        Position white_extra_pawn;
        white_extra_pawn.set_piece(Color::White, PieceType::King, make_square(4, 0));
        white_extra_pawn.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        white_extra_pawn.set_piece(Color::White, PieceType::Pawn, make_square(4, 3));

        Position black_extra_pawn;
        black_extra_pawn.set_piece(Color::White, PieceType::King, make_square(4, 0));
        black_extra_pawn.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        black_extra_pawn.set_piece(Color::Black, PieceType::Pawn, make_square(4, 4));

        assert(evaluate(white_extra_pawn) == -evaluate(black_extra_pawn));
    }
}
