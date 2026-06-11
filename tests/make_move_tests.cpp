#include "move.hpp"
#include "position.hpp"

#include <cassert>

using namespace chess;

int main() {
    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Knight, make_square(6, 0));
        pos.side_to_move = Color::White;
        pos.en_passant_square = make_square(0, 2);
        pos.halfmove_clock = 4;
        pos.fullmove_number = 7;

        pos.make_move(make_move(make_square(6, 0), make_square(5, 2)));

        assert(pos.is_empty(make_square(6, 0)));
        assert(pos.color_on_occupied(make_square(5, 2)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(5, 2)) == PieceType::Knight);
        assert(pos.side_to_move == Color::Black);
        assert(pos.en_passant_square == NoSquare);
        assert(pos.halfmove_clock == 5);
        assert(pos.fullmove_number == 7);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Knight, make_square(3, 3));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(5, 4));
        pos.side_to_move = Color::White;
        pos.halfmove_clock = 4;

        pos.make_move(make_move(make_square(3, 3), make_square(5, 4), MoveFlag::Capture));

        assert(pos.is_empty(make_square(3, 3)));
        assert(pos.color_on_occupied(make_square(5, 4)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(5, 4)) == PieceType::Knight);
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.side_to_move = Color::White;
        pos.halfmove_clock = 8;

        pos.make_move(make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush));

        assert(pos.is_empty(make_square(4, 1)));
        assert(pos.color_on_occupied(make_square(4, 3)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(4, 3)) == PieceType::Pawn);
        assert(pos.en_passant_square == make_square(4, 2));
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 4));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.side_to_move = Color::White;
        pos.en_passant_square = make_square(3, 5);
        pos.halfmove_clock = 8;

        pos.make_move(make_move(make_square(4, 4), make_square(3, 5), MoveFlag::EnPassant));

        assert(pos.is_empty(make_square(4, 4)));
        assert(pos.is_empty(make_square(3, 4)));
        assert(pos.color_on_occupied(make_square(3, 5)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(3, 5)) == PieceType::Pawn);
        assert(pos.en_passant_square == NoSquare);
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.side_to_move = Color::White;
        pos.halfmove_clock = 8;

        pos.make_move(make_move(make_square(4, 6), make_square(4, 7), MoveFlag::QueenPromotion));

        assert(pos.is_empty(make_square(4, 6)));
        assert(pos.color_on_occupied(make_square(4, 7)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(4, 7)) == PieceType::Queen);
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(5, 7));
        pos.side_to_move = Color::White;
        pos.halfmove_clock = 8;

        pos.make_move(make_move(make_square(4, 6), make_square(5, 7), MoveFlag::KnightPromotionCapture));

        assert(pos.is_empty(make_square(4, 6)));
        assert(pos.color_on_occupied(make_square(5, 7)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(5, 7)) == PieceType::Knight);
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(6, 7));
        pos.side_to_move = Color::Black;
        pos.halfmove_clock = 10;
        pos.fullmove_number = 23;

        pos.make_move(make_move(make_square(6, 7), make_square(5, 5)));

        assert(pos.side_to_move == Color::White);
        assert(pos.halfmove_clock == 11);
        assert(pos.fullmove_number == 24);
    }
}
