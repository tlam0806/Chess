#include "evaluate.hpp"

#include <cassert>

using namespace chess;

namespace {

Position empty_kings(Color side_to_move = Color::White) {
    Position pos;
    pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
    pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
    pos.side_to_move = side_to_move;
    return pos;
}

void assert_see_matches_overload(const Position& pos, Move move) {
    const PieceType moving_piece = pos.piece_type_on_occupied(move.from());
    const PieceType captured_piece = move.flag() == MoveFlag::EnPassant
        ? PieceType::Pawn
        : pos.piece_type_on_occupied(move.to());
    assert(static_exchange_eval(pos, move)
        == static_exchange_eval(pos, move, moving_piece, captured_piece));
}

} // namespace

int main() {
    {
        Position pos = empty_kings(Color::White);
        pos.set_piece(Color::White, PieceType::Rook, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));

        assert(is_pinned(pos, make_square(4, 1), make_square(3, 1)));
        assert(!is_pinned(pos, make_square(4, 1), make_square(4, 2)));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 3));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));
        pos.side_to_move = Color::White;

        assert(!king_capture_legal(pos, make_square(4, 4)));
        assert(king_capture_legal(pos, make_square(3, 3)));
    }

    {
        Position pos = empty_kings(Color::White);
        pos.set_piece(Color::Black, PieceType::Queen, make_square(3, 4));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(2, 3));
        pos.set_piece(Color::White, PieceType::Knight, make_square(5, 3));

        const SeeAttacker attacker = find_least_valuable_attacker(pos, make_square(3, 4));
        assert(attacker.square == make_square(2, 3));
        assert(attacker.piece == PieceType::Pawn);
    }

    {
        Position pos = empty_kings(Color::Black);
        pos.set_piece(Color::White, PieceType::Queen, make_square(3, 3));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(2, 4));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(5, 4));

        const SeeAttacker attacker = find_least_valuable_attacker(pos, make_square(3, 3));
        assert(attacker.square == make_square(2, 4));
        assert(attacker.piece == PieceType::Pawn);
    }

    {
        Position pos = empty_kings(Color::White);
        pos.set_piece(Color::Black, PieceType::Queen, make_square(3, 1));
        pos.set_piece(Color::White, PieceType::Rook, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));

        const SeeAttacker attacker = find_least_valuable_attacker(pos, make_square(3, 1));
        assert(attacker.piece == PieceType::None);
        assert(attacker.square == NoSquare);
    }

    {
        Position pos = empty_kings(Color::White);
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 3));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));

        const Move capture = make_move(make_square(4, 3), make_square(3, 4), MoveFlag::Capture);
        assert_see_matches_overload(pos, capture);
        assert(static_exchange_eval(pos, capture) == 100);
    }

    {
        Position pos = empty_kings(Color::White);
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 3));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(5, 5));

        const Move capture = make_move(make_square(4, 3), make_square(3, 4), MoveFlag::Capture);
        assert_see_matches_overload(pos, capture);
        assert(static_exchange_eval(pos, capture) == 0);
    }

    {
        Position pos = empty_kings(Color::White);
        pos.set_piece(Color::White, PieceType::Queen, make_square(3, 0));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(5, 5));

        const Move capture = make_move(make_square(3, 0), make_square(3, 4), MoveFlag::Capture);
        assert_see_matches_overload(pos, capture);
        assert(static_exchange_eval(pos, capture) == -800);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(6, 6));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(7, 7));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(5, 6));
        pos.side_to_move = Color::White;

        const Move capture = make_move(
            make_square(6, 6),
            make_square(7, 7),
            MoveFlag::QueenPromotionCapture);
        assert_see_matches_overload(pos, capture);
        assert(static_exchange_eval(pos, capture) == -400);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 4));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.en_passant_square = make_square(3, 5);
        pos.side_to_move = Color::White;

        const Move capture = make_move(
            make_square(4, 4),
            make_square(3, 5),
            MoveFlag::EnPassant);
        assert_see_matches_overload(pos, capture);
        assert(static_exchange_eval(pos, capture) == 0);
    }
}
