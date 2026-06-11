#include "move.hpp"

#include <cassert>
#include <string>

using namespace chess;

int main() {
    assert(square_to_string(make_square(0, 0)) == "a1");
    assert(square_to_string(make_square(4, 1)) == "e2");
    assert(square_to_string(make_square(7, 7)) == "h8");

    const Move e2e4 = make_move(make_square(4, 1), make_square(4, 3));
    assert(e2e4.from() == make_square(4, 1));
    assert(e2e4.to() == make_square(4, 3));
    assert(e2e4.flag() == MoveFlag::Quiet);
    assert(promotion_piece(e2e4) == PieceType::None);
    assert(!is_capture(e2e4));
    assert(move_to_string(e2e4) == "e2e4");

    const Move promotion = make_move(
        make_square(0, 6),
        make_square(0, 7),
        MoveFlag::QueenPromotion
    );
    assert(promotion_piece(promotion) == PieceType::Queen);
    assert(!is_capture(promotion));
    assert(move_to_string(promotion) == "a7a8q");

    const Move same_e2e4 = make_move(make_square(4, 1), make_square(4, 3));
    assert(e2e4 == same_e2e4);

    Move mutable_move = make_move(make_square(1, 0), make_square(2, 2));
    mutable_move.set_from(make_square(6, 0));
    mutable_move.set_to(make_square(5, 2));
    mutable_move.set_flag(MoveFlag::Capture);
    assert(mutable_move.from() == make_square(6, 0));
    assert(mutable_move.to() == make_square(5, 2));
    assert(mutable_move.flag() == MoveFlag::Capture);
    assert(is_capture(mutable_move));
    assert(move_to_string(mutable_move) == "g1f3");

    assert(is_capture(MoveFlag::EnPassant));
    assert(is_capture(make_move(make_square(4, 6), make_square(5, 7), MoveFlag::QueenPromotionCapture)));
    assert(!is_capture(MoveFlag::DoublePawnPush));
}
