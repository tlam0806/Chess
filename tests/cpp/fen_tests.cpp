#include "position.hpp"

#include <cassert>

using namespace chess;

int main() {
    Position pos;

    assert(pos.set_fen("8/8/8/8/8/8/8/8 w"));
    assert(pos.occupancy() == EmptyBB);
    assert(pos.side_to_move == Color::White);
    assert(!pos.white_can_castle_kingside);
    assert(!pos.white_can_castle_queenside);
    assert(!pos.black_can_castle_kingside);
    assert(!pos.black_can_castle_queenside);

    assert(pos.set_fen("8/8/8/3k4/8/8/4K3/8 b"));
    const Square d5 = make_square(3, 4);
    const Square e2 = make_square(4, 1);
    assert(pos.occupancy(Color::White) == bit(e2));
    assert(pos.occupancy(Color::Black) == bit(d5));
    assert(pos.color_on_occupied(e2) == Color::White);
    assert(pos.piece_type_on_occupied(e2) == PieceType::King);
    assert(pos.color_on_occupied(d5) == Color::Black);
    assert(pos.piece_type_on_occupied(d5) == PieceType::King);
    assert(pos.side_to_move == Color::Black);

    assert(pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
    assert(popcount(pos.occupancy(Color::White)) == 16);
    assert(popcount(pos.occupancy(Color::Black)) == 16);
    assert(popcount(pos.occupancy()) == 32);
    assert(pos.color_on_occupied(make_square(4, 0)) == Color::White);
    assert(pos.piece_type_on_occupied(make_square(4, 0)) == PieceType::King);
    assert(pos.color_on_occupied(make_square(3, 7)) == Color::Black);
    assert(pos.piece_type_on_occupied(make_square(3, 7)) == PieceType::Queen);
    assert(pos.side_to_move == Color::White);
    assert(pos.white_can_castle_kingside);
    assert(pos.white_can_castle_queenside);
    assert(pos.black_can_castle_kingside);
    assert(pos.black_can_castle_queenside);
    assert(pos.en_passant_square == NoSquare);
    assert(pos.halfmove_clock == 0);
    assert(pos.fullmove_number == 1);

    assert(pos.set_fen("8/8/8/8/8/8/8/8 b q e3 17 42"));
    assert(pos.side_to_move == Color::Black);
    assert(!pos.white_can_castle_kingside);
    assert(!pos.white_can_castle_queenside);
    assert(!pos.black_can_castle_kingside);
    assert(pos.black_can_castle_queenside);
    assert(pos.en_passant_square == make_square(4, 2));
    assert(pos.halfmove_clock == 17);
    assert(pos.fullmove_number == 42);

    assert(pos.set_fen("8/8/8/8/8/8/8/8 w - - 0 1"));
    assert(!pos.white_can_castle_kingside);
    assert(!pos.white_can_castle_queenside);
    assert(!pos.black_can_castle_kingside);
    assert(!pos.black_can_castle_queenside);
    assert(pos.en_passant_square == NoSquare);
    assert(pos.halfmove_clock == 0);
    assert(pos.fullmove_number == 1);

    const Bitboard old_occupancy = pos.occupancy();
    const Color old_side = pos.side_to_move;

    assert(!pos.set_fen("8/8/8/8/8/8/8/X7 w"));
    assert(pos.occupancy() == old_occupancy);
    assert(pos.side_to_move == old_side);

    assert(!pos.set_fen("8/8/8/8/8/8/8/9 w"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/7 w"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8 x"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8 w KK"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8 w A"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8 w - i3 0 1"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8 w - e9 0 1"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8 w - - x 1"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8 w - - 0 0"));
    assert(!pos.set_fen("8/8/8/8/8/8/8/8 w - - 0 1 extra"));
}
