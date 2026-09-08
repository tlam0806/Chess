#include "position.hpp"

#include <cassert>

using namespace chess;

int main() {
    Position pos;
    pos.set_startpos();

    assert(popcount(pos.occupancy(Color::White)) == 16);
    assert(popcount(pos.occupancy(Color::Black)) == 16);
    assert(popcount(pos.occupancy()) == 32);

    const Square a1 = make_square(0, 0);
    const Square d1 = make_square(3, 0);
    const Square e1 = make_square(4, 0);
    const Square h1 = make_square(7, 0);
    const Square a2 = make_square(0, 1);
    const Square h2 = make_square(7, 1);

    const Square a7 = make_square(0, 6);
    const Square h7 = make_square(7, 6);
    const Square a8 = make_square(0, 7);
    const Square d8 = make_square(3, 7);
    const Square e8 = make_square(4, 7);
    const Square h8 = make_square(7, 7);

    assert(pos.color_on_occupied(a1) == Color::White);
    assert(pos.piece_type_on_occupied(a1) == PieceType::Rook);
    assert(pos.color_on_occupied(h1) == Color::White);
    assert(pos.piece_type_on_occupied(h1) == PieceType::Rook);
    assert(pos.color_on_occupied(d1) == Color::White);
    assert(pos.piece_type_on_occupied(d1) == PieceType::Queen);
    assert(pos.color_on_occupied(e1) == Color::White);
    assert(pos.piece_type_on_occupied(e1) == PieceType::King);
    assert(pos.color_on_occupied(a2) == Color::White);
    assert(pos.piece_type_on_occupied(a2) == PieceType::Pawn);
    assert(pos.color_on_occupied(h2) == Color::White);
    assert(pos.piece_type_on_occupied(h2) == PieceType::Pawn);

    assert(pos.color_on_occupied(a8) == Color::Black);
    assert(pos.piece_type_on_occupied(a8) == PieceType::Rook);
    assert(pos.color_on_occupied(h8) == Color::Black);
    assert(pos.piece_type_on_occupied(h8) == PieceType::Rook);
    assert(pos.color_on_occupied(d8) == Color::Black);
    assert(pos.piece_type_on_occupied(d8) == PieceType::Queen);
    assert(pos.color_on_occupied(e8) == Color::Black);
    assert(pos.piece_type_on_occupied(e8) == PieceType::King);
    assert(pos.color_on_occupied(a7) == Color::Black);
    assert(pos.piece_type_on_occupied(a7) == PieceType::Pawn);
    assert(pos.color_on_occupied(h7) == Color::Black);
    assert(pos.piece_type_on_occupied(h7) == PieceType::Pawn);

    assert(pos.is_empty(make_square(0, 2)));
    assert(pos.is_empty(make_square(4, 3)));
    assert(pos.is_empty(make_square(7, 5)));

    assert(pos.side_to_move == Color::White);
    assert(pos.white_can_castle_kingside);
    assert(pos.white_can_castle_queenside);
    assert(pos.black_can_castle_kingside);
    assert(pos.black_can_castle_queenside);
    assert(pos.en_passant_square == NoSquare);
    assert(pos.halfmove_clock == 0);
    assert(pos.fullmove_number == 1);
}
