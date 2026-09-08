#include "position.hpp"

#include <cassert>

using namespace chess;

int main() {
    Position pos;

    const Square e1 = make_square(4, 0);
    const Square a1 = make_square(0, 0);
    const Square h8 = make_square(7, 7);

    assert(pos.occupancy(Color::White) == EmptyBB);
    assert(pos.occupancy(Color::Black) == EmptyBB);
    assert(pos.occupancy() == EmptyBB);
    assert(pos.is_empty(e1));
    assert(pos.is_empty(a1));
    assert(pos.is_empty(h8));

    pos.set_piece(Color::White, PieceType::King, e1);
    pos.set_piece(Color::White, PieceType::Rook, a1);
    pos.set_piece(Color::Black, PieceType::Queen, h8);

    assert(pos.occupancy(Color::White) == (bit(e1) | bit(a1)));
    assert(pos.occupancy(Color::Black) == bit(h8));
    assert(pos.occupancy() == (bit(e1) | bit(a1) | bit(h8)));
    assert(!pos.is_empty(e1));
    assert(!pos.is_empty(a1));
    assert(!pos.is_empty(h8));
    assert(pos.color_on_occupied(e1) == Color::White);
    assert(pos.color_on_occupied(a1) == Color::White);
    assert(pos.color_on_occupied(h8) == Color::Black);
    assert(pos.piece_type_on_occupied(e1) == PieceType::King);
    assert(pos.piece_type_on_occupied(a1) == PieceType::Rook);
    assert(pos.piece_type_on_occupied(h8) == PieceType::Queen);

    pos.clear_square(a1);
    assert(pos.occupancy(Color::White) == bit(e1));
    assert(pos.occupancy(Color::Black) == bit(h8));
    assert(pos.occupancy() == (bit(e1) | bit(h8)));
    assert(pos.is_empty(a1));
    assert(!pos.is_empty(e1));
    assert(!pos.is_empty(h8));

    pos.clear_square(h8);
    assert(pos.occupancy(Color::White) == bit(e1));
    assert(pos.occupancy(Color::Black) == EmptyBB);
    assert(pos.occupancy() == bit(e1));
    assert(pos.is_empty(h8));
    assert(pos.color_on_occupied(e1) == Color::White);
    assert(pos.piece_type_on_occupied(e1) == PieceType::King);
}
