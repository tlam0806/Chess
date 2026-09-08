#include "attacks.hpp"

#include <cassert>

using namespace chess;

int main() {
    assert(pawn_attacks(Color::White, make_square(4, 1)) ==
           (bit(make_square(3, 2)) | bit(make_square(5, 2))));

    assert(pawn_attacks(Color::Black, make_square(4, 6)) ==
           (bit(make_square(3, 5)) | bit(make_square(5, 5))));

    assert(pawn_attacks(Color::White, make_square(0, 1)) ==
           bit(make_square(1, 2)));

    assert(pawn_attacks(Color::White, make_square(7, 1)) ==
           bit(make_square(6, 2)));

    assert(pawn_attacks(Color::Black, make_square(0, 6)) ==
           bit(make_square(1, 5)));

    assert(pawn_attacks(Color::Black, make_square(7, 6)) ==
           bit(make_square(6, 5)));

    assert(pawn_attacks(Color::White, make_square(3, 7)) == EmptyBB);
    assert(pawn_attacks(Color::Black, make_square(3, 0)) == EmptyBB);
}
