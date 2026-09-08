#include "attacks.hpp"

#include <cassert>

using namespace chess;

int main() {
    const Square d4 = make_square(3, 3);

    assert(queen_attacks(d4, EmptyBB) ==
           (rook_attacks(d4, EmptyBB) | bishop_attacks(d4, EmptyBB)));
    assert(popcount(queen_attacks(d4, EmptyBB)) == 27);

    const Square a1 = make_square(0, 0);
    assert(queen_attacks(a1, EmptyBB) ==
           (rook_attacks(a1, EmptyBB) | bishop_attacks(a1, EmptyBB)));
    assert(popcount(queen_attacks(a1, EmptyBB)) == 21);

    const Bitboard blockers =
        bit(make_square(3, 5)) |
        bit(make_square(5, 3)) |
        bit(make_square(3, 1)) |
        bit(make_square(1, 3)) |
        bit(make_square(5, 5)) |
        bit(make_square(1, 5)) |
        bit(make_square(5, 1)) |
        bit(make_square(1, 1));

    assert(queen_attacks(d4, blockers) ==
           (rook_attacks(d4, blockers) | bishop_attacks(d4, blockers)));
    assert(popcount(queen_attacks(d4, blockers)) == 16);
}
