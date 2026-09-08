#include "attacks.hpp"

#include <cassert>

using namespace chess;

int main() {
    const Square d4 = make_square(3, 3);

    const Bitboard expected_empty_d4 =
        bit(make_square(4, 4)) | bit(make_square(5, 5)) |
        bit(make_square(6, 6)) | bit(make_square(7, 7)) |
        bit(make_square(4, 2)) | bit(make_square(5, 1)) |
        bit(make_square(6, 0)) |
        bit(make_square(2, 2)) | bit(make_square(1, 1)) |
        bit(make_square(0, 0)) |
        bit(make_square(2, 4)) | bit(make_square(1, 5)) |
        bit(make_square(0, 6));

    assert(bishop_attacks(d4, EmptyBB) == expected_empty_d4);
    assert(popcount(bishop_attacks(d4, EmptyBB)) == 13);

    const Square a1 = make_square(0, 0);
    const Bitboard expected_empty_a1 =
        bit(make_square(1, 1)) | bit(make_square(2, 2)) |
        bit(make_square(3, 3)) | bit(make_square(4, 4)) |
        bit(make_square(5, 5)) | bit(make_square(6, 6)) |
        bit(make_square(7, 7));

    assert(bishop_attacks(a1, EmptyBB) == expected_empty_a1);
    assert(popcount(bishop_attacks(a1, EmptyBB)) == 7);

    const Bitboard blockers =
        bit(make_square(5, 5)) |
        bit(make_square(1, 5)) |
        bit(make_square(5, 1)) |
        bit(make_square(1, 1));

    const Bitboard expected_blocked =
        bit(make_square(4, 4)) | bit(make_square(5, 5)) |
        bit(make_square(2, 4)) | bit(make_square(1, 5)) |
        bit(make_square(4, 2)) | bit(make_square(5, 1)) |
        bit(make_square(2, 2)) | bit(make_square(1, 1));

    assert(bishop_attacks(d4, blockers) == expected_blocked);
    assert(popcount(bishop_attacks(d4, blockers)) == 8);
}
