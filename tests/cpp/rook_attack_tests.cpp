#include "attacks.hpp"

#include <cassert>

using namespace chess;

int main() {
    const Square d4 = make_square(3, 3);

    Bitboard expected_empty_d4 = EmptyBB;
    for (int rank = 0; rank < 8; ++rank) {
        if (rank != 3) {
            expected_empty_d4 |= bit(make_square(3, rank));
        }
    }
    for (int file = 0; file < 8; ++file) {
        if (file != 3) {
            expected_empty_d4 |= bit(make_square(file, 3));
        }
    }

    assert(rook_attacks(d4, EmptyBB) == expected_empty_d4);
    assert(popcount(rook_attacks(d4, EmptyBB)) == 14);

    const Square a1 = make_square(0, 0);
    Bitboard expected_empty_a1 = EmptyBB;
    for (int file = 1; file < 8; ++file) {
        expected_empty_a1 |= bit(make_square(file, 0));
    }
    for (int rank = 1; rank < 8; ++rank) {
        expected_empty_a1 |= bit(make_square(0, rank));
    }

    assert(rook_attacks(a1, EmptyBB) == expected_empty_a1);
    assert(popcount(rook_attacks(a1, EmptyBB)) == 14);

    const Bitboard blockers =
        bit(make_square(3, 5)) |
        bit(make_square(5, 3)) |
        bit(make_square(3, 1)) |
        bit(make_square(1, 3));

    const Bitboard expected_blocked =
        bit(make_square(3, 4)) |
        bit(make_square(3, 5)) |
        bit(make_square(4, 3)) |
        bit(make_square(5, 3)) |
        bit(make_square(3, 2)) |
        bit(make_square(3, 1)) |
        bit(make_square(2, 3)) |
        bit(make_square(1, 3));

    assert(rook_attacks(d4, blockers) == expected_blocked);
    assert(popcount(rook_attacks(d4, blockers)) == 8);
}
