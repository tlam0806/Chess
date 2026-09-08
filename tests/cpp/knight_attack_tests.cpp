#include "attacks.hpp"

#include <cassert>

using namespace chess;

int main() {
    assert(knight_attacks(make_square(0, 0)) ==
           (bit(make_square(1, 2)) | bit(make_square(2, 1))));

    assert(knight_attacks(make_square(7, 0)) ==
           (bit(make_square(5, 1)) | bit(make_square(6, 2))));

    assert(knight_attacks(make_square(0, 7)) ==
           (bit(make_square(1, 5)) | bit(make_square(2, 6))));

    assert(knight_attacks(make_square(7, 7)) ==
           (bit(make_square(5, 6)) | bit(make_square(6, 5))));

    assert(knight_attacks(make_square(3, 3)) ==
           (bit(make_square(4, 5)) | bit(make_square(5, 4)) |
            bit(make_square(5, 2)) | bit(make_square(4, 1)) |
            bit(make_square(2, 1)) | bit(make_square(1, 2)) |
            bit(make_square(1, 4)) | bit(make_square(2, 5))));

    assert(popcount(knight_attacks(make_square(3, 3))) == 8);
    assert(popcount(knight_attacks(make_square(0, 0))) == 2);
    assert(popcount(knight_attacks(make_square(1, 0))) == 3);
}
