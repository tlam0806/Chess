#include "attacks.hpp"

#include <cassert>

using namespace chess;

int main() {
    assert(king_attacks(make_square(0, 0)) ==
           (bit(make_square(1, 0)) |
            bit(make_square(0, 1)) |
            bit(make_square(1, 1))));

    assert(king_attacks(make_square(7, 0)) ==
           (bit(make_square(6, 0)) |
            bit(make_square(6, 1)) |
            bit(make_square(7, 1))));

    assert(king_attacks(make_square(0, 7)) ==
           (bit(make_square(0, 6)) |
            bit(make_square(1, 6)) |
            bit(make_square(1, 7))));

    assert(king_attacks(make_square(7, 7)) ==
           (bit(make_square(6, 6)) |
            bit(make_square(7, 6)) |
            bit(make_square(6, 7))));

    assert(king_attacks(make_square(3, 3)) ==
           (bit(make_square(2, 2)) | bit(make_square(3, 2)) |
            bit(make_square(4, 2)) | bit(make_square(2, 3)) |
            bit(make_square(4, 3)) | bit(make_square(2, 4)) |
            bit(make_square(3, 4)) | bit(make_square(4, 4))));

    assert(popcount(king_attacks(make_square(0, 0))) == 3);
    assert(popcount(king_attacks(make_square(3, 0))) == 5);
    assert(popcount(king_attacks(make_square(3, 3))) == 8);
}
