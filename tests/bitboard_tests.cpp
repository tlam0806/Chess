#include "bitboard.hpp"

#include <cassert>

using namespace chess;

int main() {
    static_assert(make_square(0, 0) == 0);
    static_assert(make_square(7, 0) == 7);
    static_assert(make_square(0, 7) == 56);
    static_assert(make_square(7, 7) == 63);

    static_assert(file_of(0) == 0);
    static_assert(rank_of(0) == 0);
    static_assert(file_of(7) == 7);
    static_assert(rank_of(7) == 0);
    static_assert(file_of(56) == 0);
    static_assert(rank_of(56) == 7);
    static_assert(file_of(63) == 7);
    static_assert(rank_of(63) == 7);

    assert(bit(0) == 1ULL);
    assert(bit(7) == 128ULL);
    assert(bit(63) == (1ULL << 63));

    Bitboard bb = bit(0) | bit(7) | bit(63);
    assert(popcount(bb) == 3);

    assert(pop_lsb(bb) == 0);
    assert(popcount(bb) == 2);
    assert(pop_lsb(bb) == 7);
    assert(popcount(bb) == 1);
    assert(pop_lsb(bb) == 63);
    assert(bb == 0);

    assert(FileMask[0] & bit(0));
    assert(FileMask[0] & bit(56));
    assert(!(FileMask[0] & bit(1)));

    assert(FileMask[7] & bit(7));
    assert(FileMask[7] & bit(63));
    assert(!(FileMask[7] & bit(6)));

    assert(RankMask[0] & bit(0));
    assert(RankMask[0] & bit(7));
    assert(!(RankMask[0] & bit(8)));

    assert(RankMask[7] & bit(56));
    assert(RankMask[7] & bit(63));
    assert(!(RankMask[7] & bit(55)));
}
