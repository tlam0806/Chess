#pragma once

#include "types.hpp"

#include <array>
#include <iosfwd>

namespace chess {

constexpr Bitboard EmptyBB = 0ULL;
constexpr Bitboard FullBB = ~0ULL;

constexpr Bitboard bit(Square sq) {
    return 1ULL << sq;
}

int popcount(Bitboard bb);

Square pop_lsb(Bitboard& bb);

extern const std::array<Bitboard, 8> FileMask;
extern const std::array<Bitboard, 8> RankMask;

void print_bitboard(Bitboard bb, std::ostream& os);


} // namespace chess