#include "bitboard.hpp"

#include <bit>
#include <cassert>
#include <ostream>

namespace chess {

namespace {

constexpr std::array<Bitboard, 8> make_file_masks() {
    std::array<Bitboard, 8> masks{};

    for (int file = 0; file < 8; ++file) {
        Bitboard bb = 0;
        for (int rank = 0; rank < 8; ++rank) {
            bb |= bit(make_square(file, rank));
        }
        masks[file] = bb;
    }

    return masks;
}

constexpr std::array<Bitboard, 8> make_rank_masks() {
    std::array<Bitboard, 8> masks{};

    for (int rank = 0; rank < 8; ++rank) {
        Bitboard bb = 0;
        for (int file = 0; file < 8; ++file) {
            bb |= bit(make_square(file, rank));
        }
        masks[rank] = bb;
    }

    return masks;
}

} // namespace

const std::array<Bitboard, 8> FileMask = make_file_masks();
const std::array<Bitboard, 8> RankMask = make_rank_masks();

int popcount(Bitboard bb) {
    return std::popcount(bb);
}

Square pop_lsb(Bitboard& bb) {
    assert(bb != 0);

    Square sq = std::countr_zero(bb);
    bb &= bb - 1;

    return sq;
}

void print_bitboard(Bitboard bb, std::ostream& os) {
    for (int rank = 7; rank >= 0; --rank) {
        for (int file = 0; file < 8; ++file) {
            Square sq = make_square(file, rank);
            os << ((bb & bit(sq)) ? '1' : '.');

            if (file != 7) {
                os << ' ';
            }
        }
        os << '\n';
    }
}

} // namespace chess