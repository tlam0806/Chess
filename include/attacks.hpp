#pragma once

#include "bitboard.hpp"
#include "types.hpp"
#include "position.hpp"

namespace chess {
    Bitboard knight_attacks(Square square);
    Bitboard king_attacks(Square square);
    Bitboard pawn_attacks(Color color, Square square);

    Bitboard rook_attacks(Square square, Bitboard occupancy);
    Bitboard bishop_attacks(Square square, Bitboard occupancy);
    Bitboard queen_attacks(Square square, Bitboard occupancy);

    bool is_square_attacked(const Position& pos, Square square, Color by_color);
    bool is_square_attacked(
        const Position& pos,
        Square square,
        Color by_color,
        Bitboard occupancy
    );
    bool is_square_attacked(
        const Position& pos,
        Square square,
        Color by_color,
        Bitboard occupancy,
        Square excluded_attacker_square
    );

    Square king_square(const Position& pos, Color king_color);
    bool in_check(const Position& pos, Color king_color);
}
