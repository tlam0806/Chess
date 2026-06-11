#pragma once
#include <cstdint>
#include <cassert>

namespace chess {
    using Bitboard = std::uint64_t;

    enum class PieceType : int { Pawn = 0, Knight, Bishop, Rook, Queen, King, None};
    enum class Color : int { White = 0, Black = 1};

    enum class File : int { A = 0, B, C, D, E, F, G, H };
    enum class Rank : int { R1 = 0, R2, R3, R4, R5, R6, R7, R8 };

    using Square = int;

    constexpr int BoardSize = 64;
    constexpr Square NoSquare = -1;

    constexpr bool is_valid_square(int file, int rank) {
        return 0 <= rank && rank < 8 && 0 <= file && file < 8;
    }

    constexpr bool is_valid_square(Square square) {
        return 0 <= square && square < BoardSize;
    }

    constexpr int file_of(Square square) {
        assert(is_valid_square(square));
        return square & 7;
    }

    constexpr int rank_of(Square square) {
        assert(is_valid_square(square));
        return square >> 3;
    }

    constexpr Square make_square (int file, int rank) {
        assert(0 <= file && file < 8);
        assert(0 <= rank && rank < 8);
        return rank * 8 + file;
    }

    constexpr Color opposite(Color color) {
        return color == Color::White ? Color::Black : Color::White;
    }

    constexpr Square relative_square(Color color, Square square) {
        return color == Color::White ? square : square ^ 56;
    }
}
