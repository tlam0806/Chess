#pragma once

#include "bitboard.hpp"
#include "types.hpp"

#include <array>
#include <iosfwd>
#include <string_view>

namespace chess {

struct Move;

struct Position {
    std::array<std::array<Bitboard, 6>, 2> pieces{};
    Color side_to_move = Color::White;
    bool white_can_castle_kingside = false;
    bool white_can_castle_queenside = false;
    bool black_can_castle_kingside = false;
    bool black_can_castle_queenside = false;
    Square en_passant_square = NoSquare;
    int halfmove_clock = 0;
    int fullmove_number = 1;

    void print(std::ostream& os) const;

    Bitboard occupancy(Color color) const;
    Bitboard occupancy() const;

    void set_piece(Color color, PieceType piece, Square square);
    void clear_square(Square square);

    Color color_on_occupied(Square square) const;
    PieceType piece_type_on_occupied(Square square) const;
    bool is_empty(Square square) const;

    void clear();
    void set_startpos();
    bool set_fen(std::string_view fen);
    void make_move(Move move);
};

} 
