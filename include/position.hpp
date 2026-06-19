#pragma once

#include "bitboard.hpp"
#include "types.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string_view>

namespace chess {

struct Move;

struct UndoState {
    HashKey zobrist_key = 0;
    std::array<Bitboard, 2> occupancies{EmptyBB, EmptyBB};
    std::array<Square, 2> king_squares{NoSquare, NoSquare};
    std::array<Bitboard, 2> king_checkers{EmptyBB, EmptyBB};
    std::array<Bitboard, 2> king_pinned{EmptyBB, EmptyBB};
    std::array<Bitboard, 2> king_block_masks{FullBB, FullBB};
    int eval_score = 0;
    int halfmove_clock = 0;
    int fullmove_number = 1;
    Square en_passant_square = NoSquare;
    std::int8_t captured_square = NoSquare;
    std::uint8_t castling_rights = 0;
    PieceType moved_piece = PieceType::None;
    PieceType captured_piece = PieceType::None;
};

struct Position {
    std::array<std::array<Bitboard, 6>, 2> pieces{};
    std::array<Bitboard, 2> occupancies{EmptyBB, EmptyBB};
    std::array<std::uint8_t, BoardSize> board{};
    Color side_to_move = Color::White;
    bool white_can_castle_kingside = false;
    bool white_can_castle_queenside = false;
    bool black_can_castle_kingside = false;
    bool black_can_castle_queenside = false;
    Square en_passant_square = NoSquare;
    int halfmove_clock = 0;
    int fullmove_number = 1;
    int eval_score = 0;
    HashKey zobrist_key = 0;
    std::array<Square, 2> king_squares{NoSquare, NoSquare};
    std::array<Bitboard, 2> king_checkers{EmptyBB, EmptyBB};
    std::array<Bitboard, 2> king_pinned{EmptyBB, EmptyBB};
    std::array<Bitboard, 2> king_block_masks{FullBB, FullBB};

    void print(std::ostream& os) const;

    Bitboard occupancy(Color color) const;
    Bitboard occupancy() const;
    Bitboard occupancy(Color color, PieceType piece) const;

    void set_piece(Color color, PieceType piece, Square square);
    void clear_square(Square square);
    void clear_piece(Color color, PieceType piece, Square square);

    Color color_on_occupied(Square square) const;
    PieceType piece_type_on_occupied(Square square) const;
    PieceType piece_type_on_occupied(Color color, Square square) const;
    bool is_empty(Square square) const;

    void clear();
    void set_startpos();
    bool set_fen(std::string_view fen);
    void make_move(Move move);
    void make_move(Move move, PieceType moved_piece);
    void make_move(Move move, PieceType moved_piece, PieceType captured_piece);
    void make_move(Move move, UndoState& undo);
    void make_move(Move move, PieceType moved_piece, UndoState& undo);
    void make_move(Move move, PieceType moved_piece, PieceType captured_piece, UndoState& undo);
    void unmake_move(Move move, const UndoState& undo);

};

} 
