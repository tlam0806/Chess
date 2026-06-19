#pragma once

#include "bitboard.hpp"
#include "move.hpp"
#include "position.hpp"
#include "types.hpp"

namespace chess {

struct KingSafetyContext {
    Square king_square = NoSquare;
    Bitboard checkers = EmptyBB;
    Bitboard pinned = EmptyBB;
    Bitboard block_mask = FullBB;
};

KingSafetyContext make_king_safety_context(const Position& pos);
KingSafetyContext make_king_safety_context(const Position& pos, Color color);
KingSafetyContext make_old_king_safety_context(const Position& pos);
KingSafetyContext cached_king_safety_context(const Position& pos, Color color);
void refresh_king_safety(Position& pos);
void update_king_safety_after_move(
    Position& pos,
    Move move,
    Color moved_color,
    PieceType moved_piece,
    PieceType captured_piece,
    Square captured_square
);

bool is_pseudo_move_legal(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Move move,
    PieceType moved_piece,
    PieceType captured_piece
);

bool gives_check_fast(
    const Position& pos,
    Move move,
    PieceType moved_piece,
    PieceType captured_piece
);

} // namespace chess
