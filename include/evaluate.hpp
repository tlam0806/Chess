#pragma once

#include "move.hpp"
#include "position.hpp"

namespace chess {

struct SeeAttacker {
    Square square = NoSquare;
    PieceType piece = PieceType::None;
};

int evaluate(const Position& pos);
int evaluate_for_side_to_move(const Position& pos);

bool is_pinned(Position pos, Square prev, Square next);
bool king_capture_legal(const Position& pos, Square square);
SeeAttacker find_least_valuable_attacker(const Position& pos, Square square);
int static_exchange_eval(const Position& pos, Move move);
int static_exchange_eval(
    const Position& pos,
    Move move,
    PieceType moving_piece,
    PieceType captured_piece
);

} // namespace chess
