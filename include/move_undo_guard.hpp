#pragma once

#include "move.hpp"
#include "position.hpp"
#include "king_safety.hpp"

#include <cassert>

namespace chess {

class MoveUndoGuard {
public:
    MoveUndoGuard(
        Position& pos,
        Move move,
        PieceType moved_piece,
        PieceType captured_piece
    ) noexcept
        : pos_(pos),
          move_(move) {
        pos_.make_move(move_, moved_piece, captured_piece, undo_);
#ifndef NDEBUG
        const Color moved_color = opposite(pos_.side_to_move);
        assert(pos_.king_checkers[static_cast<int>(moved_color)] == EmptyBB);
        for (Color color : {Color::White, Color::Black}) {
            const int color_idx = static_cast<int>(color);
            const KingSafetyContext rebuilt = make_king_safety_context(pos_, color);
            assert(pos_.king_squares[color_idx] == rebuilt.king_square);
            assert(pos_.king_checkers[color_idx] == rebuilt.checkers);
            assert(pos_.king_pinned[color_idx] == rebuilt.pinned);
            assert(pos_.king_block_masks[color_idx] == rebuilt.block_mask);
        }
#endif
    }

    ~MoveUndoGuard() noexcept {
        pos_.unmake_move(move_, undo_);
    }

    MoveUndoGuard(const MoveUndoGuard&) = delete;
    MoveUndoGuard& operator=(const MoveUndoGuard&) = delete;

private:
    Position& pos_;
    Move move_{};
    UndoState undo_{};
};

class SnapshotMoveUndoGuard {
public:
    SnapshotMoveUndoGuard(
        Position& pos,
        const PositionStateSnapshot& snapshot,
        Move move,
        PieceType moved_piece,
        PieceType captured_piece
    ) noexcept
        : pos_(pos),
          snapshot_(snapshot),
          move_(move) {
        pos_.make_move(move_, moved_piece, captured_piece, undo_);
#ifndef NDEBUG
        const Color moved_color = opposite(pos_.side_to_move);
        assert(pos_.king_checkers[static_cast<int>(moved_color)] == EmptyBB);
        for (Color color : {Color::White, Color::Black}) {
            const int color_idx = static_cast<int>(color);
            const KingSafetyContext rebuilt = make_king_safety_context(pos_, color);
            assert(pos_.king_squares[color_idx] == rebuilt.king_square);
            assert(pos_.king_checkers[color_idx] == rebuilt.checkers);
            assert(pos_.king_pinned[color_idx] == rebuilt.pinned);
            assert(pos_.king_block_masks[color_idx] == rebuilt.block_mask);
        }
#endif
    }

    ~SnapshotMoveUndoGuard() noexcept {
        pos_.unmake_move(move_, snapshot_, undo_);
    }

    SnapshotMoveUndoGuard(const SnapshotMoveUndoGuard&) = delete;
    SnapshotMoveUndoGuard& operator=(const SnapshotMoveUndoGuard&) = delete;

private:
    Position& pos_;
    const PositionStateSnapshot& snapshot_;
    Move move_{};
    MoveUndoState undo_{};
};

} // namespace chess
