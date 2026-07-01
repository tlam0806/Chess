#pragma once

#include "attacks.hpp"
#include "king_safety.hpp"
#include "move.hpp"
#include "position.hpp"

#include <cassert>
#include <cstdlib>

namespace chess {

namespace detail {

struct LegalNonCaptureMoveGenerationContext {
    Color us;
    Color them;
    int us_idx;
    Bitboard our_occupancy;
    Bitboard occupancy;
};

inline LegalNonCaptureMoveGenerationContext make_legal_non_capture_context(const Position& pos) {
    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Bitboard our_occupancy = pos.occupancy(us);
    const Bitboard their_occupancy = pos.occupancy(them);
    return LegalNonCaptureMoveGenerationContext{
        us,
        them,
        static_cast<int>(us),
        our_occupancy,
        our_occupancy | their_occupancy
    };
}

inline bool has_rook_on_for_non_capture(const Position& pos, Color color, Square square) {
    return (pos.pieces[static_cast<int>(color)][static_cast<int>(PieceType::Rook)] & bit(square))
        != EmptyBB;
}

inline bool same_line_for_non_capture(Square a, Square b, Square c) {
    const int af = file_of(a);
    const int ar = rank_of(a);
    const int bf = file_of(b);
    const int br = rank_of(b);
    const int cf = file_of(c);
    const int cr = rank_of(c);

    const int df1 = bf - af;
    const int dr1 = br - ar;
    const int df2 = cf - af;
    const int dr2 = cr - ar;
    return (df1 == 0 && df2 == 0)
        || (dr1 == 0 && dr2 == 0)
        || (std::abs(df1) == std::abs(dr1)
            && std::abs(df2) == std::abs(dr2)
            && df1 * dr2 == dr1 * df2);
}

inline bool legal_non_king_non_capture_target(
    const KingSafetyContext& king_safety,
    Square from,
    Square to
) {
    if ((king_safety.pinned & bit(from)) == EmptyBB) {
        return true;
    }
    return same_line_for_non_capture(king_safety.king_square, from, to);
}

template <typename Callback>
inline void try_emit_legal_king_non_capture_move(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Move move,
    Callback&& callback
) {
    if (is_pseudo_move_legal(pos, king_safety, move, PieceType::King, PieceType::None)) {
        callback(move, PieceType::King);
    }
}

template <typename Callback>
inline void try_emit_legal_non_king_non_capture_move(
    const KingSafetyContext& king_safety,
    Move move,
    PieceType moved_piece,
    Callback&& callback
) {
    if (legal_non_king_non_capture_target(king_safety, from_square(move), to_square(move))) {
        callback(move, moved_piece);
    }
}

template <typename Callback>
inline void generate_castling_moves_with_info(
    const Position& pos,
    const LegalNonCaptureMoveGenerationContext& context,
    Square king_square,
    Callback&& callback
) {
    const Color color = pos.side_to_move;
    const Color enemy = opposite(color);
    const Bitboard occupancy = context.occupancy;

    if (color == Color::White) {
        const Square e1 = make_square(4, 0);
        const Square f1 = make_square(5, 0);
        const Square g1 = make_square(6, 0);
        const Square d1 = make_square(3, 0);
        const Square c1 = make_square(2, 0);
        const Square b1 = make_square(1, 0);
        const Square h1 = make_square(7, 0);
        const Square a1 = make_square(0, 0);

        if (king_square == e1 && pos.white_can_castle_kingside
            && has_rook_on_for_non_capture(pos, color, h1)
            && !(occupancy & (bit(f1) | bit(g1)))
            && !is_square_attacked(pos, e1, enemy)
            && !is_square_attacked(pos, f1, enemy)
            && !is_square_attacked(pos, g1, enemy)) {
            callback(make_move(e1, g1, MoveFlag::KingCastle), PieceType::King);
        }

        if (king_square == e1 && pos.white_can_castle_queenside
            && has_rook_on_for_non_capture(pos, color, a1)
            && !(occupancy & (bit(d1) | bit(c1) | bit(b1)))
            && !is_square_attacked(pos, e1, enemy)
            && !is_square_attacked(pos, d1, enemy)
            && !is_square_attacked(pos, c1, enemy)) {
            callback(make_move(e1, c1, MoveFlag::QueenCastle), PieceType::King);
        }
    } else {
        const Square e8 = make_square(4, 7);
        const Square f8 = make_square(5, 7);
        const Square g8 = make_square(6, 7);
        const Square d8 = make_square(3, 7);
        const Square c8 = make_square(2, 7);
        const Square b8 = make_square(1, 7);
        const Square h8 = make_square(7, 7);
        const Square a8 = make_square(0, 7);

        if (king_square == e8 && pos.black_can_castle_kingside
            && has_rook_on_for_non_capture(pos, color, h8)
            && !(occupancy & (bit(f8) | bit(g8)))
            && !is_square_attacked(pos, e8, enemy)
            && !is_square_attacked(pos, f8, enemy)
            && !is_square_attacked(pos, g8, enemy)) {
            callback(make_move(e8, g8, MoveFlag::KingCastle), PieceType::King);
        }

        if (king_square == e8 && pos.black_can_castle_queenside
            && has_rook_on_for_non_capture(pos, color, a8)
            && !(occupancy & (bit(d8) | bit(c8) | bit(b8)))
            && !is_square_attacked(pos, e8, enemy)
            && !is_square_attacked(pos, d8, enemy)
            && !is_square_attacked(pos, c8, enemy)) {
            callback(make_move(e8, c8, MoveFlag::QueenCastle), PieceType::King);
        }
    }
}

template <typename Callback>
inline void generate_legal_king_non_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNonCaptureMoveGenerationContext& context,
    Callback&& callback
) {
    Bitboard kings = pos.pieces[context.us_idx][static_cast<int>(PieceType::King)];
    while (kings != EmptyBB) {
        const Square from = pop_lsb(kings);
        Bitboard targets = king_attacks(from) & ~context.occupancy;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_king_non_capture_move(
                pos,
                king_safety,
                make_move(from, to, MoveFlag::Quiet),
                callback);
        }

        generate_castling_moves_with_info(pos, context, from, [&](Move move, PieceType) {
            try_emit_legal_king_non_capture_move(pos, king_safety, move, callback);
        });
    }
}

template <typename Callback>
inline void generate_legal_pawn_quiet_non_promotion_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNonCaptureMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pawns = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    const int starting_rank = context.us == Color::White ? static_cast<int>(Rank::R2) : static_cast<int>(Rank::R7);
    const int promotion_rank = context.us == Color::White ? static_cast<int>(Rank::R8) : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;

    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        const int rank = rank_of(from);
        const int file = file_of(from);
        assert(rank != promotion_rank);

        const int next_rank = rank + offset;
        if (next_rank == promotion_rank) {
            continue;
        }

        const Square to = make_square(file, next_rank);
        const Bitboard to_mask = bit(to);
        if ((context.occupancy & to_mask) != EmptyBB || (allowed_targets & to_mask) == EmptyBB) {
            continue;
        }

        try_emit_legal_non_king_non_capture_move(
            king_safety,
            make_move(from, to, MoveFlag::Quiet),
            PieceType::Pawn,
            callback);
        if (rank == starting_rank) {
            const Square double_to = make_square(file, rank + offset + offset);
            const Bitboard double_to_mask = bit(double_to);
            if ((context.occupancy & double_to_mask) == EmptyBB
                && (allowed_targets & double_to_mask) != EmptyBB) {
                try_emit_legal_non_king_non_capture_move(
                    king_safety,
                    make_move(from, double_to, MoveFlag::DoublePawnPush),
                    PieceType::Pawn,
                    callback);
            }
        }
    }
}

template <typename Callback>
inline void generate_legal_knight_non_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNonCaptureMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = knight_attacks(from) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_non_king_non_capture_move(
                king_safety,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::Knight,
                callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_bishop_non_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNonCaptureMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = bishop_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_non_king_non_capture_move(
                king_safety,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::Bishop,
                callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_rook_non_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNonCaptureMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = rook_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_non_king_non_capture_move(
                king_safety,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::Rook,
                callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_queen_non_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNonCaptureMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = queen_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_non_king_non_capture_move(
                king_safety,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::Queen,
                callback);
        }
    }
}

} // namespace detail

template <typename Callback>
inline void generate_legal_quiet_non_promotion_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Callback&& callback
) {
    const detail::LegalNonCaptureMoveGenerationContext context =
        detail::make_legal_non_capture_context(pos);
    detail::generate_legal_king_non_capture_moves_with_info(pos, king_safety, context, callback);

    if (popcount(king_safety.checkers) >= 2) {
        return;
    }

    const Bitboard allowed_targets = king_safety.block_mask & ~context.occupancy;
    detail::generate_legal_pawn_quiet_non_promotion_moves_with_info(
        pos,
        king_safety,
        context,
        allowed_targets,
        callback);
    detail::generate_legal_knight_non_capture_moves_with_info(
        pos,
        king_safety,
        context,
        allowed_targets,
        callback);
    detail::generate_legal_bishop_non_capture_moves_with_info(
        pos,
        king_safety,
        context,
        allowed_targets,
        callback);
    detail::generate_legal_rook_non_capture_moves_with_info(
        pos,
        king_safety,
        context,
        allowed_targets,
        callback);
    detail::generate_legal_queen_non_capture_moves_with_info(
        pos,
        king_safety,
        context,
        allowed_targets,
        callback);
}

} // namespace chess
