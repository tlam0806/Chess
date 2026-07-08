#pragma once

#include "attacks.hpp"
#include "king_safety.hpp"
#include "move.hpp"
#include "position.hpp"

#include <cassert>

namespace chess {

namespace detail {

struct LegalNoisyMoveGenerationContext {
    Color us;
    Color them;
    int us_idx;
    int them_idx;
    Bitboard our_occupancy;
    Bitboard their_occupancy;
    Bitboard occupancy;
    Bitboard their_king;
    Bitboard capturable;
};

inline LegalNoisyMoveGenerationContext make_legal_noisy_context(const Position& pos) {
    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const int us_idx = static_cast<int>(us);
    const int them_idx = static_cast<int>(them);
    const Bitboard our_occupancy = pos.occupancy(us);
    const Bitboard their_occupancy = pos.occupancy(them);
    const Bitboard their_king = pos.pieces[them_idx][static_cast<int>(PieceType::King)];
    return LegalNoisyMoveGenerationContext{
        us,
        them,
        us_idx,
        them_idx,
        our_occupancy,
        their_occupancy,
        our_occupancy | their_occupancy,
        their_king,
        their_occupancy & ~their_king
    };
}

inline Bitboard noisy_promotion_from_rank_mask(Color color) {
    return color == Color::White
        ? RankMask[static_cast<int>(Rank::R7)]
        : RankMask[static_cast<int>(Rank::R2)];
}

inline PieceType noisy_captured_piece(
    const Position& pos,
    const LegalNoisyMoveGenerationContext& context,
    Move move
) {
    if (!is_capture(move)) {
        return PieceType::None;
    }
    if (move_flag(move) == MoveFlag::EnPassant) {
        return PieceType::Pawn;
    }
    return pos.piece_type_on_occupied(context.them, to_square(move));
}

template <typename Callback>
inline void try_emit_legal_noisy_move(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Move move,
    PieceType moved_piece,
    Callback&& callback
) {
    const PieceType captured_piece = noisy_captured_piece(pos, context, move);
    if (is_pseudo_move_legal(pos, king_safety, move, moved_piece, captured_piece)) {
        callback(move, moved_piece, captured_piece);
    }
}

template <typename Callback>
inline void generate_legal_king_noisy_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Callback&& callback
) {
    Bitboard kings = pos.pieces[context.us_idx][static_cast<int>(PieceType::King)];
    while (kings != EmptyBB) {
        const Square from = pop_lsb(kings);
        Bitboard targets = king_attacks(from) & context.capturable;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_noisy_move(
                pos,
                king_safety,
                context,
                make_move(from, to, MoveFlag::Capture),
                PieceType::King,
                callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_pawn_noisy_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Bitboard allowed_capture_targets,
    Bitboard allowed_promotion_targets,
    Callback&& callback
) {
    const Bitboard all_pawns = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    const Bitboard promotion_pawns = all_pawns & noisy_promotion_from_rank_mask(context.us);
    const Bitboard normal_pawns = all_pawns & ~promotion_pawns;

    auto emit = [&](Square from, Square to, MoveFlag flag) {
        try_emit_legal_noisy_move(
            pos,
            king_safety,
            context,
            make_move(from, to, flag),
            PieceType::Pawn,
            callback);
    };
    auto emit_promotion_capture = [&](Square from, Square to) {
        emit(from, to, MoveFlag::QueenPromotionCapture);
        emit(from, to, MoveFlag::RookPromotionCapture);
        emit(from, to, MoveFlag::BishopPromotionCapture);
        emit(from, to, MoveFlag::KnightPromotionCapture);
    };
    auto emit_quiet_promotion = [&](Square from, Square to) {
        emit(from, to, MoveFlag::QueenPromotion);
        emit(from, to, MoveFlag::RookPromotion);
        emit(from, to, MoveFlag::BishopPromotion);
        emit(from, to, MoveFlag::KnightPromotion);
    };

    if (context.us == Color::White) {
        Bitboard normal_left_targets = ((normal_pawns & ~FileMask[0]) << 7) & allowed_capture_targets;
        while (normal_left_targets != EmptyBB) {
            const Square to = pop_lsb(normal_left_targets);
            emit(to - 7, to, MoveFlag::Capture);
        }

        Bitboard normal_right_targets = ((normal_pawns & ~FileMask[7]) << 9) & allowed_capture_targets;
        while (normal_right_targets != EmptyBB) {
            const Square to = pop_lsb(normal_right_targets);
            emit(to - 9, to, MoveFlag::Capture);
        }

        Bitboard promotion_left_targets = ((promotion_pawns & ~FileMask[0]) << 7) & allowed_capture_targets;
        while (promotion_left_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_left_targets);
            emit_promotion_capture(to - 7, to);
        }

        Bitboard promotion_right_targets = ((promotion_pawns & ~FileMask[7]) << 9) & allowed_capture_targets;
        while (promotion_right_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_right_targets);
            emit_promotion_capture(to - 9, to);
        }

        Bitboard promotion_targets = (promotion_pawns << 8) & ~context.occupancy & allowed_promotion_targets;
        while (promotion_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_targets);
            emit_quiet_promotion(to - 8, to);
        }

        if (pos.en_passant_square != NoSquare) {
            const Bitboard ep_target = bit(pos.en_passant_square);
            Bitboard ep_from = normal_pawns & (((ep_target & ~FileMask[7]) >> 7) | ((ep_target & ~FileMask[0]) >> 9));
            while (ep_from != EmptyBB) {
                emit(pop_lsb(ep_from), pos.en_passant_square, MoveFlag::EnPassant);
            }
        }
    } else {
        Bitboard normal_left_targets = ((normal_pawns & ~FileMask[0]) >> 9) & allowed_capture_targets;
        while (normal_left_targets != EmptyBB) {
            const Square to = pop_lsb(normal_left_targets);
            emit(to + 9, to, MoveFlag::Capture);
        }

        Bitboard normal_right_targets = ((normal_pawns & ~FileMask[7]) >> 7) & allowed_capture_targets;
        while (normal_right_targets != EmptyBB) {
            const Square to = pop_lsb(normal_right_targets);
            emit(to + 7, to, MoveFlag::Capture);
        }

        Bitboard promotion_left_targets = ((promotion_pawns & ~FileMask[0]) >> 9) & allowed_capture_targets;
        while (promotion_left_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_left_targets);
            emit_promotion_capture(to + 9, to);
        }

        Bitboard promotion_right_targets = ((promotion_pawns & ~FileMask[7]) >> 7) & allowed_capture_targets;
        while (promotion_right_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_right_targets);
            emit_promotion_capture(to + 7, to);
        }

        Bitboard promotion_targets = (promotion_pawns >> 8) & ~context.occupancy & allowed_promotion_targets;
        while (promotion_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_targets);
            emit_quiet_promotion(to + 8, to);
        }

        if (pos.en_passant_square != NoSquare) {
            const Bitboard ep_target = bit(pos.en_passant_square);
            Bitboard ep_from = normal_pawns & (((ep_target & ~FileMask[7]) << 9) | ((ep_target & ~FileMask[0]) << 7));
            while (ep_from != EmptyBB) {
                emit(pop_lsb(ep_from), pos.en_passant_square, MoveFlag::EnPassant);
            }
        }
    }
}

template <typename Callback>
inline void generate_legal_pawn_non_promotion_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Bitboard allowed_capture_targets,
    Callback&& callback
) {
    Bitboard pawns =
        pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)]
        & ~noisy_promotion_from_rank_mask(context.us);

    auto emit = [&](Square from, Square to, MoveFlag flag) {
        try_emit_legal_noisy_move(
            pos,
            king_safety,
            context,
            make_move(from, to, flag),
            PieceType::Pawn,
            callback);
    };

    if (context.us == Color::White) {
        Bitboard left_targets = ((pawns & ~FileMask[0]) << 7) & allowed_capture_targets;
        while (left_targets != EmptyBB) {
            const Square to = pop_lsb(left_targets);
            emit(to - 7, to, MoveFlag::Capture);
        }

        Bitboard right_targets = ((pawns & ~FileMask[7]) << 9) & allowed_capture_targets;
        while (right_targets != EmptyBB) {
            const Square to = pop_lsb(right_targets);
            emit(to - 9, to, MoveFlag::Capture);
        }

        if (pos.en_passant_square != NoSquare) {
            const Bitboard ep_target = bit(pos.en_passant_square);
            Bitboard ep_from = pawns & (((ep_target & ~FileMask[7]) >> 7) | ((ep_target & ~FileMask[0]) >> 9));
            while (ep_from != EmptyBB) {
                emit(pop_lsb(ep_from), pos.en_passant_square, MoveFlag::EnPassant);
            }
        }
    } else {
        Bitboard left_targets = ((pawns & ~FileMask[0]) >> 9) & allowed_capture_targets;
        while (left_targets != EmptyBB) {
            const Square to = pop_lsb(left_targets);
            emit(to + 9, to, MoveFlag::Capture);
        }

        Bitboard right_targets = ((pawns & ~FileMask[7]) >> 7) & allowed_capture_targets;
        while (right_targets != EmptyBB) {
            const Square to = pop_lsb(right_targets);
            emit(to + 7, to, MoveFlag::Capture);
        }

        if (pos.en_passant_square != NoSquare) {
            const Bitboard ep_target = bit(pos.en_passant_square);
            Bitboard ep_from = pawns & (((ep_target & ~FileMask[7]) << 9) | ((ep_target & ~FileMask[0]) << 7));
            while (ep_from != EmptyBB) {
                emit(pop_lsb(ep_from), pos.en_passant_square, MoveFlag::EnPassant);
            }
        }
    }
}

template <typename Callback>
inline void generate_legal_knight_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = knight_attacks(from) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_noisy_move(
                pos,
                king_safety,
                context,
                make_move(from, to, MoveFlag::Capture),
                PieceType::Knight,
                callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_bishop_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = bishop_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_noisy_move(
                pos,
                king_safety,
                context,
                make_move(from, to, MoveFlag::Capture),
                PieceType::Bishop,
                callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_rook_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = rook_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_noisy_move(
                pos,
                king_safety,
                context,
                make_move(from, to, MoveFlag::Capture),
                PieceType::Rook,
                callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_queen_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Bitboard allowed_targets,
    Callback&& callback
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = queen_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_emit_legal_noisy_move(
                pos,
                king_safety,
                context,
                make_move(from, to, MoveFlag::Capture),
                PieceType::Queen,
                callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_pawn_captures_to_square_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Square target,
    Callback&& callback
) {
    [[maybe_unused]] const int promotion_rank = context.us == Color::White
        ? static_cast<int>(Rank::R8)
        : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;
    Bitboard pawns =
        pawn_attacks(context.them, target)
        & pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        const int rank = rank_of(from);
        assert(rank != promotion_rank);
        if (rank + offset == promotion_rank) {
            try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, target, MoveFlag::QueenPromotionCapture), PieceType::Pawn, callback);
            try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, target, MoveFlag::RookPromotionCapture), PieceType::Pawn, callback);
            try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, target, MoveFlag::BishopPromotionCapture), PieceType::Pawn, callback);
            try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, target, MoveFlag::KnightPromotionCapture), PieceType::Pawn, callback);
        } else {
            try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, target, MoveFlag::Capture), PieceType::Pawn, callback);
        }
    }
}

template <typename Callback>
inline void generate_legal_en_passant_capture_checker_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Square checker,
    Callback&& callback
) {
    if (pos.en_passant_square == NoSquare) {
        return;
    }

    const Square captured_square = context.us == Color::White
        ? pos.en_passant_square - 8
        : pos.en_passant_square + 8;
    if (captured_square != checker) {
        return;
    }

    Bitboard pawns =
        pawn_attacks(context.them, pos.en_passant_square)
        & pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        try_emit_legal_noisy_move(
            pos,
            king_safety,
            context,
            make_move(from, pos.en_passant_square, MoveFlag::EnPassant),
            PieceType::Pawn,
            callback);
    }
}

template <typename Callback>
inline void generate_legal_piece_captures_to_square_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    PieceType piece,
    Bitboard pieces,
    Square target,
    Callback&& callback
) {
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        try_emit_legal_noisy_move(
            pos,
            king_safety,
            context,
            make_move(from, target, MoveFlag::Capture),
            piece,
            callback);
    }
}

template <typename Callback>
inline void generate_legal_quiet_promotion_blocks_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const LegalNoisyMoveGenerationContext& context,
    Callback&& callback
) {
    [[maybe_unused]] const int promotion_rank = context.us == Color::White
        ? static_cast<int>(Rank::R8)
        : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;
    Bitboard pawns = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)]
        & noisy_promotion_from_rank_mask(context.us);
    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        const int rank = rank_of(from);
        assert(rank != promotion_rank);
        assert(rank + offset == promotion_rank);
        const Square to = make_square(file_of(from), rank + offset);
        if ((context.occupancy & bit(to)) != EmptyBB
            || (king_safety.block_mask & bit(to)) == EmptyBB) {
            continue;
        }

        try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, to, MoveFlag::QueenPromotion), PieceType::Pawn, callback);
        try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, to, MoveFlag::RookPromotion), PieceType::Pawn, callback);
        try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, to, MoveFlag::BishopPromotion), PieceType::Pawn, callback);
        try_emit_legal_noisy_move(pos, king_safety, context, make_move(from, to, MoveFlag::KnightPromotion), PieceType::Pawn, callback);
    }
}

} // namespace detail

template <typename Callback>
inline void generate_legal_promotion_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Callback&& callback
) {
    const detail::LegalNoisyMoveGenerationContext context = detail::make_legal_noisy_context(pos);
    auto emit_promotion = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
        if (is_promotion(move)) {
            callback(move, moved_piece, captured_piece);
        }
    };

    if (popcount(king_safety.checkers) >= 2) {
        return;
    }

    if (king_safety.checkers != EmptyBB) {
        const Square checker = std::countr_zero(king_safety.checkers);
        detail::generate_legal_pawn_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            checker,
            emit_promotion);
        detail::generate_legal_quiet_promotion_blocks_with_info(
            pos,
            king_safety,
            context,
            emit_promotion);
        return;
    }

    detail::generate_legal_pawn_noisy_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        FullBB,
        emit_promotion);
}

template <typename Callback>
inline void generate_legal_non_promotion_capture_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Callback&& callback
) {
    const detail::LegalNoisyMoveGenerationContext context = detail::make_legal_noisy_context(pos);
    auto emit_non_promotion_capture = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
        if (is_capture(move) && !is_promotion(move)) {
            callback(move, moved_piece, captured_piece);
        }
    };

    detail::generate_legal_king_noisy_moves_with_info(pos, king_safety, context, emit_non_promotion_capture);

    if (popcount(king_safety.checkers) >= 2) {
        return;
    }

    if (king_safety.checkers != EmptyBB) {
        const Square checker = std::countr_zero(king_safety.checkers);
        detail::generate_legal_pawn_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            checker,
            emit_non_promotion_capture);
        detail::generate_legal_en_passant_capture_checker_with_info(
            pos,
            king_safety,
            context,
            checker,
            emit_non_promotion_capture);
        detail::generate_legal_piece_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            PieceType::Knight,
            knight_attacks(checker) & pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)],
            checker,
            emit_non_promotion_capture);
        detail::generate_legal_piece_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            PieceType::Bishop,
            bishop_attacks(checker, context.occupancy) & pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)],
            checker,
            emit_non_promotion_capture);
        detail::generate_legal_piece_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            PieceType::Rook,
            rook_attacks(checker, context.occupancy) & pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)],
            checker,
            emit_non_promotion_capture);
        detail::generate_legal_piece_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            PieceType::Queen,
            queen_attacks(checker, context.occupancy) & pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)],
            checker,
            emit_non_promotion_capture);
        return;
    }

    detail::generate_legal_pawn_non_promotion_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        emit_non_promotion_capture);
    detail::generate_legal_knight_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        emit_non_promotion_capture);
    detail::generate_legal_bishop_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        emit_non_promotion_capture);
    detail::generate_legal_rook_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        emit_non_promotion_capture);
    detail::generate_legal_queen_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        emit_non_promotion_capture);
}

template <typename Callback>
inline void generate_legal_noisy_moves_with_info(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Callback&& callback
) {
    const detail::LegalNoisyMoveGenerationContext context = detail::make_legal_noisy_context(pos);
    detail::generate_legal_king_noisy_moves_with_info(pos, king_safety, context, callback);

    if (popcount(king_safety.checkers) >= 2) {
        return;
    }

    if (king_safety.checkers != EmptyBB) {
        const Square checker = std::countr_zero(king_safety.checkers);
        detail::generate_legal_pawn_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            checker,
            callback);
        detail::generate_legal_en_passant_capture_checker_with_info(
            pos,
            king_safety,
            context,
            checker,
            callback);
        detail::generate_legal_piece_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            PieceType::Knight,
            knight_attacks(checker) & pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)],
            checker,
            callback);
        detail::generate_legal_piece_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            PieceType::Bishop,
            bishop_attacks(checker, context.occupancy) & pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)],
            checker,
            callback);
        detail::generate_legal_piece_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            PieceType::Rook,
            rook_attacks(checker, context.occupancy) & pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)],
            checker,
            callback);
        detail::generate_legal_piece_captures_to_square_with_info(
            pos,
            king_safety,
            context,
            PieceType::Queen,
            queen_attacks(checker, context.occupancy) & pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)],
            checker,
            callback);
        detail::generate_legal_quiet_promotion_blocks_with_info(
            pos,
            king_safety,
            context,
            callback);
        return;
    }

    detail::generate_legal_pawn_noisy_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        FullBB,
        callback);
    detail::generate_legal_knight_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        callback);
    detail::generate_legal_bishop_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        callback);
    detail::generate_legal_rook_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        callback);
    detail::generate_legal_queen_capture_moves_with_info(
        pos,
        king_safety,
        context,
        context.capturable,
        callback);
}

} // namespace chess
