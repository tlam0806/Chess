#include "king_safety.hpp"

#include "attacks.hpp"

#include <bit>
#include <cassert>
#include <cstdlib>

namespace chess {

namespace {

struct Direction {
    int file_delta = 0;
    int rank_delta = 0;
    bool rook_like = false;
};

constexpr Direction KingRayDirections[] = {
    {1, 0, true},
    {-1, 0, true},
    {0, 1, true},
    {0, -1, true},
    {1, 1, false},
    {1, -1, false},
    {-1, 1, false},
    {-1, -1, false},
};

bool is_slider_for_direction(const Position& pos, Color color, Square square, bool rook_like) {
    const Bitboard square_mask = bit(square);
    const int color_idx = static_cast<int>(color);
    if (rook_like) {
        return (pos.pieces[color_idx][static_cast<int>(PieceType::Rook)] & square_mask)
            || (pos.pieces[color_idx][static_cast<int>(PieceType::Queen)] & square_mask);
    }
    return (pos.pieces[color_idx][static_cast<int>(PieceType::Bishop)] & square_mask)
        || (pos.pieces[color_idx][static_cast<int>(PieceType::Queen)] & square_mask);
}

bool same_line(Square a, Square b, Square c) {
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

Bitboard ray_between_exclusive(Square from, Square to) {
    const int from_file = file_of(from);
    const int from_rank = rank_of(from);
    const int to_file = file_of(to);
    const int to_rank = rank_of(to);
    const int file_delta = (to_file > from_file) - (to_file < from_file);
    const int rank_delta = (to_rank > from_rank) - (to_rank < from_rank);

    Bitboard ray = EmptyBB;
    int file = from_file + file_delta;
    int rank = from_rank + rank_delta;
    while (file != to_file || rank != to_rank) {
        ray |= bit(make_square(file, rank));
        file += file_delta;
        rank += rank_delta;
    }
    return ray;
}

bool is_square_attacked_with_occupancy(
    const Position& pos,
    Square square,
    Color by_color,
    Bitboard occupancy,
    Square excluded_attacker_square
) {
    assert(is_valid_square(square));
    const Bitboard excluded = excluded_attacker_square == NoSquare ? EmptyBB : bit(excluded_attacker_square);
    const int color_idx = static_cast<int>(by_color);
    const Color other_color = opposite(by_color);
    return (pawn_attacks(other_color, square)
                & (pos.pieces[color_idx][static_cast<int>(PieceType::Pawn)] & ~excluded))
        || (knight_attacks(square)
                & (pos.pieces[color_idx][static_cast<int>(PieceType::Knight)] & ~excluded))
        || (bishop_attacks(square, occupancy)
                & (pos.pieces[color_idx][static_cast<int>(PieceType::Bishop)] & ~excluded))
        || (rook_attacks(square, occupancy)
                & (pos.pieces[color_idx][static_cast<int>(PieceType::Rook)] & ~excluded))
        || (queen_attacks(square, occupancy)
                & (pos.pieces[color_idx][static_cast<int>(PieceType::Queen)] & ~excluded))
        || (king_attacks(square)
                & (pos.pieces[color_idx][static_cast<int>(PieceType::King)] & ~excluded));
}

Bitboard occupancy_after_move(const Position& pos, Move move) {
    const Square from = from_square(move);
    const Square to = to_square(move);
    Bitboard occupancy = pos.occupancy();
    occupancy &= ~bit(from);
    if (is_capture(move) && move_flag(move) != MoveFlag::EnPassant) {
        occupancy &= ~bit(to);
    }
    occupancy |= bit(to);
    return occupancy;
}

bool piece_attacks_square(
    PieceType piece,
    Color color,
    Square from,
    Square target,
    Bitboard occupancy
) {
    const Bitboard target_mask = bit(target);
    switch (piece) {
        case PieceType::Pawn:
            return (pawn_attacks(color, from) & target_mask) != EmptyBB;
        case PieceType::Knight:
            return (knight_attacks(from) & target_mask) != EmptyBB;
        case PieceType::Bishop:
            return (bishop_attacks(from, occupancy) & target_mask) != EmptyBB;
        case PieceType::Rook:
            return (rook_attacks(from, occupancy) & target_mask) != EmptyBB;
        case PieceType::Queen:
            return (queen_attacks(from, occupancy) & target_mask) != EmptyBB;
        case PieceType::King:
            return (king_attacks(from) & target_mask) != EmptyBB;
        case PieceType::None:
            return false;
    }
    return false;
}

} // namespace

KingSafetyContext make_king_safety_context(const Position& pos) {
    KingSafetyContext context;
    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const int them_idx = static_cast<int>(them);
    context.king_square = king_square(pos, us);
    const Bitboard occupancy = pos.occupancy();

    context.checkers =
        (pawn_attacks(us, context.king_square)
            & pos.pieces[them_idx][static_cast<int>(PieceType::Pawn)])
        | (knight_attacks(context.king_square)
            & pos.pieces[them_idx][static_cast<int>(PieceType::Knight)])
        | (king_attacks(context.king_square)
            & pos.pieces[them_idx][static_cast<int>(PieceType::King)])
        | (bishop_attacks(context.king_square, occupancy)
            & (pos.pieces[them_idx][static_cast<int>(PieceType::Bishop)]
                | pos.pieces[them_idx][static_cast<int>(PieceType::Queen)]))
        | (rook_attacks(context.king_square, occupancy)
            & (pos.pieces[them_idx][static_cast<int>(PieceType::Rook)]
                | pos.pieces[them_idx][static_cast<int>(PieceType::Queen)]));

    if (context.checkers == EmptyBB) {
        context.block_mask = FullBB;
    } else if (popcount(context.checkers) == 1) {
        const Square checker = std::countr_zero(context.checkers);
        if (same_line(context.king_square, checker, checker)
            && is_slider_for_direction(
                pos,
                them,
                checker,
                file_of(context.king_square) == file_of(checker)
                    || rank_of(context.king_square) == rank_of(checker))) {
            context.block_mask = ray_between_exclusive(context.king_square, checker) | bit(checker);
        } else {
            context.block_mask = bit(checker);
        }
    } else {
        context.block_mask = EmptyBB;
    }

    const Bitboard our_pieces = pos.occupancy(us);
    const Bitboard their_pieces = pos.occupancy(them);
    for (const Direction direction : KingRayDirections) {
        Square blocker = NoSquare;
        int file = file_of(context.king_square) + direction.file_delta;
        int rank = rank_of(context.king_square) + direction.rank_delta;
        while (is_valid_square(file, rank)) {
            const Square square = make_square(file, rank);
            const Bitboard square_mask = bit(square);
            if (our_pieces & square_mask) {
                if (blocker != NoSquare) {
                    break;
                }
                blocker = square;
            } else if (their_pieces & square_mask) {
                if (blocker != NoSquare
                    && is_slider_for_direction(pos, them, square, direction.rook_like)) {
                    context.pinned |= bit(blocker);
                }
                break;
            }
            file += direction.file_delta;
            rank += direction.rank_delta;
        }
    }

    return context;
}

bool is_pseudo_move_legal(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Move move,
    PieceType moved_piece,
    PieceType captured_piece
) {
    if (move_flag(move) == MoveFlag::EnPassant) {
        Position next = pos;
        next.make_move(move, moved_piece, captured_piece);
        return !in_check(next, pos.side_to_move);
    }

    const Square from = from_square(move);
    const Square to = to_square(move);
    if (moved_piece == PieceType::King) {
        if (move_flag(move) == MoveFlag::KingCastle || move_flag(move) == MoveFlag::QueenCastle) {
            if (king_safety.checkers != EmptyBB) {
                return false;
            }
            const int direction = move_flag(move) == MoveFlag::KingCastle ? 1 : -1;
            const Square transit = make_square(file_of(from) + direction, rank_of(from));
            if (is_square_attacked(pos, transit, opposite(pos.side_to_move))) {
                return false;
            }
        }

        Bitboard occupancy = pos.occupancy();
        occupancy &= ~bit(from);
        Square excluded_attacker_square = NoSquare;
        if (is_capture(move)) {
            occupancy &= ~bit(to);
            excluded_attacker_square = to;
        }
        occupancy |= bit(to);
        return !is_square_attacked_with_occupancy(
            pos,
            to,
            opposite(pos.side_to_move),
            occupancy,
            excluded_attacker_square);
    }

    if (popcount(king_safety.checkers) >= 2) {
        return false;
    }
    if (king_safety.checkers != EmptyBB
        && (king_safety.block_mask & bit(to)) == EmptyBB) {
        return false;
    }
    if ((king_safety.pinned & bit(from)) != EmptyBB
        && !same_line(king_safety.king_square, from, to)) {
        return false;
    }

    return true;
}

bool gives_check_fast(
    const Position& pos,
    Move move,
    PieceType moved_piece,
    PieceType captured_piece
) {
    const MoveFlag flag = move_flag(move);
    if (flag == MoveFlag::EnPassant
        || flag == MoveFlag::KingCastle
        || flag == MoveFlag::QueenCastle) {
        Position next = pos;
        next.make_move(move, moved_piece, captured_piece);
        return in_check(next, next.side_to_move);
    }

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Square from = from_square(move);
    const Square to = to_square(move);
    const Square enemy_king = king_square(pos, them);
    const Bitboard occupancy = occupancy_after_move(pos, move);

    PieceType checking_piece = promotion_piece(move);
    if (checking_piece == PieceType::None) {
        checking_piece = moved_piece;
    }
    if (piece_attacks_square(checking_piece, us, to, enemy_king, occupancy)) {
        return true;
    }

    const int us_idx = static_cast<int>(us);
    const Bitboard from_mask = bit(from);
    const Bitboard bishops_or_queens =
        (pos.pieces[us_idx][static_cast<int>(PieceType::Bishop)]
            | pos.pieces[us_idx][static_cast<int>(PieceType::Queen)])
        & ~from_mask;
    const Bitboard rooks_or_queens =
        (pos.pieces[us_idx][static_cast<int>(PieceType::Rook)]
            | pos.pieces[us_idx][static_cast<int>(PieceType::Queen)])
        & ~from_mask;

    return (bishop_attacks(enemy_king, occupancy) & bishops_or_queens) != EmptyBB
        || (rook_attacks(enemy_king, occupancy) & rooks_or_queens) != EmptyBB;
}

} // namespace chess
