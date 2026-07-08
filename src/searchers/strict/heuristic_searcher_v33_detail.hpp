#pragma once

#include "heuristic_searcher_v33.hpp"

#include "attacks.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>

namespace chess {

namespace v33_detail {

using Clock = std::chrono::steady_clock;


constexpr int MaxQuiescenceDepth = 16;
constexpr int MaxCheckEvasionQuiescenceDepth = 16;
constexpr int MateScoreThreshold = CheckmateScore - 1024;

constexpr bool EnableAspirationWindow = true;
constexpr bool EnableTt = true;
constexpr bool EnableTtExactStore = true;

inline bool is_valid_move(Move move) {
    return move.value != 0;
}

inline bool is_promotion(Move move) {
    return promotion_piece(move) != PieceType::None;
}

inline int ordering_piece_value(PieceType piece) {
    switch (piece) {
    case PieceType::Pawn:
        return 100;
    case PieceType::Knight:
        return 320;
    case PieceType::Bishop:
        return 330;
    case PieceType::Rook:
        return 500;
    case PieceType::Queen:
        return 900;
    default:
        return 0;
    }
}

inline ScoreRange exact_range(int score) {
    return ScoreRange{score, score};
}

inline ScoreRange lower_range(int score) {
    return ScoreRange{score, Infinity};
}

inline ScoreRange negate_range(ScoreRange range) {
    return ScoreRange{-range.upper, -range.lower};
}

inline ScoreRange intersect_ranges(ScoreRange lhs, ScoreRange rhs) {
    return ScoreRange{
        std::max(lhs.lower, rhs.lower),
        std::min(lhs.upper, rhs.upper)
    };
}

inline bool is_exact_range(ScoreRange range) {
    return range.lower == range.upper;
}

inline int representative_score(ScoreRange range) {
    return range.lower != -Infinity ? range.lower : range.upper;
}

inline Move preferred_tt_move(MoveRange moves) {
    return moves.lower.value != 0 ? moves.lower : moves.upper;
}

inline PieceType piece_type_on_square_for_color(const Position& pos, Color color, Square square) {
    if ((pos.occupancy(color) & bit(square)) == EmptyBB) {
        return PieceType::None;
    }
    return pos.piece_type_on_occupied(color, square);
}

inline bool is_pseudo_move_shape_valid(const Position& pos, Move move, PieceType moved_piece) {
    if (!is_valid_move(move) || moved_piece == PieceType::None) {
        return false;
    }

    const Square from = from_square(move);
    const Square to = to_square(move);
    if (from == to) {
        return false;
    }

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Bitboard to_mask = bit(to);
    const Bitboard occupancy = pos.occupancy();
    const bool target_is_ours = (pos.occupancy(us) & to_mask) != EmptyBB;
    const bool target_is_theirs = (pos.occupancy(them) & to_mask) != EmptyBB;
    const bool target_is_empty = !target_is_ours && !target_is_theirs;

    if (target_is_ours) {
        return false;
    }

    const MoveFlag flag = move_flag(move);
    const bool capture = is_capture(move);
    const bool promotion = promotion_piece(move) != PieceType::None;
    if (promotion && moved_piece != PieceType::Pawn) {
        return false;
    }

    if (moved_piece != PieceType::Pawn
        && flag != MoveFlag::Quiet
        && flag != MoveFlag::Capture
        && flag != MoveFlag::KingCastle
        && flag != MoveFlag::QueenCastle) {
        return false;
    }

    if (capture) {
        if (flag == MoveFlag::EnPassant) {
            if (moved_piece != PieceType::Pawn || to != pos.en_passant_square) {
                return false;
            }
        } else if (!target_is_theirs) {
            return false;
        }
    } else if (!target_is_empty) {
        return false;
    }
    switch (moved_piece) {
    case PieceType::Knight:
        return (knight_attacks(from) & to_mask) != EmptyBB;
    case PieceType::Bishop:
        return (bishop_attacks(from, occupancy) & to_mask) != EmptyBB;
    case PieceType::Rook:
        return (rook_attacks(from, occupancy) & to_mask) != EmptyBB;
    case PieceType::Queen:
        return (queen_attacks(from, occupancy) & to_mask) != EmptyBB;
    case PieceType::King:
        if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
            return false;
        }
        return (king_attacks(from) & to_mask) != EmptyBB;
    case PieceType::Pawn: {
        const int direction = us == Color::White ? 1 : -1;
        const int from_rank = rank_of(from);
        const int to_rank = rank_of(to);
        const int file_delta = file_of(to) - file_of(from);
        const int rank_delta = to_rank - from_rank;
        const int promotion_rank = us == Color::White
            ? static_cast<int>(Rank::R8)
            : static_cast<int>(Rank::R1);

        if (promotion != (to_rank == promotion_rank)) {
            return false;
        }
        if (capture) {
            return rank_delta == direction && (file_delta == -1 || file_delta == 1);
        }
        if (file_delta != 0) {
            return false;
        }
        if (rank_delta == direction) {
            return true;
        }
        if (flag != MoveFlag::DoublePawnPush || rank_delta != 2 * direction) {
            return false;
        }
        const int start_rank = us == Color::White
            ? static_cast<int>(Rank::R2)
            : static_cast<int>(Rank::R7);
        const Square middle = make_square(file_of(from), from_rank + direction);
        return from_rank == start_rank && pos.is_empty(middle);
    }
    case PieceType::None:
        return false;
    }

    return false;
}

inline bool has_rook_on(const Position& pos, Color color, Square square) {
    return (pos.pieces[static_cast<int>(color)][static_cast<int>(PieceType::Rook)] & bit(square)) != EmptyBB;
}

inline bool castle_shape_valid(const Position& pos, Move move, PieceType moved_piece) {
    if (moved_piece != PieceType::King) {
        return false;
    }
    const Color us = pos.side_to_move;
    const Square from = from_square(move);
    const Square to = to_square(move);
    const MoveFlag flag = move_flag(move);
    const Bitboard occupancy = pos.occupancy();

    if (us == Color::White) {
        const Square e1 = make_square(4, 0);
        if (from != e1) {
            return false;
        }
        if (flag == MoveFlag::KingCastle) {
            const Square f1 = make_square(5, 0);
            const Square g1 = make_square(6, 0);
            const Square h1 = make_square(7, 0);
            return to == g1
                && pos.white_can_castle_kingside
                && has_rook_on(pos, us, h1)
                && (occupancy & (bit(f1) | bit(g1))) == EmptyBB;
        }
        if (flag == MoveFlag::QueenCastle) {
            const Square d1 = make_square(3, 0);
            const Square c1 = make_square(2, 0);
            const Square b1 = make_square(1, 0);
            const Square a1 = make_square(0, 0);
            return to == c1
                && pos.white_can_castle_queenside
                && has_rook_on(pos, us, a1)
                && (occupancy & (bit(d1) | bit(c1) | bit(b1))) == EmptyBB;
        }
        return false;
    }

    const Square e8 = make_square(4, 7);
    if (from != e8) {
        return false;
    }
    if (flag == MoveFlag::KingCastle) {
        const Square f8 = make_square(5, 7);
        const Square g8 = make_square(6, 7);
        const Square h8 = make_square(7, 7);
        return to == g8
            && pos.black_can_castle_kingside
            && has_rook_on(pos, us, h8)
            && (occupancy & (bit(f8) | bit(g8))) == EmptyBB;
    }
    if (flag == MoveFlag::QueenCastle) {
        const Square d8 = make_square(3, 7);
        const Square c8 = make_square(2, 7);
        const Square b8 = make_square(1, 7);
        const Square a8 = make_square(0, 7);
        return to == c8
            && pos.black_can_castle_queenside
            && has_rook_on(pos, us, a8)
            && (occupancy & (bit(d8) | bit(c8) | bit(b8))) == EmptyBB;
    }
    return false;
}

inline bool is_move_legal_by_attack_check(
    const Position& pos,
    Move move,
    PieceType moved_piece
) {
    const MoveFlag flag = move_flag(move);
    if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        if (!castle_shape_valid(pos, move, moved_piece)) {
            return false;
        }
    } else if (!is_pseudo_move_shape_valid(pos, move, moved_piece)) {
        return false;
    }

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Square from = from_square(move);
    const Square to = to_square(move);
    Bitboard occupancy = pos.occupancy();
    occupancy &= ~bit(from);

    Square excluded_attacker_square = NoSquare;
    if (flag == MoveFlag::EnPassant) {
        excluded_attacker_square = us == Color::White ? to - 8 : to + 8;
        occupancy &= ~bit(excluded_attacker_square);
    } else if (is_capture(move)) {
        excluded_attacker_square = to;
        occupancy &= ~bit(to);
    }

    if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        const int direction = flag == MoveFlag::KingCastle ? 1 : -1;
        const Square transit = make_square(file_of(from) + direction, rank_of(from));
        if (is_square_attacked(pos, from, them)
            || is_square_attacked(pos, transit, them)) {
            return false;
        }
        const Square rook_from = flag == MoveFlag::KingCastle
            ? make_square(7, rank_of(from))
            : make_square(0, rank_of(from));
        const Square rook_to = flag == MoveFlag::KingCastle
            ? make_square(5, rank_of(from))
            : make_square(3, rank_of(from));
        occupancy &= ~bit(rook_from);
        occupancy |= bit(rook_to);
    }

    occupancy |= bit(to);
    const Square king = moved_piece == PieceType::King ? to : king_square(pos, us);
    return !is_square_attacked(pos, king, them, occupancy, excluded_attacker_square);
}

inline bool is_quiet_non_promotion_move_legal_by_attack_check(
    const Position& pos,
    Move move,
    PieceType moved_piece
) {
    const MoveFlag flag = move_flag(move);
    if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        return is_move_legal_by_attack_check(pos, move, moved_piece);
    }
    if (!is_valid_move(move)
        || moved_piece == PieceType::None
        || is_capture(move)
        || is_promotion(move)) {
        return false;
    }

    const Square from = from_square(move);
    const Square to = to_square(move);
    if (from == to) {
        return false;
    }

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Bitboard from_mask = bit(from);
    const Bitboard to_mask = bit(to);
    const Bitboard our_pieces = pos.occupancy(us);
    const Bitboard occupancy = pos.occupancy();

    if ((our_pieces & from_mask) == EmptyBB
        || (occupancy & to_mask) != EmptyBB) {
        return false;
    }

    bool shape_valid = false;
    switch (moved_piece) {
    case PieceType::Knight:
        shape_valid = flag == MoveFlag::Quiet && (knight_attacks(from) & to_mask) != EmptyBB;
        break;
    case PieceType::Bishop:
        shape_valid = flag == MoveFlag::Quiet && (bishop_attacks(from, occupancy) & to_mask) != EmptyBB;
        break;
    case PieceType::Rook:
        shape_valid = flag == MoveFlag::Quiet && (rook_attacks(from, occupancy) & to_mask) != EmptyBB;
        break;
    case PieceType::Queen:
        shape_valid = flag == MoveFlag::Quiet && (queen_attacks(from, occupancy) & to_mask) != EmptyBB;
        break;
    case PieceType::King:
        shape_valid = flag == MoveFlag::Quiet && (king_attacks(from) & to_mask) != EmptyBB;
        break;
    case PieceType::Pawn: {
        const int direction = us == Color::White ? 1 : -1;
        const int from_rank = rank_of(from);
        const int to_rank = rank_of(to);
        const int file_delta = file_of(to) - file_of(from);
        const int rank_delta = to_rank - from_rank;
        const int promotion_rank = us == Color::White
            ? static_cast<int>(Rank::R8)
            : static_cast<int>(Rank::R1);
        if (to_rank == promotion_rank || file_delta != 0) {
            return false;
        }
        if (rank_delta == direction) {
            shape_valid = true;
        } else if (flag == MoveFlag::DoublePawnPush && rank_delta == 2 * direction) {
            const int start_rank = us == Color::White
                ? static_cast<int>(Rank::R2)
                : static_cast<int>(Rank::R7);
            const Square middle = make_square(file_of(from), from_rank + direction);
            shape_valid = from_rank == start_rank && pos.is_empty(middle);
        }
        break;
    }
    case PieceType::None:
        return false;
    }

    if (!shape_valid) {
        return false;
    }

    const Bitboard occupancy_after = (occupancy & ~from_mask) | to_mask;
    const Square king = moved_piece == PieceType::King ? to : king_square(pos, us);
    return !is_square_attacked(pos, king, them, occupancy_after);
}

inline bool matches_any(Move move, Move a, Move b = Move{}, Move c = Move{}, Move d = Move{}) {
    return (is_valid_move(a) && move == a)
        || (is_valid_move(b) && move == b)
        || (is_valid_move(c) && move == c)
        || (is_valid_move(d) && move == d);
}


} // namespace v33_detail

using namespace v33_detail;

} // namespace chess
