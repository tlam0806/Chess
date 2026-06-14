#pragma once

#include "types.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include "position.hpp"

namespace chess {

enum class MoveFlag : std::uint8_t {
    Quiet = 0,
    Capture = 1,
    DoublePawnPush = 2,
    KingCastle = 3,
    QueenCastle = 4,
    EnPassant = 5,
    KnightPromotion = 6,
    BishopPromotion = 7,
    RookPromotion = 8,
    QueenPromotion = 9,
    KnightPromotionCapture = 10,
    BishopPromotionCapture = 11,
    RookPromotionCapture = 12,
    QueenPromotionCapture = 13
};

struct Move {
    std::uint16_t value = 0;

    constexpr Square from() const {
        return value & 0x3F;
    }

    constexpr Square to() const {
        return (value >> 6) & 0x3F;
    }

    constexpr MoveFlag flag() const {
        return static_cast<MoveFlag>((value >> 12) & 0x0F);
    }

    constexpr void set_from(Square from_square) {
        value = static_cast<std::uint16_t>((value & ~0x003F) | (from_square & 0x3F));
    }

    constexpr void set_to(Square to_square) {
        value = static_cast<std::uint16_t>((value & ~0x0FC0) | ((to_square & 0x3F) << 6));
    }

    constexpr void set_flag(MoveFlag move_flag) {
        value = static_cast<std::uint16_t>(
            (value & ~0xF000) | ((static_cast<int>(move_flag) & 0x0F) << 12));
    }

    constexpr PieceType get_moving_piece(const Position& pos) const {
        return pos.piece_type_on_occupied(from());
    }
};

constexpr bool operator==(Move lhs, Move rhs) {
    return lhs.value == rhs.value;
}

constexpr Move make_move(Square from, Square to, MoveFlag flag = MoveFlag::Quiet) {
    return Move{static_cast<std::uint16_t>(
        (from & 0x3F)
        | ((to & 0x3F) << 6)
        | ((static_cast<int>(flag) & 0x0F) << 12))};
}

constexpr bool can_en_passant(Square attacker, Square target) {
    return rank_of(attacker) == rank_of(target) && 
        abs(file_of(attacker) - file_of(target)) == 1;
}

constexpr Square from_square(Move move) {
    return move.from();
}

constexpr Square to_square(Move move) {
    return move.to();
}

constexpr MoveFlag move_flag(Move move) {
    return move.flag();
}

constexpr void set_from_square(Move& move, Square from) {
    move.set_from(from);
}

constexpr void set_to_square(Move& move, Square to) {
    move.set_to(to);
}

constexpr void set_move_flag(Move& move, MoveFlag flag) {
    move.set_flag(flag);
}

constexpr PieceType promotion_piece(Move move) {
    switch (move.flag()) {
        case MoveFlag::KnightPromotion:
        case MoveFlag::KnightPromotionCapture:
            return PieceType::Knight;
        case MoveFlag::BishopPromotion:
        case MoveFlag::BishopPromotionCapture:
            return PieceType::Bishop;
        case MoveFlag::RookPromotion:
        case MoveFlag::RookPromotionCapture:
            return PieceType::Rook;
        case MoveFlag::QueenPromotion:
        case MoveFlag::QueenPromotionCapture:
            return PieceType::Queen;
        default:
            return PieceType::None;
    }
}

constexpr bool is_capture(MoveFlag flag) {
    switch (flag) {
        case MoveFlag::Capture:
        case MoveFlag::EnPassant:
        case MoveFlag::KnightPromotionCapture:
        case MoveFlag::BishopPromotionCapture:
        case MoveFlag::RookPromotionCapture:
        case MoveFlag::QueenPromotionCapture:
            return true;
        default:
            return false;
    }
}

constexpr bool is_capture(Move move) {
    return is_capture(move.flag());
}

std::string square_to_string(Square square);
std::string move_to_string(Move move);

std::vector<Move> generate_knight_moves(const Position& pos);
std::vector<Move> generate_king_moves(const Position& pos);
std::vector<Move> generate_bishop_moves(const Position& pos);
std::vector<Move> generate_rook_moves(const Position& pos);
std::vector<Move> generate_queen_moves(const Position& pos);
std::vector<Move> generate_pawn_moves(const Position& pos);

std::vector<Move> generate_pseudo_legal_moves(const Position& pos);

std::vector<Move> generate_legal_moves(const Position& pos);

bool on_promotion_rank(Square);


} // namespace chess
