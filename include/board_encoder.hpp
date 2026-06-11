#pragma once

#include "position.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace chess {

using FeatureIndex = std::uint32_t;

constexpr int EncoderPieceTypes = 6;
constexpr int EncoderPieceSides = 2;
constexpr int EncoderKingContexts = 2;
constexpr int EncoderSquares = 64;
constexpr int EncodedFeatureCount =
    EncoderPieceTypes * EncoderPieceSides * EncoderKingContexts * EncoderSquares * EncoderSquares;

enum class EncodedPieceSide : int {
    Friendly = 0,
    Enemy = 1
};

enum class EncodedKingContext : int {
    FriendlyKing = 0,
    EnemyKing = 1
};

enum AuxFeature : int {
    FriendlyCanCastleKingside = 0,
    FriendlyCanCastleQueenside = 1,
    EnemyCanCastleKingside = 2,
    EnemyCanCastleQueenside = 3,
    HasEnPassant = 4,
    EnPassantFileA = 5,
    AuxFeatureCount = 13
};

struct EncodedPosition {
    std::vector<FeatureIndex> features;
    std::array<std::uint8_t, AuxFeatureCount> aux{};
};

constexpr FeatureIndex feature_index(
    PieceType piece,
    EncodedPieceSide piece_side,
    EncodedKingContext king_context,
    Square king_square,
    Square piece_square
) {
    FeatureIndex index = static_cast<FeatureIndex>(piece);
    index = index * EncoderPieceSides + static_cast<FeatureIndex>(piece_side);
    index = index * EncoderKingContexts + static_cast<FeatureIndex>(king_context);
    index = index * EncoderSquares + static_cast<FeatureIndex>(king_square);
    index = index * EncoderSquares + static_cast<FeatureIndex>(piece_square);
    return index;
}

EncodedPosition encode_position(const Position& pos);

} // namespace chess
