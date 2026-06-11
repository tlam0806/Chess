#include "board_encoder.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"

#include <algorithm>

namespace chess {

namespace {

constexpr std::array<PieceType, EncoderPieceTypes> EncodedPieces{
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King
};

bool can_castle_kingside(const Position& pos, Color color) {
    return color == Color::White ? pos.white_can_castle_kingside : pos.black_can_castle_kingside;
}

bool can_castle_queenside(const Position& pos, Color color) {
    return color == Color::White ? pos.white_can_castle_queenside : pos.black_can_castle_queenside;
}

void encode_piece_side(
    const Position& pos,
    Color board_color,
    EncodedPieceSide encoded_side,
    Color perspective,
    Square friendly_king_square,
    Square enemy_king_square,
    std::vector<FeatureIndex>& features
) {
    for (PieceType piece : EncodedPieces) {
        Bitboard bb = pos.pieces[static_cast<int>(board_color)][static_cast<int>(piece)];

        while (bb) {
            const Square board_square = pop_lsb(bb);
            const Square piece_square = relative_square(perspective, board_square);

            features.push_back(feature_index(
                piece,
                encoded_side,
                EncodedKingContext::FriendlyKing,
                friendly_king_square,
                piece_square));
            features.push_back(feature_index(
                piece,
                encoded_side,
                EncodedKingContext::EnemyKing,
                enemy_king_square,
                piece_square));
        }
    }
}

} // namespace

EncodedPosition encode_position(const Position& pos) {
    EncodedPosition encoded;
    encoded.features.reserve(64);

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Square friendly_king_square = relative_square(us, king_square(pos, us));
    const Square enemy_king_square = relative_square(us, king_square(pos, them));

    encode_piece_side(
        pos,
        us,
        EncodedPieceSide::Friendly,
        us,
        friendly_king_square,
        enemy_king_square,
        encoded.features);
    encode_piece_side(
        pos,
        them,
        EncodedPieceSide::Enemy,
        us,
        friendly_king_square,
        enemy_king_square,
        encoded.features);

    std::sort(encoded.features.begin(), encoded.features.end());

    encoded.aux[FriendlyCanCastleKingside] = can_castle_kingside(pos, us);
    encoded.aux[FriendlyCanCastleQueenside] = can_castle_queenside(pos, us);
    encoded.aux[EnemyCanCastleKingside] = can_castle_kingside(pos, them);
    encoded.aux[EnemyCanCastleQueenside] = can_castle_queenside(pos, them);

    if (pos.en_passant_square != NoSquare) {
        const Square ep_square = relative_square(us, pos.en_passant_square);
        encoded.aux[HasEnPassant] = 1;
        encoded.aux[EnPassantFileA + file_of(ep_square)] = 1;
    }

    return encoded;
}

} // namespace chess
