#pragma once
#include "position.hpp"
#include "types.hpp"

namespace chess::zobrist {

enum class CastlingRight : int {
    WhiteKingside = 0,
    WhiteQueenside = 1,
    BlackKingside = 2,
    BlackQueenside = 3
};

HashKey piece_key(int colorIndex, int pieceIndex, int squareIndex);
HashKey side_key();
HashKey castling_key(int rightIndex);

HashKey piece_key(Color color, PieceType piece, Square square);
HashKey side_key();
HashKey castling_key(CastlingRight right);
HashKey en_passant_file_key(int file);

HashKey compute_hash(const Position& pos);
}
