#include "zobrist.hpp"

#include "bitboard.hpp"

#include <array>
#include <cassert>

namespace {

constexpr chess::HashKey splitmix64(chess::HashKey& seed) {
    seed += 0x9E3779B97F4A7C15ULL;
    chess::HashKey z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Keys {
    std::array<std::array<std::array<chess::HashKey, 64>, 6>, 2> piece{};
    chess::HashKey side{};
    std::array<chess::HashKey, 4> castling{};
    std::array<chess::HashKey, 8> en_passant_file{};
};

consteval Keys make_keys() {
    Keys keys{};
    chess::HashKey seed = 0x123456789ABCDEF0ULL;
    for (auto& by_color : keys.piece) { // color
        for (auto& by_piece : by_color) { // piece
            for (auto& by_square : by_piece) { // square
                by_square = splitmix64(seed);
            }
        }
    }
    keys.side = splitmix64(seed);
    for (auto& key : keys.en_passant_file) {
        key = splitmix64(seed);
    }
    for (auto& key : keys.castling) {
        key = splitmix64(seed);
    }
    return keys;
}

constexpr auto keys_table = make_keys();

}

namespace chess::zobrist {

HashKey piece_key(int colorIndex, int pieceIndex, int squareIndex) {
    return keys_table.piece[colorIndex][pieceIndex][squareIndex];
}

HashKey piece_key(Color color, PieceType piece, Square square) {
    return keys_table.piece[static_cast<int>(color)][static_cast<int>(piece)][static_cast<int>(square)];
}

HashKey side_key() {
    return keys_table.side;
}

HashKey castling_key(int rightIndex) {
    return keys_table.castling[rightIndex];
}

HashKey castling_key(CastlingRight right) {
    return keys_table.castling[static_cast<int>(right)];
}

HashKey en_passant_file_key(int file) {
    assert(0 <= file && file < 8);
    return keys_table.en_passant_file[file];
}

HashKey compute_hash(const Position& pos) {
    HashKey hash = 0;

    for (Color color : {Color::White, Color::Black}) {
        for (PieceType piece :
             {PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
              PieceType::Rook, PieceType::Queen, PieceType::King}) {
            Bitboard pieces = pos.pieces[static_cast<int>(color)][static_cast<int>(piece)];
            while (pieces) {
                const Square square = pop_lsb(pieces);
                hash ^= piece_key(color, piece, square);
            }
        }
    }

    if (pos.side_to_move == Color::Black) {
        hash ^= side_key();
    }

    if (pos.white_can_castle_kingside) {
        hash ^= castling_key(CastlingRight::WhiteKingside);
    }
    if (pos.white_can_castle_queenside) {
        hash ^= castling_key(CastlingRight::WhiteQueenside);
    }
    if (pos.black_can_castle_kingside) {
        hash ^= castling_key(CastlingRight::BlackKingside);
    }
    if (pos.black_can_castle_queenside) {
        hash ^= castling_key(CastlingRight::BlackQueenside);
    }

    if (pos.en_passant_square != NoSquare) {
        hash ^= en_passant_file_key(file_of(pos.en_passant_square));
    }

    return hash;
}
}
