#include "evaluate.hpp"

#include "bitboard.hpp"

#include <array>

namespace chess {

namespace {

constexpr std::array<int, 6> PieceValue{
    100, 320, 330, 500, 900, 0
};

constexpr std::array<int, 64> PawnPst{
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
};

constexpr std::array<int, 64> KnightPst{
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
};

constexpr std::array<int, 64> BishopPst{
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
};

constexpr std::array<int, 64> RookPst{
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
};

constexpr std::array<int, 64> QueenPst{
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
};

constexpr std::array<int, 64> KingPst{
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
};

constexpr const std::array<int, 64>& piece_square_table(PieceType piece) {
    switch (piece) {
        case PieceType::Pawn:
            return PawnPst;
        case PieceType::Knight:
            return KnightPst;
        case PieceType::Bishop:
            return BishopPst;
        case PieceType::Rook:
            return RookPst;
        case PieceType::Queen:
            return QueenPst;
        case PieceType::King:
            return KingPst;
        default:
            return KingPst;
    }
}

int evaluate_piece_set(Bitboard pieces, PieceType piece, Color color) {
    int score = 0;
    const int piece_index = static_cast<int>(piece);
    const auto& pst = piece_square_table(piece);

    while (pieces) {
        const Square square = pop_lsb(pieces);
        const Square pst_square = relative_square(color, square);
        score += PieceValue[piece_index] + pst[pst_square];
    }

    return color == Color::White ? score : -score;
}

} // namespace

int evaluate(const Position& pos) {
    int score = 0;

    for (PieceType piece :
         {PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
          PieceType::Rook, PieceType::Queen, PieceType::King}) {
        score += evaluate_piece_set(
            pos.pieces[static_cast<int>(Color::White)][static_cast<int>(piece)],
            piece,
            Color::White);
        score += evaluate_piece_set(
            pos.pieces[static_cast<int>(Color::Black)][static_cast<int>(piece)],
            piece,
            Color::Black);
    }

    return score;
}

int evaluate_for_side_to_move(const Position& pos) {
    const int white_score = evaluate(pos);
    return pos.side_to_move == Color::White ? white_score : -white_score;
}

} // namespace chess
