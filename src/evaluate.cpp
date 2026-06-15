#include "evaluate.hpp"

#include "bitboard.hpp"
#include "move.hpp"
#include "attacks.hpp"
#include "position.hpp"

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

int get_piece_value(PieceType type) {
    assert(type != PieceType::None);
    return PieceValue[static_cast<int>(type)];
}

void apply_see_capture(Position& pos, Square from, Square to) {
    const Color color = pos.color_on_occupied(from);
    assert(pos.side_to_move == color);
    PieceType moving_piece = pos.piece_type_on_occupied(from);

    pos.clear_square(from);
    pos.clear_square(to);

    pos.set_piece(color, moving_piece, to);

    pos.side_to_move = opposite(pos.side_to_move);
}

} // namespace

bool is_pinned(Position pos, Square prev, Square next) {
    const Color side = pos.side_to_move;

    pos.clear_square(prev);
    if (!pos.is_empty(next)) {
        pos.clear_square(next);
    }
    pos.set_piece(side, PieceType::Pawn, next);

    return in_check(pos, side);
}

bool king_capture_legal(const Position& pos, Square square) {
    Position copied_pos = pos;
    const Color side = pos.side_to_move;
    const Square king_from = king_square(copied_pos, side);

    if (king_from != NoSquare) {
        copied_pos.clear_square(king_from);
    }
    if (!copied_pos.is_empty(square)) {
        copied_pos.clear_square(square);
    }
    copied_pos.set_piece(side, PieceType::King, square);

    return !is_square_attacked(copied_pos, square, opposite(side));
}

SeeAttacker find_least_valuable_attacker(const Position& pos, Square square) {
    const Color side = pos.side_to_move;
    const Bitboard occupancy = pos.occupancy();

    auto find_unpinned = [&](Bitboard attackers, PieceType piece) -> SeeAttacker {
        while (attackers) {
            const Square from = pop_lsb(attackers);
            if (!is_pinned(pos, from, square)) {
                return SeeAttacker{from, piece};
            }
        }
        return SeeAttacker{};
    };

    SeeAttacker attacker;

    if (!on_promotion_rank(square)) {
        attacker = find_unpinned(
            pos.occupancy(side, PieceType::Pawn) & pawn_attacks(opposite(side), square),
            PieceType::Pawn);
        if (attacker.piece != PieceType::None) {
            return attacker;
        }
    }

    attacker = find_unpinned(
        pos.occupancy(side, PieceType::Knight) & knight_attacks(square),
        PieceType::Knight);
    if (attacker.piece != PieceType::None) {
        return attacker;
    }

    attacker = find_unpinned(
        pos.occupancy(side, PieceType::Bishop) & bishop_attacks(square, occupancy),
        PieceType::Bishop);
    if (attacker.piece != PieceType::None) {
        return attacker;
    }

    attacker = find_unpinned(
        pos.occupancy(side, PieceType::Rook) & rook_attacks(square, occupancy),
        PieceType::Rook);
    if (attacker.piece != PieceType::None) {
        return attacker;
    }

    if (on_promotion_rank(square)) {
        attacker = find_unpinned(
            pos.occupancy(side, PieceType::Pawn) & pawn_attacks(opposite(side), square),
            PieceType::Pawn);
        if (attacker.piece != PieceType::None) {
            return attacker;
        }
    }

    attacker = find_unpinned(
        pos.occupancy(side, PieceType::Queen) & queen_attacks(square, occupancy),
        PieceType::Queen);
    if (attacker.piece != PieceType::None) {
        return attacker;
    }

    Bitboard king_attackers = pos.occupancy(side, PieceType::King) & king_attacks(square);
    while (king_attackers) {
        const Square from = pop_lsb(king_attackers);
        (void)from;
        if (king_capture_legal(pos, square)) {
            return SeeAttacker{king_square(pos, side), PieceType::King};
        }
    }

    return SeeAttacker{};
}


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


int static_exchange_eval(
    const Position& pos,
    Move move,
    PieceType moving_piece,
    PieceType captured_piece
) {
    if (move.flag() == MoveFlag::EnPassant) {
        return 0;
    }
    assert(moving_piece != PieceType::None);
    assert(!is_capture(move) || captured_piece != PieceType::None);
    Position copied_pos = pos;
    PieceType last_attacker = moving_piece;
    auto tmp = promotion_piece(move);
    int last_attacker_value = tmp != PieceType::None ? get_piece_value(tmp) : get_piece_value(last_attacker);
    int see = get_piece_value(captured_piece);
    copied_pos.make_move(move, moving_piece, captured_piece);
    int sign = -1;
    bool promotion = on_promotion_rank(move.to());
    while (true) {
        auto attacker = find_least_valuable_attacker(copied_pos, move.to());
        if (attacker.piece == PieceType::None) return see;
        int attacker_value = get_piece_value(attacker.piece);
        if (promotion && attacker.piece == PieceType::Pawn) {
            attacker_value = get_piece_value(PieceType::Queen);
        }

        if (attacker_value > last_attacker_value) return see;
        assert(last_attacker != PieceType::King);

        see += last_attacker_value * sign;

        last_attacker = attacker.piece;
        last_attacker_value = attacker_value;
        apply_see_capture(copied_pos, attacker.square, move.to());
        sign = -sign;
    }

    return see;
}

int static_exchange_eval(const Position& pos, Move move) {
    const PieceType moving_piece = pos.piece_type_on_occupied(move.from());
    const PieceType captured_piece = move.flag() == MoveFlag::EnPassant
        ? PieceType::Pawn
        : pos.piece_type_on_occupied(move.to());
    return static_exchange_eval(pos, move, moving_piece, captured_piece);
}

} // namespace chess
