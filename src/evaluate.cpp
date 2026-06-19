#include "evaluate.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "evaluation_terms.hpp"
#include "move.hpp"
#include "position.hpp"

#include <array>
#include <bit>

namespace chess {

namespace {

int get_piece_value(PieceType type) {
    assert(type != PieceType::None);
    if (type == PieceType::King) {
        return 20'000;
    }
    return EvaluationPieceValue[static_cast<int>(type)];
}

constexpr int color_idx(Color color) {
    return static_cast<int>(color);
}

constexpr int piece_idx(PieceType piece) {
    return static_cast<int>(piece);
}

struct SeeState {
    std::array<Bitboard, 2> bishops{};
    std::array<Bitboard, 2> rooks{};
    std::array<Bitboard, 2> queens{};
    std::array<Bitboard, 2> pawn_attackers{};
    std::array<Bitboard, 2> knight_attackers{};
    std::array<Bitboard, 2> king_attackers{};
    Bitboard occupied = EmptyBB;
    Square target = NoSquare;
    bool target_is_promotion_rank = false;
};

PieceType promoted_see_piece(PieceType piece, bool target_is_promotion_rank) {
    return piece == PieceType::Pawn && target_is_promotion_rank
        ? PieceType::Queen
        : piece;
}

int see_piece_value(PieceType piece, bool target_is_promotion_rank) {
    return get_piece_value(promoted_see_piece(piece, target_is_promotion_rank));
}

void clear_see_piece(SeeState& state, Color color, PieceType piece, Square square) {
    const Bitboard mask = ~bit(square);
    const int side_index = color_idx(color);
    switch (piece) {
    case PieceType::Pawn:
        state.pawn_attackers[side_index] &= mask;
        break;
    case PieceType::Knight:
        state.knight_attackers[side_index] &= mask;
        break;
    case PieceType::Bishop:
        state.bishops[side_index] &= mask;
        break;
    case PieceType::Rook:
        state.rooks[side_index] &= mask;
        break;
    case PieceType::Queen:
        state.queens[side_index] &= mask;
        break;
    case PieceType::King:
        state.king_attackers[side_index] &= mask;
        break;
    case PieceType::None:
        break;
    }
}

SeeState make_see_state(const Position& pos, Square target, Bitboard occupancy) {
    SeeState state;
    state.occupied = occupancy;
    state.target = target;
    state.target_is_promotion_rank = on_promotion_rank(target);
    state.bishops[color_idx(Color::White)] = pos.pieces[color_idx(Color::White)][piece_idx(PieceType::Bishop)];
    state.bishops[color_idx(Color::Black)] = pos.pieces[color_idx(Color::Black)][piece_idx(PieceType::Bishop)];
    state.rooks[color_idx(Color::White)] = pos.pieces[color_idx(Color::White)][piece_idx(PieceType::Rook)];
    state.rooks[color_idx(Color::Black)] = pos.pieces[color_idx(Color::Black)][piece_idx(PieceType::Rook)];
    state.queens[color_idx(Color::White)] = pos.pieces[color_idx(Color::White)][piece_idx(PieceType::Queen)];
    state.queens[color_idx(Color::Black)] = pos.pieces[color_idx(Color::Black)][piece_idx(PieceType::Queen)];

    const Bitboard white_pawn_sources = pawn_attacks(Color::Black, target);
    const Bitboard black_pawn_sources = pawn_attacks(Color::White, target);
    state.pawn_attackers[color_idx(Color::White)] =
        white_pawn_sources & pos.pieces[color_idx(Color::White)][piece_idx(PieceType::Pawn)];
    state.pawn_attackers[color_idx(Color::Black)] =
        black_pawn_sources & pos.pieces[color_idx(Color::Black)][piece_idx(PieceType::Pawn)];

    const Bitboard knight_sources = knight_attacks(target);
    state.knight_attackers[color_idx(Color::White)] =
        knight_sources & pos.pieces[color_idx(Color::White)][piece_idx(PieceType::Knight)];
    state.knight_attackers[color_idx(Color::Black)] =
        knight_sources & pos.pieces[color_idx(Color::Black)][piece_idx(PieceType::Knight)];

    const Bitboard king_sources = king_attacks(target);
    state.king_attackers[color_idx(Color::White)] =
        king_sources & pos.pieces[color_idx(Color::White)][piece_idx(PieceType::King)];
    state.king_attackers[color_idx(Color::Black)] =
        king_sources & pos.pieces[color_idx(Color::Black)][piece_idx(PieceType::King)];

    return state;
}

SeeAttacker find_least_valuable_pseudo_attacker(const SeeState& state, int side_index) {
    Bitboard attackers = EmptyBB;

    if (!state.target_is_promotion_rank) {
        attackers = state.pawn_attackers[side_index];
        if (attackers != EmptyBB) {
            return SeeAttacker{static_cast<Square>(std::countr_zero(attackers)), PieceType::Pawn};
        }
    }

    attackers = state.knight_attackers[side_index];
    if (attackers != EmptyBB) {
        return SeeAttacker{static_cast<Square>(std::countr_zero(attackers)), PieceType::Knight};
    }

    Bitboard bishop_like = EmptyBB;
    const Bitboard bishop_like_pieces = state.bishops[side_index] | state.queens[side_index];
    if (bishop_like_pieces != EmptyBB) {
        bishop_like = bishop_attacks(state.target, state.occupied);
        attackers = bishop_like & state.bishops[side_index];
        if (attackers != EmptyBB) {
            return SeeAttacker{static_cast<Square>(std::countr_zero(attackers)), PieceType::Bishop};
        }
    }

    Bitboard rook_like = EmptyBB;
    const Bitboard rook_like_pieces = state.rooks[side_index] | state.queens[side_index];
    if (rook_like_pieces != EmptyBB) {
        rook_like = rook_attacks(state.target, state.occupied);
        attackers = rook_like & state.rooks[side_index];
        if (attackers != EmptyBB) {
            return SeeAttacker{static_cast<Square>(std::countr_zero(attackers)), PieceType::Rook};
        }
    }

    if (state.target_is_promotion_rank) {
        attackers = state.pawn_attackers[side_index];
        if (attackers != EmptyBB) {
            return SeeAttacker{static_cast<Square>(std::countr_zero(attackers)), PieceType::Pawn};
        }
    }

    attackers = (bishop_like | rook_like) & state.queens[side_index];
    if (attackers != EmptyBB) {
        return SeeAttacker{static_cast<Square>(std::countr_zero(attackers)), PieceType::Queen};
    }

    attackers = state.king_attackers[side_index];
    if (attackers != EmptyBB) {
        return SeeAttacker{static_cast<Square>(std::countr_zero(attackers)), PieceType::King};
    }

    return SeeAttacker{};
}

void apply_see_recapture(
    SeeState& state,
    Color side,
    PieceType attacker_piece,
    Square from
) {
    clear_see_piece(state, side, attacker_piece, from);
    state.occupied &= ~bit(from);
    state.occupied |= bit(state.target);
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
    return pos.eval_score;
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

    SeeState state = make_see_state(pos, move.to(), pos.occupancy());
    Color side = pos.side_to_move;
    PieceType last_attacker = promoted_see_piece(moving_piece, state.target_is_promotion_rank);
    const PieceType promoted_piece = promotion_piece(move);
    int last_attacker_value = promoted_piece != PieceType::None
        ? get_piece_value(promoted_piece)
        : get_piece_value(last_attacker);
    int see = get_piece_value(captured_piece);
    if (last_attacker == PieceType::King) {
        return see;
    }
    clear_see_piece(state, side, moving_piece, move.from());
    clear_see_piece(state, opposite(side), captured_piece, move.to());
    state.occupied &= ~bit(move.from());
    state.occupied &= ~bit(move.to());
    state.occupied |= bit(move.to());
    side = opposite(side);
    int side_index = color_idx(side);
    int sign = -1;

    while (true) {
        assert(last_attacker != PieceType::King);
        const SeeAttacker attacker = find_least_valuable_pseudo_attacker(state, side_index);
        if (attacker.piece == PieceType::None) {
            return see;
        }

        const int attacker_value = see_piece_value(attacker.piece, state.target_is_promotion_rank);

        if (attacker_value > last_attacker_value) {
            apply_see_recapture(
                state,
                side,
                attacker.piece,
                attacker.square);
            if (find_least_valuable_pseudo_attacker(state, side_index ^ 1).piece != PieceType::None) {
                return see;
            }
            return see + last_attacker_value * sign;
        }
        see += last_attacker_value * sign;

        last_attacker_value = attacker_value;
        apply_see_recapture(state, side, attacker.piece, attacker.square);
        last_attacker = promoted_see_piece(attacker.piece, state.target_is_promotion_rank);
        side = opposite(side);
        side_index ^= 1;
        sign = -sign;
    }
}

int static_exchange_eval(const Position& pos, Move move) {
    const PieceType moving_piece = pos.piece_type_on_occupied(pos.side_to_move, move.from());
    const PieceType captured_piece = move.flag() == MoveFlag::EnPassant
        ? PieceType::Pawn
        : pos.piece_type_on_occupied(opposite(pos.side_to_move), move.to());
    return static_exchange_eval(pos, move, moving_piece, captured_piece);
}

} // namespace chess
