#include "move.hpp"
#include "attacks.hpp"
#include "evaluation_terms.hpp"
#include "king_safety.hpp"
#include "zobrist.hpp"

#include <cassert>
#include <cstdlib>
#include <string>

namespace chess {

namespace {

bool has_rook_on(const Position& pos, Color color, Square square) {
    const Bitboard rooks = pos.pieces[static_cast<int>(color)][static_cast<int>(PieceType::Rook)];
    return (rooks & bit(square)) != EmptyBB;
}

std::uint8_t encode_piece_for_move(Color color, PieceType piece) {
    assert(piece != PieceType::None);
    return static_cast<std::uint8_t>(1 + static_cast<int>(color) * 6 + static_cast<int>(piece));
}

std::uint8_t pack_castling_rights(const Position& pos) {
    return static_cast<std::uint8_t>(
        (pos.white_can_castle_kingside ? 1 : 0)
        | (pos.white_can_castle_queenside ? 2 : 0)
        | (pos.black_can_castle_kingside ? 4 : 0)
        | (pos.black_can_castle_queenside ? 8 : 0));
}

void restore_castling_rights(Position& pos, std::uint8_t rights) {
    pos.white_can_castle_kingside = (rights & 1) != 0;
    pos.white_can_castle_queenside = (rights & 2) != 0;
    pos.black_can_castle_kingside = (rights & 4) != 0;
    pos.black_can_castle_queenside = (rights & 8) != 0;
}

void move_piece_fast(Position& pos, Color color, PieceType piece, Square from, Square to) {
    assert(piece != PieceType::None);
    const std::uint8_t encoded = encode_piece_for_move(color, piece);
    assert(pos.board[from] == encoded);
    assert(pos.board[to] == 0);

    const int color_idx = static_cast<int>(color);
    const int piece_idx = static_cast<int>(piece);
    Bitboard& bb = pos.pieces[color_idx][piece_idx];
    const Bitboard from_mask = bit(from);
    const Bitboard to_mask = bit(to);
    assert((bb & from_mask) != EmptyBB);

    bb ^= from_mask | to_mask;
    pos.occupancies[color_idx] ^= from_mask | to_mask;
    pos.board[from] = 0;
    pos.board[to] = encoded;
    if (piece == PieceType::King) {
        pos.king_squares[color_idx] = to;
    }
    pos.eval_score += evaluation_piece_contribution(color, piece, to)
        - evaluation_piece_contribution(color, piece, from);
    pos.zobrist_key ^= zobrist::piece_key(color, piece, from);
    pos.zobrist_key ^= zobrist::piece_key(color, piece, to);
}

void capture_piece_fast(
    Position& pos,
    Color color,
    PieceType moved_piece,
    Color enemy,
    PieceType captured_piece,
    Square from,
    Square to
) {
    assert(moved_piece != PieceType::None);
    assert(captured_piece != PieceType::None);
    const std::uint8_t moved_encoded = encode_piece_for_move(color, moved_piece);
    [[maybe_unused]] const std::uint8_t captured_encoded = encode_piece_for_move(enemy, captured_piece);
    assert(pos.board[from] == moved_encoded);
    assert(pos.board[to] == captured_encoded);

    const Bitboard from_mask = bit(from);
    const Bitboard to_mask = bit(to);
    const int color_idx = static_cast<int>(color);
    const int enemy_idx = static_cast<int>(enemy);
    Bitboard& moved_bb = pos.pieces[color_idx][static_cast<int>(moved_piece)];
    Bitboard& captured_bb = pos.pieces[enemy_idx][static_cast<int>(captured_piece)];
    assert((moved_bb & from_mask) != EmptyBB);
    assert((captured_bb & to_mask) != EmptyBB);

    moved_bb ^= from_mask | to_mask;
    captured_bb &= ~to_mask;
    pos.occupancies[color_idx] ^= from_mask | to_mask;
    pos.occupancies[enemy_idx] &= ~to_mask;
    pos.board[from] = 0;
    pos.board[to] = moved_encoded;
    if (moved_piece == PieceType::King) {
        pos.king_squares[static_cast<int>(color)] = to;
    }
    pos.eval_score += evaluation_piece_contribution(color, moved_piece, to)
        - evaluation_piece_contribution(color, moved_piece, from)
        - evaluation_piece_contribution(enemy, captured_piece, to);
    pos.zobrist_key ^= zobrist::piece_key(color, moved_piece, from);
    pos.zobrist_key ^= zobrist::piece_key(color, moved_piece, to);
    pos.zobrist_key ^= zobrist::piece_key(enemy, captured_piece, to);
}

void finish_move_metadata(
    Position& pos,
    MoveFlag flag,
    Color color,
    Color enemy,
    Square to,
    bool pawn_move,
    bool capture
) {
    if (pos.en_passant_square != NoSquare) {
        pos.zobrist_key ^= zobrist::en_passant_file_key(file_of(pos.en_passant_square));
    }
    pos.en_passant_square = flag == MoveFlag::DoublePawnPush
        ? (color == Color::White ? to - 8 : to + 8)
        : NoSquare;
    if (pos.en_passant_square != NoSquare) {
        pos.zobrist_key ^= zobrist::en_passant_file_key(file_of(pos.en_passant_square));
    }

    pos.halfmove_clock = (pawn_move || capture) ? 0 : pos.halfmove_clock + 1;
    if (color == Color::Black) {
        ++pos.fullmove_number;
    }
    pos.zobrist_key ^= zobrist::side_key();
    pos.side_to_move = enemy;
}

void push_generated_move(
    MoveList& moves,
    Move move,
    PieceType moved_piece,
    PieceType captured_piece = PieceType::None
) {
    (void)moved_piece;
    (void)captured_piece;
    moves.push_back(move);
}

void remove_castling_right_for_rook_square(Position& pos, Square square) {
    if (square == make_square(7, 0)) {
        if (pos.white_can_castle_kingside) {
            pos.zobrist_key ^= zobrist::castling_key(zobrist::CastlingRight::WhiteKingside);
            pos.white_can_castle_kingside = false;
        }
    } else if (square == make_square(0, 0)) {
        if (pos.white_can_castle_queenside) {
            pos.zobrist_key ^= zobrist::castling_key(zobrist::CastlingRight::WhiteQueenside);
            pos.white_can_castle_queenside = false;
        }
    } else if (square == make_square(7, 7)) {
        if (pos.black_can_castle_kingside) {
            pos.zobrist_key ^= zobrist::castling_key(zobrist::CastlingRight::BlackKingside);
            pos.black_can_castle_kingside = false;
        }
    } else if (square == make_square(0, 7)) {
        if (pos.black_can_castle_queenside) {
            pos.zobrist_key ^= zobrist::castling_key(zobrist::CastlingRight::BlackQueenside);
            pos.black_can_castle_queenside = false;
        }
    }
}

void remove_castling_rights_for_king(Position& pos, Color color) {
    if (color == Color::White) {
        if (pos.white_can_castle_kingside) {
            pos.zobrist_key ^= zobrist::castling_key(zobrist::CastlingRight::WhiteKingside);
            pos.white_can_castle_kingside = false;
        }
        if (pos.white_can_castle_queenside) {
            pos.zobrist_key ^= zobrist::castling_key(zobrist::CastlingRight::WhiteQueenside);
            pos.white_can_castle_queenside = false;
        }
    } else {
        if (pos.black_can_castle_kingside) {
            pos.zobrist_key ^= zobrist::castling_key(zobrist::CastlingRight::BlackKingside);
            pos.black_can_castle_kingside = false;
        }
        if (pos.black_can_castle_queenside) {
            pos.zobrist_key ^= zobrist::castling_key(zobrist::CastlingRight::BlackQueenside);
            pos.black_can_castle_queenside = false;
        }
    }
}

struct MoveGenerationContext {
    Color us;
    Color them;
    int us_idx;
    int them_idx;
    Bitboard our_occupancy;
    Bitboard their_occupancy;
    Bitboard occupancy;
    Bitboard their_king;
    Bitboard capturable;
};

Bitboard promotion_from_rank_mask(Color color) {
    return color == Color::White
        ? RankMask[static_cast<int>(Rank::R7)]
        : RankMask[static_cast<int>(Rank::R2)];
}

MoveGenerationContext make_move_generation_context(const Position& pos) {
    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const int us_idx = static_cast<int>(us);
    const int them_idx = static_cast<int>(them);
    const Bitboard our_occupancy = pos.occupancy(us);
    const Bitboard their_occupancy = pos.occupancy(them);
    const Bitboard their_king = pos.pieces[them_idx][static_cast<int>(PieceType::King)];
    return MoveGenerationContext{
        us,
        them,
        us_idx,
        them_idx,
        our_occupancy,
        their_occupancy,
        our_occupancy | their_occupancy,
        their_king,
        their_occupancy & ~their_king
    };
}

void add_castling_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves,
    Square king_square
) {
    const Color color = pos.side_to_move;
    const Color enemy = opposite(color);
    const Bitboard occupancy = context.occupancy;

    if (color == Color::White) {
        const Square e1 = make_square(4, 0);
        const Square f1 = make_square(5, 0);
        const Square g1 = make_square(6, 0);
        const Square d1 = make_square(3, 0);
        const Square c1 = make_square(2, 0);
        const Square b1 = make_square(1, 0);
        const Square h1 = make_square(7, 0);
        const Square a1 = make_square(0, 0);

        if (king_square == e1 && pos.white_can_castle_kingside && has_rook_on(pos, color, h1)
            && !(occupancy & (bit(f1) | bit(g1)))
            && !is_square_attacked(pos, e1, enemy)
            && !is_square_attacked(pos, f1, enemy)
            && !is_square_attacked(pos, g1, enemy)) {
            push_generated_move(moves, make_move(e1, g1, MoveFlag::KingCastle), PieceType::King);
        }

        if (king_square == e1 && pos.white_can_castle_queenside && has_rook_on(pos, color, a1)
            && !(occupancy & (bit(d1) | bit(c1) | bit(b1)))
            && !is_square_attacked(pos, e1, enemy)
            && !is_square_attacked(pos, d1, enemy)
            && !is_square_attacked(pos, c1, enemy)) {
            push_generated_move(moves, make_move(e1, c1, MoveFlag::QueenCastle), PieceType::King);
        }
    } else {
        const Square e8 = make_square(4, 7);
        const Square f8 = make_square(5, 7);
        const Square g8 = make_square(6, 7);
        const Square d8 = make_square(3, 7);
        const Square c8 = make_square(2, 7);
        const Square b8 = make_square(1, 7);
        const Square h8 = make_square(7, 7);
        const Square a8 = make_square(0, 7);

        if (king_square == e8 && pos.black_can_castle_kingside && has_rook_on(pos, color, h8)
            && !(occupancy & (bit(f8) | bit(g8)))
            && !is_square_attacked(pos, e8, enemy)
            && !is_square_attacked(pos, f8, enemy)
            && !is_square_attacked(pos, g8, enemy)) {
            push_generated_move(moves, make_move(e8, g8, MoveFlag::KingCastle), PieceType::King);
        }

        if (king_square == e8 && pos.black_can_castle_queenside && has_rook_on(pos, color, a8)
            && !(occupancy & (bit(d8) | bit(c8) | bit(b8)))
            && !is_square_attacked(pos, e8, enemy)
            && !is_square_attacked(pos, d8, enemy)
            && !is_square_attacked(pos, c8, enemy)) {
            push_generated_move(moves, make_move(e8, c8, MoveFlag::QueenCastle), PieceType::King);
        }
    }
}

void generate_knight_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)];
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = knight_attacks(from) & ~context.our_occupancy & ~context.their_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            const bool capture = (context.capturable & bit(to)) != EmptyBB;
            push_generated_move(
                moves,
                make_move(from, to, capture ? MoveFlag::Capture : MoveFlag::Quiet),
                PieceType::Knight,
                PieceType::None);
        }
    }
}

void generate_king_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::King)];
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = king_attacks(from) & ~context.our_occupancy & ~context.their_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            const bool capture = (context.capturable & bit(to)) != EmptyBB;
            push_generated_move(
                moves,
                make_move(from, to, capture ? MoveFlag::Capture : MoveFlag::Quiet),
                PieceType::King,
                PieceType::None);
        }
        add_castling_moves(pos, context, moves, from);
    }
}

void generate_bishop_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)];
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = bishop_attacks(from, context.occupancy)
            & ~context.our_occupancy
            & ~context.their_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            const bool capture = (context.capturable & bit(to)) != EmptyBB;
            push_generated_move(
                moves,
                make_move(from, to, capture ? MoveFlag::Capture : MoveFlag::Quiet),
                PieceType::Bishop,
                PieceType::None);
        }
    }
}

void generate_rook_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)];
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = rook_attacks(from, context.occupancy)
            & ~context.our_occupancy
            & ~context.their_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            const bool capture = (context.capturable & bit(to)) != EmptyBB;
            push_generated_move(
                moves,
                make_move(from, to, capture ? MoveFlag::Capture : MoveFlag::Quiet),
                PieceType::Rook,
                PieceType::None);
        }
    }
}

void generate_queen_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)];
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = queen_attacks(from, context.occupancy)
            & ~context.our_occupancy
            & ~context.their_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            const bool capture = (context.capturable & bit(to)) != EmptyBB;
            push_generated_move(
                moves,
                make_move(from, to, capture ? MoveFlag::Capture : MoveFlag::Quiet),
                PieceType::Queen,
                PieceType::None);
        }
    }
}

void generate_pawn_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    int starting_rank = context.us == Color::White ? static_cast<int>(Rank::R2) : static_cast<int>(Rank::R7);
    int promotion_rank = context.us == Color::White ? static_cast<int>(Rank::R8) : static_cast<int>(Rank::R1);
    int offset = context.us == Color::White ? 1 : -1;
    while (positions) {
        Square from = pop_lsb(positions);
        int rank = rank_of(from);
        int file = file_of(from);
        assert(rank != promotion_rank);
        Bitboard target = pawn_attacks(context.us, from);
        if (rank + offset == promotion_rank) {
            while (target) {
                Square to = pop_lsb(target);
                if (context.capturable & bit(to)) {
                    push_generated_move(moves, make_move(from, to, MoveFlag::QueenPromotionCapture), PieceType::Pawn);
                    push_generated_move(moves, make_move(from, to, MoveFlag::KnightPromotionCapture), PieceType::Pawn);
                    push_generated_move(moves, make_move(from, to, MoveFlag::BishopPromotionCapture), PieceType::Pawn);
                    push_generated_move(moves, make_move(from, to, MoveFlag::RookPromotionCapture), PieceType::Pawn);
                }
            }
            Square to = make_square(file, rank + offset);
            if (!(context.occupancy & bit(to))) {
                push_generated_move(moves, make_move(from, to, MoveFlag::QueenPromotion), PieceType::Pawn);
                push_generated_move(moves, make_move(from, to, MoveFlag::KnightPromotion), PieceType::Pawn);
                push_generated_move(moves, make_move(from, to, MoveFlag::BishopPromotion), PieceType::Pawn);
                push_generated_move(moves, make_move(from, to, MoveFlag::RookPromotion), PieceType::Pawn);
            }
        } else {
            while (target) {
                Square to = pop_lsb(target);
                if (context.capturable & bit(to)) {
                    push_generated_move(
                        moves,
                        make_move(from, to, MoveFlag::Capture),
                        PieceType::Pawn,
                        PieceType::None);
                } else if (to == pos.en_passant_square) {
                    push_generated_move(
                        moves,
                        make_move(from, to, MoveFlag::EnPassant),
                        PieceType::Pawn,
                        PieceType::Pawn);
                }
            }
            Square to = make_square(file, rank + offset);
            if (!(context.occupancy & bit(to))) {
                push_generated_move(moves, make_move(from, to, MoveFlag::Quiet), PieceType::Pawn);
                if (rank == starting_rank) {
                    Square double_to = make_square(file, rank + offset + offset);
                    if (!(context.occupancy & bit(double_to))) {
                        push_generated_move(moves, make_move(from, double_to, MoveFlag::DoublePawnPush), PieceType::Pawn);
                    }
                }
            }
        }
    }
}

void generate_knight_non_capture_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)];
    while (positions != EmptyBB) {
        const Square from = pop_lsb(positions);
        Bitboard targets = knight_attacks(from) & ~context.occupancy;
        while (targets != EmptyBB) {
            push_generated_move(
                moves,
                make_move(from, pop_lsb(targets), MoveFlag::Quiet),
                PieceType::Knight);
        }
    }
}

void generate_king_non_capture_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::King)];
    while (positions != EmptyBB) {
        const Square from = pop_lsb(positions);
        Bitboard targets = king_attacks(from) & ~context.occupancy;
        while (targets != EmptyBB) {
            push_generated_move(
                moves,
                make_move(from, pop_lsb(targets), MoveFlag::Quiet),
                PieceType::King);
        }
        add_castling_moves(pos, context, moves, from);
    }
}

void generate_bishop_non_capture_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)];
    while (positions != EmptyBB) {
        const Square from = pop_lsb(positions);
        Bitboard targets = bishop_attacks(from, context.occupancy) & ~context.occupancy;
        while (targets != EmptyBB) {
            push_generated_move(
                moves,
                make_move(from, pop_lsb(targets), MoveFlag::Quiet),
                PieceType::Bishop);
        }
    }
}

void generate_rook_non_capture_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)];
    while (positions != EmptyBB) {
        const Square from = pop_lsb(positions);
        Bitboard targets = rook_attacks(from, context.occupancy) & ~context.occupancy;
        while (targets != EmptyBB) {
            push_generated_move(
                moves,
                make_move(from, pop_lsb(targets), MoveFlag::Quiet),
                PieceType::Rook);
        }
    }
}

void generate_queen_non_capture_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)];
    while (positions != EmptyBB) {
        const Square from = pop_lsb(positions);
        Bitboard targets = queen_attacks(from, context.occupancy) & ~context.occupancy;
        while (targets != EmptyBB) {
            push_generated_move(
                moves,
                make_move(from, pop_lsb(targets), MoveFlag::Quiet),
                PieceType::Queen);
        }
    }
}

void generate_pawn_non_capture_moves(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard positions = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    const int starting_rank = context.us == Color::White ? static_cast<int>(Rank::R2) : static_cast<int>(Rank::R7);
    [[maybe_unused]] const int promotion_rank = context.us == Color::White
        ? static_cast<int>(Rank::R8)
        : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;

    while (positions != EmptyBB) {
        const Square from = pop_lsb(positions);
        const int rank = rank_of(from);
        const int file = file_of(from);
        assert(rank != promotion_rank);

        const Square to = make_square(file, rank + offset);
        if ((context.occupancy & bit(to)) != EmptyBB) {
            continue;
        }

        if (rank + offset == promotion_rank) {
            push_generated_move(moves, make_move(from, to, MoveFlag::QueenPromotion), PieceType::Pawn);
            push_generated_move(moves, make_move(from, to, MoveFlag::KnightPromotion), PieceType::Pawn);
            push_generated_move(moves, make_move(from, to, MoveFlag::BishopPromotion), PieceType::Pawn);
            push_generated_move(moves, make_move(from, to, MoveFlag::RookPromotion), PieceType::Pawn);
            continue;
        }

        push_generated_move(moves, make_move(from, to, MoveFlag::Quiet), PieceType::Pawn);
        if (rank == starting_rank) {
            const Square double_to = make_square(file, rank + offset + offset);
            if ((context.occupancy & bit(double_to)) == EmptyBB) {
                push_generated_move(moves, make_move(from, double_to, MoveFlag::DoublePawnPush), PieceType::Pawn);
            }
        }
    }
}

} // namespace

std::string square_to_string(Square square) {
    assert(is_valid_square(square));

    std::string result;
    result += static_cast<char>('a' + file_of(square));
    result += static_cast<char>('1' + rank_of(square));
    return result;
}

std::string move_to_string(Move move) {
    std::string result = square_to_string(move.from()) + square_to_string(move.to());

    const PieceType promotion = promotion_piece(move);
    if (promotion != PieceType::None) {
        switch (promotion) {
            case PieceType::Knight:
                result += 'n';
                break;
            case PieceType::Bishop:
                result += 'b';
                break;
            case PieceType::Rook:
                result += 'r';
                break;
            case PieceType::Queen:
                result += 'q';
                break;
            default:
                assert(false);
                break;
        }
    }
    return result;
}

std::vector<Move> to_vector(const MoveList& moves) {
    return std::vector<Move>(moves.begin(), moves.end());
}

void generate_knight_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_knight_moves(pos, context, moves);
}

void generate_king_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_king_moves(pos, context, moves);
}

void generate_bishop_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_bishop_moves(pos, context, moves);
}

void generate_rook_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_rook_moves(pos, context, moves);
}

void generate_queen_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_queen_moves(pos, context, moves);
}

void generate_pawn_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_pawn_moves(pos, context, moves);
}

void generate_pseudo_capture_moves_with_context(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    [[maybe_unused]] const int promotion_rank = context.us == Color::White
        ? static_cast<int>(Rank::R8)
        : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;

    const Bitboard all_pawns = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    Bitboard pawns = EmptyBB;
    if (context.us == Color::White) {
        pawns |= all_pawns & ((context.capturable & ~FileMask[7]) >> 7);
        pawns |= all_pawns & ((context.capturable & ~FileMask[0]) >> 9);

        if (pos.en_passant_square != NoSquare) {
            const Bitboard ep_target = bit(pos.en_passant_square);
            pawns |= all_pawns & ((ep_target & ~FileMask[7]) >> 7);
            pawns |= all_pawns & ((ep_target & ~FileMask[0]) >> 9);
        }
    } else {
        pawns |= all_pawns & ((context.capturable & ~FileMask[7]) << 9);
        pawns |= all_pawns & ((context.capturable & ~FileMask[0]) << 7);

        if (pos.en_passant_square != NoSquare) {
            const Bitboard ep_target = bit(pos.en_passant_square);
            pawns |= all_pawns & ((ep_target & ~FileMask[7]) << 9);
            pawns |= all_pawns & ((ep_target & ~FileMask[0]) << 7);
        }
    }

    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        const int rank = rank_of(from);
        assert(rank != promotion_rank);
        Bitboard targets = pawn_attacks(context.us, from);
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            if ((context.capturable & bit(to)) != EmptyBB) {
                if (rank + offset == promotion_rank) {
                    push_generated_move(moves, make_move(from, to, MoveFlag::QueenPromotionCapture), PieceType::Pawn);
                    push_generated_move(moves, make_move(from, to, MoveFlag::KnightPromotionCapture), PieceType::Pawn);
                    push_generated_move(moves, make_move(from, to, MoveFlag::BishopPromotionCapture), PieceType::Pawn);
                    push_generated_move(moves, make_move(from, to, MoveFlag::RookPromotionCapture), PieceType::Pawn);
                } else {
                    push_generated_move(moves, make_move(from, to, MoveFlag::Capture), PieceType::Pawn);
                }
            } else if (to == pos.en_passant_square) {
                push_generated_move(moves, make_move(from, to, MoveFlag::EnPassant), PieceType::Pawn, PieceType::Pawn);
            }
        }
    }

    Bitboard knights = pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)];
    while (knights != EmptyBB) {
        const Square from = pop_lsb(knights);
        Bitboard targets = knight_attacks(from) & context.capturable;
        while (targets != EmptyBB) {
            push_generated_move(moves, make_move(from, pop_lsb(targets), MoveFlag::Capture), PieceType::Knight);
        }
    }

    Bitboard bishops = pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)];
    while (bishops != EmptyBB) {
        const Square from = pop_lsb(bishops);
        Bitboard targets = bishop_attacks(from, context.occupancy) & context.capturable;
        while (targets != EmptyBB) {
            push_generated_move(moves, make_move(from, pop_lsb(targets), MoveFlag::Capture), PieceType::Bishop);
        }
    }

    Bitboard rooks = pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)];
    while (rooks != EmptyBB) {
        const Square from = pop_lsb(rooks);
        Bitboard targets = rook_attacks(from, context.occupancy) & context.capturable;
        while (targets != EmptyBB) {
            push_generated_move(moves, make_move(from, pop_lsb(targets), MoveFlag::Capture), PieceType::Rook);
        }
    }

    Bitboard queens = pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)];
    while (queens != EmptyBB) {
        const Square from = pop_lsb(queens);
        Bitboard targets = queen_attacks(from, context.occupancy) & context.capturable;
        while (targets != EmptyBB) {
            push_generated_move(moves, make_move(from, pop_lsb(targets), MoveFlag::Capture), PieceType::Queen);
        }
    }

    Bitboard kings = pos.pieces[context.us_idx][static_cast<int>(PieceType::King)];
    while (kings != EmptyBB) {
        const Square from = pop_lsb(kings);
        Bitboard targets = king_attacks(from) & context.capturable;
        while (targets != EmptyBB) {
            push_generated_move(moves, make_move(from, pop_lsb(targets), MoveFlag::Capture), PieceType::King);
        }
    }
}

void generate_pseudo_promotion_moves_with_context(
    const Position& pos,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    [[maybe_unused]] const int promotion_rank = context.us == Color::White
        ? static_cast<int>(Rank::R8)
        : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;

    Bitboard pawns = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)]
        & promotion_from_rank_mask(context.us);
    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        const int rank = rank_of(from);
        assert(rank != promotion_rank);
        assert(rank + offset == promotion_rank);
        const Square to = make_square(file_of(from), rank + offset);
        if ((context.occupancy & bit(to)) == EmptyBB) {
            push_generated_move(moves, make_move(from, to, MoveFlag::QueenPromotion), PieceType::Pawn);
            push_generated_move(moves, make_move(from, to, MoveFlag::KnightPromotion), PieceType::Pawn);
            push_generated_move(moves, make_move(from, to, MoveFlag::BishopPromotion), PieceType::Pawn);
            push_generated_move(moves, make_move(from, to, MoveFlag::RookPromotion), PieceType::Pawn);
        }
    }
}

PieceType captured_piece_for_noisy_move(
    const Position& pos,
    const MoveGenerationContext& context,
    Move move
) {
    if (!is_capture(move)) {
        return PieceType::None;
    }
    if (move_flag(move) == MoveFlag::EnPassant) {
        return PieceType::Pawn;
    }
    return pos.piece_type_on_occupied(context.them, to_square(move));
}

void try_push_legal_noisy_move(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Move move,
    PieceType moved_piece
) {
    const PieceType captured_piece = captured_piece_for_noisy_move(pos, context, move);
    if (is_pseudo_move_legal(pos, king_safety, move, moved_piece, captured_piece)) {
        push_generated_move(moves, move, moved_piece, captured_piece);
    }
}

void try_push_legal_non_capture_move(
    const Position& pos,
    const KingSafetyContext& king_safety,
    MoveList& moves,
    Move move,
    PieceType moved_piece
) {
    if (is_pseudo_move_legal(pos, king_safety, move, moved_piece, PieceType::None)) {
        push_generated_move(moves, move, moved_piece);
    }
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

bool is_legal_non_king_non_capture_target(
    const KingSafetyContext& king_safety,
    Square from,
    Square to
) {
    if ((king_safety.pinned & bit(from)) == EmptyBB) {
        return true;
    }
    return same_line(king_safety.king_square, from, to);
}

void push_legal_non_king_non_capture_move(
    const KingSafetyContext& king_safety,
    MoveList& moves,
    Move move,
    PieceType moved_piece
) {
    if (is_legal_non_king_non_capture_target(
            king_safety,
            from_square(move),
            to_square(move))) {
        push_generated_move(moves, move, moved_piece);
    }
}

void generate_legal_king_noisy_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard kings = pos.pieces[context.us_idx][static_cast<int>(PieceType::King)];
    while (kings != EmptyBB) {
        const Square from = pop_lsb(kings);
        Bitboard targets = king_attacks(from) & context.capturable;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, to, MoveFlag::Capture),
                PieceType::King);
        }
    }
}

void generate_legal_king_non_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    Bitboard kings = pos.pieces[context.us_idx][static_cast<int>(PieceType::King)];
    while (kings != EmptyBB) {
        const Square from = pop_lsb(kings);
        Bitboard targets = king_attacks(from) & ~context.occupancy;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_push_legal_non_capture_move(
                pos,
                king_safety,
                moves,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::King);
        }

        MoveList castling_moves;
        add_castling_moves(pos, context, castling_moves, from);
        for (Move move : castling_moves) {
            try_push_legal_non_capture_move(pos, king_safety, moves, move, PieceType::King);
        }
    }
}

void generate_legal_pawn_non_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pawns = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    const int starting_rank = context.us == Color::White ? static_cast<int>(Rank::R2) : static_cast<int>(Rank::R7);
    [[maybe_unused]] const int promotion_rank = context.us == Color::White
        ? static_cast<int>(Rank::R8)
        : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;

    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        const int rank = rank_of(from);
        const int file = file_of(from);
        assert(rank != promotion_rank);

        const Square to = make_square(file, rank + offset);
        const Bitboard to_mask = bit(to);
        if ((context.occupancy & to_mask) != EmptyBB || (allowed_targets & to_mask) == EmptyBB) {
            continue;
        }

        if (rank + offset == promotion_rank) {
            push_legal_non_king_non_capture_move(
                king_safety,
                moves,
                make_move(from, to, MoveFlag::QueenPromotion),
                PieceType::Pawn);
            push_legal_non_king_non_capture_move(
                king_safety,
                moves,
                make_move(from, to, MoveFlag::KnightPromotion),
                PieceType::Pawn);
            push_legal_non_king_non_capture_move(
                king_safety,
                moves,
                make_move(from, to, MoveFlag::BishopPromotion),
                PieceType::Pawn);
            push_legal_non_king_non_capture_move(
                king_safety,
                moves,
                make_move(from, to, MoveFlag::RookPromotion),
                PieceType::Pawn);
            continue;
        }

        push_legal_non_king_non_capture_move(
            king_safety,
            moves,
            make_move(from, to, MoveFlag::Quiet),
            PieceType::Pawn);
        if (rank == starting_rank) {
            const Square double_to = make_square(file, rank + offset + offset);
            const Bitboard double_to_mask = bit(double_to);
            if ((context.occupancy & double_to_mask) == EmptyBB
                && (allowed_targets & double_to_mask) != EmptyBB) {
                push_legal_non_king_non_capture_move(
                    king_safety,
                    moves,
                    make_move(from, double_to, MoveFlag::DoublePawnPush),
                    PieceType::Pawn);
            }
        }
    }
}

void generate_legal_knight_non_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = knight_attacks(from) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            push_legal_non_king_non_capture_move(
                king_safety,
                moves,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::Knight);
        }
    }
}

void generate_legal_bishop_non_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = bishop_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            push_legal_non_king_non_capture_move(
                king_safety,
                moves,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::Bishop);
        }
    }
}

void generate_legal_rook_non_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = rook_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            push_legal_non_king_non_capture_move(
                king_safety,
                moves,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::Rook);
        }
    }
}

void generate_legal_queen_non_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = queen_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            push_legal_non_king_non_capture_move(
                king_safety,
                moves,
                make_move(from, to, MoveFlag::Quiet),
                PieceType::Queen);
        }
    }
}

void generate_legal_pawn_noisy_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_capture_targets,
    Bitboard allowed_promotion_targets
) {
    const Bitboard all_pawns = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    const Bitboard promotion_pawns = all_pawns & promotion_from_rank_mask(context.us);
    const Bitboard normal_pawns = all_pawns & ~promotion_pawns;

    auto push_promotion_capture = [&](Square from, Square to) {
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::QueenPromotionCapture),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::KnightPromotionCapture),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::BishopPromotionCapture),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::RookPromotionCapture),
            PieceType::Pawn);
    };

    auto push_quiet_promotion = [&](Square from, Square to) {
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::QueenPromotion),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::KnightPromotion),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::BishopPromotion),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::RookPromotion),
            PieceType::Pawn);
    };

    auto push_normal_capture = [&](Square from, Square to) {
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::Capture),
            PieceType::Pawn);
    };

    if (context.us == Color::White) {
        Bitboard normal_left_targets = ((normal_pawns & ~FileMask[0]) << 7) & allowed_capture_targets;
        while (normal_left_targets != EmptyBB) {
            const Square to = pop_lsb(normal_left_targets);
            push_normal_capture(to - 7, to);
        }

        Bitboard normal_right_targets = ((normal_pawns & ~FileMask[7]) << 9) & allowed_capture_targets;
        while (normal_right_targets != EmptyBB) {
            const Square to = pop_lsb(normal_right_targets);
            push_normal_capture(to - 9, to);
        }

        Bitboard promotion_left_targets = ((promotion_pawns & ~FileMask[0]) << 7) & allowed_capture_targets;
        while (promotion_left_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_left_targets);
            push_promotion_capture(to - 7, to);
        }

        Bitboard promotion_right_targets = ((promotion_pawns & ~FileMask[7]) << 9) & allowed_capture_targets;
        while (promotion_right_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_right_targets);
            push_promotion_capture(to - 9, to);
        }

        Bitboard promotion_targets = (promotion_pawns << 8) & ~context.occupancy & allowed_promotion_targets;
        while (promotion_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_targets);
            push_quiet_promotion(to - 8, to);
        }

        if (pos.en_passant_square != NoSquare) {
            const Bitboard ep_target = bit(pos.en_passant_square);
            Bitboard ep_from = normal_pawns & (((ep_target & ~FileMask[7]) >> 7) | ((ep_target & ~FileMask[0]) >> 9));
            while (ep_from != EmptyBB) {
                const Square from = pop_lsb(ep_from);
                try_push_legal_noisy_move(
                    pos,
                    king_safety,
                    context,
                    moves,
                    make_move(from, pos.en_passant_square, MoveFlag::EnPassant),
                    PieceType::Pawn);
            }
        }
    } else {
        Bitboard normal_left_targets = ((normal_pawns & ~FileMask[0]) >> 9) & allowed_capture_targets;
        while (normal_left_targets != EmptyBB) {
            const Square to = pop_lsb(normal_left_targets);
            push_normal_capture(to + 9, to);
        }

        Bitboard normal_right_targets = ((normal_pawns & ~FileMask[7]) >> 7) & allowed_capture_targets;
        while (normal_right_targets != EmptyBB) {
            const Square to = pop_lsb(normal_right_targets);
            push_normal_capture(to + 7, to);
        }

        Bitboard promotion_left_targets = ((promotion_pawns & ~FileMask[0]) >> 9) & allowed_capture_targets;
        while (promotion_left_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_left_targets);
            push_promotion_capture(to + 9, to);
        }

        Bitboard promotion_right_targets = ((promotion_pawns & ~FileMask[7]) >> 7) & allowed_capture_targets;
        while (promotion_right_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_right_targets);
            push_promotion_capture(to + 7, to);
        }

        Bitboard promotion_targets = (promotion_pawns >> 8) & ~context.occupancy & allowed_promotion_targets;
        while (promotion_targets != EmptyBB) {
            const Square to = pop_lsb(promotion_targets);
            push_quiet_promotion(to + 8, to);
        }

        if (pos.en_passant_square != NoSquare) {
            const Bitboard ep_target = bit(pos.en_passant_square);
            Bitboard ep_from = normal_pawns & (((ep_target & ~FileMask[7]) << 9) | ((ep_target & ~FileMask[0]) << 7));
            while (ep_from != EmptyBB) {
                const Square from = pop_lsb(ep_from);
                try_push_legal_noisy_move(
                    pos,
                    king_safety,
                    context,
                    moves,
                    make_move(from, pos.en_passant_square, MoveFlag::EnPassant),
                    PieceType::Pawn);
            }
        }
    }
}

void generate_legal_knight_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = knight_attacks(from) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, to, MoveFlag::Capture),
                PieceType::Knight);
        }
    }
}

void generate_legal_bishop_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = bishop_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, to, MoveFlag::Capture),
                PieceType::Bishop);
        }
    }
}

void generate_legal_rook_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = rook_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, to, MoveFlag::Capture),
                PieceType::Rook);
        }
    }
}

void generate_legal_queen_capture_moves_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Bitboard allowed_targets
) {
    Bitboard pieces = pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = queen_attacks(from, context.occupancy) & allowed_targets;
        while (targets != EmptyBB) {
            const Square to = pop_lsb(targets);
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, to, MoveFlag::Capture),
                PieceType::Queen);
        }
    }
}

void generate_legal_pawn_captures_to_square_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Square target
) {
    [[maybe_unused]] const int promotion_rank = context.us == Color::White
        ? static_cast<int>(Rank::R8)
        : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;
    Bitboard pawns =
        pawn_attacks(context.them, target)
        & pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        const int rank = rank_of(from);
        assert(rank != promotion_rank);
        if (rank + offset == promotion_rank) {
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, target, MoveFlag::QueenPromotionCapture),
                PieceType::Pawn);
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, target, MoveFlag::KnightPromotionCapture),
                PieceType::Pawn);
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, target, MoveFlag::BishopPromotionCapture),
                PieceType::Pawn);
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, target, MoveFlag::RookPromotionCapture),
                PieceType::Pawn);
        } else {
            try_push_legal_noisy_move(
                pos,
                king_safety,
                context,
                moves,
                make_move(from, target, MoveFlag::Capture),
                PieceType::Pawn);
        }
    }
}

void generate_legal_en_passant_capture_checker_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Square checker
) {
    if (pos.en_passant_square == NoSquare) {
        return;
    }

    const Square captured_square = context.us == Color::White
        ? pos.en_passant_square - 8
        : pos.en_passant_square + 8;
    if (captured_square != checker) {
        return;
    }

    Bitboard pawns =
        pawn_attacks(context.them, pos.en_passant_square)
        & pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)];
    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, pos.en_passant_square, MoveFlag::EnPassant),
            PieceType::Pawn);
    }
}

void generate_legal_knight_captures_to_square_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Square target
) {
    Bitboard pieces =
        knight_attacks(target)
        & pos.pieces[context.us_idx][static_cast<int>(PieceType::Knight)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, target, MoveFlag::Capture),
            PieceType::Knight);
    }
}

void generate_legal_bishop_captures_to_square_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Square target
) {
    Bitboard pieces =
        bishop_attacks(target, context.occupancy)
        & pos.pieces[context.us_idx][static_cast<int>(PieceType::Bishop)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, target, MoveFlag::Capture),
            PieceType::Bishop);
    }
}

void generate_legal_rook_captures_to_square_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Square target
) {
    Bitboard pieces =
        rook_attacks(target, context.occupancy)
        & pos.pieces[context.us_idx][static_cast<int>(PieceType::Rook)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, target, MoveFlag::Capture),
            PieceType::Rook);
    }
}

void generate_legal_queen_captures_to_square_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves,
    Square target
) {
    Bitboard pieces =
        queen_attacks(target, context.occupancy)
        & pos.pieces[context.us_idx][static_cast<int>(PieceType::Queen)];
    while (pieces != EmptyBB) {
        const Square from = pop_lsb(pieces);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, target, MoveFlag::Capture),
            PieceType::Queen);
    }
}

void generate_legal_quiet_promotion_blocks_with_context(
    const Position& pos,
    const KingSafetyContext& king_safety,
    const MoveGenerationContext& context,
    MoveList& moves
) {
    [[maybe_unused]] const int promotion_rank = context.us == Color::White
        ? static_cast<int>(Rank::R8)
        : static_cast<int>(Rank::R1);
    const int offset = context.us == Color::White ? 1 : -1;
    Bitboard pawns = pos.pieces[context.us_idx][static_cast<int>(PieceType::Pawn)]
        & promotion_from_rank_mask(context.us);
    while (pawns != EmptyBB) {
        const Square from = pop_lsb(pawns);
        const int rank = rank_of(from);
        assert(rank != promotion_rank);
        assert(rank + offset == promotion_rank);
        const Square to = make_square(file_of(from), rank + offset);
        if ((context.occupancy & bit(to)) != EmptyBB
            || (king_safety.block_mask & bit(to)) == EmptyBB) {
            continue;
        }

        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::QueenPromotion),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::KnightPromotion),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::BishopPromotion),
            PieceType::Pawn);
        try_push_legal_noisy_move(
            pos,
            king_safety,
            context,
            moves,
            make_move(from, to, MoveFlag::RookPromotion),
            PieceType::Pawn);
    }
}

void generate_pseudo_capture_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_pseudo_capture_moves_with_context(pos, context, moves);
}

void generate_pseudo_non_capture_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_pawn_non_capture_moves(pos, context, moves);
    generate_knight_non_capture_moves(pos, context, moves);
    generate_bishop_non_capture_moves(pos, context, moves);
    generate_rook_non_capture_moves(pos, context, moves);
    generate_queen_non_capture_moves(pos, context, moves);
    generate_king_non_capture_moves(pos, context, moves);
}

void generate_pseudo_promotion_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_pseudo_promotion_moves_with_context(pos, context, moves);
}

void generate_pseudo_noisy_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_pseudo_capture_moves_with_context(pos, context, moves);
    generate_pseudo_promotion_moves_with_context(pos, context, moves);
}

void generate_legal_noisy_moves(
    const Position& pos,
    const KingSafetyContext& king_safety,
    MoveList& moves
) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_legal_king_noisy_moves_with_context(pos, king_safety, context, moves);

    if (popcount(king_safety.checkers) >= 2) {
        return;
    }

    if (king_safety.checkers != EmptyBB) {
        const Square checker = std::countr_zero(king_safety.checkers);
        generate_legal_pawn_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_en_passant_capture_checker_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_knight_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_bishop_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_rook_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_queen_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_quiet_promotion_blocks_with_context(
            pos,
            king_safety,
            context,
            moves);
        return;
    }

    generate_legal_pawn_noisy_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable,
        FullBB);
    generate_legal_knight_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable);
    generate_legal_bishop_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable);
    generate_legal_rook_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable);
    generate_legal_queen_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable);
}

void generate_legal_capture_moves(
    const Position& pos,
    const KingSafetyContext& king_safety,
    MoveList& moves
) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_legal_king_noisy_moves_with_context(pos, king_safety, context, moves);

    if (popcount(king_safety.checkers) >= 2) {
        return;
    }

    if (king_safety.checkers != EmptyBB) {
        const Square checker = std::countr_zero(king_safety.checkers);
        generate_legal_pawn_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_en_passant_capture_checker_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_knight_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_bishop_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_rook_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        generate_legal_queen_captures_to_square_with_context(
            pos,
            king_safety,
            context,
            moves,
            checker);
        return;
    }

    generate_legal_pawn_noisy_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable,
        EmptyBB);
    generate_legal_knight_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable);
    generate_legal_bishop_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable);
    generate_legal_rook_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable);
    generate_legal_queen_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        context.capturable);
}

void generate_legal_non_capture_moves(
    const Position& pos,
    const KingSafetyContext& king_safety,
    MoveList& moves
) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_legal_king_non_capture_moves_with_context(pos, king_safety, context, moves);

    if (popcount(king_safety.checkers) >= 2) {
        return;
    }

    const Bitboard allowed_targets = king_safety.block_mask & ~context.occupancy;
    generate_legal_pawn_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
    generate_legal_knight_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
    generate_legal_bishop_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
    generate_legal_rook_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
    generate_legal_queen_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
}

void generate_legal_evasion_moves(
    const Position& pos,
    const KingSafetyContext& king_safety,
    MoveList& moves
) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_legal_king_noisy_moves_with_context(pos, king_safety, context, moves);
    generate_legal_king_non_capture_moves_with_context(pos, king_safety, context, moves);

    if (popcount(king_safety.checkers) >= 2) {
        return;
    }

    const Square checker = std::countr_zero(king_safety.checkers);
    generate_legal_pawn_captures_to_square_with_context(
        pos,
        king_safety,
        context,
        moves,
        checker);
    generate_legal_en_passant_capture_checker_with_context(
        pos,
        king_safety,
        context,
        moves,
        checker);
    generate_legal_knight_captures_to_square_with_context(
        pos,
        king_safety,
        context,
        moves,
        checker);
    generate_legal_bishop_captures_to_square_with_context(
        pos,
        king_safety,
        context,
        moves,
        checker);
    generate_legal_rook_captures_to_square_with_context(
        pos,
        king_safety,
        context,
        moves,
        checker);
    generate_legal_queen_captures_to_square_with_context(
        pos,
        king_safety,
        context,
        moves,
        checker);

    const Bitboard allowed_targets = king_safety.block_mask & ~context.occupancy;
    generate_legal_pawn_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
    generate_legal_knight_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
    generate_legal_bishop_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
    generate_legal_rook_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
    generate_legal_queen_non_capture_moves_with_context(
        pos,
        king_safety,
        context,
        moves,
        allowed_targets);
}

void generate_pseudo_legal_moves(const Position& pos, MoveList& moves) {
    const MoveGenerationContext context = make_move_generation_context(pos);
    generate_pawn_moves(pos, context, moves);
    generate_knight_moves(pos, context, moves);
    generate_bishop_moves(pos, context, moves);
    generate_rook_moves(pos, context, moves);
    generate_queen_moves(pos, context, moves);
    generate_king_moves(pos, context, moves);
}

std::vector<Move> generate_knight_moves(const Position& pos) {
    MoveList moves;
    generate_knight_moves(pos, moves);
    return to_vector(moves);
}

std::vector<Move> generate_king_moves(const Position& pos) {
    MoveList moves;
    generate_king_moves(pos, moves);
    return to_vector(moves);
}

std::vector<Move> generate_bishop_moves(const Position& pos) {
    MoveList moves;
    generate_bishop_moves(pos, moves);
    return to_vector(moves);
}

std::vector<Move> generate_rook_moves(const Position& pos) {
    MoveList moves;
    generate_rook_moves(pos, moves);
    return to_vector(moves);
}

std::vector<Move> generate_queen_moves(const Position& pos) {
    MoveList moves;
    generate_queen_moves(pos, moves);
    return to_vector(moves);
}

std::vector<Move> generate_pawn_moves(const Position& pos) {
    MoveList moves;
    generate_pawn_moves(pos, moves);
    return to_vector(moves);
}

std::vector<Move> generate_pseudo_legal_moves(const Position& pos) {
    MoveList moves;
    generate_pseudo_legal_moves(pos, moves);
    return to_vector(moves);
}

std::vector<Move> generate_pseudo_non_capture_moves(const Position& pos) {
    MoveList moves;
    generate_pseudo_non_capture_moves(pos, moves);
    return to_vector(moves);
}

std::vector<Move> generate_pseudo_noisy_moves(const Position& pos) {
    MoveList moves;
    generate_pseudo_noisy_moves(pos, moves);
    return to_vector(moves);
}

void Position::make_move(Move move) {
    make_move(move, piece_type_on_occupied(side_to_move, move.from()));
}

void Position::make_move(Move move, PieceType moved_piece) {
    const MoveFlag flag = move.flag();
    const PieceType captured_piece =
        (is_capture(flag) && flag != MoveFlag::EnPassant)
            ? piece_type_on_occupied(opposite(side_to_move), move.to())
            : PieceType::None;
    make_move(move, moved_piece, captured_piece);
}

void Position::make_move(Move move, PieceType moved_piece, PieceType captured_piece) {
    assert(moved_piece != PieceType::None);

    const Square from = move.from();
    const Square to = move.to();
    const Color color = side_to_move;
    const MoveFlag flag = move.flag();
    assert(side_to_move == color);
    PieceType placed_piece = promotion_piece(move);
    if (placed_piece == PieceType::None) {
        placed_piece = moved_piece;
    }
    const bool pawn_move = moved_piece == PieceType::Pawn;
    const bool capture = is_capture(flag);
    const Color enemy = opposite(color);
    Square captured_square = NoSquare;
    if (capture && flag != MoveFlag::EnPassant && captured_piece == PieceType::None) {
        captured_piece = piece_type_on_occupied(enemy, to);
    }
    assert(!capture || flag == MoveFlag::EnPassant || captured_piece != PieceType::None);
    if (capture) {
        captured_square = flag == MoveFlag::EnPassant
            ? (color == Color::White ? to - 8 : to + 8)
            : to;
        if (flag == MoveFlag::EnPassant) {
            captured_piece = PieceType::Pawn;
        }
    }

    if (moved_piece == PieceType::King) {
        remove_castling_rights_for_king(*this, color);
    } else if (moved_piece == PieceType::Rook) {
        remove_castling_right_for_rook_square(*this, from);
    }

    if (capture && flag != MoveFlag::EnPassant) {
        remove_castling_right_for_rook_square(*this, to);
    }

    if (flag == MoveFlag::Quiet || flag == MoveFlag::DoublePawnPush) {
        move_piece_fast(*this, color, moved_piece, from, to);
        finish_move_metadata(*this, flag, color, enemy, to, pawn_move, capture);
        update_king_safety_after_move(*this, move, color, moved_piece, captured_piece, captured_square);
        return;
    }

    if (flag == MoveFlag::Capture) {
        capture_piece_fast(*this, color, moved_piece, enemy, captured_piece, from, to);
        finish_move_metadata(*this, flag, color, enemy, to, pawn_move, capture);
        update_king_safety_after_move(*this, move, color, moved_piece, captured_piece, captured_square);
        return;
    }

    clear_piece(color, moved_piece, from);

  
    if (flag == MoveFlag::EnPassant) {
        assert(to == en_passant_square);
        clear_piece(enemy, PieceType::Pawn, captured_square);
    } else if (flag == MoveFlag::KingCastle) {
        const Square rook_from = color == Color::White ? make_square(7, 0) : make_square(7, 7);
        const Square rook_to = color == Color::White ? make_square(5, 0) : make_square(5, 7);
        clear_piece(color, PieceType::Rook, rook_from);
        set_piece(color, PieceType::Rook, rook_to);
    } else if (flag == MoveFlag::QueenCastle) {
        const Square rook_from = color == Color::White ? make_square(0, 0) : make_square(0, 7);
        const Square rook_to = color == Color::White ? make_square(3, 0) : make_square(3, 7);
        clear_piece(color, PieceType::Rook, rook_from);
        set_piece(color, PieceType::Rook, rook_to);
    } else if (capture) {
        clear_piece(enemy, captured_piece, to);
    }


    set_piece(color, placed_piece, to);

    finish_move_metadata(*this, flag, color, enemy, to, pawn_move, capture);
    update_king_safety_after_move(*this, move, color, moved_piece, captured_piece, captured_square);
}

void Position::make_move(Move move, UndoState& undo) {
    make_move(move, piece_type_on_occupied(side_to_move, move.from()), undo);
}

void Position::make_move(Move move, PieceType moved_piece, UndoState& undo) {
    const MoveFlag flag = move.flag();
    const PieceType captured_piece =
        (is_capture(flag) && flag != MoveFlag::EnPassant)
            ? piece_type_on_occupied(opposite(side_to_move), move.to())
            : PieceType::None;
    make_move(move, moved_piece, captured_piece, undo);
}

void Position::make_move(
    Move move,
    PieceType moved_piece,
    PieceType captured_piece,
    UndoState& undo
) {
    const MoveFlag flag = move.flag();
    const bool capture = is_capture(flag);
    const Color enemy = opposite(side_to_move);

    undo.castling_rights = pack_castling_rights(*this);
    undo.en_passant_square = en_passant_square;
    undo.halfmove_clock = halfmove_clock;
    undo.fullmove_number = fullmove_number;
    undo.eval_score = eval_score;
    undo.zobrist_key = zobrist_key;
    undo.occupancies = occupancies;
    undo.king_squares = king_squares;
    undo.king_checkers = king_checkers;
    undo.king_pinned = king_pinned;
    undo.king_block_masks = king_block_masks;
    undo.moved_piece = moved_piece;
    undo.captured_piece = PieceType::None;
    undo.captured_square = NoSquare;

    if (capture) {
        if (flag == MoveFlag::EnPassant) {
            undo.captured_piece = PieceType::Pawn;
            undo.captured_square = static_cast<std::int8_t>(
                side_to_move == Color::White ? move.to() - 8 : move.to() + 8);
        } else {
            if (captured_piece == PieceType::None) {
                captured_piece = piece_type_on_occupied(enemy, move.to());
            }
            undo.captured_piece = captured_piece;
            undo.captured_square = static_cast<std::int8_t>(move.to());
        }
    }

    make_move(move, moved_piece, captured_piece);
}

void Position::unmake_move(Move move, const UndoState& undo) {
    assert(undo.moved_piece != PieceType::None);

    const Color enemy = side_to_move;
    const Color color = opposite(enemy);
    const int color_idx = static_cast<int>(color);
    const int enemy_idx = static_cast<int>(enemy);
    const Square from = move.from();
    const Square to = move.to();
    const MoveFlag flag = move.flag();
    PieceType placed_piece = promotion_piece(move);
    if (placed_piece == PieceType::None) {
        placed_piece = undo.moved_piece;
    }
    const auto clear_raw = [this](int color_index, PieceType piece, Square square) {
        assert(piece != PieceType::None);
        const int piece_index = static_cast<int>(piece);
        const Bitboard square_mask = bit(square);
        assert((pieces[color_index][piece_index] & square_mask) != EmptyBB);
        pieces[color_index][piece_index] &= ~square_mask;
        board[square] = 0;
    };
    const auto set_raw = [this](int color_index, Color piece_color, PieceType piece, Square square) {
        assert(piece != PieceType::None);
        const int piece_index = static_cast<int>(piece);
        const Bitboard square_mask = bit(square);
        assert(board[square] == 0);
        pieces[color_index][piece_index] |= square_mask;
        board[square] = encode_piece_for_move(piece_color, piece);
    };

    if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        clear_raw(color_idx, PieceType::King, to);
        set_raw(color_idx, color, PieceType::King, from);

        const bool king_side = flag == MoveFlag::KingCastle;
        const Square rook_from = color == Color::White
            ? make_square(king_side ? 7 : 0, 0)
            : make_square(king_side ? 7 : 0, 7);
        const Square rook_to = color == Color::White
            ? make_square(king_side ? 5 : 3, 0)
            : make_square(king_side ? 5 : 3, 7);
        clear_raw(color_idx, PieceType::Rook, rook_to);
        set_raw(color_idx, color, PieceType::Rook, rook_from);
    } else {
        clear_raw(color_idx, placed_piece, to);
        set_raw(color_idx, color, undo.moved_piece, from);

        if (undo.captured_piece != PieceType::None) {
            assert(undo.captured_square != NoSquare);
            set_raw(enemy_idx, enemy, undo.captured_piece, static_cast<Square>(undo.captured_square));
        }
    }

    side_to_move = color;
    restore_castling_rights(*this, undo.castling_rights);
    en_passant_square = undo.en_passant_square;
    halfmove_clock = undo.halfmove_clock;
    fullmove_number = undo.fullmove_number;
    eval_score = undo.eval_score;
    zobrist_key = undo.zobrist_key;
    occupancies = undo.occupancies;
    king_squares = undo.king_squares;
    king_checkers = undo.king_checkers;
    king_pinned = undo.king_pinned;
    king_block_masks = undo.king_block_masks;
}

void generate_legal_moves(const Position& pos, MoveList& legal_moves) {
    const Color us = pos.side_to_move;
    MoveList pseudo_moves;
    generate_pseudo_legal_moves(pos, pseudo_moves);

    for (Move move : pseudo_moves) {
        Position next = pos;
        next.make_move(move);

        if (!in_check(next, us)) {
            legal_moves.push_back(move);
        }
    }
}

std::vector<Move> generate_legal_moves(const Position& pos) {
    MoveList legal_moves;
    generate_legal_moves(pos, legal_moves);
    return to_vector(legal_moves);
}

bool on_promotion_rank(Square square) {
    return rank_of(square) == static_cast<int>(Rank::R1) || rank_of(square) == static_cast<int>(Rank::R8);
}

} // namespace chess
