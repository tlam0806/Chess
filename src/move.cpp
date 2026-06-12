#include "move.hpp"
#include "attacks.hpp"
#include "zobrist.hpp"

#include <cassert>
#include <string>

namespace chess {

namespace {

bool has_rook_on(const Position& pos, Color color, Square square) {
    const Bitboard rooks = pos.pieces[static_cast<int>(color)][static_cast<int>(PieceType::Rook)];
    return (rooks & bit(square)) != EmptyBB;
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

void add_castling_moves(const Position& pos, std::vector<Move>& moves, Square king_square) {
    const Color color = pos.side_to_move;
    const Color enemy = opposite(color);
    const Bitboard occupancy = pos.occupancy();

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
            moves.push_back(make_move(e1, g1, MoveFlag::KingCastle));
        }

        if (king_square == e1 && pos.white_can_castle_queenside && has_rook_on(pos, color, a1)
            && !(occupancy & (bit(d1) | bit(c1) | bit(b1)))
            && !is_square_attacked(pos, e1, enemy)
            && !is_square_attacked(pos, d1, enemy)
            && !is_square_attacked(pos, c1, enemy)) {
            moves.push_back(make_move(e1, c1, MoveFlag::QueenCastle));
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
            moves.push_back(make_move(e8, g8, MoveFlag::KingCastle));
        }

        if (king_square == e8 && pos.black_can_castle_queenside && has_rook_on(pos, color, a8)
            && !(occupancy & (bit(d8) | bit(c8) | bit(b8)))
            && !is_square_attacked(pos, e8, enemy)
            && !is_square_attacked(pos, d8, enemy)
            && !is_square_attacked(pos, c8, enemy)) {
            moves.push_back(make_move(e8, c8, MoveFlag::QueenCastle));
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

std::vector<Move> generate_knight_moves(const Position& pos) {
    std::vector<Move> moves;
    Bitboard positions = pos.pieces[static_cast<int>(pos.side_to_move)][static_cast<int>(PieceType::Knight)];
    const Color op_side = opposite(pos.side_to_move);
    const Bitboard op_occupancy = pos.occupancy(op_side);
    const Bitboard op_king = pos.pieces[static_cast<int>(op_side)][static_cast<int>(PieceType::King)];
    const Bitboard capturable_occupancy = op_occupancy & ~op_king;
    const Bitboard my_occupancy = pos.occupancy(pos.side_to_move);
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = knight_attacks(from) & ~my_occupancy & ~op_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            moves.push_back(make_move(from, to, (capturable_occupancy & bit(to)) ? MoveFlag::Capture : MoveFlag::Quiet));
        }
    }
    return moves;
}

std::vector<Move> generate_king_moves(const Position& pos) {
    std::vector<Move> moves;
    Bitboard positions = pos.pieces[static_cast<int>(pos.side_to_move)][static_cast<int>(PieceType::King)];
    const Color op_side = opposite(pos.side_to_move);
    const Bitboard op_occupancy = pos.occupancy(op_side);
    const Bitboard op_king = pos.pieces[static_cast<int>(op_side)][static_cast<int>(PieceType::King)];
    const Bitboard capturable_occupancy = op_occupancy & ~op_king;
    const Bitboard my_occupancy = pos.occupancy(pos.side_to_move);
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = king_attacks(from) & ~my_occupancy & ~op_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            moves.push_back(make_move(from, to, (capturable_occupancy & bit(to)) ? MoveFlag::Capture : MoveFlag::Quiet));
        }
        add_castling_moves(pos, moves, from);
    }
    return moves;
}

std::vector<Move> generate_bishop_moves(const Position& pos) {
    std::vector<Move> moves;
    Bitboard positions = pos.pieces[static_cast<int>(pos.side_to_move)][static_cast<int>(PieceType::Bishop)];
    const Color op_side = opposite(pos.side_to_move);
    const Bitboard op_occupancy = pos.occupancy(op_side);
    const Bitboard op_king = pos.pieces[static_cast<int>(op_side)][static_cast<int>(PieceType::King)];
    const Bitboard capturable_occupancy = op_occupancy & ~op_king;
    const Bitboard my_occupancy = pos.occupancy(pos.side_to_move);
    const Bitboard occupancy = op_occupancy | my_occupancy;
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = bishop_attacks(from, occupancy) & ~my_occupancy & ~op_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            moves.push_back(make_move(from, to, (capturable_occupancy & bit(to)) ? MoveFlag::Capture : MoveFlag::Quiet));
        }
    }
    return moves;
}

std::vector<Move> generate_rook_moves(const Position& pos) {
    std::vector<Move> moves;
    Bitboard positions = pos.pieces[static_cast<int>(pos.side_to_move)][static_cast<int>(PieceType::Rook)];
    const Color op_side = opposite(pos.side_to_move);
    const Bitboard op_occupancy = pos.occupancy(op_side);
    const Bitboard op_king = pos.pieces[static_cast<int>(op_side)][static_cast<int>(PieceType::King)];
    const Bitboard capturable_occupancy = op_occupancy & ~op_king;
    const Bitboard my_occupancy = pos.occupancy(pos.side_to_move);
    const Bitboard occupancy = op_occupancy | my_occupancy;
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = rook_attacks(from, occupancy) & ~my_occupancy & ~op_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            moves.push_back(make_move(from, to, (capturable_occupancy & bit(to)) ? MoveFlag::Capture : MoveFlag::Quiet));
        }
    }
    return moves;
}

std::vector<Move> generate_queen_moves(const Position& pos) {
    std::vector<Move> moves;
    Bitboard positions = pos.pieces[static_cast<int>(pos.side_to_move)][static_cast<int>(PieceType::Queen)];
    const Color op_side = opposite(pos.side_to_move);
    const Bitboard op_occupancy = pos.occupancy(op_side);
    const Bitboard op_king = pos.pieces[static_cast<int>(op_side)][static_cast<int>(PieceType::King)];
    const Bitboard capturable_occupancy = op_occupancy & ~op_king;
    const Bitboard my_occupancy = pos.occupancy(pos.side_to_move);
    const Bitboard occupancy = op_occupancy | my_occupancy;
    while (positions) {
        Square from = pop_lsb(positions);
        Bitboard to_mask = queen_attacks(from, occupancy) & ~my_occupancy & ~op_king;
        while (to_mask) {
            Square to = pop_lsb(to_mask);
            moves.push_back(make_move(from, to, (capturable_occupancy & bit(to)) ? MoveFlag::Capture : MoveFlag::Quiet));
        }
    }
    return moves;
}

std::vector<Move> generate_pawn_moves(const Position& pos) {
    std::vector<Move> moves;
    Bitboard positions = pos.pieces[static_cast<int>(pos.side_to_move)][static_cast<int>(PieceType::Pawn)];
    const Color op_side = opposite(pos.side_to_move);
    const Bitboard op_occupancy = pos.occupancy(op_side);
    const Bitboard op_king = pos.pieces[static_cast<int>(op_side)][static_cast<int>(PieceType::King)];
    const Bitboard capturable_occupancy = op_occupancy & ~op_king;
    const Bitboard my_occupancy = pos.occupancy(pos.side_to_move);
    const Bitboard occupancy = op_occupancy | my_occupancy;
    int starting_rank = pos.side_to_move == Color::White ? static_cast<int>(Rank::R2) : static_cast<int>(Rank::R7);
    int promotion_rank = pos.side_to_move == Color::White ? static_cast<int>(Rank::R8) : static_cast<int>(Rank::R1);
    int offset = pos.side_to_move == Color::White ? 1 : -1;
    while (positions) {
        Square from = pop_lsb(positions);
        int rank = rank_of(from);
        int file = file_of(from);
        assert(rank != promotion_rank);
        Bitboard target = pawn_attacks(pos.side_to_move, from);
        if (rank + offset == promotion_rank) {
            while (target) {
                Square to = pop_lsb(target);
                if (capturable_occupancy & bit(to)) {
                    moves.push_back(make_move(from, to, MoveFlag::QueenPromotionCapture));
                    moves.push_back(make_move(from, to, MoveFlag::KnightPromotionCapture));
                    moves.push_back(make_move(from, to, MoveFlag::BishopPromotionCapture));
                    moves.push_back(make_move(from, to, MoveFlag::RookPromotionCapture));
                }
            }
            Square to = make_square(file, rank + offset);
            if (!(occupancy & bit(to))) {
                moves.push_back(make_move(from, to, MoveFlag::QueenPromotion));
                moves.push_back(make_move(from, to, MoveFlag::KnightPromotion));
                moves.push_back(make_move(from, to, MoveFlag::BishopPromotion));
                moves.push_back(make_move(from, to, MoveFlag::RookPromotion));
            }
        } else {
            while (target) {
                Square to = pop_lsb(target);
                if (capturable_occupancy & bit(to)) {
                    moves.push_back(make_move(from, to, MoveFlag::Capture));
                } else if (to == pos.en_passant_square) {
                    moves.push_back(make_move(from, to, MoveFlag::EnPassant));
                }
            }
            Square to = make_square(file, rank + offset);
            if (!(occupancy & bit(to))) {
                moves.push_back(make_move(from, to, MoveFlag::Quiet));
                if (rank == starting_rank) {
                    Square double_to = make_square(file, rank + offset + offset);
                    if (!(occupancy & bit(double_to))) {
                        moves.push_back(make_move(from, double_to, MoveFlag::DoublePawnPush));
                    }
                }
            }
        }
    }
    return moves;
}

std::vector<Move> generate_pseudo_legal_moves(const Position& pos) {
    std::vector<Move> moves;
    auto append = [&moves](std::vector<Move> part) {
        moves.insert(moves.end(), part.begin(), part.end());
    };
    append(generate_pawn_moves(pos));
    append(generate_knight_moves(pos));
    append(generate_bishop_moves(pos));
    append(generate_rook_moves(pos));
    append(generate_queen_moves(pos));
    append(generate_king_moves(pos));
    return moves;
}

void Position::make_move(Move move) {
    const Square from = move.from();
    const Square to = move.to();
    const Color color = color_on_occupied(from);
    const MoveFlag flag = move.flag();
    assert(side_to_move == color);
    PieceType moving_piece = promotion_piece(move);
    if (moving_piece == PieceType::None) moving_piece = piece_type_on_occupied(from);
    const bool pawn_move = piece_type_on_occupied(from) == PieceType::Pawn;
    const bool capture = is_capture(flag);

    if (moving_piece == PieceType::King) {
        remove_castling_rights_for_king(*this, color);
    } else if (moving_piece == PieceType::Rook) {
        remove_castling_right_for_rook_square(*this, from);
    }

    if (is_capture(flag) && flag != MoveFlag::EnPassant) {
        remove_castling_right_for_rook_square(*this, to);
    }

    clear_square(from);

  
    if (flag == MoveFlag::EnPassant) {
        assert(to == en_passant_square);
        Square captured_square = color == Color::White ? to - 8 : to + 8;
        clear_square(captured_square);
    } else if (flag == MoveFlag::KingCastle) {
        const Square rook_from = color == Color::White ? make_square(7, 0) : make_square(7, 7);
        const Square rook_to = color == Color::White ? make_square(5, 0) : make_square(5, 7);
        clear_square(rook_from);
        set_piece(color, PieceType::Rook, rook_to);
    } else if (flag == MoveFlag::QueenCastle) {
        const Square rook_from = color == Color::White ? make_square(0, 0) : make_square(0, 7);
        const Square rook_to = color == Color::White ? make_square(3, 0) : make_square(3, 7);
        clear_square(rook_from);
        set_piece(color, PieceType::Rook, rook_to);
    } else if (is_capture(flag)) {
        clear_square(to);
    }


    set_piece(color, moving_piece, to);

    if (en_passant_square != NoSquare) {
        zobrist_key ^= zobrist::en_passant_file_key(file_of(en_passant_square));
    }
    en_passant_square = flag == MoveFlag::DoublePawnPush ? color == Color::White ? to - 8 : to + 8 
                                                         : NoSquare;
    if (en_passant_square != NoSquare) {
        zobrist_key ^= zobrist::en_passant_file_key(file_of(en_passant_square));
    }

    halfmove_clock = (pawn_move || capture) ? 0 : halfmove_clock + 1;
    if (color == Color::Black) {
        ++fullmove_number;
    }
    zobrist_key ^= zobrist::side_key();
    side_to_move = opposite(side_to_move);
}

std::vector<Move> generate_legal_moves(const Position& pos) {
    std::vector<Move> legal_moves;
    const Color us = pos.side_to_move;
    const std::vector<Move> pseudo_moves = generate_pseudo_legal_moves(pos);

    for (Move move : pseudo_moves) {
        Position next = pos;
        next.make_move(move);

        if (!in_check(next, us)) {
            legal_moves.push_back(move);
        }
    }
    return legal_moves;
}

bool on_promotion_rank(Square square) {
    return rank_of(square) == static_cast<int>(Rank::R1) || rank_of(square) == static_cast<int>(Rank::R8);
}

} // namespace chess
