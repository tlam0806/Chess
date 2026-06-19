#include "move.hpp"
#include "position.hpp"
#include "evaluate.hpp"
#include "evaluation_terms.hpp"
#include "king_safety.hpp"

#include <cassert>
#include <iostream>
#include <random>

using namespace chess;

namespace {

int recompute_evaluate(const Position& pos) {
    int score = 0;
    for (Color color : {Color::White, Color::Black}) {
        for (PieceType piece :
             {PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
              PieceType::Rook, PieceType::Queen, PieceType::King}) {
            Bitboard pieces = pos.pieces[static_cast<int>(color)][static_cast<int>(piece)];
            while (pieces != EmptyBB) {
                score += evaluation_piece_contribution(color, piece, pop_lsb(pieces));
            }
        }
    }
    return score;
}

void assert_incremental_eval_matches(const Position& pos) {
    assert(evaluate(pos) == recompute_evaluate(pos));
}

void assert_king_safety_cache_matches_rebuild(const Position& pos) {
    for (Color color : {Color::White, Color::Black}) {
        const int color_idx = static_cast<int>(color);
        if (popcount(pos.pieces[color_idx][static_cast<int>(PieceType::King)]) != 1) {
            continue;
        }
        const KingSafetyContext rebuilt = make_king_safety_context(pos, color);
        if (pos.king_checkers[color_idx] != rebuilt.checkers
            || pos.king_pinned[color_idx] != rebuilt.pinned
            || pos.king_block_masks[color_idx] != rebuilt.block_mask
            || pos.king_squares[color_idx] != rebuilt.king_square) {
            std::cerr << "king safety mismatch color=" << color_idx
                      << " cached_king=" << pos.king_squares[color_idx]
                      << " rebuilt_king=" << rebuilt.king_square
                      << " cached_checkers=" << pos.king_checkers[color_idx]
                      << " rebuilt_checkers=" << rebuilt.checkers
                      << " cached_pinned=" << pos.king_pinned[color_idx]
                      << " rebuilt_pinned=" << rebuilt.pinned
                      << " cached_block=" << pos.king_block_masks[color_idx]
                      << " rebuilt_block=" << rebuilt.block_mask
                      << '\n';
            pos.print(std::cerr);
        }
        assert(pos.king_squares[color_idx] == rebuilt.king_square);
        assert(pos.king_checkers[color_idx] == rebuilt.checkers);
        assert(pos.king_pinned[color_idx] == rebuilt.pinned);
        assert(pos.king_block_masks[color_idx] == rebuilt.block_mask);
    }
}

void assert_same_position(const Position& lhs, const Position& rhs) {
    assert(lhs.pieces == rhs.pieces);
    assert(lhs.occupancies == rhs.occupancies);
    assert(lhs.board == rhs.board);
    assert(lhs.side_to_move == rhs.side_to_move);
    assert(lhs.white_can_castle_kingside == rhs.white_can_castle_kingside);
    assert(lhs.white_can_castle_queenside == rhs.white_can_castle_queenside);
    assert(lhs.black_can_castle_kingside == rhs.black_can_castle_kingside);
    assert(lhs.black_can_castle_queenside == rhs.black_can_castle_queenside);
    assert(lhs.en_passant_square == rhs.en_passant_square);
    assert(lhs.halfmove_clock == rhs.halfmove_clock);
    assert(lhs.fullmove_number == rhs.fullmove_number);
    assert(lhs.eval_score == rhs.eval_score);
    assert(lhs.zobrist_key == rhs.zobrist_key);
    assert(lhs.king_squares == rhs.king_squares);
    assert(lhs.king_checkers == rhs.king_checkers);
    assert(lhs.king_pinned == rhs.king_pinned);
    assert(lhs.king_block_masks == rhs.king_block_masks);
}

void assert_make_unmake_round_trip(Position pos, Move move) {
    refresh_king_safety(pos);
    assert_king_safety_cache_matches_rebuild(pos);
    const Position before = pos;
    UndoState undo;
    pos.make_move(move, undo);
    assert_king_safety_cache_matches_rebuild(pos);
    pos.unmake_move(move, undo);
    assert_king_safety_cache_matches_rebuild(pos);
    assert_same_position(pos, before);
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Knight, make_square(6, 0));
        pos.side_to_move = Color::White;
        pos.en_passant_square = make_square(0, 2);
        pos.halfmove_clock = 4;
        pos.fullmove_number = 7;

        pos.make_move(make_move(make_square(6, 0), make_square(5, 2)));

        assert_incremental_eval_matches(pos);
        assert(pos.is_empty(make_square(6, 0)));
        assert(pos.color_on_occupied(make_square(5, 2)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(5, 2)) == PieceType::Knight);
        assert(pos.side_to_move == Color::Black);
        assert(pos.en_passant_square == NoSquare);
        assert(pos.halfmove_clock == 5);
        assert(pos.fullmove_number == 7);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Knight, make_square(3, 3));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(5, 4));
        pos.side_to_move = Color::White;
        pos.halfmove_clock = 4;

        pos.make_move(make_move(make_square(3, 3), make_square(5, 4), MoveFlag::Capture));

        assert_incremental_eval_matches(pos);
        assert(pos.is_empty(make_square(3, 3)));
        assert(pos.color_on_occupied(make_square(5, 4)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(5, 4)) == PieceType::Knight);
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.side_to_move = Color::White;
        pos.halfmove_clock = 8;

        pos.make_move(make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush));

        assert_incremental_eval_matches(pos);
        assert(pos.is_empty(make_square(4, 1)));
        assert(pos.color_on_occupied(make_square(4, 3)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(4, 3)) == PieceType::Pawn);
        assert(pos.en_passant_square == make_square(4, 2));
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 4));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.side_to_move = Color::White;
        pos.en_passant_square = make_square(3, 5);
        pos.halfmove_clock = 8;

        pos.make_move(make_move(make_square(4, 4), make_square(3, 5), MoveFlag::EnPassant));

        assert_incremental_eval_matches(pos);
        assert(pos.is_empty(make_square(4, 4)));
        assert(pos.is_empty(make_square(3, 4)));
        assert(pos.color_on_occupied(make_square(3, 5)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(3, 5)) == PieceType::Pawn);
        assert(pos.en_passant_square == NoSquare);
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.side_to_move = Color::White;
        pos.halfmove_clock = 8;

        pos.make_move(make_move(make_square(4, 6), make_square(4, 7), MoveFlag::QueenPromotion));

        assert_incremental_eval_matches(pos);
        assert(pos.is_empty(make_square(4, 6)));
        assert(pos.color_on_occupied(make_square(4, 7)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(4, 7)) == PieceType::Queen);
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(5, 7));
        pos.side_to_move = Color::White;
        pos.halfmove_clock = 8;

        pos.make_move(make_move(make_square(4, 6), make_square(5, 7), MoveFlag::KnightPromotionCapture));

        assert_incremental_eval_matches(pos);
        assert(pos.is_empty(make_square(4, 6)));
        assert(pos.color_on_occupied(make_square(5, 7)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(5, 7)) == PieceType::Knight);
        assert(pos.side_to_move == Color::Black);
        assert(pos.halfmove_clock == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(6, 7));
        pos.side_to_move = Color::Black;
        pos.halfmove_clock = 10;
        pos.fullmove_number = 23;

        pos.make_move(make_move(make_square(6, 7), make_square(5, 5)));

        assert_incremental_eval_matches(pos);
        assert(pos.side_to_move == Color::White);
        assert(pos.halfmove_clock == 11);
        assert(pos.fullmove_number == 24);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Knight, make_square(6, 0));
        pos.side_to_move = Color::White;
        pos.en_passant_square = make_square(0, 2);
        pos.halfmove_clock = 4;
        pos.fullmove_number = 7;
        assert_make_unmake_round_trip(pos, make_move(make_square(6, 0), make_square(5, 2)));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Knight, make_square(3, 3));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(5, 4));
        pos.side_to_move = Color::White;
        assert_make_unmake_round_trip(
            pos,
            make_move(make_square(3, 3), make_square(5, 4), MoveFlag::Capture));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.side_to_move = Color::White;
        assert_make_unmake_round_trip(
            pos,
            make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 4));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.side_to_move = Color::White;
        pos.en_passant_square = make_square(3, 5);
        assert_make_unmake_round_trip(
            pos,
            make_move(make_square(4, 4), make_square(3, 5), MoveFlag::EnPassant));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.side_to_move = Color::White;
        assert_make_unmake_round_trip(
            pos,
            make_move(make_square(4, 6), make_square(4, 7), MoveFlag::QueenPromotion));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(5, 7));
        pos.side_to_move = Color::White;
        assert_make_unmake_round_trip(
            pos,
            make_move(make_square(4, 6), make_square(5, 7), MoveFlag::KnightPromotionCapture));
    }

    {
        Position pos;
        pos.set_startpos();
        pos.clear_square(make_square(5, 0));
        pos.clear_square(make_square(6, 0));
        pos.side_to_move = Color::White;
        assert_make_unmake_round_trip(
            pos,
            make_move(make_square(4, 0), make_square(6, 0), MoveFlag::KingCastle));
    }

    {
        Position pos;
        pos.set_startpos();
        std::mt19937 rng(20260619);
        for (int ply = 0; ply < 200; ++ply) {
            MoveList moves;
            generate_legal_moves(pos, moves);
            if (moves.empty()) {
                break;
            }
            std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
            const Move move = moves[dist(rng)];
            assert_make_unmake_round_trip(pos, move);
            pos.make_move(move);
        }
    }
}
