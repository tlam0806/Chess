#include "evaluate.hpp"
#include "evaluation_terms.hpp"
#include "move.hpp"

#include <cassert>

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

} // namespace

int main() {
    {
        assert(relative_square(Color::White, make_square(4, 1)) == make_square(4, 1));
        assert(relative_square(Color::Black, make_square(4, 1)) == make_square(4, 6));
        assert(relative_square(Color::Black, make_square(0, 0)) == make_square(0, 7));
        assert(relative_square(Color::Black, make_square(7, 7)) == make_square(7, 0));
    }

    {
        Position pos;
        pos.set_startpos();

        assert_incremental_eval_matches(pos);
        assert(evaluate(pos) == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Queen, make_square(3, 0));

        assert_incremental_eval_matches(pos);
        assert(evaluate(pos) > 800);
        pos.side_to_move = Color::White;
        assert(evaluate_for_side_to_move(pos) == evaluate(pos));
        pos.side_to_move = Color::Black;
        assert(evaluate_for_side_to_move(pos) == -evaluate(pos));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(0, 7));

        assert_incremental_eval_matches(pos);
        assert(evaluate(pos) < -400);
    }

    {
        Position corner_knight;
        corner_knight.set_piece(Color::White, PieceType::King, make_square(4, 0));
        corner_knight.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        corner_knight.set_piece(Color::White, PieceType::Knight, make_square(0, 0));

        Position center_knight;
        center_knight.set_piece(Color::White, PieceType::King, make_square(4, 0));
        center_knight.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        center_knight.set_piece(Color::White, PieceType::Knight, make_square(3, 3));

        assert_incremental_eval_matches(corner_knight);
        assert_incremental_eval_matches(center_knight);
        assert(evaluate(center_knight) > evaluate(corner_knight));
    }

    {
        Position white_extra_pawn;
        white_extra_pawn.set_piece(Color::White, PieceType::King, make_square(4, 0));
        white_extra_pawn.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        white_extra_pawn.set_piece(Color::White, PieceType::Pawn, make_square(4, 3));

        Position black_extra_pawn;
        black_extra_pawn.set_piece(Color::White, PieceType::King, make_square(4, 0));
        black_extra_pawn.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        black_extra_pawn.set_piece(Color::Black, PieceType::Pawn, make_square(4, 4));

        assert_incremental_eval_matches(white_extra_pawn);
        assert_incremental_eval_matches(black_extra_pawn);
        assert(evaluate(white_extra_pawn) == -evaluate(black_extra_pawn));
    }

    {
        const char* fens[] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/pppq1ppp/2npbn2/3Np3/2B1P3/2N2Q2/PPP2PPP/R3K2R w KQkq - 0 1",
            "4k3/4P3/8/8/8/8/4p3/4K3 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "8/4p1kn/p2pP2p/3Pb3/2P4K/8/1r6/7q w - - 0 1"
        };

        for (const char* fen : fens) {
            Position pos;
            assert(pos.set_fen(fen));
            assert_incremental_eval_matches(pos);

            MoveList moves;
            generate_legal_moves(pos, moves);
            for (Move move : moves) {
                Position next = pos;
                next.make_move(move);
                assert_incremental_eval_matches(next);
            }
        }
    }
}
