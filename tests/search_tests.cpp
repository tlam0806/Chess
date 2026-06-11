#include "search.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

using namespace chess;

namespace {

int reference_negamax(Position pos, int depth, int ply = 0) {
    const std::vector<Move> moves = generate_legal_moves(pos);

    if (moves.empty()) {
        return in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0;
    }

    if (depth == 0) {
        return evaluate_for_side_to_move(pos);
    }

    int best_score = -Infinity;
    for (Move move : moves) {
        Position next = pos;
        next.make_move(move);
        best_score = std::max(best_score, -reference_negamax(next, depth - 1, ply + 1));
    }

    return best_score;
}

int root_score_from_best_move(Position pos, Move move, int depth) {
    Position next = pos;
    next.make_move(move);
    return -negamax(next, depth - 1);
}

void assert_search_matches_reference(Position pos, int max_depth) {
    for (int depth = 0; depth <= max_depth; ++depth) {
        const int expected = reference_negamax(pos, depth);
        assert(negamax(pos, depth) == expected);
        assert(negamax(pos, depth, -Infinity, Infinity) == expected);

        const SearchResult result = search_best_move(pos, depth);
        assert(result.score == expected);

        if (depth > 0 && !generate_legal_moves(pos).empty()) {
            assert(root_score_from_best_move(pos, result.best_move, depth) == result.score);
        }
    }
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Queen, make_square(3, 0));
        pos.side_to_move = Color::White;

        assert(negamax(pos, 0) == evaluate_for_side_to_move(pos));
        assert(negamax(pos, 0, -Infinity, Infinity) == negamax(pos, 0));
        const SearchResult result = search_best_move(pos, 0);
        assert(result.score == evaluate_for_side_to_move(pos));
        assert(result.nodes == 1);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Rook, make_square(0, 0));
        pos.set_piece(Color::Black, PieceType::Queen, make_square(0, 7));
        pos.side_to_move = Color::White;

        const SearchResult result = search_best_move(pos, 1);

        assert(result.best_move == make_move(make_square(0, 0), make_square(0, 7), MoveFlag::Capture));
        assert(result.score > 300);
        assert(negamax(pos, 1, -Infinity, Infinity) == negamax(pos, 1));
        assert(negamax(pos, 2, -Infinity, Infinity) == negamax(pos, 2));
        assert(result.nodes > 0);
    }

    {
        Position pos;
        assert(pos.set_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"));

        assert(generate_legal_moves(pos).empty());
        assert(!in_check(pos, pos.side_to_move));
        assert(negamax(pos, 1) == 0);

        const SearchResult result = search_best_move(pos, 3);
        assert(result.score == 0);
        assert(result.nodes == 1);
    }

    {
        Position pos;
        assert(pos.set_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));

        assert(generate_legal_moves(pos).empty());
        assert(in_check(pos, pos.side_to_move));
        assert(negamax(pos, 1) == -CheckmateScore);

        const SearchResult result = search_best_move(pos, 3);
        assert(result.score == -CheckmateScore);
        assert(result.nodes == 1);
    }

    {
        Position pos;
        assert(pos.set_fen("7k/5K2/8/8/8/8/1Q6/8 w - - 0 1"));

        const SearchResult result = search_best_move(pos, 1);

        assert(result.score == CheckmateScore - 1);
        Position next = pos;
        next.make_move(result.best_move);
        assert(generate_legal_moves(next).empty());
        assert(in_check(next, next.side_to_move));
    }

    {
        Position pos;
        pos.set_startpos();
        assert_search_matches_reference(pos, 3);
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));
        assert_search_matches_reference(pos, 2);
    }

    {
        Position pos;
        assert(pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"));
        assert_search_matches_reference(pos, 3);
    }

    {
        Position pos;
        assert(pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"));
        assert_search_matches_reference(pos, 2);
    }
}
