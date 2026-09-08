#include "move.hpp"
#include "position.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

using namespace chess;

namespace {

bool contains_move(const std::vector<Move>& moves, Move move) {
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

}

int main() {
    {
        Position pos;
        pos.set_startpos();

        const std::vector<Move> moves = generate_king_moves(pos);
        assert(moves.empty());
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_king_moves(pos);

        assert(moves.size() == 5);
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(3, 0))));
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(5, 0))));
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(3, 1))));
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(4, 1))));
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(5, 1))));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(5, 1));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_king_moves(pos);

        assert(!contains_move(moves, make_move(make_square(4, 0), make_square(4, 1))));
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(5, 1), MoveFlag::Capture)));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.side_to_move = Color::Black;

        const std::vector<Move> moves = generate_king_moves(pos);

        assert(moves.size() == 5);
        assert(contains_move(moves, make_move(make_square(4, 7), make_square(3, 7))));
        assert(contains_move(moves, make_move(make_square(4, 7), make_square(5, 7))));
        assert(contains_move(moves, make_move(make_square(4, 7), make_square(3, 6))));
        assert(contains_move(moves, make_move(make_square(4, 7), make_square(4, 6))));
        assert(contains_move(moves, make_move(make_square(4, 7), make_square(5, 6))));
    }
}
