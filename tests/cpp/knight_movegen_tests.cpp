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

        const std::vector<Move> moves = generate_knight_moves(pos);

        assert(moves.size() == 4);
        assert(contains_move(moves, make_move(make_square(1, 0), make_square(0, 2))));
        assert(contains_move(moves, make_move(make_square(1, 0), make_square(2, 2))));
        assert(contains_move(moves, make_move(make_square(6, 0), make_square(5, 2))));
        assert(contains_move(moves, make_move(make_square(6, 0), make_square(7, 2))));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Knight, make_square(3, 3));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 5));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(5, 4));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_knight_moves(pos);

        assert(!contains_move(moves, make_move(make_square(3, 3), make_square(4, 5))));
        assert(contains_move(moves, make_move(make_square(3, 3), make_square(5, 4), MoveFlag::Capture)));
    }

    {
        Position pos;
        pos.set_startpos();
        pos.side_to_move = Color::Black;

        const std::vector<Move> moves = generate_knight_moves(pos);

        assert(moves.size() == 4);
        assert(contains_move(moves, make_move(make_square(1, 7), make_square(0, 5))));
        assert(contains_move(moves, make_move(make_square(1, 7), make_square(2, 5))));
        assert(contains_move(moves, make_move(make_square(6, 7), make_square(5, 5))));
        assert(contains_move(moves, make_move(make_square(6, 7), make_square(7, 5))));
    }
}
