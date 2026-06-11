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

} // namespace

int main() {
    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::White, PieceType::Rook, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_legal_moves(pos);

        assert(!contains_move(moves, make_move(make_square(4, 1), make_square(3, 1))));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 7), MoveFlag::Capture)));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_legal_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 0), make_square(3, 0))));
        assert(!contains_move(moves, make_move(make_square(4, 0), make_square(4, 1))));
    }
}
