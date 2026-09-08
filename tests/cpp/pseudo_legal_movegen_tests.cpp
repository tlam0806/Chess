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

std::size_t total_piece_generator_count(const Position& pos) {
    return generate_pawn_moves(pos).size()
         + generate_knight_moves(pos).size()
         + generate_bishop_moves(pos).size()
         + generate_rook_moves(pos).size()
         + generate_queen_moves(pos).size()
         + generate_king_moves(pos).size();
}

}

int main() {
    {
        Position pos;
        pos.set_startpos();

        const std::vector<Move> moves = generate_pseudo_legal_moves(pos);

        assert(moves.size() == 20);
        assert(moves.size() == total_piece_generator_count(pos));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush)));
        assert(contains_move(moves, make_move(make_square(6, 0), make_square(5, 2))));
    }

    {
        Position pos;
        pos.set_startpos();
        pos.side_to_move = Color::Black;

        const std::vector<Move> moves = generate_pseudo_legal_moves(pos);

        assert(moves.size() == 20);
        assert(moves.size() == total_piece_generator_count(pos));
        assert(contains_move(moves, make_move(make_square(4, 6), make_square(4, 4), MoveFlag::DoublePawnPush)));
        assert(contains_move(moves, make_move(make_square(6, 7), make_square(5, 5))));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::White, PieceType::Queen, make_square(3, 3));
        pos.set_piece(Color::White, PieceType::Rook, make_square(0, 0));
        pos.set_piece(Color::White, PieceType::Bishop, make_square(2, 2));
        pos.set_piece(Color::White, PieceType::Knight, make_square(5, 2));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(6, 4));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_pseudo_legal_moves(pos);

        assert(moves.size() == total_piece_generator_count(pos));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush)));
        assert(contains_move(moves, make_move(make_square(3, 3), make_square(3, 7))));
        assert(contains_move(moves, make_move(make_square(0, 0), make_square(0, 7))));
        assert(contains_move(moves, make_move(make_square(2, 2), make_square(0, 4))));
        assert(contains_move(moves, make_move(make_square(5, 2), make_square(7, 3))));
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(5, 0))));
    }
}
