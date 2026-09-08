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
        assert(pos.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));

        const std::vector<Move> moves = generate_legal_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 0), make_square(6, 0), MoveFlag::KingCastle)));
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(2, 0), MoveFlag::QueenCastle)));
    }

    {
        Position pos;
        pos.set_startpos();

        const std::vector<Move> moves = generate_legal_moves(pos);

        assert(!contains_move(moves, make_move(make_square(4, 0), make_square(6, 0), MoveFlag::KingCastle)));
        assert(!contains_move(moves, make_move(make_square(4, 0), make_square(2, 0), MoveFlag::QueenCastle)));
    }

    {
        Position pos;
        assert(pos.set_fen("k4r2/8/8/8/8/8/8/4K2R w K - 0 1"));

        const std::vector<Move> moves = generate_legal_moves(pos);

        assert(!contains_move(moves, make_move(make_square(4, 0), make_square(6, 0), MoveFlag::KingCastle)));
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));

        pos.make_move(make_move(make_square(4, 0), make_square(6, 0), MoveFlag::KingCastle));

        assert(pos.is_empty(make_square(4, 0)));
        assert(pos.is_empty(make_square(7, 0)));
        assert(pos.color_on_occupied(make_square(6, 0)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(6, 0)) == PieceType::King);
        assert(pos.color_on_occupied(make_square(5, 0)) == Color::White);
        assert(pos.piece_type_on_occupied(make_square(5, 0)) == PieceType::Rook);
        assert(!pos.white_can_castle_kingside);
        assert(!pos.white_can_castle_queenside);
        assert(pos.black_can_castle_kingside);
        assert(pos.black_can_castle_queenside);
        assert(pos.side_to_move == Color::Black);
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"));

        pos.make_move(make_move(make_square(4, 7), make_square(2, 7), MoveFlag::QueenCastle));

        assert(pos.is_empty(make_square(4, 7)));
        assert(pos.is_empty(make_square(0, 7)));
        assert(pos.color_on_occupied(make_square(2, 7)) == Color::Black);
        assert(pos.piece_type_on_occupied(make_square(2, 7)) == PieceType::King);
        assert(pos.color_on_occupied(make_square(3, 7)) == Color::Black);
        assert(pos.piece_type_on_occupied(make_square(3, 7)) == PieceType::Rook);
        assert(pos.white_can_castle_kingside);
        assert(pos.white_can_castle_queenside);
        assert(!pos.black_can_castle_kingside);
        assert(!pos.black_can_castle_queenside);
        assert(pos.side_to_move == Color::White);
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));

        pos.make_move(make_move(make_square(7, 0), make_square(7, 1)));

        assert(!pos.white_can_castle_kingside);
        assert(pos.white_can_castle_queenside);
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));

        pos.make_move(make_move(make_square(0, 0), make_square(0, 7), MoveFlag::Capture));

        assert(!pos.white_can_castle_queenside);
        assert(!pos.black_can_castle_queenside);
        assert(pos.black_can_castle_kingside);
    }
}
