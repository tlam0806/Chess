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
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 2))));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush)));
        assert(moves.size() == 2);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(4, 2));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(moves.empty());
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(0, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.set_piece(Color::White, PieceType::Knight, make_square(4, 2));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(moves.empty());
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(3, 2));
        pos.set_piece(Color::Black, PieceType::Bishop, make_square(5, 2));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 1), make_square(3, 2), MoveFlag::Capture)));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(5, 2), MoveFlag::Capture)));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 2))));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush)));
        assert(moves.size() == 4);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(0, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(4, 6));
        pos.side_to_move = Color::Black;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 6), make_square(4, 5))));
        assert(contains_move(moves, make_move(make_square(4, 6), make_square(4, 4), MoveFlag::DoublePawnPush)));
        assert(moves.size() == 2);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 6), make_square(4, 7), MoveFlag::QueenPromotion)));
        assert(contains_move(moves, make_move(make_square(4, 6), make_square(4, 7), MoveFlag::KnightPromotion)));
        assert(contains_move(moves, make_move(make_square(4, 6), make_square(4, 7), MoveFlag::BishopPromotion)));
        assert(contains_move(moves, make_move(make_square(4, 6), make_square(4, 7), MoveFlag::RookPromotion)));
        assert(moves.size() == 4);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(5, 7));
        pos.side_to_move = Color::White;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 6), make_square(5, 7), MoveFlag::QueenPromotionCapture)));
        assert(contains_move(moves, make_move(make_square(4, 6), make_square(5, 7), MoveFlag::KnightPromotionCapture)));
        assert(contains_move(moves, make_move(make_square(4, 6), make_square(5, 7), MoveFlag::BishopPromotionCapture)));
        assert(contains_move(moves, make_move(make_square(4, 6), make_square(5, 7), MoveFlag::RookPromotionCapture)));
        assert(moves.size() == 8);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(0, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(4, 1));
        pos.side_to_move = Color::Black;

        const std::vector<Move> moves = generate_pawn_moves(pos);

        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 0), MoveFlag::QueenPromotion)));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 0), MoveFlag::KnightPromotion)));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 0), MoveFlag::BishopPromotion)));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 0), MoveFlag::RookPromotion)));
        assert(moves.size() == 4);
    }
}
