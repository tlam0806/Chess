#include "attacks.hpp"
#include "move.hpp"
#include "position.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

using namespace chess;

namespace {

std::string move_key(Move move) {
    return move_to_string(move) + "#" + std::to_string(static_cast<int>(move_flag(move)));
}

std::set<std::string> move_keys(const std::vector<Move>& moves) {
    std::set<std::string> keys;
    for (Move move : moves) {
        const bool inserted = keys.insert(move_key(move)).second;
        assert(inserted && "duplicate move generated");
    }
    return keys;
}

bool contains_move(const std::vector<Move>& moves, Move move) {
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

std::size_t count_flag(const std::vector<Move>& moves, MoveFlag flag) {
    return static_cast<std::size_t>(std::count_if(moves.begin(), moves.end(), [flag](Move move) {
        return move_flag(move) == flag;
    }));
}

std::size_t total_piece_generator_count(const Position& pos) {
    return generate_pawn_moves(pos).size()
         + generate_knight_moves(pos).size()
         + generate_bishop_moves(pos).size()
         + generate_rook_moves(pos).size()
         + generate_queen_moves(pos).size()
         + generate_king_moves(pos).size();
}

void assert_legal_moves_are_unique_and_safe(Position pos, std::size_t expected_count) {
    const Color us = pos.side_to_move;
    const std::vector<Move> legal_moves = generate_legal_moves(pos);
    assert(legal_moves.size() == expected_count);
    (void)move_keys(legal_moves);

    for (Move move : legal_moves) {
        Position next = pos;
        next.make_move(move);
        assert(!in_check(next, us));
    }
}

void assert_pseudo_matches_piece_generators(const Position& pos) {
    const std::vector<Move> pseudo_moves = generate_pseudo_legal_moves(pos);
    assert(pseudo_moves.size() == total_piece_generator_count(pos));
    (void)move_keys(pseudo_moves);
}

void assert_fen_legal_count(std::string_view fen, std::size_t expected_count) {
    Position pos;
    assert(pos.set_fen(fen));
    assert_pseudo_matches_piece_generators(pos);
    assert_legal_moves_are_unique_and_safe(pos, expected_count);
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_startpos();
        assert_pseudo_matches_piece_generators(pos);
        assert_legal_moves_are_unique_and_safe(pos, 20);
        const std::vector<Move> moves = generate_legal_moves(pos);
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush)));
        assert(contains_move(moves, make_move(make_square(6, 0), make_square(5, 2))));
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));
        assert_pseudo_matches_piece_generators(pos);
        assert_legal_moves_are_unique_and_safe(pos, 48);
        const std::vector<Move> moves = generate_legal_moves(pos);
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(6, 0), MoveFlag::KingCastle)));
        assert(contains_move(moves, make_move(make_square(4, 0), make_square(2, 0), MoveFlag::QueenCastle)));
    }

    {
        Position pos;
        assert(pos.set_fen("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"));
        assert_pseudo_matches_piece_generators(pos);
        const std::vector<Move> moves = generate_legal_moves(pos);
        (void)move_keys(moves);
        assert(contains_move(moves, make_move(make_square(4, 4), make_square(3, 5), MoveFlag::EnPassant)));
    }

    {
        Position pos;
        assert(pos.set_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
        assert_pseudo_matches_piece_generators(pos);
        const std::vector<Move> moves = generate_legal_moves(pos);
        (void)move_keys(moves);
        assert(contains_move(moves, make_move(make_square(0, 6), make_square(0, 7), MoveFlag::KnightPromotion)));
        assert(contains_move(moves, make_move(make_square(0, 6), make_square(0, 7), MoveFlag::BishopPromotion)));
        assert(contains_move(moves, make_move(make_square(0, 6), make_square(0, 7), MoveFlag::RookPromotion)));
        assert(contains_move(moves, make_move(make_square(0, 6), make_square(0, 7), MoveFlag::QueenPromotion)));
        assert(contains_move(moves, make_move(make_square(0, 6), make_square(1, 7), MoveFlag::KnightPromotionCapture)));
        assert(contains_move(moves, make_move(make_square(0, 6), make_square(1, 7), MoveFlag::BishopPromotionCapture)));
        assert(contains_move(moves, make_move(make_square(0, 6), make_square(1, 7), MoveFlag::RookPromotionCapture)));
        assert(contains_move(moves, make_move(make_square(0, 6), make_square(1, 7), MoveFlag::QueenPromotionCapture)));
        assert(count_flag(moves, MoveFlag::QueenPromotion) == 1);
        assert(count_flag(moves, MoveFlag::QueenPromotionCapture) == 1);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::White, PieceType::Rook, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::Black, PieceType::Rook, make_square(4, 7));
        pos.side_to_move = Color::White;

        assert_pseudo_matches_piece_generators(pos);
        const std::vector<Move> moves = generate_legal_moves(pos);
        (void)move_keys(moves);
        assert(!contains_move(moves, make_move(make_square(4, 1), make_square(3, 1))));
        assert(contains_move(moves, make_move(make_square(4, 1), make_square(4, 7), MoveFlag::Capture)));
        assert_legal_moves_are_unique_and_safe(pos, moves.size());
    }

    assert_fen_legal_count("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 14);
    assert_fen_legal_count("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 44);
    assert_fen_legal_count("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 46);
}
