#include "attacks.hpp"
#include "heuristic_searcher_v21.hpp"

#include <cassert>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string_view>
#include <vector>

using namespace chess;

namespace {

bool contains_move(const std::vector<Move>& moves, Move target) {
    for (Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

Move require_legal_move(const Position& pos, std::string_view text) {
    for (Move move : generate_legal_moves(pos)) {
        if (move_to_string(move) == text) {
            return move;
        }
    }
    assert(false && "required legal move not found");
    return Move{};
}

char piece_char(Color color, PieceType piece) {
    char result = '?';
    switch (piece) {
        case PieceType::Pawn: result = 'p'; break;
        case PieceType::Knight: result = 'n'; break;
        case PieceType::Bishop: result = 'b'; break;
        case PieceType::Rook: result = 'r'; break;
        case PieceType::Queen: result = 'q'; break;
        case PieceType::King: result = 'k'; break;
        case PieceType::None: result = '?'; break;
    }
    if (color == Color::White && result >= 'a' && result <= 'z') {
        result = static_cast<char>(result - 'a' + 'A');
    }
    return result;
}

std::string test_fen(const Position& pos) {
    std::ostringstream out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const Square square = make_square(file, rank);
            if (pos.is_empty(square)) {
                ++empty;
                continue;
            }
            if (empty != 0) {
                out << empty;
                empty = 0;
            }
            const Color color = pos.color_on_occupied(square);
            const PieceType piece = pos.piece_type_on_occupied(square);
            out << piece_char(color, piece);
        }
        if (empty != 0) {
            out << empty;
        }
        if (rank != 0) {
            out << '/';
        }
    }
    out << (pos.side_to_move == Color::White ? " w " : " b ");
    std::string castling;
    if (pos.white_can_castle_kingside) castling += 'K';
    if (pos.white_can_castle_queenside) castling += 'Q';
    if (pos.black_can_castle_kingside) castling += 'k';
    if (pos.black_can_castle_queenside) castling += 'q';
    out << (castling.empty() ? "-" : castling);
    out << ' ';
    if (pos.en_passant_square == NoSquare) {
        out << '-';
    } else {
        out << static_cast<char>('a' + file_of(pos.en_passant_square))
            << static_cast<char>('1' + rank_of(pos.en_passant_square));
    }
    out << ' ' << pos.halfmove_clock << ' ' << pos.fullmove_number;
    return out.str();
}

void assert_v21_search_is_stable(Position pos, int max_depth) {
    HeuristicSearcherV21 v21(1);

    for (int depth = 0; depth <= max_depth; ++depth) {
        v21.clear_tt();
        const SearchResult cold = v21.search_best_move(pos, depth);
        const SearchResult warm = v21.search_best_move(pos, depth);

        assert(cold.score == warm.score);
        assert(cold.best_move == warm.best_move);
        assert(warm.nodes <= cold.nodes);

        const std::vector<Move> moves = generate_legal_moves(pos);
        if (!moves.empty() && depth > 0) {
            assert(contains_move(moves, cold.best_move));
        }
    }
}

void assert_v21_search_is_stable_from_fen(std::string_view fen, int max_depth) {
    Position pos;
    assert(pos.set_fen(fen));
    assert_v21_search_is_stable(pos, max_depth);
}

void stress_v21_search_is_stable(std::uint32_t seed, int samples, int max_random_plies, int max_depth) {
    std::mt19937 rng(seed);

    for (int sample = 0; sample < samples; ++sample) {
        Position pos;
        pos.set_startpos();

        std::uniform_int_distribution<int> plies_dist(0, max_random_plies);
        const int plies = plies_dist(rng);
        for (int ply = 0; ply < plies; ++ply) {
            const std::vector<Move> moves = generate_legal_moves(pos);
            if (moves.empty()) {
                break;
            }
            std::uniform_int_distribution<std::size_t> move_dist(0, moves.size() - 1);
            pos.make_move(moves[move_dist(rng)]);
        }

        assert_v21_search_is_stable(pos, max_depth);
    }
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_startpos();
        assert_v21_search_is_stable(pos, 3);
    }

    assert_v21_search_is_stable_from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        2);

    assert_v21_search_is_stable_from_fen(
        "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
        3);

    assert_v21_search_is_stable_from_fen(
        "8/8/8/8/8/2k5/4K3/8 w - - 0 1",
        3);

    {
        Position pos;
        pos.set_startpos();
        assert_v21_search_is_stable(pos, 4);
    }

    assert_v21_search_is_stable_from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        4);

    assert_v21_search_is_stable_from_fen(
        "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
        4);

    assert_v21_search_is_stable_from_fen(
        "7k/5K2/8/8/8/8/1Q6/8 w - - 0 1",
        3);

    assert_v21_search_is_stable_from_fen(
        "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4",
        4);

    assert_v21_search_is_stable_from_fen(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        4);

    assert_v21_search_is_stable_from_fen(
        "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
        4);

    assert_v21_search_is_stable_from_fen(
        "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
        4);

    assert_v21_search_is_stable_from_fen(
        "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23",
        4);

    {
        Position pos;
        assert(pos.set_fen("7k/5K2/8/8/8/8/1Q6/8 w - - 0 1"));

        HeuristicSearcherV21 v21(1);
        for (int depth = 1; depth <= 3; ++depth) {
            v21.clear_tt();
            const SearchResult result = v21.search_best_move(pos, depth);
            assert(result.score == CheckmateScore - 1);
            assert(contains_move(generate_legal_moves(pos), result.best_move));

            Position next = pos;
            next.make_move(result.best_move);
            assert(generate_legal_moves(next).empty());
            assert(in_check(next, next.side_to_move));
        }
    }

    {
        Position pos;
        assert(pos.set_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"));
        assert(generate_legal_moves(pos).empty());
        assert(!in_check(pos, pos.side_to_move));

        HeuristicSearcherV21 v21(1);
        for (int depth = 1; depth <= 3; ++depth) {
            v21.clear_tt();
            const SearchResult result = v21.search_best_move(pos, depth);
            assert(result.score == 0);
            assert(result.nodes == 1);
            assert(result.best_move.value == 0);
        }
    }

    {
        Position pos;
        assert(pos.set_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));
        assert(generate_legal_moves(pos).empty());
        assert(in_check(pos, pos.side_to_move));

        HeuristicSearcherV21 v21(1);
        for (int depth = 1; depth <= 3; ++depth) {
            v21.clear_tt();
            const SearchResult result = v21.search_best_move(pos, depth);
            assert(result.score == -CheckmateScore);
            assert(result.nodes == 1);
            assert(result.best_move.value == 0);
        }
    }

    {
        Position pos;
        assert(pos.set_fen("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1"));
        const Move stalemate_move = require_legal_move(pos, "g5g6");

        Position stalemate_child = pos;
        stalemate_child.make_move(stalemate_move);
        assert(generate_legal_moves(stalemate_child).empty());
        assert(!in_check(stalemate_child, stalemate_child.side_to_move));

        HeuristicSearcherV21 v21(1);
        const SearchResult child_result = v21.search_best_move(stalemate_child, 1);
        assert(child_result.score == 0);
        assert(child_result.nodes == 1);

        const SearchResult parent_depth_2 = v21.search_best_move(pos, 2);
        assert(parent_depth_2.best_move != stalemate_move);
        assert(parent_depth_2.score > 0);
        assert(contains_move(generate_legal_moves(pos), parent_depth_2.best_move));
    }

    {
        Position pos;
        assert(pos.set_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));

        HeuristicSearcherV21 v21(1);
        const SearchResult qsearch_result = v21.search_best_move(pos, 0);
        assert(qsearch_result.score == -CheckmateScore);
        assert(qsearch_result.best_move.value == 0);
    }

    {
        std::mt19937 rng(20260615);
        Position pos;
        pos.set_startpos();

        for (int sample = 0; sample < 80; ++sample) {
            assert_v21_search_is_stable(pos, 3);

            const std::vector<Move> moves = generate_legal_moves(pos);
            if (moves.empty()) {
                pos.set_startpos();
                continue;
            }

            std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
            pos.make_move(moves[dist(rng)]);
        }
    }

    stress_v21_search_is_stable(20260616, 250, 80, 4);
    stress_v21_search_is_stable(20260617, 40, 100, 5);

    {
        Position pos;
        pos.set_startpos();

        HeuristicSearcherV21 v21(1);
        const SearchResult result = v21.search_best_move(pos, SearchLimits{
            .max_depth = 64,
            .move_time = std::chrono::milliseconds{1}
        });

        assert(result.depth >= 0);
        assert(result.depth <= 64);
        assert(contains_move(generate_legal_moves(pos), result.best_move));
    }

    {
        Position pos;
        assert(pos.set_fen("r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9"));

        HeuristicSearcherV21 v21(4);
        const SearchLimits limits{
            .max_depth = 5,
            .move_time = std::chrono::milliseconds{0}
        };
        const SearchResult result = v21.search_best_move(pos, limits);
        assert(result.depth == 5);
        assert(contains_move(generate_legal_moves(pos), result.best_move));
    }
}
