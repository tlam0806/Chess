#include "heuristic_searcher.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"
#include "lower_move_range_bucket_transposition_table.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
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
    return -reference_negamax(next, depth - 1, 1);
}

void assert_search_matches_reference(Position pos, int max_depth) {
    HeuristicSearcher searcher;

    for (int depth = 0; depth <= max_depth; ++depth) {
        const int expected = reference_negamax(pos, depth);

        const SearchResult result = searcher.search_best_move(pos, depth);
        assert(result.score == expected);

        if (depth > 0 && !generate_legal_moves(pos).empty()) {
            assert(root_score_from_best_move(pos, result.best_move, depth) == result.score);
        }
    }
}

bool contains_move(const std::vector<Move>& moves, Move target) {
    for (Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    {
        LowerMoveRangeBucketTranspositionTable::TTValue value{};
        assert(value.lower_score() == -Infinity);
        assert(value.upper_score() == Infinity);
        assert(value.generation() == 0);

        constexpr int Scores[] = {
            -Infinity, -CheckmateScore, -1, 0, 1, CheckmateScore, Infinity};
        for (const int score : Scores) {
            value.set_lower_score(score);
            value.set_upper_score(score);
            assert(value.lower_score() == score);
            assert(value.upper_score() == score);
        }

        for (const std::uint8_t generation : {
                 std::uint8_t{0}, std::uint8_t{1}, std::uint8_t{127},
                 std::uint8_t{254}, std::uint8_t{255}}) {
            value.set_generation(generation);
            assert(value.generation() == generation);
            assert(value.lower_score() == Infinity);
            assert(value.upper_score() == Infinity);
        }
    }

    {
        LowerMoveRangeBucketTranspositionTable tt(1, 4);
        constexpr HashKey key = 0x123456789abcdef0ULL;
        const Move old_hint = make_move(
            make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush);
        const Move new_hint = make_move(
            make_square(3, 1), make_square(3, 3), MoveFlag::DoublePawnPush);

        tt.advance_generation();
        tt.store(key, 3, ScoreRange{42, 42}, MoveRange{old_hint, old_hint});

        ScoreRange window{-100, 100};
        ScoreRange stored_score;
        MoveRange stored_move;
        bool score_available = false;
        assert(tt.probe(
            key, 3, window, stored_score, stored_move, 0, score_available,
            TTDepthPolicy::Exact));
        assert(score_available);
        assert(stored_score.lower == 42 && stored_score.upper == 42);
        assert(stored_move.lower == old_hint);

        // A new root history starts a new score generation.  The previous
        // move remains useful for ordering, but its path-dependent score must
        // not narrow the new search window or produce a cutoff.
        tt.advance_generation();
        window = ScoreRange{-100, 100};
        stored_score = {};
        stored_move = {};
        score_available = false;
        assert(!tt.probe(
            key, 3, window, stored_score, stored_move, 0, score_available,
            TTDepthPolicy::Exact));
        assert(!score_available);
        assert(window.lower == -100 && window.upper == 100);
        assert(stored_move.lower == old_hint);

        // Writing the same key in the current generation replaces, rather
        // than merges with, the stale score.
        tt.store(key, 1, ScoreRange{-7, -7}, MoveRange{new_hint, new_hint});
        window = ScoreRange{-100, 100};
        stored_score = {};
        stored_move = {};
        score_available = false;
        assert(tt.probe(
            key, 1, window, stored_score, stored_move, 0, score_available,
            TTDepthPolicy::Exact));
        assert(score_available);
        assert(stored_score.lower == -7 && stored_score.upper == -7);
        assert(stored_move.lower == new_hint);

        tt.clear();
        tt.advance_generation();
        window = ScoreRange{-100, 100};
        stored_move = {};
        score_available = false;
        assert(!tt.probe(
            key, 1, window, stored_score, stored_move, 0, score_available,
            TTDepthPolicy::Exact));
        assert(stored_move.lower.value == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Queen, make_square(3, 0));
        pos.side_to_move = Color::White;

        HeuristicSearcher searcher;
        const SearchResult result = searcher.search_best_move(pos, 0);
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

        HeuristicSearcher searcher;
        const SearchResult result = searcher.search_best_move(pos, 1);

        assert(result.best_move == make_move(make_square(0, 0), make_square(0, 7), MoveFlag::Capture));
        assert(result.score > 300);
        assert(result.nodes > 0);
    }

    {
        Position pos;
        assert(pos.set_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"));

        assert(generate_legal_moves(pos).empty());
        assert(!in_check(pos, pos.side_to_move));

        HeuristicSearcher searcher;
        const SearchResult result = searcher.search_best_move(pos, 3);
        assert(result.score == 0);
        assert(result.nodes == 1);
    }

    {
        Position pos;
        assert(pos.set_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));

        assert(generate_legal_moves(pos).empty());
        assert(in_check(pos, pos.side_to_move));

        HeuristicSearcher searcher;
        const SearchResult result = searcher.search_best_move(pos, 3);
        assert(result.score == -CheckmateScore);
        assert(result.nodes == 1);
    }

    {
        Position pos;
        assert(pos.set_fen("7k/5K2/8/8/8/8/1Q6/8 w - - 0 1"));

        HeuristicSearcher searcher;
        const SearchResult result = searcher.search_best_move(pos, 1);

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

    {
        Position pos;
        pos.set_startpos();

        HeuristicSearcher searcher;
        const SearchResult fixed = searcher.search_best_move(pos, 3);
        const SearchResult iterative = searcher.search_best_move(pos, SearchLimits{
            .max_depth = 3,
            .move_time = std::chrono::milliseconds{0}
        });

        assert(iterative.score == fixed.score);
        assert(iterative.best_move == fixed.best_move);
        assert(iterative.depth == 3);
        assert(!iterative.stopped);

        const SearchResult timed = searcher.search_best_move(pos, SearchLimits{
            .max_depth = 64,
            .move_time = std::chrono::milliseconds{1}
        });

        assert(timed.depth >= 0);
        assert(timed.depth <= 64);
        assert(contains_move(generate_legal_moves(pos), timed.best_move));
    }
}
