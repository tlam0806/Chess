#include "attacks.hpp"
#include "move.hpp"
#include "nnue_searcher_v40.hpp"
#include "nnue_searcher_v41.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

chess::Position forced_king_evasion(int halfmove_clock) {
    chess::Position pos;
    const std::string fen =
        "8/8/8/8/8/1k6/r7/K7 w - - "
        + std::to_string(halfmove_clock)
        + " 1";
    assert(pos.set_fen(fen));
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    assert(moves.size() == 1);
    return pos;
}

std::vector<chess::HashKey> history_making_only_child_threefold(
    const chess::Position& root
) {
    chess::Position child = root;
    const std::vector<chess::Move> moves = chess::generate_legal_moves(root);
    assert(moves.size() == 1);
    child.make_move(moves.front());
    return {
        child.zobrist_key,
        0x1111111111111111ULL,
        0x2222222222222222ULL,
        0x3333333333333333ULL,
        child.zobrist_key,
        0x4444444444444444ULL,
        0x5555555555555555ULL,
        root.zobrist_key,
    };
}

chess::Move find_uci_move(
    const chess::Position& pos,
    std::string_view uci
) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    assert(false && "expected legal UCI move");
    return {};
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: nnue_searcher_v41_repetition_tests <nnue-model.bin>\n";
        return 2;
    }

    chess::PhaseQuantizedNnueModel model;
    assert(model.load(argv[1]));
    chess::NnueSearcherV40 v40(model);
    chess::NnueSearcherV41 v41(model);
    assert(v41.name() == "nnue_repetition_v41");

    // The only legal move reaches a position already present twice. V41 must
    // search the child and score the third occurrence as a draw.
    const chess::Position repeat_root = forced_king_evasion(8);
    const std::vector<chess::HashKey> repeat_history =
        history_making_only_child_threefold(repeat_root);
    const chess::SearchResult cold =
        v41.search_best_move(repeat_root, 3, repeat_history);
    assert(cold.score == 0);
    assert(v41.repetition_stats().threefold_draws > 0);
    assert(v41.repetition_stats().tt_score_suppressions > 0);

    // A warm TT must not bypass the path-dependent draw. The dangerous
    // history disables TT score cutoffs but still permits its move hint.
    const chess::SearchResult warm =
        v41.search_best_move(repeat_root, 3, repeat_history);
    assert(warm.score == 0);
    assert(warm.best_move == cold.best_move);
    assert(v41.repetition_stats().threefold_draws > 0);
    assert(v41.repetition_stats().tt_score_suppressions > 0);

    // A pawn/capture boundary (represented by halfmove_clock=0 at the root)
    // excludes the older synthetic occurrences. V41 then matches V40.
    const chess::Position reset_root = forced_king_evasion(0);
    const std::vector<chess::HashKey> stale_history =
        history_making_only_child_threefold(reset_root);
    v40.clear_tt();
    v41.clear_tt();
    const chess::SearchResult control = v40.search_best_move(reset_root, 1);
    const chess::SearchResult after_reset =
        v41.search_best_move(reset_root, 1, stale_history);
    assert(after_reset.best_move == control.best_move);
    assert(after_reset.score == control.score);
    assert(v41.repetition_stats().threefold_draws == 0);

    // If the root itself has already occurred three times, the searcher
    // returns draw immediately while retaining a legal fallback move.
    chess::Position root_threefold = forced_king_evasion(8);
    const std::array<chess::HashKey, 9> root_history{
        root_threefold.zobrist_key,
        0x1111111111111111ULL,
        0x2222222222222222ULL,
        0x3333333333333333ULL,
        root_threefold.zobrist_key,
        0x4444444444444444ULL,
        0x5555555555555555ULL,
        0x6666666666666666ULL,
        root_threefold.zobrist_key,
    };
    const chess::SearchResult terminal =
        v41.search_best_move(root_threefold, 4, root_history);
    assert(terminal.score == 0);
    assert(terminal.nodes == 1);
    assert(terminal.best_move == chess::generate_legal_moves(root_threefold).front());
    assert(v41.repetition_stats().threefold_draws == 1);

    // 99 halfmoves is not yet claimable. At depth zero no child can advance
    // the clock, so no 50-move draw should be observed.
    chess::Position clock_99;
    assert(clock_99.set_fen("7k/8/8/8/8/8/8/KR6 w - - 99 1"));
    const std::array<chess::HashKey, 1> clock_99_history{
        clock_99.zobrist_key};
    (void)v41.search_best_move(clock_99, 0, clock_99_history);
    assert(v41.repetition_stats().fifty_move_draws == 0);

    // Scores from a previous root must not cross game-history generations.
    // Zobrist intentionally excludes halfmove_clock, so these positions have
    // the same key even though a depth-one search at clock 99 reaches an
    // immediate 50-move draw while clock 0 does not.
    chess::Position clock_0_same_board;
    assert(clock_0_same_board.set_fen("7k/8/8/8/8/8/8/KR6 w - - 0 1"));
    assert(clock_0_same_board.zobrist_key == clock_99.zobrist_key);
    const std::array<chess::HashKey, 1> clock_0_history{
        clock_0_same_board.zobrist_key};
    v41.clear_tt();
    const chess::SearchResult clock_0_result =
        v41.search_best_move(clock_0_same_board, 1, clock_0_history);
    assert(clock_0_result.score != 0);
    const chess::SearchResult clock_99_after_retained_tt =
        v41.search_best_move(clock_99, 1, clock_99_history);
    chess::NnueSearcherV41 fresh_v41(model);
    const chess::SearchResult clock_99_fresh =
        fresh_v41.search_best_move(clock_99, 1, clock_99_history);
    assert(clock_99_fresh.score == 0);
    assert(clock_99_after_retained_tt.score == clock_99_fresh.score);
    const std::vector<chess::Move> clock_99_legal_moves =
        chess::generate_legal_moves(clock_99);
    assert(std::find(
        clock_99_legal_moves.begin(),
        clock_99_legal_moves.end(),
        clock_99_after_retained_tt.best_move) != clock_99_legal_moves.end());
    assert(clock_99_after_retained_tt.nodes > 1);

    // At 100 halfmoves a normal legal position is an immediate claimable
    // draw, including with a warm TT, while retaining a legal fallback move.
    chess::Position clock_100;
    assert(clock_100.set_fen("7k/8/8/8/8/8/8/KR6 w - - 100 1"));
    const std::array<chess::HashKey, 1> clock_100_history{
        clock_100.zobrist_key};
    const chess::SearchResult fifty_cold =
        v41.search_best_move(clock_100, 4, clock_100_history);
    assert(fifty_cold.score == 0);
    assert(fifty_cold.nodes == 1);
    assert(v41.repetition_stats().fifty_move_draws == 1);
    const std::vector<chess::Move> fifty_legal_moves =
        chess::generate_legal_moves(clock_100);
    assert(std::find(
        fifty_legal_moves.begin(),
        fifty_legal_moves.end(),
        fifty_cold.best_move) != fifty_legal_moves.end());
    const chess::SearchResult fifty_warm =
        v41.search_best_move(clock_100, 4, clock_100_history);
    assert(fifty_warm.score == 0);
    assert(fifty_warm.nodes == 1);
    assert(fifty_warm.best_move == fifty_cold.best_move);
    assert(v41.repetition_stats().fifty_move_draws == 1);

    // Checkmate on the 100th halfmove takes precedence over the draw rule.
    chess::Position checkmate_100;
    assert(checkmate_100.set_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1"));
    assert(chess::in_check(checkmate_100, checkmate_100.side_to_move));
    assert(chess::generate_legal_moves(checkmate_100).empty());
    const std::array<chess::HashKey, 1> checkmate_history{
        checkmate_100.zobrist_key};
    const chess::SearchResult checkmate =
        v41.search_best_move(checkmate_100, 4, checkmate_history);
    assert(checkmate.score == -chess::CheckmateScore);
    assert(v41.repetition_stats().fifty_move_draws == 0);

    // A quiet mate-in-one reached from clock 99 produces a child at clock 100.
    // That child must remain checkmate rather than being converted to a draw.
    chess::Position mate_in_one;
    assert(mate_in_one.set_fen("7k/8/5KQ1/8/8/8/8/8 w - - 99 1"));
    const std::array<chess::HashKey, 1> mate_in_one_history{
        mate_in_one.zobrist_key};
    const chess::SearchResult mating =
        v41.search_best_move(mate_in_one, 1, mate_in_one_history);
    assert(mating.best_move == find_uci_move(mate_in_one, "g6g7"));
    assert(mating.score >= chess::CheckmateScore / 2);

    // Pawn moves and captures reset halfmove_clock, so their children must not
    // inherit a stale 50-move draw.
    chess::Position pawn_reset;
    assert(pawn_reset.set_fen("7k/8/8/8/8/8/P7/K7 w - - 99 1"));
    pawn_reset.make_move(find_uci_move(pawn_reset, "a2a3"));
    assert(pawn_reset.halfmove_clock == 0);
    const std::array<chess::HashKey, 1> pawn_history{pawn_reset.zobrist_key};
    (void)v41.search_best_move(pawn_reset, 0, pawn_history);
    assert(v41.repetition_stats().fifty_move_draws == 0);

    chess::Position capture_reset;
    assert(capture_reset.set_fen("7k/8/8/8/8/8/1n6/KR6 w - - 99 1"));
    capture_reset.make_move(find_uci_move(capture_reset, "b1b2"));
    assert(capture_reset.halfmove_clock == 0);
    const std::array<chess::HashKey, 1> capture_history{
        capture_reset.zobrist_key};
    (void)v41.search_best_move(capture_reset, 0, capture_history);
    assert(v41.repetition_stats().fifty_move_draws == 0);

    std::cout << "nnue v41 repetition checks passed\n";
}
