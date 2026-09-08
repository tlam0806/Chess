#include "attacks.hpp"
#include "move.hpp"
#include "nnue_searcher_v43.hpp"
#include "nnue_searcher_v44.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<std::string_view, 159> GameMoves{
    "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "c7c5",
    "a2a3", "b4c3", "b2c3", "b7b6", "f1d3", "d7d5", "c4d5", "e6d5",
    "g1e2", "e8g8", "e1g1", "c8a6", "f2f3", "f8e8", "e2g3", "a6d3",
    "d1d3", "b8c6", "c1b2", "c5c4", "d3e2", "h7h5", "e3e4", "h5h4",
    "g3f5", "d5e4", "f3e4", "e8e4", "e2c4", "c6e5", "c4a6", "e5g4",
    "a6d3", "d8c7", "d3h3", "g7g6", "f5h4", "a8e8", "h4f3", "c7c6",
    "a1b1", "e4e2", "b2c1", "g4e3", "c1e3", "e8e3", "b1c1", "g8g7",
    "a3a4", "a7a5", "c3c4", "c6a4", "h3g3", "f6h5", "g3c7", "a4a2",
    "g1h1", "e2f2", "f1g1", "a2b2", "c1b1", "b2c2", "b1b6", "a5a4",
    "b6b8", "f2e2", "d4d5", "h5f6", "f3g5", "e3e7", "c7f4", "f6e4",
    "d5d6", "c2d2", "f4h4", "e4f2", "h4f2", "e2f2", "d6e7", "f2e2",
    "e7e8q", "e2e8", "b8e8", "d2g5", "e8a8", "g5e3", "a8a4", "e3b3",
    "g1a1", "g7h6", "c4c5", "b3c3", "a4a6", "f7f5", "c5c6", "f5f4",
    "a1f1", "f4f3", "g2f3", "c3c4", "f1a1", "c4e2", "a6a3", "e2c2",
    "a1a2", "c2c1", "h1g2", "c1g5", "g2f2", "g5h4", "f2f1", "h4c4",
    "f1g2", "c4c6", "a3a4", "h6h5", "a4g4", "c6b5", "a2d2", "h5h6",
    "h2h4", "b5c5", "g4e4", "c5c3", "d2a2", "c3b3", "a2e2", "b3c3",
    "e4e8", "c3d3", "e2b2", "d3c4", "e8e4", "c4c3", "b2e2", "c3d3",
    "e4e8", "d3c4", "e2b2", "c4a4", "e8e4", "a4a1", "e4e2", "a1a4",
    "e2e1", "a4c4", "b2f2", "c4c3", "e1e4", "h6h5", "f2a2",
};

// Completed depths reported by the production game for Black's searches
// after White moves 70 through 79. Keep these fixed: the regression depends
// on reproducing the exact cross-root search path, not merely the final FEN.
constexpr std::array<int, 10> DefaultWarmDepths{
    15, 14, 17, 14, 15, 14, 15, 13, 13, 13,
};

constexpr std::string_view ExpectedTargetFen =
    "8/8/6p1/7k/4R2P/2q2P2/R5K1/8 b - - 30 80";

struct ReplayState {
    chess::Position position;
    std::vector<chess::HashKey> history;
};

std::uint8_t castling_rights(const chess::Position& pos) {
    return static_cast<std::uint8_t>(
        (pos.white_can_castle_kingside ? 1 : 0)
        | (pos.white_can_castle_queenside ? 2 : 0)
        | (pos.black_can_castle_kingside ? 4 : 0)
        | (pos.black_can_castle_queenside ? 8 : 0));
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
    assert(false && "expected legal game move");
    return {};
}

void apply_game_move(
    ReplayState& replay,
    std::string_view uci
) {
    const std::uint8_t rights_before = castling_rights(replay.position);
    replay.position.make_move(find_uci_move(replay.position, uci));
    const bool irreversible = replay.position.halfmove_clock == 0
        || castling_rights(replay.position) != rights_before;
    if (irreversible) {
        replay.history.clear();
    }
    replay.history.push_back(replay.position.zobrist_key);
}

ReplayState initial_replay() {
    ReplayState replay;
    replay.position.set_startpos();
    replay.history.push_back(replay.position.zobrist_key);
    return replay;
}

chess::SearchLimits depth_limits(int depth) {
    chess::SearchLimits limits;
    limits.max_depth = depth;
    return limits;
}

template <typename Searcher>
ReplayState replay_with_black_warmup(Searcher& searcher) {
    ReplayState replay = initial_replay();
    std::size_t warm_index = 0;
    for (std::size_t index = 0; index < GameMoves.size(); ++index) {
        apply_game_move(replay, GameMoves[index]);
        const int applied_ply = static_cast<int>(index + 1);
        if (applied_ply >= 139 && applied_ply <= 157
            && (applied_ply & 1) != 0) {
            const chess::SearchResult result = searcher.search_best_move(
                replay.position,
                depth_limits(DefaultWarmDepths[warm_index]),
                replay.history);
            assert(!result.stopped);
            assert(result.depth == DefaultWarmDepths[warm_index]);
            ++warm_index;
        }
    }
    assert(warm_index == DefaultWarmDepths.size());
    return replay;
}

ReplayState replay_without_search() {
    ReplayState replay = initial_replay();
    for (std::string_view move : GameMoves) {
        apply_game_move(replay, move);
    }
    return replay;
}

std::string move_text(chess::Move move) {
    return chess::move_to_string(move);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "usage: nnue_searcher_v44_sable_tt_regression_tests <model>\n";
        return 2;
    }

    chess::PhaseQuantizedNnueModel model;
    assert(model.load(argv[1]));

    chess::NnueSearcherV43 warm_v43(model, 64);
    warm_v43.set_twofold_search_draw_enabled(true);
    ReplayState v43_replay = replay_with_black_warmup(warm_v43);
    chess::Position expected_target;
    assert(expected_target.set_fen(std::string(ExpectedTargetFen)));
    assert(v43_replay.position.zobrist_key == expected_target.zobrist_key);
    assert(v43_replay.position.halfmove_clock == 30);
    const chess::SearchResult v43_result = warm_v43.search_best_move(
        v43_replay.position,
        depth_limits(13),
        v43_replay.history);

    chess::NnueSearcherV44 warm_v44(model, 64);
    warm_v44.set_twofold_search_draw_enabled(true);
    ReplayState v44_replay = replay_with_black_warmup(warm_v44);
    const chess::SearchResult warm_v44_result = warm_v44.search_best_move(
        v44_replay.position,
        depth_limits(13),
        v44_replay.history);

    chess::NnueSearcherV44 cold_v44(model, 64);
    cold_v44.set_twofold_search_draw_enabled(true);
    ReplayState cold_replay = replay_without_search();
    const chess::SearchResult cold_v44_result = cold_v44.search_best_move(
        cold_replay.position,
        depth_limits(13),
        cold_replay.history);

    std::cout
        << "warm_v43=" << move_text(v43_result.best_move)
        << " score=" << v43_result.score
        << " nodes=" << v43_result.nodes << '\n'
        << "warm_v44=" << move_text(warm_v44_result.best_move)
        << " score=" << warm_v44_result.score
        << " nodes=" << warm_v44_result.nodes << '\n'
        << "cold_v44=" << move_text(cold_v44_result.best_move)
        << " score=" << cold_v44_result.score
        << " nodes=" << cold_v44_result.nodes << '\n';

    // The deployed V43 lifecycle reproduces 80...Kh6. V44 must discard the
    // preceding roots' table contents and agree with a fresh V44 search on
    // 80...Qc5. Persistent history heuristics may change score/nodes, but not
    // the selected move in this regression.
    assert(move_text(v43_result.best_move) == "h5h6");
    assert(move_text(warm_v44_result.best_move) == "c3c5");
    assert(warm_v44_result.best_move == cold_v44_result.best_move);
    return 0;
}
