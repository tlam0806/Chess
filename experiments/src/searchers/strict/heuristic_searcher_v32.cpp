#include "heuristic_searcher_v32.hpp"

#include "attacks.hpp"
#include "heuristic_searcher_v32_detail.hpp"
#include "evaluate.hpp"
#include "legal_noisy_generator.hpp"
#include "legal_non_capture_generator.hpp"
#include "move_undo_guard.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace chess {

void HeuristicSearcherV32::reward_quiet_cutoff(
    Color side_to_move,
    int depth,
    int ply,
    const ScoredMove& scored_move,
    Move prev_move,
    PieceType prev_moved_piece
) {
    if (!is_quiet_move(scored_move)) {
        return;
    }
    killer_table_.store(ply, scored_move.move);
    const PieceType moved_piece = scored_moved_piece(scored_move);
    history_table_.store(side_to_move, moved_piece, scored_move.move, depth);
    if (is_valid_move(prev_move) && prev_moved_piece != PieceType::None) {
        counter_move_table_.store(side_to_move, prev_moved_piece, prev_move, scored_move.move);
        counter_history_table_.store(
            side_to_move,
            prev_moved_piece,
            prev_move,
            moved_piece,
            scored_move.move,
            depth
        );
    }
}

void HeuristicSearcherV32::penalize_failed_quiets(
    Color side_to_move,
    int depth,
    Move prev_move,
    PieceType prev_moved_piece,
    const HeuristicSearcherV32::ScoredMoveList& failed_quiet_moves
) {
    for (const ScoredMove& failed_quiet : failed_quiet_moves) {
        const PieceType moved_piece = scored_moved_piece(failed_quiet);
        history_table_.penalize(side_to_move, moved_piece, failed_quiet.move, depth);
        if (is_valid_move(prev_move) && prev_moved_piece != PieceType::None) {
            counter_history_table_.penalize(
                side_to_move,
                prev_moved_piece,
                prev_move,
                moved_piece,
                failed_quiet.move,
                depth
            );
        }
    }
}

bool HeuristicSearcherV32::should_stop(SearchState& state) const {
    if (!state.has_deadline) {
        return false;
    }
    if ((state.nodes & 1023ULL) != 0) {
        return false;
    }
    if (Clock::now() >= state.deadline) {
        state.stopped = true;
        return true;
    }
    return false;
}

KingSafetyContext HeuristicSearcherV32::current_king_safety_context(const Position& pos) const {
    return cached_king_safety_context(pos, pos.side_to_move);
}

int HeuristicSearcherV32::evaluate_current_position(const Position& pos) const {
    return evaluate_for_side_to_move(pos);
}



HeuristicSearcherV32::SearchValue HeuristicSearcherV32::quiescence(
    Position& pos,
    int alpha,
    int beta,
    int ply,
    int q_depth,
    SearchState& state
) {
    assert(alpha < beta);
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{exact_range(0)};
    }

    std::optional<PositionStateSnapshot> node_snapshot;
    auto get_node_snapshot = [&]() -> const PositionStateSnapshot& {
        if (!node_snapshot) {
            node_snapshot = pos.make_state_snapshot();
        }
        return *node_snapshot;
    };
    const KingSafetyContext king_safety = current_king_safety_context(pos);
    const bool side_in_check = king_safety.checkers != EmptyBB;
    auto generate_qsearch_capture_stage = [&]() {
        ScoredMoveList moves;
        auto score_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
            moves.push_back(make_qsearch_scored_move<true, false>(
                move, moved_piece, captured_piece));
        };
        generate_legal_non_promotion_capture_moves_with_info(pos, king_safety, score_move);
        sort_scored_moves(moves);
        return moves;
    };
    auto generate_qsearch_quiet_evasion_stage = [&]() {
        ScoredMoveList moves;
        auto score_move = [&](Move move, PieceType moved_piece) {
            moves.push_back(make_qsearch_scored_move<false, false>(
                move, moved_piece, PieceType::None));
        };
        generate_legal_quiet_non_promotion_moves_with_info(pos, king_safety, score_move);
        sort_scored_moves(moves);
        return moves;
    };
    auto search_qsearch_move = [&](const ScoredMove& scored_move, ScoreRange& node_range) {
        const PieceType moved_piece = scored_moved_piece(scored_move);
        const PieceType captured_piece = scored_captured_piece(scored_move);
        ScoreRange move_range;
        {
            SnapshotMoveUndoGuard move_guard(
                pos,
                get_node_snapshot(),
                scored_move.move,
                moved_piece,
                captured_piece);
            SearchValue child = quiescence(
                pos,
                -beta,
                -alpha,
                ply + 1,
                q_depth + 1,
                state);
            move_range = negate_range(child.range);
        }
        if (state.stopped) {
            return true;
        }

        node_range.lower = std::max(node_range.lower, move_range.lower);
        node_range.upper = std::max(node_range.upper, move_range.upper);
        if (node_range.lower >= beta) {
            return true;
        }
        if (move_range.lower != -Infinity && move_range.lower > alpha) {
            alpha = move_range.lower;
        }
        return false;
    };
    auto search_qsearch_promotion_stage = [&](ScoreRange& node_range, bool allow_search, bool& has_legal_move) {
        bool stage_done = false;
        auto search_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
            has_legal_move = true;
            if (!allow_search || stage_done) {
                return;
            }
            const ScoredMove scored_move = captured_piece == PieceType::None
                ? make_qsearch_scored_move<false, true>(move, moved_piece, captured_piece)
                : make_qsearch_scored_move<true, true>(move, moved_piece, captured_piece);
            stage_done = search_qsearch_move(scored_move, node_range);
        };
        generate_legal_promotion_moves_with_info(pos, king_safety, search_move);
        return stage_done;
    };
    auto search_qsearch_stage = [&](const ScoredMoveList& staged_moves, ScoreRange& node_range) {
        for (std::size_t move_index = 0; move_index < staged_moves.size(); ++move_index) {
            if (search_qsearch_move(staged_moves[move_index], node_range)) {
                return true;
            }
        }
        return false;
    };

    if (side_in_check) {
        ScoreRange node_range;
        node_range.lower = -Infinity;
        node_range.upper = -Infinity;
        bool has_legal_move = false;

        if (search_qsearch_promotion_stage(
                node_range,
                q_depth < MaxCheckEvasionQuiescenceDepth,
                has_legal_move)) {
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }
            return SearchValue{ScoreRange{node_range.lower, Infinity}};
        }

        const ScoredMoveList capture_moves = generate_qsearch_capture_stage();
        has_legal_move = has_legal_move || !capture_moves.empty();
        if (q_depth < MaxCheckEvasionQuiescenceDepth
            && search_qsearch_stage(capture_moves, node_range)) {
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }
            return SearchValue{ScoreRange{node_range.lower, Infinity}};
        }

        const ScoredMoveList quiet_evasion_moves = generate_qsearch_quiet_evasion_stage();
        has_legal_move = has_legal_move || !quiet_evasion_moves.empty();
        if (!has_legal_move) {
            return SearchValue{exact_range(-CheckmateScore + ply)};
        }
        if (q_depth >= MaxCheckEvasionQuiescenceDepth) {
            return SearchValue{exact_range(evaluate_current_position(pos))};
        }
        if (search_qsearch_stage(quiet_evasion_moves, node_range)) {
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }
            return SearchValue{ScoreRange{node_range.lower, Infinity}};
        }

        return SearchValue{node_range};
    }

    int best_score = evaluate_current_position(pos);
    if (best_score >= beta) {
        return SearchValue{lower_range(best_score)};
    }
    if (best_score > alpha) {
        alpha = best_score;
    }

    if (q_depth >= MaxQuiescenceDepth) {
        return SearchValue{exact_range(best_score)};
    }

    ScoreRange node_range = exact_range(best_score);
    bool has_legal_promotion = false;
    if (search_qsearch_promotion_stage(node_range, true, has_legal_promotion)) {
        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }
        return SearchValue{ScoreRange{node_range.lower, Infinity}};
    }

    const ScoredMoveList capture_moves = generate_qsearch_capture_stage();
    if (search_qsearch_stage(capture_moves, node_range)) {
        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }
        return SearchValue{ScoreRange{node_range.lower, Infinity}};
    }

    return SearchValue{node_range};
}

HeuristicSearcherV32::SearchValue HeuristicSearcherV32::negamax(
    Position& pos,
    int depth,
    int ply,
    Move prev_move,
    PieceType prev_moved_piece,
    int alpha,
    int beta,
    SearchState& state
) {
    assert(depth >= 0);
    assert(alpha < beta);
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{exact_range(0)};
    }

    TTProbeResult tt_probe = probe_tt(pos.zobrist_key, depth, alpha, beta, ply, true);
    if (tt_probe.hit) {
        return SearchValue{tt_probe.range};
    }

    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply, 0, state);
    }

    PositionStateSnapshot node_snapshot;
    bool node_snapshot_ready = false;
    auto get_node_snapshot = [&]() -> const PositionStateSnapshot& {
        if (!node_snapshot_ready) {
            node_snapshot = pos.make_state_snapshot();
            node_snapshot_ready = true;
        }
        return node_snapshot;
    };
    ScoreRange node_range;
    node_range.lower = -Infinity;
    node_range.upper = -Infinity;
    Move best_lower_move{};
    Move best_upper_move{};
    Move fallback_best_move{};
    ScoredMoveList failed_quiet_moves;
    bool searched_any_move = false;
    Move searched_tt_move = tt_probe.moves.lower;
    std::optional<KingSafetyContext> king_safety_cache;
    auto get_king_safety = [&]() -> const KingSafetyContext& {
        if (!king_safety_cache) {
            king_safety_cache = current_king_safety_context(pos);
        }
        return *king_safety_cache;
    };

    const Move tt_lower_move = tt_probe.moves.lower;
    if (is_valid_move(tt_lower_move)) {
        ScoredMove scored_move = make_tt_lower_scored_move(pos, tt_lower_move);
        if (is_valid_move(scored_move.move)) {
            searched_any_move = true;
            searched_tt_move = scored_move.move;
            fallback_best_move = scored_move.move;

            ScoreRange move_range;
            {
                SnapshotMoveUndoGuard move_guard(
                    pos,
                    get_node_snapshot(),
                    scored_move.move,
                    scored_moved_piece(scored_move),
                    scored_captured_piece(scored_move));
                SearchValue child = negamax(
                    pos,
                    depth - 1,
                    ply + 1,
                    scored_move.move,
                    scored_moved_piece(scored_move),
                    -beta,
                    -alpha,
                    state
                );
                move_range = negate_range(child.range);
            }
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }

            update_best_range(node_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (node_range.lower >= beta) {
                reward_quiet_cutoff(pos.side_to_move, depth, ply, scored_move, prev_move, prev_moved_piece);
                node_range.upper = Infinity;
                if (tt_probe.has_score) {
                    node_range = intersect_ranges(node_range, tt_probe.range);
                }
                assert(node_range.lower <= node_range.upper);
                store_tt_if_needed(
                    pos.zobrist_key,
                    depth,
                    ply,
                    node_range,
                    best_lower_move,
                    best_upper_move,
                    fallback_best_move);
                return SearchValue{node_range};
            }
            if (is_quiet_move(scored_move)) {
                failed_quiet_moves.push_back(scored_move);
            }
        }
    }

    bool cutoff = node_range.lower >= beta;
    bool stage_stopped = false;
    auto search_one_scored_move =
        [&]<MoveGenerationStage Stage>(const ScoredMove& scored_move, bool has_unsearched_stage) {
        if (cutoff || stage_stopped) {
            return;
        }
        const PieceType moved_piece = scored_moved_piece(scored_move);
        const PieceType captured_piece = scored_captured_piece(scored_move);
        if (!is_valid_move(fallback_best_move)) {
            fallback_best_move = scored_move.move;
        }

        ScoreRange move_range;
        {
            SnapshotMoveUndoGuard move_guard(
                pos,
                get_node_snapshot(),
                scored_move.move,
                moved_piece,
                captured_piece);

            if (searched_any_move && beta > alpha + 1) {
                SearchValue scout = negamax(
                    pos,
                    depth - 1,
                    ply + 1,
                    scored_move.move,
                    moved_piece,
                    -alpha - 1,
                    -alpha,
                    state
                );
                move_range = negate_range(scout.range);
                if (state.stopped) {
                    stage_stopped = true;
                    return;
                }

                const bool scout_proves_fail_low = move_range.upper <= alpha;
                const bool scout_proves_fail_high = move_range.lower >= beta;
                if (!scout_proves_fail_low && !scout_proves_fail_high) {
                    SearchValue full = negamax(
                        pos,
                        depth - 1,
                        ply + 1,
                        scored_move.move,
                        moved_piece,
                        -beta,
                        -alpha,
                        state
                    );
                    move_range = negate_range(full.range);
                    if (state.stopped) {
                        stage_stopped = true;
                        return;
                    }
                }
            } else {
                SearchValue child = negamax(
                    pos,
                    depth - 1,
                    ply + 1,
                    scored_move.move,
                    moved_piece,
                    -beta,
                    -alpha,
                    state
                );
                move_range = negate_range(child.range);
                if (state.stopped) {
                    stage_stopped = true;
                    return;
                }
            }
        }
        searched_any_move = true;
        update_best_range(node_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
        if (node_range.lower >= beta) {
            if constexpr (Stage == MoveGenerationStage::QuietNonPromotion
                || Stage == MoveGenerationStage::Killer) {
                if (!scored_gives_check(scored_move)) {
                    reward_quiet_cutoff(
                        pos.side_to_move,
                        depth,
                        ply,
                        scored_move,
                        prev_move,
                        prev_moved_piece);
                }
            }
            penalize_failed_quiets(pos.side_to_move, depth, prev_move, prev_moved_piece, failed_quiet_moves);
            node_range.upper = has_unsearched_stage ? Infinity : node_range.upper;
            cutoff = true;
            return;
        }
        if constexpr (Stage == MoveGenerationStage::QuietNonPromotion
            || Stage == MoveGenerationStage::Killer) {
            if (!scored_gives_check(scored_move)) {
                failed_quiet_moves.push_back(scored_move);
            }
        }
    };
    auto search_scored_moves = [&]<MoveGenerationStage Stage>(const ScoredMoveList& moves) {
        if (cutoff || stage_stopped) {
            return;
        }
        for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
            const bool has_unsearched_stage =
                Stage != MoveGenerationStage::BadCapture
                || move_index + 1 < moves.size();
            search_one_scored_move.template operator()<Stage>(moves[move_index], has_unsearched_stage);
            if (cutoff || stage_stopped) {
                return;
            }
        }
    };
    auto search_promotion_stage = [&](const KingSafetyContext& king_safety) {
        auto search_promotion_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
            if (cutoff || stage_stopped || matches_any(move, searched_tt_move)) {
                return;
            }
            const ScoredMove scored_move = captured_piece == PieceType::None
                ? make_scored_legal_move<ScoringMode::MainSearch, false, true>(
                    pos, move, moved_piece, captured_piece, ply, tt_probe.moves, prev_move, prev_moved_piece)
                : make_scored_legal_move<ScoringMode::MainSearch, true, true>(
                    pos, move, moved_piece, captured_piece, ply, tt_probe.moves, prev_move, prev_moved_piece);
            search_one_scored_move.template operator()<MoveGenerationStage::Promotion>(scored_move, true);
        };
        generate_legal_promotion_moves_with_info(pos, king_safety, search_promotion_move);
    };
    if (!cutoff && !stage_stopped) {
        const KingSafetyContext& king_safety = get_king_safety();
        search_promotion_stage(king_safety);
    }
    if (!cutoff && !stage_stopped) {
        const KingSafetyContext& king_safety = get_king_safety();
        CaptureMoveLists capture_moves = generate_legal_capture_scored_move_lists_for_searcher(
            pos,
            king_safety,
            ply,
            tt_probe.moves,
            prev_move,
            prev_moved_piece,
            searched_tt_move);
        sort_scored_moves(capture_moves.good);
        search_scored_moves.template operator()<MoveGenerationStage::GoodCapture>(capture_moves.good);
        Move searched_priority1{};
        Move searched_priority2{};
        Move searched_priority3{};
        if (!cutoff && !stage_stopped) {
            const ScoredMoveList priority_moves = ordered_priority_quiet_moves_for_stage(
                pos,
                ply,
                tt_probe.moves,
                prev_move,
                prev_moved_piece,
                searched_tt_move);
            if (priority_moves.size() > 0) {
                searched_priority1 = priority_moves[0].move;
            }
            if (priority_moves.size() > 1) {
                searched_priority2 = priority_moves[1].move;
            }
            if (priority_moves.size() > 2) {
                searched_priority3 = priority_moves[2].move;
            }
            search_scored_moves.template operator()<MoveGenerationStage::Killer>(priority_moves);
        }
        if (!cutoff && !stage_stopped) {
            const ScoredMoveList quiet_moves = ordered_moves_for_stage<MoveGenerationStage::QuietNonPromotion>(
                pos,
                king_safety,
                ply,
                tt_probe.moves,
                prev_move,
                prev_moved_piece,
                searched_tt_move,
                searched_priority1,
                searched_priority2,
                searched_priority3);
            search_scored_moves.template operator()<MoveGenerationStage::QuietNonPromotion>(quiet_moves);
        }
        if (!cutoff && !stage_stopped) {
            sort_scored_moves(capture_moves.bad);
            search_scored_moves.template operator()<MoveGenerationStage::BadCapture>(capture_moves.bad);
        }
    }
    if (stage_stopped) {
        return SearchValue{exact_range(0)};
    }
    if (!searched_any_move) {
        return SearchValue{exact_range(in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0)};
    }
    assert(node_range.lower <= node_range.upper);
    if (tt_probe.has_score) {
        node_range = intersect_ranges(node_range, tt_probe.range);
    }
    assert(node_range.lower <= node_range.upper);

    store_tt_if_needed(pos.zobrist_key, depth, ply, node_range, best_lower_move, best_upper_move, fallback_best_move);

    return SearchValue{node_range};
}

SearchResult HeuristicSearcherV32::make_fallback_result(const Position& pos) const {
    SearchResult result;
    const ScoredMoveList moves = ordered_moves(pos, 0);
    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }
    result.best_move = moves[0].move;
    result.score = evaluate_current_position(pos);
    result.nodes = 1;
    return result;
}

HeuristicSearcherV32::RootSearchResult HeuristicSearcherV32::search_fixed_depth(
    Position pos,
    int depth,
    SearchState& state,
    int alpha,
    int beta,
    bool allow_root_tt_probe
) {
    assert(depth >= 0);

    RootSearchResult root_result;
    SearchResult& result = root_result.result;
    result.depth = depth;

    if (depth == 0) {
        root_result.range = quiescence(pos, -Infinity, Infinity, 0, 0, state).range;
        result.score = representative_score(root_result.range);
        result.nodes = state.nodes;
        return root_result;
    }

    TTProbeResult tt_probe = probe_tt(pos.zobrist_key, depth, alpha, beta, 0, allow_root_tt_probe);
    if (tt_probe.hit) {
        const Move tt_best_move = preferred_tt_move(tt_probe.moves);
        if (is_valid_move(tt_best_move)) {
            root_result.range = tt_probe.range;
            result.score = representative_score(root_result.range);
            result.best_move = tt_best_move;
            result.nodes = 1;
            return root_result;
        }
    }

    const PositionStateSnapshot node_snapshot = pos.make_state_snapshot();
    ScoreRange root_range;
    root_range.lower = -Infinity;
    root_range.upper = -Infinity;
    Move best_lower_move{};
    Move best_upper_move{};
    Move fallback_best_move{};
    bool searched_any_move = false;
    Move searched_tt_move = tt_probe.moves.lower;
    std::optional<KingSafetyContext> king_safety_cache;
    auto get_king_safety = [&]() -> const KingSafetyContext& {
        if (!king_safety_cache) {
            king_safety_cache = current_king_safety_context(pos);
        }
        return *king_safety_cache;
    };

    const Move tt_lower_move = tt_probe.moves.lower;
    if (is_valid_move(tt_lower_move)) {
        ScoredMove scored_move = make_tt_lower_scored_move(pos, tt_lower_move);
        if (is_valid_move(scored_move.move)) {
            searched_any_move = true;
            searched_tt_move = scored_move.move;
            fallback_best_move = scored_move.move;

            ScoreRange move_range;
            {
                SnapshotMoveUndoGuard move_guard(
                    pos,
                    node_snapshot,
                    scored_move.move,
                    scored_moved_piece(scored_move),
                    scored_captured_piece(scored_move));
                SearchValue child = negamax(
                    pos,
                    depth - 1,
                    1,
                    scored_move.move,
                    scored_moved_piece(scored_move),
                    -beta,
                    -alpha,
                    state
                );
                move_range = negate_range(child.range);
            }
            if (state.stopped) {
                result.stopped = true;
                result.nodes = state.nodes;
                return root_result;
            }

            update_best_range(root_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (root_range.lower >= beta) {
                root_range.upper = Infinity;
            }
        }
    }

    bool cutoff = root_range.lower >= beta;
    bool root_stage_stopped = false;
    auto search_one_root_move =
        [&]<MoveGenerationStage Stage>(const ScoredMove& scored_move, bool has_unsearched_stage) {
        if (cutoff || root_stage_stopped) {
            return;
        }
        const PieceType moved_piece = scored_moved_piece(scored_move);
        const PieceType captured_piece = scored_captured_piece(scored_move);
        if (!is_valid_move(fallback_best_move)) {
            fallback_best_move = scored_move.move;
        }

        ScoreRange move_range;
        {
            SnapshotMoveUndoGuard move_guard(
                pos,
                node_snapshot,
                scored_move.move,
                moved_piece,
                captured_piece);
            SearchValue child = negamax(
                pos,
                depth - 1,
                1,
                scored_move.move,
                moved_piece,
                -beta,
                -alpha,
                state
            );
            move_range = negate_range(child.range);
        }
        if (state.stopped) {
            result.stopped = true;
            result.nodes = state.nodes;
            root_stage_stopped = true;
            return;
        }

        searched_any_move = true;

        update_best_range(root_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
        if (root_range.lower >= beta) {
            root_range.upper = has_unsearched_stage ? Infinity : root_range.upper;
            cutoff = true;
            return;
        }
    };
    auto search_root_scored_moves = [&]<MoveGenerationStage Stage>(const ScoredMoveList& moves) {
        if (cutoff || root_stage_stopped) {
            return;
        }
        for (std::size_t move_index = 0; root_range.lower < beta && move_index < moves.size(); ++move_index) {
            const bool has_unsearched_stage =
                Stage != MoveGenerationStage::BadCapture
                || move_index + 1 < moves.size();
            search_one_root_move.template operator()<Stage>(moves[move_index], has_unsearched_stage);
            if (cutoff || root_stage_stopped) {
                return;
            }
        }
    };
    auto search_root_promotion_stage = [&](const KingSafetyContext& king_safety) {
        auto search_promotion_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
            if (cutoff || root_stage_stopped || matches_any(move, searched_tt_move)) {
                return;
            }
            const ScoredMove scored_move = captured_piece == PieceType::None
                ? make_scored_legal_move<ScoringMode::MainSearch, false, true>(
                    pos, move, moved_piece, captured_piece, 0, tt_probe.moves, Move{}, PieceType::None)
                : make_scored_legal_move<ScoringMode::MainSearch, true, true>(
                    pos, move, moved_piece, captured_piece, 0, tt_probe.moves, Move{}, PieceType::None);
            search_one_root_move.template operator()<MoveGenerationStage::Promotion>(scored_move, true);
        };
        generate_legal_promotion_moves_with_info(pos, king_safety, search_promotion_move);
    };
    if (!cutoff && !root_stage_stopped) {
        const KingSafetyContext& king_safety = get_king_safety();
        search_root_promotion_stage(king_safety);
    }
    if (!cutoff && !root_stage_stopped) {
        const KingSafetyContext& king_safety = get_king_safety();
        CaptureMoveLists capture_moves = generate_legal_capture_scored_move_lists_for_searcher(
            pos,
            king_safety,
            0,
            tt_probe.moves,
            Move{},
            PieceType::None,
            searched_tt_move);
        sort_scored_moves(capture_moves.good);
        search_root_scored_moves.template operator()<MoveGenerationStage::GoodCapture>(capture_moves.good);
        Move searched_priority1{};
        Move searched_priority2{};
        Move searched_priority3{};
        if (!cutoff && !root_stage_stopped) {
            const ScoredMoveList priority_moves = ordered_priority_quiet_moves_for_stage(
                pos,
                0,
                tt_probe.moves,
                Move{},
                PieceType::None,
                searched_tt_move);
            if (priority_moves.size() > 0) {
                searched_priority1 = priority_moves[0].move;
            }
            if (priority_moves.size() > 1) {
                searched_priority2 = priority_moves[1].move;
            }
            if (priority_moves.size() > 2) {
                searched_priority3 = priority_moves[2].move;
            }
            search_root_scored_moves.template operator()<MoveGenerationStage::Killer>(priority_moves);
        }
        if (!cutoff && !root_stage_stopped) {
            const ScoredMoveList quiet_moves = ordered_moves_for_stage<MoveGenerationStage::QuietNonPromotion>(
                pos,
                king_safety,
                0,
                tt_probe.moves,
                Move{},
                PieceType::None,
                searched_tt_move,
                searched_priority1,
                searched_priority2,
                searched_priority3);
            search_root_scored_moves.template operator()<MoveGenerationStage::QuietNonPromotion>(quiet_moves);
        }
        if (!cutoff && !root_stage_stopped) {
            sort_scored_moves(capture_moves.bad);
            search_root_scored_moves.template operator()<MoveGenerationStage::BadCapture>(capture_moves.bad);
        }
    }
    if (root_stage_stopped) {
        return root_result;
    }

    if (!searched_any_move) {
        root_result.range = exact_range(in_check(pos, pos.side_to_move) ? -CheckmateScore : 0);
        result.score = representative_score(root_result.range);
        result.nodes = 1;
        return root_result;
    }

    if (root_range.lower != -Infinity && root_range.upper <= root_range.lower) {
        root_range.upper = root_range.lower;
    }
    if (tt_probe.has_score) {
        root_range = intersect_ranges(root_range, tt_probe.range);
    }
    assert(root_range.lower <= root_range.upper);

    root_result.range = root_range;
    result.score = representative_score(root_result.range);
    result.best_move = root_range.lower != -Infinity ? best_lower_move : best_upper_move;
    if (!is_valid_move(result.best_move)) {
        result.best_move = fallback_best_move;
    }
    if (!is_valid_move(result.best_move)) {
        result.best_move = make_fallback_result(pos).best_move;
    }

    store_tt_if_needed(pos.zobrist_key, depth, 0, root_range, best_lower_move, best_upper_move, fallback_best_move);

    result.nodes = state.nodes;
    return root_result;
}

SearchResult HeuristicSearcherV32::search_root_without_tt_probe(
    const Position& pos,
    int depth,
    SearchState& state
) {
    return search_fixed_depth(pos, depth, state, -Infinity, Infinity, false).result;
}

SearchResult HeuristicSearcherV32::search_best_move(const Position& pos, int depth) {
    killer_table_.clear();
    counter_move_table_.clear();
    SearchState state;
    RootSearchResult current = search_fixed_depth(pos, depth, state);
    if (!current.result.stopped && !state.stopped && !is_exact_range(current.range)) {
        current = search_fixed_depth(pos, depth, state, -Infinity, Infinity, false);
    }
    return current.result;
}

SearchResult HeuristicSearcherV32::search_best_move(const Position& pos, const SearchLimits& limits) {
    assert(limits.max_depth >= 0);

    killer_table_.clear();
    counter_move_table_.clear();
    SearchResult best = make_fallback_result(pos);
    best.depth = 0;

    SearchState state;
    state.has_deadline = limits.move_time.count() > 0;
    if (state.has_deadline) {
        state.deadline = Clock::now() + limits.move_time;
    }

    const int aspiration_window_cp = 50;

    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        if (depth == 1) {
            RootSearchResult current = search_fixed_depth(pos, depth, state);
            if (current.result.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return best;
            }
            best = current.result;
        } else if (EnableAspirationWindow) {
            int alpha = best.score - aspiration_window_cp;
            int beta = best.score + aspiration_window_cp;
            RootSearchResult current;
            bool retried_without_root_tt_probe = false;
            for (;;) {
                current = search_fixed_depth(pos, depth, state, alpha, beta);
                if (current.result.stopped || state.stopped) {
                    best.stopped = true;
                    best.nodes = state.nodes;
                    return best;
                }
                const bool exact = is_exact_range(current.range);
                const bool full_window = alpha == -Infinity && beta == Infinity;
                if (exact) {
                    break;
                }
                if (full_window) {
                    if (!retried_without_root_tt_probe) {
                        current = search_fixed_depth(pos, depth, state, -Infinity, Infinity, false);
                        retried_without_root_tt_probe = true;
                        if (current.result.stopped || state.stopped) {
                            best.stopped = true;
                            best.nodes = state.nodes;
                            return best;
                        }
                        if (is_exact_range(current.range)) {
                            break;
                        }
                    }
                    break;
                }
                if (current.range.upper <= alpha) {
                    beta = alpha;
                    alpha = -Infinity;
                } else if (current.range.lower >= beta) {
                    alpha = beta;
                    beta = Infinity;
                } else {
                    alpha = -Infinity;
                    beta = Infinity;
                }
            }
            best = current.result;
        } else {
            RootSearchResult current = search_fixed_depth(pos, depth, state);
            if (current.result.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return best;
            }
            if (!is_exact_range(current.range)) {
                current = search_fixed_depth(pos, depth, state, -Infinity, Infinity, false);
                if (current.result.stopped || state.stopped) {
                    best.stopped = true;
                    best.nodes = state.nodes;
                    return best;
                }
            }
            best = current.result;
        }
    }

    best.nodes = state.nodes;
    return best;
}

} // namespace chess
