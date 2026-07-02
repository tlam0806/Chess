#include "heuristic_searcher_v29_see.hpp"

#include "attacks.hpp"
#include "heuristic_searcher_v29_see_detail.hpp"
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

void HeuristicSearcherV29See::reward_quiet_cutoff(
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
    history_table_.store(side_to_move, scored_move.moved_piece, scored_move.move, depth);
    if (is_valid_move(prev_move) && prev_moved_piece != PieceType::None) {
        counter_history_table_.store(
            side_to_move,
            prev_moved_piece,
            prev_move,
            scored_move.moved_piece,
            scored_move.move,
            depth
        );
    }
}

void HeuristicSearcherV29See::penalize_failed_quiets(
    Color side_to_move,
    int depth,
    Move prev_move,
    PieceType prev_moved_piece,
    const HeuristicSearcherV29See::ScoredMoveList& failed_quiet_moves
) {
    for (const ScoredMove& failed_quiet : failed_quiet_moves) {
        history_table_.penalize(side_to_move, failed_quiet.moved_piece, failed_quiet.move, depth);
        if (is_valid_move(prev_move) && prev_moved_piece != PieceType::None) {
            counter_history_table_.penalize(
                side_to_move,
                prev_moved_piece,
                prev_move,
                failed_quiet.moved_piece,
                failed_quiet.move,
                depth
            );
        }
    }
}

bool HeuristicSearcherV29See::should_stop(SearchState& state) const {
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

KingSafetyContext HeuristicSearcherV29See::current_king_safety_context(const Position& pos) const {
    return cached_king_safety_context(pos, pos.side_to_move);
}

int HeuristicSearcherV29See::evaluate_current_position(const Position& pos) const {
    return evaluate_for_side_to_move(pos);
}



HeuristicSearcherV29See::SearchValue HeuristicSearcherV29See::quiescence(
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

    PositionStateSnapshot node_snapshot;
    bool node_snapshot_ready = false;
    auto get_node_snapshot = [&]() -> const PositionStateSnapshot& {
        if (!node_snapshot_ready) {
            node_snapshot = pos.make_state_snapshot();
            node_snapshot_ready = true;
        }
        return node_snapshot;
    };
    const KingSafetyContext king_safety = current_king_safety_context(pos);
    const bool side_in_check = king_safety.checkers != EmptyBB;
    ScoredMoveList moves;
    bool has_legal_move = false;
    auto score_qsearch_move =
        [&](Move move, PieceType moved_piece, PieceType captured_piece) {
            ScoredMove scored_move =
                make_scored_legal_move(
                    pos,
                    move,
                    moved_piece,
                    captured_piece,
                    ply,
                    MoveRange{},
                    Move{},
                    PieceType::None,
                    ScoringMode::Quiescence);
            has_legal_move = true;
            moves.push_back(scored_move);
        };
    if (side_in_check) {
        auto score_qsearch_noisy_move =
            [&](Move move, PieceType moved_piece, PieceType captured_piece) {
                score_qsearch_move(move, moved_piece, captured_piece);
            };
        auto score_qsearch_quiet_move = [&](Move move, PieceType moved_piece) {
            score_qsearch_move(move, moved_piece, PieceType::None);
        };
        {
            generate_legal_noisy_moves_with_info(pos, king_safety, score_qsearch_noisy_move);
        }
        {
            generate_legal_quiet_non_promotion_moves_with_info(
                pos,
                king_safety,
                score_qsearch_quiet_move);
        }
    } else {
        auto score_qsearch_noisy_move =
            [&](Move move, PieceType moved_piece, PieceType captured_piece) {
                score_qsearch_move(move, moved_piece, captured_piece);
            };
        {
            generate_legal_noisy_moves_with_info(pos, king_safety, score_qsearch_noisy_move);
        }
    }

    if (!has_legal_move && side_in_check) {
        return SearchValue{exact_range(-CheckmateScore + ply)};
    }
    // Non-check quiescence intentionally does not detect stalemate.
    // Root/negamax handle terminal positions when depth remains.
 
    if (side_in_check) {
        if (q_depth >= MaxCheckEvasionQuiescenceDepth) {
            return SearchValue{exact_range(evaluate_current_position(pos))};
        }

        ScoreRange node_range;
        node_range.lower = -Infinity;
        node_range.upper = -Infinity;
        sort_scored_moves(moves);

        for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
            const ScoredMove& scored_move = moves[move_index];
            ScoreRange move_range;
            {
                SnapshotMoveUndoGuard move_guard(pos, get_node_snapshot(), scored_move.move, scored_move.moved_piece, scored_move.captured_piece);
                SearchValue child = quiescence(pos, -beta, -alpha, ply + 1, q_depth + 1, state);
                move_range = negate_range(child.range);
            }
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }

            node_range.lower = std::max(node_range.lower, move_range.lower);
            node_range.upper = std::max(node_range.upper, move_range.upper);
            if (node_range.lower >= beta) {
                return SearchValue{ScoreRange{node_range.lower, Infinity}};
            }
            if (move_range.lower != -Infinity && move_range.lower > alpha) {
                alpha = move_range.lower;
            }
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
    sort_scored_moves(moves);
    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];
        ScoreRange move_range;
        {
            SnapshotMoveUndoGuard move_guard(pos, get_node_snapshot(), scored_move.move, scored_move.moved_piece, scored_move.captured_piece);
            SearchValue child = quiescence(pos, -beta, -alpha, ply + 1, q_depth + 1, state);
            move_range = negate_range(child.range);
        }
        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }

        node_range.lower = std::max(node_range.lower, move_range.lower);
        node_range.upper = std::max(node_range.upper, move_range.upper);
        if (node_range.lower >= beta) {
            return SearchValue{ScoreRange{node_range.lower, Infinity}};
        }
        if (move_range.lower != -Infinity && move_range.lower > alpha) {
            alpha = move_range.lower;
        }
    }

    return SearchValue{node_range};
}

HeuristicSearcherV29See::SearchValue HeuristicSearcherV29See::negamax(
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
                SnapshotMoveUndoGuard move_guard(pos, get_node_snapshot(), scored_move.move, scored_move.moved_piece, scored_move.captured_piece);
                SearchValue child = negamax(
                    pos,
                    depth - 1,
                    ply + 1,
                    scored_move.move,
                    scored_move.moved_piece,
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
    for (MoveGenerationStage generation_stage : {
             MoveGenerationStage::Promotion,
             MoveGenerationStage::GoodCapture,
             MoveGenerationStage::QuietNonPromotion,
             MoveGenerationStage::BadCapture
         }) {
        if (cutoff) {
            break;
        }
        const KingSafetyContext& king_safety = get_king_safety();
        const ScoredMoveList moves = ordered_moves_for_stage(
            pos,
            king_safety,
            ply,
            tt_probe.moves,
            prev_move,
            prev_moved_piece,
            searched_tt_move,
            generation_stage);

        for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
            const ScoredMove& scored_move = moves[move_index];
            const PieceType moved_piece = scored_move.moved_piece;
            if (!is_valid_move(fallback_best_move)) {
                fallback_best_move = scored_move.move;
            }

            ScoreRange move_range;
            {
                SnapshotMoveUndoGuard move_guard(pos, get_node_snapshot(), scored_move.move, scored_move.moved_piece, scored_move.captured_piece);

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
                        return SearchValue{exact_range(0)};
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
                            return SearchValue{exact_range(0)};
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
                        return SearchValue{exact_range(0)};
                    }
                }
            }
            searched_any_move = true;
            update_best_range(node_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (node_range.lower >= beta) {
                reward_quiet_cutoff(pos.side_to_move, depth, ply, scored_move, prev_move, prev_moved_piece);
                penalize_failed_quiets(pos.side_to_move, depth, prev_move, prev_moved_piece, failed_quiet_moves);
                const bool has_unsearched_stage =
                    generation_stage != MoveGenerationStage::BadCapture
                    || move_index + 1 < moves.size();
                node_range.upper = has_unsearched_stage ? Infinity : node_range.upper;
                cutoff = true;
                break;
            }
            if (is_quiet_move(scored_move)) {
                failed_quiet_moves.push_back(scored_move);
            }
        }
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

SearchResult HeuristicSearcherV29See::make_fallback_result(const Position& pos) const {
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

HeuristicSearcherV29See::RootSearchResult HeuristicSearcherV29See::search_fixed_depth(
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
                SnapshotMoveUndoGuard move_guard(pos, node_snapshot, scored_move.move, scored_move.moved_piece, scored_move.captured_piece);
                SearchValue child = negamax(
                    pos,
                    depth - 1,
                    1,
                    scored_move.move,
                    scored_move.moved_piece,
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
    for (MoveGenerationStage generation_stage : {
             MoveGenerationStage::Promotion,
             MoveGenerationStage::GoodCapture,
             MoveGenerationStage::QuietNonPromotion,
             MoveGenerationStage::BadCapture
         }) {
        if (cutoff) {
            break;
        }
        const KingSafetyContext& king_safety = get_king_safety();
        const ScoredMoveList moves = ordered_moves_for_stage(
            pos,
            king_safety,
            0,
            tt_probe.moves,
            Move{},
            PieceType::None,
            searched_tt_move,
            generation_stage);

        for (std::size_t move_index = 0; root_range.lower < beta && move_index < moves.size(); ++move_index) {
            const ScoredMove& scored_move = moves[move_index];
            const PieceType moved_piece = scored_move.moved_piece;
            if (!is_valid_move(fallback_best_move)) {
                fallback_best_move = scored_move.move;
            }

            ScoreRange move_range;
            {
                SnapshotMoveUndoGuard move_guard(pos, node_snapshot, scored_move.move, scored_move.moved_piece, scored_move.captured_piece);
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
                return root_result;
            }

            searched_any_move = true;

            update_best_range(root_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (root_range.lower >= beta) {
                const bool has_unsearched_stage =
                    generation_stage != MoveGenerationStage::BadCapture
                    || move_index + 1 < moves.size();
                root_range.upper = has_unsearched_stage ? Infinity : root_range.upper;
                cutoff = true;
                break;
            }
        }
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

SearchResult HeuristicSearcherV29See::search_root_without_tt_probe(
    const Position& pos,
    int depth,
    SearchState& state
) {
    return search_fixed_depth(pos, depth, state, -Infinity, Infinity, false).result;
}

SearchResult HeuristicSearcherV29See::search_best_move(const Position& pos, int depth) {
    killer_table_.clear();
    SearchState state;
    return search_fixed_depth(pos, depth, state).result;
}

SearchResult HeuristicSearcherV29See::search_best_move(const Position& pos, const SearchLimits& limits) {
    assert(limits.max_depth >= 0);

    killer_table_.clear();
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
            for (;;) {
                current = search_fixed_depth(pos, depth, state, alpha, beta);
                if (current.result.stopped || state.stopped) {
                    best.stopped = true;
                    best.nodes = state.nodes;
                    return best;
                }
                const bool exact = current.range.lower == current.range.upper;
                const bool full_window = alpha == -Infinity && beta == Infinity;
                if (exact || full_window) {
                    break;
                }
                if (current.range.upper <= alpha) {
                    beta = alpha;
                    alpha = -Infinity;
                } else if (current.range.lower >= beta) {
                    alpha = beta;
                    beta = Infinity;
                } else {
                    break;
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
            best = current.result;
        }
    }

    best.nodes = state.nodes;
    return best;
}

} // namespace chess
