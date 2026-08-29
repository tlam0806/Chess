#include "nnue_searcher_v38.hpp"

#include "attacks.hpp"
#include "nnue_searcher_v38_detail.hpp"
#include "evaluate.hpp"
#include "legal_noisy_generator.hpp"
#include "legal_non_capture_generator.hpp"
#include "move_undo_guard.hpp"
#include "zobrist.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace chess {

namespace {

void assert_accumulator_matches_full_recompute(
    const PhaseQuantizedNnueModel& model,
    const PhaseQuantizedNnueAccumulator& accumulator,
    const Position& pos
) {
#ifndef NDEBUG
    assert(accumulator.matches_full_recompute(pos));
    assert(accumulator.evaluate_cp_rounded(pos) == model.evaluate_cp_rounded(pos));
#else
    static_cast<void>(model);
    static_cast<void>(accumulator);
    static_cast<void>(pos);
#endif
}

class ScopedNnueUpdate {
public:
    ScopedNnueUpdate(
        PhaseQuantizedNnueAccumulator& accumulator,
        Move move,
        Color moving_color,
        PieceType moved_piece,
        PieceType captured_piece,
        const Position& after
    )
        : accumulator_(&accumulator),
          undo_(accumulator.make_move_with_undo(
              move,
              moving_color,
              moved_piece,
              captured_piece,
              after)) {
    }

    ~ScopedNnueUpdate() {
        if (accumulator_ != nullptr) {
            accumulator_->undo(undo_);
        }
    }

    ScopedNnueUpdate(const ScopedNnueUpdate&) = delete;
    ScopedNnueUpdate& operator=(const ScopedNnueUpdate&) = delete;

    ScopedNnueUpdate(ScopedNnueUpdate&& other) noexcept
        : accumulator_(other.accumulator_),
          undo_(std::move(other.undo_)) {
        other.accumulator_ = nullptr;
    }

    ScopedNnueUpdate& operator=(ScopedNnueUpdate&&) = delete;

private:
    PhaseQuantizedNnueAccumulator* accumulator_ = nullptr;
    PhaseQuantizedNnueUndo undo_;
};

class ScopedRepetitionStatsCommit {
public:
    ScopedRepetitionStatsCommit(
        RepetitionStack::Stats& destination,
        const RepetitionStack& source
    ) noexcept
        : destination_(destination), source_(source) {
        destination_ = {};
    }

    ~ScopedRepetitionStatsCommit() noexcept {
        destination_ = source_.stats();
    }

    ScopedRepetitionStatsCommit(const ScopedRepetitionStatsCommit&) = delete;
    ScopedRepetitionStatsCommit& operator=(const ScopedRepetitionStatsCommit&) = delete;

private:
    RepetitionStack::Stats& destination_;
    const RepetitionStack& source_;
};

SearchResult ensure_legal_root_move(
    const Position& pos,
    SearchResult result
) {
    const std::vector<Move> legal_moves = generate_legal_moves(pos);
    if (legal_moves.empty()) {
        result.ponder_move = {};
        return result;
    }
    if (std::find(
            legal_moves.begin(), legal_moves.end(), result.best_move)
        == legal_moves.end()) {
        result.best_move = legal_moves.front();
        result.ponder_move = {};
    }
    return result;
}

void inherit_ponder_move(
    SearchResult& current,
    const SearchResult& previous
) {
    if (current.ponder_move.value == 0
        && current.best_move == previous.best_move) {
        current.ponder_move = previous.ponder_move;
    }
}

bool has_non_pawn_material(const Position& pos, Color color) {
    const int color_index = static_cast<int>(color);
    for (PieceType piece :
         {PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen}) {
        if (pos.pieces[color_index][static_cast<int>(piece)] != EmptyBB) {
            return true;
        }
    }
    return false;
}

} // namespace

bool NnueSearcherV38::can_null_move_prune(
    const Position& pos,
    int depth,
    const SearchState& state
) const {
    return selective_config_.enable_null_move
        && selective_config_.null_move_reduction >= 1
        && depth >= selective_config_.null_move_min_depth
        && !state.in_null_move
        && !in_check(pos, pos.side_to_move)
        && has_non_pawn_material(pos, pos.side_to_move);
}

void NnueSearcherV38::make_null_move(Position& pos) {
    if (pos.en_passant_square != NoSquare) {
        pos.zobrist_key ^= zobrist::en_passant_file_key(file_of(pos.en_passant_square));
        pos.en_passant_square = NoSquare;
    }
    pos.zobrist_key ^= zobrist::side_key();
    pos.side_to_move = opposite(pos.side_to_move);
}

bool NnueSearcherV38::castling_rights_changed(
    const PositionStateSnapshot& before,
    const Position& after
) {
    const std::uint8_t after_rights = static_cast<std::uint8_t>(
        (after.white_can_castle_kingside ? 1 : 0)
        | (after.white_can_castle_queenside ? 2 : 0)
        | (after.black_can_castle_kingside ? 4 : 0)
        | (after.black_can_castle_queenside ? 8 : 0));
    return before.castling_rights != after_rights;
}

void NnueSearcherV38::initialize_repetition(
    SearchState& state,
    const Position& pos,
    std::span<const HashKey> game_history
) const {
    state.repetition.reset(
        game_history,
        pos.zobrist_key,
        pos.halfmove_clock,
        twofold_search_draw_enabled_);
}

bool NnueSearcherV38::history_draw(
    const Position& pos,
    SearchState& state
) const {
    if (!state.repetition_enabled || state.in_null_move) {
        return false;
    }
    if (state.repetition.current_is_threefold()) {
        state.repetition.record_threefold_draw();
        return true;
    }
    if (twofold_search_draw_enabled_
        && state.repetition.current_repeats_in_search_path()) {
        state.repetition.record_search_cycle_draw();
        return true;
    }
    if (pos.halfmove_clock < 100) {
        return false;
    }

    // A claimable 50-move draw does not override checkmate. Generating moves
    // is confined to the extremely rare >=100-and-in-check case; the usual
    // hot path is only the predictable halfmove_clock comparison above.
    if (in_check(pos, pos.side_to_move)
        && generate_legal_moves(pos).empty()) {
        return false;
    }
    state.repetition.record_fifty_move_draw();
    return true;
}

bool NnueSearcherV38::allow_repetition_tt_score(SearchState& state) const {
    if (!state.repetition_enabled
        || state.in_null_move
        || !state.repetition.has_twofold_position()) {
        return true;
    }
    state.repetition.record_tt_score_suppression();
    return false;
}

void NnueSearcherV38::reward_quiet_cutoff(
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

void NnueSearcherV38::penalize_failed_quiets(
    Color side_to_move,
    int depth,
    Move prev_move,
    PieceType prev_moved_piece,
    const NnueSearcherV38::ScoredMoveList& failed_quiet_moves
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

void NnueSearcherV38::record_beta_cutoff(
    const ScoredMove& scored_move,
    std::size_t move_index,
    CutoffStage stage
) {
    if (!move_ordering_stats_enabled_) {
        return;
    }
    ++move_ordering_stats_.beta_cutoffs;
    ++move_ordering_stats_.cutoff_move_index[
        std::min<std::size_t>(
            move_index,
            move_ordering_stats_.cutoff_move_index.size() - 1)];
    ++move_ordering_stats_.cutoff_stage[static_cast<std::size_t>(stage)];

    const bool capture =
        scored_captured_piece(scored_move) != PieceType::None;
    const bool promotion =
        promotion_piece(scored_move.move) != PieceType::None;
    if (capture) {
        ++move_ordering_stats_.cutoff_by_capture;
    }
    if (scored_gives_check(scored_move)) {
        ++move_ordering_stats_.cutoff_by_check;
    }
    if (promotion) {
        ++move_ordering_stats_.cutoff_by_promotion;
    }
    if (!capture && !promotion) {
        ++move_ordering_stats_.cutoff_by_quiet;
    }
}

bool NnueSearcherV38::should_stop(SearchState& state) const {
    if ((state.nodes & 1023ULL) != 0) {
        return false;
    }
    if (state.stop_requested != nullptr
        && state.stop_requested->load(std::memory_order_relaxed)) {
        state.stopped = true;
        return true;
    }
    if (!state.has_deadline) {
        return false;
    }
    if (Clock::now() >= state.deadline) {
        state.stopped = true;
        return true;
    }
    return false;
}

KingSafetyContext NnueSearcherV38::current_king_safety_context(const Position& pos) const {
    return cached_king_safety_context(pos, pos.side_to_move);
}

int NnueSearcherV38::evaluate_current_position(const Position& pos, const SearchState& state) const {
    assert_accumulator_matches_full_recompute(model_, state.accumulator, pos);
    return state.accumulator.evaluate_cp_rounded(pos);
}



NnueSearcherV38::SearchValue NnueSearcherV38::quiescence(
    Position& pos,
    int alpha,
    int beta,
    int ply,
    int q_depth,
    SearchState& state,
    bool check_current_repetition
) {
    assert(alpha < beta);
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{exact_range(0)};
    }

    if (check_current_repetition && history_draw(pos, state)) {
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
    auto generate_qsearch_capture_stage = [&](bool allow_see_pruning) {
        ScoredMoveList moves;
        auto score_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
            if (allow_see_pruning
                && selective_config_.enable_qsearch_see_pruning) {
                ++selective_stats_.qsearch_see_evaluations;
                const int see = static_exchange_eval(
                    pos, move, moved_piece, captured_piece);
                if (see < selective_config_.qsearch_see_threshold
                    && !gives_check_fast(
                        pos, move, moved_piece, captured_piece)) {
                    ++selective_stats_.qsearch_see_pruned_moves;
                    return;
                }
            }
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
            const Color moving_color = pos.side_to_move;
            SnapshotMoveUndoGuard move_guard(
                pos,
                get_node_snapshot(),
                scored_move.move,
                moved_piece,
                captured_piece);
            const ScopedRepetitionPush repetition_push(
                state.repetition_enabled && !state.in_null_move
                    ? &state.repetition
                    : nullptr,
                pos.zobrist_key,
                pos.halfmove_clock == 0
                    || ((moved_piece == PieceType::King
                         || moved_piece == PieceType::Rook)
                        && castling_rights_changed(get_node_snapshot(), pos)),
                pos.halfmove_clock);
            const ScopedNnueUpdate nnue_update(
                state.accumulator,
                scored_move.move,
                moving_color,
                moved_piece,
                captured_piece,
                pos);
            SearchValue child = quiescence(
                pos,
                -beta,
                -alpha,
                ply + 1,
                q_depth + 1,
                state,
                true);
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

        const ScoredMoveList capture_moves =
            generate_qsearch_capture_stage(false);
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
            return SearchValue{exact_range(evaluate_current_position(pos, state))};
        }
        if (search_qsearch_stage(quiet_evasion_moves, node_range)) {
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }
            return SearchValue{ScoreRange{node_range.lower, Infinity}};
        }

        return SearchValue{node_range};
    }

    int best_score = evaluate_current_position(pos, state);
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

    const ScoredMoveList capture_moves = generate_qsearch_capture_stage(true);
    if (search_qsearch_stage(capture_moves, node_range)) {
        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }
        return SearchValue{ScoreRange{node_range.lower, Infinity}};
    }

    return SearchValue{node_range};
}

NnueSearcherV38::SearchValue NnueSearcherV38::negamax(
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
    if (ply == 1) {
        state.ply_one_best_move = {};
    }
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{exact_range(0)};
    }


    if (history_draw(pos, state)) {
        return SearchValue{exact_range(0)};
    }

    const bool allow_tt_score = allow_repetition_tt_score(state);
    TTProbeResult tt_probe = probe_tt(
        pos.zobrist_key,
        depth,
        alpha,
        beta,
        ply,
        true,
        allow_tt_score);
    if (tt_probe.hit) {
        if (ply == 1) {
            state.ply_one_best_move = preferred_tt_move(tt_probe.moves);
        }
        return SearchValue{tt_probe.range};
    }

    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply, 0, state, false);
    }

    const bool reverse_futility_allowed =
        selective_config_.enable_reverse_futility
        && selective_config_.reverse_futility_max_depth >= 1
        && selective_config_.reverse_futility_base_margin >= 0
        && selective_config_.reverse_futility_margin_per_depth >= 0
        && depth <= selective_config_.reverse_futility_max_depth
        && beta == alpha + 1
        && beta > -MateScoreThreshold
        && beta < MateScoreThreshold
        && !state.in_null_move
        && !in_check(pos, pos.side_to_move)
        && has_non_pawn_material(pos, pos.side_to_move);
    if (reverse_futility_allowed) {
        ++selective_stats_.reverse_futility_evaluations;
        const int static_eval = evaluate_current_position(pos, state);
        const std::int64_t margin =
            static_cast<std::int64_t>(
                selective_config_.reverse_futility_base_margin)
            + static_cast<std::int64_t>(
                  selective_config_.reverse_futility_margin_per_depth)
                * depth;
        if (static_cast<std::int64_t>(static_eval) - margin >= beta) {
            ++selective_stats_.reverse_futility_cutoffs;
            return SearchValue{lower_range(beta)};
        }
    }

    if (can_null_move_prune(pos, depth, state)) {
        Position null_pos = pos;
        make_null_move(null_pos);

        const bool previous_in_null_move = state.in_null_move;
        state.in_null_move = true;
        ++selective_stats_.null_move_searches;
        const int null_depth = std::max(
            0,
            depth - 1 - selective_config_.null_move_reduction);
        const SearchValue child = negamax(
            null_pos,
            null_depth,
            ply + 1,
            Move{},
            PieceType::None,
            -beta,
            -beta + 1,
            state);
        state.in_null_move = previous_in_null_move;

        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }
        const ScoreRange null_range = negate_range(child.range);
        if (null_range.lower >= beta) {
            ++selective_stats_.null_move_cutoffs;
            return SearchValue{lower_range(beta)};
        }
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
    std::size_t searched_move_count = 0;
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
                const Color moving_color = pos.side_to_move;
                const PieceType moved_piece = scored_moved_piece(scored_move);
                const PieceType captured_piece = scored_captured_piece(scored_move);
                SnapshotMoveUndoGuard move_guard(
                    pos,
                    get_node_snapshot(),
                    scored_move.move,
                    moved_piece,
                    captured_piece);
                const ScopedRepetitionPush repetition_push(
                    state.repetition_enabled && !state.in_null_move
                        ? &state.repetition
                        : nullptr,
                    pos.zobrist_key,
                    pos.halfmove_clock == 0
                        || ((moved_piece == PieceType::King
                             || moved_piece == PieceType::Rook)
                            && castling_rights_changed(get_node_snapshot(), pos)),
                    pos.halfmove_clock);
                const ScopedNnueUpdate nnue_update(
                    state.accumulator,
                    scored_move.move,
                    moving_color,
                    moved_piece,
                    captured_piece,
                    pos);
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
            if (ply == 1) {
                state.ply_one_best_move = node_range.lower != -Infinity
                    ? best_lower_move
                    : best_upper_move;
            }
            if (node_range.lower >= beta) {
                record_beta_cutoff(
                    scored_move,
                    searched_move_count,
                    CutoffStage::TtLower);
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
                    fallback_best_move,
                    allow_tt_score);
                return SearchValue{node_range};
            }
            ++searched_move_count;
            if (is_quiet_move(scored_move)) {
                failed_quiet_moves.push_back(scored_move);
            }
        }
    }

    bool cutoff = node_range.lower >= beta;
    bool stage_stopped = false;
    bool late_move_pruned_at_node = false;
    auto search_one_scored_move =
        [&]<MoveGenerationStage Stage>(const ScoredMove& scored_move, bool has_unsearched_stage) {
        if (cutoff || stage_stopped) {
            return;
        }
        const std::size_t current_move_index = searched_move_count++;
        const PieceType moved_piece = scored_moved_piece(scored_move);
        const PieceType captured_piece = scored_captured_piece(scored_move);
        if (!is_valid_move(fallback_best_move)) {
            fallback_best_move = scored_move.move;
        }

        ScoreRange move_range;
        {
            const Color moving_color = pos.side_to_move;
            SnapshotMoveUndoGuard move_guard(
                pos,
                get_node_snapshot(),
                scored_move.move,
                moved_piece,
                captured_piece);
            const ScopedRepetitionPush repetition_push(
                state.repetition_enabled && !state.in_null_move
                    ? &state.repetition
                    : nullptr,
                pos.zobrist_key,
                pos.halfmove_clock == 0
                    || ((moved_piece == PieceType::King
                         || moved_piece == PieceType::Rook)
                        && castling_rights_changed(get_node_snapshot(), pos)),
                pos.halfmove_clock);
            const ScopedNnueUpdate nnue_update(
                state.accumulator,
                scored_move.move,
                moving_color,
                moved_piece,
                captured_piece,
                pos);

            const bool can_lmr =
                selective_config_.enable_lmr
                && selective_config_.lmr_divisor > 0.0
                && depth > 2
                && depth >= selective_config_.lmr_min_depth
                && current_move_index >= selective_config_.lmr_min_move_index
                && king_safety_cache
                && king_safety_cache->checkers == EmptyBB
                && !scored_gives_check(scored_move)
                && captured_piece == PieceType::None
                && promotion_piece(scored_move.move) == PieceType::None;
            const int full_child_depth = depth - 1;
            int child_depth = full_child_depth;
            bool reduced_search = false;
            if (can_lmr) {
                int reduction = static_cast<int>(
                    selective_config_.lmr_base
                    + std::log(static_cast<double>(depth))
                        * std::log(static_cast<double>(current_move_index + 1))
                        / selective_config_.lmr_divisor);
                reduction = std::clamp(reduction, 1, depth - 2);
                child_depth = full_child_depth - reduction;
                reduced_search = true;
                ++selective_stats_.lmr_searches;
            }

            if (searched_any_move && beta > alpha + 1) {
                SearchValue scout = negamax(
                    pos,
                    child_depth,
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
                if (reduced_search
                    && (move_range.lower > alpha || move_range.upper > alpha)) {
                    ++selective_stats_.lmr_researches;
                    SearchValue full = negamax(
                        pos,
                        full_child_depth,
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
                } else if (!scout_proves_fail_low && !scout_proves_fail_high) {
                    SearchValue full = negamax(
                        pos,
                        full_child_depth,
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
                    child_depth,
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
                if (reduced_search
                    && (move_range.lower > alpha || move_range.upper > alpha)) {
                    ++selective_stats_.lmr_researches;
                    SearchValue full = negamax(
                        pos,
                        full_child_depth,
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
            }
        }
        searched_any_move = true;
        update_best_range(node_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
        if (ply == 1) {
            state.ply_one_best_move = node_range.lower != -Infinity
                ? best_lower_move
                : best_upper_move;
        }
        if (node_range.lower >= beta) {
            constexpr CutoffStage cutoff_stage = [] {
                if constexpr (Stage == MoveGenerationStage::Promotion) {
                    return CutoffStage::Promotion;
                } else if constexpr (Stage == MoveGenerationStage::GoodCapture) {
                    return CutoffStage::GoodCapture;
                } else if constexpr (Stage == MoveGenerationStage::Killer) {
                    return CutoffStage::PriorityQuiet;
                } else if constexpr (Stage == MoveGenerationStage::QuietNonPromotion) {
                    return CutoffStage::Quiet;
                } else {
                    return CutoffStage::BadCapture;
                }
            }();
            record_beta_cutoff(
                scored_move,
                current_move_index,
                cutoff_stage);
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
            if constexpr (Stage == MoveGenerationStage::BadCapture) {
                const ScoredMove& scored_move = moves[move_index];
                const PieceType moved_piece = scored_moved_piece(scored_move);
                const PieceType captured_piece = scored_captured_piece(scored_move);
                const bool can_prune_bad_capture =
                    selective_config_.enable_main_search_see_pruning
                    && selective_config_.main_search_see_max_depth >= 1
                    && selective_config_.main_search_see_margin_per_depth >= 0
                    && ply > 0
                    && depth <= selective_config_.main_search_see_max_depth
                    && beta == alpha + 1
                    && searched_any_move
                    && king_safety_cache
                    && king_safety_cache->checkers == EmptyBB
                    && !gives_check_fast(
                        pos,
                        scored_move.move,
                        moved_piece,
                        captured_piece);
                if (can_prune_bad_capture) {
                    ++selective_stats_.main_search_see_evaluations;
                    const int see = static_exchange_eval(
                        pos,
                        scored_move.move,
                        moved_piece,
                        captured_piece);
                    const std::int64_t margin =
                        static_cast<std::int64_t>(
                            selective_config_.main_search_see_margin_per_depth)
                        * depth;
                    if (static_cast<std::int64_t>(see) < -margin) {
                        ++selective_stats_.main_search_see_pruned_moves;
                        continue;
                    }
                }
            }
            if constexpr (Stage == MoveGenerationStage::QuietNonPromotion) {
                const std::uint64_t depth_squared =
                    static_cast<std::uint64_t>(depth)
                    * static_cast<std::uint64_t>(depth);
                const std::uint64_t move_threshold =
                    static_cast<std::uint64_t>(
                        selective_config_.late_move_pruning_base)
                    + static_cast<std::uint64_t>(
                          selective_config_.late_move_pruning_depth_multiplier)
                        * depth_squared;
                const bool can_prune_late_move =
                    selective_config_.enable_late_move_pruning
                    && selective_config_.late_move_pruning_max_depth >= 1
                    && depth <= selective_config_.late_move_pruning_max_depth
                    && beta == alpha + 1
                    && searched_any_move
                    && king_safety_cache
                    && king_safety_cache->checkers == EmptyBB
                    && !scored_gives_check(moves[move_index])
                    && searched_move_count >= move_threshold;
                if (can_prune_late_move) {
                    if (!late_move_pruned_at_node) {
                        late_move_pruned_at_node = true;
                        ++selective_stats_.late_move_pruned_nodes;
                    }
                    ++selective_stats_.late_move_pruned_moves;
                    continue;
                }
            }
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
    // LMP deliberately assumes that the skipped quiet moves cannot improve the
    // current result.  Widening the upper bound here would undo that
    // assumption: the caller would regard every pruned node as unresolved and
    // repeatedly re-search it, which can make enabling LMP increase the node
    // count by an order of magnitude.
    assert(node_range.lower <= node_range.upper);
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
        fallback_best_move,
        allow_tt_score);

    return SearchValue{node_range};
}

SearchResult NnueSearcherV38::make_fallback_result(const Position& pos) const {
    SearchResult result;
    const ScoredMoveList moves = ordered_moves(pos, 0);
    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }
    result.best_move = moves[0].move;
    result.score = model_.evaluate_cp_rounded(pos);
    result.nodes = 1;
    return result;
}

NnueSearcherV38::RootSearchResult NnueSearcherV38::search_fixed_depth(
    Position pos,
    int depth,
    SearchState& state,
    int alpha,
    int beta,
    bool allow_root_tt_probe
) {
    assert(depth >= 0);
    state.accumulator.reset(model_, pos);

    RootSearchResult root_result;
    SearchResult& result = root_result.result;
    result.depth = depth;

    if (depth == 0) {
        root_result.range = quiescence(
            pos, -Infinity, Infinity, 0, 0, state, true).range;
        result.score = representative_score(root_result.range);
        result.nodes = state.nodes;
        return root_result;
    }

    const bool allow_tt_score = allow_repetition_tt_score(state);
    TTProbeResult tt_probe = probe_tt(
        pos.zobrist_key,
        depth,
        alpha,
        beta,
        0,
        allow_root_tt_probe,
        allow_tt_score);
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
    Move best_lower_reply{};
    Move best_upper_reply{};
    Move fallback_reply{};
    bool searched_any_move = false;
    std::size_t searched_move_count = 0;
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
            Move candidate_reply{};
            {
                const Color moving_color = pos.side_to_move;
                const PieceType moved_piece = scored_moved_piece(scored_move);
                const PieceType captured_piece = scored_captured_piece(scored_move);
                SnapshotMoveUndoGuard move_guard(
                    pos,
                    node_snapshot,
                    scored_move.move,
                    moved_piece,
                    captured_piece);
                const ScopedRepetitionPush repetition_push(
                    state.repetition_enabled
                        ? &state.repetition
                        : nullptr,
                    pos.zobrist_key,
                    pos.halfmove_clock == 0
                        || ((moved_piece == PieceType::King
                             || moved_piece == PieceType::Rook)
                            && castling_rights_changed(node_snapshot, pos)),
                    pos.halfmove_clock);
                const ScopedNnueUpdate nnue_update(
                    state.accumulator,
                    scored_move.move,
                    moving_color,
                    moved_piece,
                    captured_piece,
                    pos);
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
                candidate_reply = state.ply_one_best_move;
            }
            if (state.stopped) {
                result.stopped = true;
                result.nodes = state.nodes;
                return root_result;
            }

            update_best_range(root_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (fallback_best_move == scored_move.move) {
                fallback_reply = candidate_reply;
            }
            if (best_lower_move == scored_move.move) {
                best_lower_reply = candidate_reply;
            }
            if (best_upper_move == scored_move.move) {
                best_upper_reply = candidate_reply;
            }
            if (root_range.lower >= beta) {
                record_beta_cutoff(
                    scored_move,
                    searched_move_count,
                    CutoffStage::TtLower);
                root_range.upper = Infinity;
            } else {
                ++searched_move_count;
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
        const std::size_t current_move_index = searched_move_count++;
        const PieceType moved_piece = scored_moved_piece(scored_move);
        const PieceType captured_piece = scored_captured_piece(scored_move);
        if (!is_valid_move(fallback_best_move)) {
            fallback_best_move = scored_move.move;
        }

        ScoreRange move_range;
        Move candidate_reply{};
        {
            const Color moving_color = pos.side_to_move;
            SnapshotMoveUndoGuard move_guard(
                pos,
                node_snapshot,
                scored_move.move,
                moved_piece,
                captured_piece);
            const ScopedRepetitionPush repetition_push(
                state.repetition_enabled
                    ? &state.repetition
                    : nullptr,
                pos.zobrist_key,
                pos.halfmove_clock == 0
                    || ((moved_piece == PieceType::King
                         || moved_piece == PieceType::Rook)
                        && castling_rights_changed(node_snapshot, pos)),
                pos.halfmove_clock);
            const ScopedNnueUpdate nnue_update(
                state.accumulator,
                scored_move.move,
                moving_color,
                moved_piece,
                captured_piece,
                pos);
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
            candidate_reply = state.ply_one_best_move;
        }
        if (state.stopped) {
            result.stopped = true;
            result.nodes = state.nodes;
            root_stage_stopped = true;
            return;
        }

        searched_any_move = true;

        update_best_range(root_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
        if (fallback_best_move == scored_move.move) {
            fallback_reply = candidate_reply;
        }
        if (best_lower_move == scored_move.move) {
            best_lower_reply = candidate_reply;
        }
        if (best_upper_move == scored_move.move) {
            best_upper_reply = candidate_reply;
        }
        if (root_range.lower >= beta) {
            constexpr CutoffStage cutoff_stage = [] {
                if constexpr (Stage == MoveGenerationStage::Promotion) {
                    return CutoffStage::Promotion;
                } else if constexpr (Stage == MoveGenerationStage::GoodCapture) {
                    return CutoffStage::GoodCapture;
                } else if constexpr (Stage == MoveGenerationStage::Killer) {
                    return CutoffStage::PriorityQuiet;
                } else if constexpr (Stage == MoveGenerationStage::QuietNonPromotion) {
                    return CutoffStage::Quiet;
                } else {
                    return CutoffStage::BadCapture;
                }
            }();
            record_beta_cutoff(
                scored_move,
                current_move_index,
                cutoff_stage);
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
    result.ponder_move = root_range.lower != -Infinity
        ? best_lower_reply
        : best_upper_reply;
    if (!is_valid_move(result.best_move)) {
        result.best_move = fallback_best_move;
        result.ponder_move = fallback_reply;
    }
    if (!is_valid_move(result.best_move)) {
        result.best_move = make_fallback_result(pos).best_move;
        result.ponder_move = {};
    }

    store_tt_if_needed(
        pos.zobrist_key,
        depth,
        0,
        root_range,
        best_lower_move,
        best_upper_move,
        fallback_best_move,
        allow_tt_score);

    result.nodes = state.nodes;
    return root_result;
}

SearchResult NnueSearcherV38::search_root_without_tt_probe(
    const Position& pos,
    int depth,
    SearchState& state
) {
    return search_fixed_depth(pos, depth, state, -Infinity, Infinity, false).result;
}

SearchResult NnueSearcherV38::search_best_move(const Position& pos, int depth) {
    return search_best_move_impl(pos, depth, {}, false);
}

SearchResult NnueSearcherV38::search_best_move(
    const Position& pos,
    int depth,
    std::span<const HashKey> game_history
) {
    return search_best_move_impl(pos, depth, game_history, true);
}

SearchResult NnueSearcherV38::search_best_move_impl(
    const Position& pos,
    int depth,
    std::span<const HashKey> game_history,
    bool enable_repetition
) {
    if (enable_repetition) {
        tt_.advance_generation();
    }
    killer_table_.clear();
    counter_move_table_.clear();
    SearchState state;
    const ScopedRepetitionStatsCommit repetition_stats_commit(
        repetition_stats_, state.repetition);
    state.repetition_enabled = enable_repetition;
    if (enable_repetition) {
        initialize_repetition(state, pos, game_history);
        if (history_draw(pos, state)) {
            SearchResult result = make_fallback_result(pos);
            result.score = 0;
            result.depth = depth;
            result.nodes = 1;
            return ensure_legal_root_move(pos, result);
        }
    }
    RootSearchResult current = search_fixed_depth(pos, depth, state);
    if (!current.result.stopped && !state.stopped && !is_exact_range(current.range)) {
        current = search_fixed_depth(pos, depth, state, -Infinity, Infinity, false);
    }
    return ensure_legal_root_move(pos, current.result);
}

SearchResult NnueSearcherV38::search_best_move(const Position& pos, const SearchLimits& limits) {
    return search_best_move_impl(pos, limits, {}, false);
}

SearchResult NnueSearcherV38::search_best_move(
    const Position& pos,
    const SearchLimits& limits,
    std::span<const HashKey> game_history
) {
    return search_best_move_impl(pos, limits, game_history, true);
}

SearchResult NnueSearcherV38::search_best_move_impl(
    const Position& pos,
    const SearchLimits& limits,
    std::span<const HashKey> game_history,
    bool enable_repetition
) {
    assert(limits.max_depth >= 0);

    if (enable_repetition) {
        tt_.advance_generation();
    }
    killer_table_.clear();
    counter_move_table_.clear();
    SearchResult best = make_fallback_result(pos);
    best.depth = 0;

    SearchState state;
    const ScopedRepetitionStatsCommit repetition_stats_commit(
        repetition_stats_, state.repetition);
    state.repetition_enabled = enable_repetition;
    state.stop_requested = limits.stop_requested;
    if (enable_repetition) {
        initialize_repetition(state, pos, game_history);
        if (history_draw(pos, state)) {
            best.score = 0;
            best.nodes = 1;
            return ensure_legal_root_move(pos, best);
        }
    }
    state.has_deadline = limits.move_time.count() > 0;
    if (state.has_deadline) {
        state.deadline = Clock::now() + limits.move_time;
    }

    const int aspiration_window_cp = 50;

    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        if (state.stop_requested != nullptr
            && state.stop_requested->load(std::memory_order_relaxed)) {
            best.stopped = true;
            best.nodes = state.nodes;
            return ensure_legal_root_move(pos, best);
        }
        if (depth == 1) {
            RootSearchResult current = search_fixed_depth(pos, depth, state);
            if (current.result.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return ensure_legal_root_move(pos, best);
            }
            inherit_ponder_move(current.result, best);
            best = current.result;
        } else if (EnableAspirationWindow) {
            RootSearchResult current = search_fixed_depth(
                pos,
                depth,
                state,
                best.score - aspiration_window_cp,
                best.score + aspiration_window_cp);
            if (current.result.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return ensure_legal_root_move(pos, best);
            }
            if (!is_exact_range(current.range)) {
                // A fail-low and fail-high can meet at the same boundary and
                // make two half-windows alternate forever. Retry once with a
                // full window and no root TT cutoff instead.
                current = search_fixed_depth(
                    pos, depth, state, -Infinity, Infinity, false);
                if (current.result.stopped || state.stopped) {
                    best.stopped = true;
                    best.nodes = state.nodes;
                    return ensure_legal_root_move(pos, best);
                }
            }
            inherit_ponder_move(current.result, best);
            best = current.result;
        } else {
            RootSearchResult current = search_fixed_depth(pos, depth, state);
            if (current.result.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return ensure_legal_root_move(pos, best);
            }
            if (!is_exact_range(current.range)) {
                current = search_fixed_depth(pos, depth, state, -Infinity, Infinity, false);
                if (current.result.stopped || state.stopped) {
                    best.stopped = true;
                    best.nodes = state.nodes;
                    return ensure_legal_root_move(pos, best);
                }
            }
            inherit_ponder_move(current.result, best);
            best = current.result;
        }
    }

    best.nodes = state.nodes;
    return ensure_legal_root_move(pos, best);
}

} // namespace chess
