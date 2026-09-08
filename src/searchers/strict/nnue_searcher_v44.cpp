#include "nnue_searcher_v44.hpp"

#include "attacks.hpp"
#include "nnue_searcher_v44_detail.hpp"
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

int adaptive_aspiration_delta(
    const NnueSearcherV44::AspirationConfig& config,
    int mean_score
) {
    const std::int64_t bounded_mean = std::clamp<std::int64_t>(
        mean_score,
        -static_cast<std::int64_t>(config.mean_score_clamp_cp),
        static_cast<std::int64_t>(config.mean_score_clamp_cp));
    const std::int64_t score_term =
        bounded_mean * bounded_mean / config.delta_divisor;
    return static_cast<int>(std::clamp<std::int64_t>(
        static_cast<std::int64_t>(config.delta_base_cp) + score_term,
        1,
        Infinity));
}

int expand_aspiration_delta(
    const NnueSearcherV44::AspirationConfig& config,
    int delta
) {
    const std::int64_t scaled =
        (static_cast<std::int64_t>(delta)
             * config.expansion_factor_per_mille
         + 999)
        / 1'000;
    return static_cast<int>(std::clamp<std::int64_t>(
        std::max<std::int64_t>(scaled, static_cast<std::int64_t>(delta) + 1),
        1,
        Infinity));
}

int update_aspiration_mean_score(
    const NnueSearcherV44::AspirationConfig& config,
    int previous_mean,
    int exact_score
) {
    const std::int64_t bounded_score = std::clamp<std::int64_t>(
        exact_score,
        -static_cast<std::int64_t>(config.mean_score_clamp_cp),
        static_cast<std::int64_t>(config.mean_score_clamp_cp));
    const std::int64_t new_weight = config.mean_score_new_weight_per_mille;
    const std::int64_t old_weight = 1'000 - new_weight;
    return static_cast<int>(
        (static_cast<std::int64_t>(previous_mean) * old_weight
         + bounded_score * new_weight)
        / 1'000);
}

int aspiration_lower_bound(int score, int delta) {
    return static_cast<int>(std::max<std::int64_t>(
        -Infinity,
        static_cast<std::int64_t>(score) - delta));
}

int aspiration_upper_bound(int score, int delta) {
    return static_cast<int>(std::min<std::int64_t>(
        Infinity,
        static_cast<std::int64_t>(score) + delta));
}

int midpoint_without_overflow(int lhs, int rhs) {
    return static_cast<int>(
        (static_cast<std::int64_t>(lhs) + rhs) / 2);
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

bool NnueSearcherV44::can_null_move_prune(
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

void NnueSearcherV44::make_null_move(Position& pos) {
    if (pos.en_passant_square != NoSquare) {
        pos.zobrist_key ^= zobrist::en_passant_file_key(file_of(pos.en_passant_square));
        pos.en_passant_square = NoSquare;
    }
    pos.zobrist_key ^= zobrist::side_key();
    pos.side_to_move = opposite(pos.side_to_move);
}

bool NnueSearcherV44::castling_rights_changed(
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

void NnueSearcherV44::initialize_repetition(
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

bool NnueSearcherV44::history_draw(
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

bool NnueSearcherV44::allow_repetition_tt_score(SearchState& state) const {
    if (!state.tt_scores_enabled) {
        return false;
    }
    if (!state.repetition_enabled
        || state.in_null_move
        || !state.repetition.has_twofold_position()) {
        return true;
    }
    state.repetition.record_tt_score_suppression();
    return false;
}

void NnueSearcherV44::reward_quiet_cutoff(
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

void NnueSearcherV44::penalize_failed_quiets(
    Color side_to_move,
    int depth,
    Move prev_move,
    PieceType prev_moved_piece,
    const NnueSearcherV44::ScoredMoveList& failed_quiet_moves
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

void NnueSearcherV44::record_beta_cutoff(
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

bool NnueSearcherV44::should_stop(
    SearchState& state,
    bool force_poll
) const {
    if (!force_poll && (state.nodes & 1023ULL) != 0) {
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

KingSafetyContext NnueSearcherV44::current_king_safety_context(const Position& pos) const {
    return cached_king_safety_context(pos, pos.side_to_move);
}

int NnueSearcherV44::evaluate_current_position(const Position& pos, const SearchState& state) const {
    assert_accumulator_matches_full_recompute(model_, state.accumulator, pos);
    return state.accumulator.evaluate_cp_rounded(pos);
}



NnueSearcherV44::SearchValue NnueSearcherV44::quiescence(
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
        return SearchValue{0};
    }

    if (check_current_repetition && history_draw(pos, state)) {
        return SearchValue{0};
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
    auto search_qsearch_move = [&](const ScoredMove& scored_move, int& node_score) {
        const PieceType moved_piece = scored_moved_piece(scored_move);
        const PieceType captured_piece = scored_captured_piece(scored_move);
        int move_score = 0;
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
            move_score = -child.score;
        }
        if (state.stopped) {
            return true;
        }

        node_score = std::max(node_score, move_score);
        if (node_score >= beta) {
            return true;
        }
        if (move_score > alpha) {
            alpha = move_score;
        }
        return false;
    };
    auto search_qsearch_promotion_stage = [&](int& node_score, bool allow_search, bool& has_legal_move) {
        bool stage_done = false;
        auto search_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
            has_legal_move = true;
            if (!allow_search || stage_done) {
                return;
            }
            const ScoredMove scored_move = captured_piece == PieceType::None
                ? make_qsearch_scored_move<false, true>(move, moved_piece, captured_piece)
                : make_qsearch_scored_move<true, true>(move, moved_piece, captured_piece);
            stage_done = search_qsearch_move(scored_move, node_score);
        };
        generate_legal_promotion_moves_with_info(pos, king_safety, search_move);
        return stage_done;
    };
    auto search_qsearch_stage = [&](const ScoredMoveList& staged_moves, int& node_score) {
        for (std::size_t move_index = 0; move_index < staged_moves.size(); ++move_index) {
            if (search_qsearch_move(staged_moves[move_index], node_score)) {
                return true;
            }
        }
        return false;
    };

    if (side_in_check) {
        int node_score = -Infinity;
        bool has_legal_move = false;

        if (search_qsearch_promotion_stage(
                node_score,
                q_depth < MaxCheckEvasionQuiescenceDepth,
                has_legal_move)) {
            if (state.stopped) {
                return SearchValue{0};
            }
            return SearchValue{node_score};
        }

        const ScoredMoveList capture_moves =
            generate_qsearch_capture_stage(false);
        has_legal_move = has_legal_move || !capture_moves.empty();
        if (q_depth < MaxCheckEvasionQuiescenceDepth
            && search_qsearch_stage(capture_moves, node_score)) {
            if (state.stopped) {
                return SearchValue{0};
            }
            return SearchValue{node_score};
        }

        const ScoredMoveList quiet_evasion_moves = generate_qsearch_quiet_evasion_stage();
        has_legal_move = has_legal_move || !quiet_evasion_moves.empty();
        if (!has_legal_move) {
            return SearchValue{-CheckmateScore + ply};
        }
        if (q_depth >= MaxCheckEvasionQuiescenceDepth) {
            return SearchValue{evaluate_current_position(pos, state)};
        }
        if (search_qsearch_stage(quiet_evasion_moves, node_score)) {
            if (state.stopped) {
                return SearchValue{0};
            }
            return SearchValue{node_score};
        }

        return SearchValue{node_score};
    }

    int best_score = evaluate_current_position(pos, state);
    if (best_score >= beta) {
        return SearchValue{best_score};
    }
    if (best_score > alpha) {
        alpha = best_score;
    }

    if (q_depth >= MaxQuiescenceDepth) {
        return SearchValue{best_score};
    }

    bool has_legal_promotion = false;
    if (search_qsearch_promotion_stage(best_score, true, has_legal_promotion)) {
        if (state.stopped) {
            return SearchValue{0};
        }
        return SearchValue{best_score};
    }

    const ScoredMoveList capture_moves = generate_qsearch_capture_stage(true);
    if (search_qsearch_stage(capture_moves, best_score)) {
        if (state.stopped) {
            return SearchValue{0};
        }
        return SearchValue{best_score};
    }

    return SearchValue{best_score};
}

NnueSearcherV44::SearchValue NnueSearcherV44::negamax(
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
    const int original_alpha = alpha;
    const int original_beta = beta;
    if (ply == 1) {
        state.ply_one_best_move = {};
    }
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{0};
    }


    if (history_draw(pos, state)) {
        return SearchValue{0};
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
    if (tt_probe.cutoff) {
        if (ply == 1) {
            state.ply_one_best_move = tt_probe.move;
        }
        return SearchValue{tt_probe.score};
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
            return SearchValue{beta};
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
            return SearchValue{0};
        }
        const int null_score = -child.score;
        if (null_score >= beta) {
            ++selective_stats_.null_move_cutoffs;
            return SearchValue{beta};
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
    int best_score = -Infinity;
    Move best_move{};
    Move fallback_best_move{};
    ScoredMoveList failed_quiet_moves;
    bool searched_any_move = false;
    std::size_t searched_move_count = 0;
    const MoveHints tt_moves{tt_probe.move, Move{}};
    Move searched_tt_move = tt_probe.move;
    std::optional<KingSafetyContext> king_safety_cache;
    auto get_king_safety = [&]() -> const KingSafetyContext& {
        if (!king_safety_cache) {
            king_safety_cache = current_king_safety_context(pos);
        }
        return *king_safety_cache;
    };

    const Move tt_move = tt_probe.move;
    if (is_valid_move(tt_move)) {
        ScoredMove scored_move = make_tt_lower_scored_move(pos, tt_move);
        if (is_valid_move(scored_move.move)) {
            searched_any_move = true;
            searched_tt_move = scored_move.move;
            fallback_best_move = scored_move.move;

            int move_score = 0;
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
                move_score = -child.score;
            }
            if (state.stopped) {
                return SearchValue{0};
            }

            update_best_score(best_score, best_move, alpha, scored_move.move, move_score);
            if (ply == 1) {
                state.ply_one_best_move = best_move;
            }
            if (best_score >= beta) {
                record_beta_cutoff(
                    scored_move,
                    searched_move_count,
                    CutoffStage::TtLower);
                reward_quiet_cutoff(pos.side_to_move, depth, ply, scored_move, prev_move, prev_moved_piece);
                store_tt_if_needed(
                    pos.zobrist_key,
                    depth,
                    ply,
                    best_score,
                    original_alpha,
                    original_beta,
                    best_move,
                    fallback_best_move,
                    allow_tt_score);
                return SearchValue{best_score};
            }
            ++searched_move_count;
            if (is_quiet_move(scored_move)) {
                failed_quiet_moves.push_back(scored_move);
            }
        }
    }

    bool cutoff = best_score >= beta;
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

        int move_score = 0;
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
                move_score = -scout.score;
                if (state.stopped) {
                    stage_stopped = true;
                    return;
                }

                const bool scout_proves_fail_low = move_score <= alpha;
                const bool scout_proves_fail_high = move_score >= beta;
                if (reduced_search && move_score > alpha) {
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
                    move_score = -full.score;
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
                    move_score = -full.score;
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
                move_score = -child.score;
                if (state.stopped) {
                    stage_stopped = true;
                    return;
                }
                if (reduced_search && move_score > alpha) {
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
                    move_score = -full.score;
                    if (state.stopped) {
                        stage_stopped = true;
                        return;
                    }
                }
            }
        }
        searched_any_move = true;
        update_best_score(best_score, best_move, alpha, scored_move.move, move_score);
        if (ply == 1) {
            state.ply_one_best_move = best_move;
        }
        if (best_score >= beta) {
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
            static_cast<void>(has_unsearched_stage);
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
                    pos, move, moved_piece, captured_piece, ply, tt_moves, prev_move, prev_moved_piece)
                : make_scored_legal_move<ScoringMode::MainSearch, true, true>(
                    pos, move, moved_piece, captured_piece, ply, tt_moves, prev_move, prev_moved_piece);
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
            tt_moves,
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
                tt_moves,
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
                tt_moves,
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
        return SearchValue{0};
    }
    if (!searched_any_move) {
        return SearchValue{
            in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0};
    }

    store_tt_if_needed(
        pos.zobrist_key,
        depth,
        ply,
        best_score,
        original_alpha,
        original_beta,
        best_move,
        fallback_best_move,
        allow_tt_score);

    return SearchValue{best_score};
}

SearchResult NnueSearcherV44::make_fallback_result(const Position& pos) const {
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

NnueSearcherV44::RootSearchResult NnueSearcherV44::search_fixed_depth(
    Position pos,
    int depth,
    SearchState& state,
    int alpha,
    int beta,
    bool allow_root_tt_probe
) {
    assert(depth >= 0);
    assert(alpha < beta);
    const int original_alpha = alpha;
    const int original_beta = beta;
    state.accumulator.reset(model_, pos);

    RootSearchResult root_result;
    SearchResult& result = root_result.result;
    result.depth = depth;

    if (depth == 0) {
        root_result.score = quiescence(
            pos, -Infinity, Infinity, 0, 0, state, true).score;
        root_result.bound = V44SingleBoundTranspositionTable::Bound::Exact;
        result.score = root_result.score;
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
    const MoveHints tt_moves{tt_probe.move, Move{}};
    if (tt_probe.cutoff) {
        const Move tt_best_move = tt_probe.move;
        if (is_valid_move(tt_best_move)) {
            root_result.score = tt_probe.score;
            root_result.bound = tt_probe.bound;
            result.score = root_result.score;
            result.best_move = tt_best_move;
            result.nodes = 1;
            return root_result;
        }
    }

    const PositionStateSnapshot node_snapshot = pos.make_state_snapshot();
    int best_score = -Infinity;
    Move best_move{};
    Move fallback_best_move{};
    Move best_reply{};
    Move fallback_reply{};
    bool searched_any_move = false;
    std::size_t searched_move_count = 0;
    Move searched_tt_move = tt_probe.move;
    std::optional<KingSafetyContext> king_safety_cache;
    auto get_king_safety = [&]() -> const KingSafetyContext& {
        if (!king_safety_cache) {
            king_safety_cache = current_king_safety_context(pos);
        }
        return *king_safety_cache;
    };

    const Move tt_move = tt_probe.move;
    if (is_valid_move(tt_move)) {
        ScoredMove scored_move = make_tt_lower_scored_move(pos, tt_move);
        if (is_valid_move(scored_move.move)) {
            searched_any_move = true;
            searched_tt_move = scored_move.move;
            fallback_best_move = scored_move.move;

            int move_score = 0;
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
                move_score = -child.score;
                candidate_reply = state.ply_one_best_move;
            }
            if (state.stopped) {
                result.stopped = true;
                result.nodes = state.nodes;
                return root_result;
            }

            update_best_score(best_score, best_move, alpha, scored_move.move, move_score);
            if (fallback_best_move == scored_move.move) {
                fallback_reply = candidate_reply;
            }
            if (best_move == scored_move.move) {
                best_reply = candidate_reply;
            }
            if (best_score >= beta) {
                record_beta_cutoff(
                    scored_move,
                    searched_move_count,
                    CutoffStage::TtLower);
            } else {
                ++searched_move_count;
            }
        }
    }

    bool cutoff = best_score >= beta;
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

        int move_score = 0;
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
            move_score = -child.score;
            candidate_reply = state.ply_one_best_move;
        }
        if (state.stopped) {
            result.stopped = true;
            result.nodes = state.nodes;
            root_stage_stopped = true;
            return;
        }

        searched_any_move = true;

        update_best_score(best_score, best_move, alpha, scored_move.move, move_score);
        if (fallback_best_move == scored_move.move) {
            fallback_reply = candidate_reply;
        }
        if (best_move == scored_move.move) {
            best_reply = candidate_reply;
        }
        if (best_score >= beta) {
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
            static_cast<void>(has_unsearched_stage);
            cutoff = true;
            return;
        }
    };
    auto search_root_scored_moves = [&]<MoveGenerationStage Stage>(const ScoredMoveList& moves) {
        if (cutoff || root_stage_stopped) {
            return;
        }
        for (std::size_t move_index = 0; best_score < beta && move_index < moves.size(); ++move_index) {
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
                    pos, move, moved_piece, captured_piece, 0, tt_moves, Move{}, PieceType::None)
                : make_scored_legal_move<ScoringMode::MainSearch, true, true>(
                    pos, move, moved_piece, captured_piece, 0, tt_moves, Move{}, PieceType::None);
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
            tt_moves,
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
                tt_moves,
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
                tt_moves,
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
        root_result.score =
            in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        root_result.bound = V44SingleBoundTranspositionTable::Bound::Exact;
        result.score = root_result.score;
        result.nodes = 1;
        return root_result;
    }

    root_result.score = best_score;
    if (best_score <= original_alpha) {
        root_result.bound = V44SingleBoundTranspositionTable::Bound::Upper;
    } else if (best_score >= original_beta) {
        root_result.bound = V44SingleBoundTranspositionTable::Bound::Lower;
    } else {
        root_result.bound = V44SingleBoundTranspositionTable::Bound::Exact;
    }
    result.score = best_score;
    result.best_move = best_move;
    result.ponder_move = best_reply;
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
        best_score,
        original_alpha,
        original_beta,
        best_move,
        fallback_best_move,
        allow_tt_score);

    result.nodes = state.nodes;
    return root_result;
}

SearchResult NnueSearcherV44::search_best_move(const Position& pos, int depth) {
    return search_best_move_impl(pos, depth, {}, true);
}

SearchResult NnueSearcherV44::search_best_move(
    const Position& pos,
    int depth,
    std::span<const HashKey> game_history
) {
    return search_best_move_impl(pos, depth, game_history, true);
}

SearchResult NnueSearcherV44::search_best_move_impl(
    const Position& pos,
    int depth,
    std::span<const HashKey> game_history,
    bool enable_repetition
) {
    clear_aspiration_stats();
    // V44 deliberately prevents all cross-move TT reuse. The clear happens
    // once per public search, before any fixed-depth work starts; recursive
    // nodes within this call still share the table normally.
    tt_.clear();
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
    return ensure_legal_root_move(pos, current.result);
}

SearchResult NnueSearcherV44::search_best_move(const Position& pos, const SearchLimits& limits) {
    return search_best_move_impl(pos, limits, {}, true);
}

SearchResult NnueSearcherV44::search_best_move(
    const Position& pos,
    const SearchLimits& limits,
    std::span<const HashKey> game_history
) {
    return search_best_move_impl(pos, limits, game_history, true);
}

SearchResult NnueSearcherV44::search_best_move_impl(
    const Position& pos,
    const SearchLimits& limits,
    std::span<const HashKey> game_history,
    bool enable_repetition
) {
    assert(limits.max_depth >= 0);
    const Clock::time_point search_start = Clock::now();

    clear_aspiration_stats();
    // Charge the table clear to this move's clock. Iterative-deepening
    // iterations and aspiration retries then reuse the fresh table.
    tt_.clear();
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
        state.deadline = search_start + limits.move_time;
    }

    const int aspiration_window_cp = 50;
    bool has_mean_score = false;
    int mean_score = 0;

    auto record_completed_iteration = [&](const RootSearchResult& current) {
        if (current.bound != V44SingleBoundTranspositionTable::Bound::Exact) {
            return;
        }
        const int exact_score = current.score;
        if (!has_mean_score) {
            mean_score = std::clamp(
                exact_score,
                -aspiration_config_.mean_score_clamp_cp,
                aspiration_config_.mean_score_clamp_cp);
            has_mean_score = true;
        } else {
            mean_score = update_aspiration_mean_score(
                aspiration_config_, mean_score, exact_score);
        }
        ++aspiration_stats_.completed_iterations;
        aspiration_stats_.final_mean_score_cp = mean_score;
        aspiration_stats_.has_final_mean_score = true;
    };

    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        // Adaptive retries can hit the root TT without incrementing nodes, so
        // V44 polls both its clock and cooperative stop at every root
        // boundary. The disabled-policy branch retains the legacy cadence.
        const bool stop_at_iteration_boundary = aspiration_config_.enabled
            ? should_stop(state, true)
            : state.stop_requested != nullptr
                && state.stop_requested->load(std::memory_order_relaxed);
        if (stop_at_iteration_boundary) {
            best.stopped = true;
            best.nodes = state.nodes;
            return ensure_legal_root_move(pos, best);
        }
        if (aspiration_config_.enabled) {
            const bool use_narrow_window =
                depth >= aspiration_config_.min_depth
                && has_mean_score
                && std::abs(best.score) < MateScoreThreshold;
            int accepted_search_depth = depth;
            auto search_emergency_full_window = [&]() {
                // With a single-bound TT there is no range intersection to
                // repair. A full root window is sufficient; internal TT
                // scores remain cutoff-only and never narrow the window.
                return search_fixed_depth(
                    pos, depth, state, -Infinity, Infinity, false);
            };

            RootSearchResult current;
            if (!use_narrow_window) {
                current = search_fixed_depth(pos, depth, state);
                if (current.result.stopped || state.stopped) {
                    best.stopped = true;
                    best.nodes = state.nodes;
                    return ensure_legal_root_move(pos, best);
                }
                if (current.bound
                    != V44SingleBoundTranspositionTable::Bound::Exact) {
                    ++aspiration_stats_.full_window_fallbacks;
                    if (should_stop(state, true)) {
                        best.stopped = true;
                        best.nodes = state.nodes;
                        return ensure_legal_root_move(pos, best);
                    }
                    current = search_emergency_full_window();
                    if (current.result.stopped || state.stopped) {
                        best.stopped = true;
                        best.nodes = state.nodes;
                        return ensure_legal_root_move(pos, best);
                    }
                }
            } else {
                int delta = adaptive_aspiration_delta(
                    aspiration_config_, mean_score);
                const bool first_narrow_iteration =
                    aspiration_stats_.narrow_iterations == 0;
                ++aspiration_stats_.narrow_iterations;
                aspiration_stats_.initial_delta_sum_cp += delta;
                if (first_narrow_iteration) {
                    aspiration_stats_.initial_delta_min_cp = delta;
                    aspiration_stats_.initial_delta_max_cp = delta;
                } else {
                    aspiration_stats_.initial_delta_min_cp = std::min(
                        aspiration_stats_.initial_delta_min_cp, delta);
                    aspiration_stats_.initial_delta_max_cp = std::max(
                        aspiration_stats_.initial_delta_max_cp, delta);
                }
                aspiration_stats_.last_initial_delta_cp = delta;
                int alpha = aspiration_lower_bound(best.score, delta);
                int beta = aspiration_upper_bound(best.score, delta);
                int fail_high_reductions = 0;
                int failed_attempts = 0;
                bool require_full_window = false;

                while (true) {
                    if (should_stop(state, true)) {
                        best.stopped = true;
                        best.nodes = state.nodes;
                        return ensure_legal_root_move(pos, best);
                    }
                    const int search_depth = std::max(
                        1, depth - fail_high_reductions);
                    ++aspiration_stats_.narrow_attempts;
                    if (search_depth < depth) {
                        ++aspiration_stats_.reduced_depth_attempts;
                    }

                    current = search_fixed_depth(
                        pos,
                        search_depth,
                        state,
                        alpha,
                        beta,
                        failed_attempts == 0);
                    if (current.result.stopped || state.stopped) {
                        best.stopped = true;
                        best.nodes = state.nodes;
                        return ensure_legal_root_move(pos, best);
                    }
                    if (current.bound
                        == V44SingleBoundTranspositionTable::Bound::Exact) {
                        accepted_search_depth = search_depth;
                        if (failed_attempts == 0) {
                            ++aspiration_stats_.initial_window_successes;
                        }
                        break;
                    }

                    if (current.bound
                        == V44SingleBoundTranspositionTable::Bound::Upper) {
                        ++aspiration_stats_.fail_lows;
                        const int value = current.score;
                        beta = midpoint_without_overflow(alpha, beta);
                        alpha = aspiration_lower_bound(value, delta);
                        fail_high_reductions = 0;
                        if (value <= -MateScoreThreshold) {
                            alpha = -Infinity;
                        }
                    } else if (current.bound
                               == V44SingleBoundTranspositionTable::Bound::Lower) {
                        ++aspiration_stats_.fail_highs;
                        const int value = current.score;
                        beta = aspiration_upper_bound(value, delta);
                        fail_high_reductions = std::min(
                            fail_high_reductions + 1,
                            aspiration_config_.max_fail_high_reductions);
                        if (value >= MateScoreThreshold) {
                            beta = Infinity;
                            fail_high_reductions = 0;
                        }
                    } else {
                        require_full_window = true;
                    }

                    ++failed_attempts;
                    if (failed_attempts >= aspiration_config_.max_researches) {
                        ++aspiration_stats_.retry_limit_fallbacks;
                        require_full_window = true;
                    }
                    if (alpha >= beta) {
                        require_full_window = true;
                    }
                    if (require_full_window) {
                        break;
                    }
                    delta = expand_aspiration_delta(
                        aspiration_config_, delta);
                }

                if (require_full_window) {
                    ++aspiration_stats_.full_window_fallbacks;
                    accepted_search_depth = depth;
                    if (should_stop(state, true)) {
                        best.stopped = true;
                        best.nodes = state.nodes;
                        return ensure_legal_root_move(pos, best);
                    }
                    current = search_emergency_full_window();
                    if (current.result.stopped || state.stopped) {
                        best.stopped = true;
                        best.nodes = state.nodes;
                        return ensure_legal_root_move(pos, best);
                    }
                }
            }

            if (current.bound
                != V44SingleBoundTranspositionTable::Bound::Exact) {
                ++aspiration_stats_.unresolved_ranges;
                // A completed iterative-deepening result must be exact.  Keep
                // the previous completed iteration instead of publishing a
                // bound as though it were an exact score.
                best.nodes = state.nodes;
                return ensure_legal_root_move(pos, best);
            }
            if (use_narrow_window) {
                aspiration_stats_.accepted_narrow_nominal_depth_sum +=
                    static_cast<std::uint64_t>(depth);
                aspiration_stats_.accepted_narrow_search_depth_sum +=
                    static_cast<std::uint64_t>(accepted_search_depth);
                const int accepted_depth_reduction =
                    depth - accepted_search_depth;
                if (accepted_depth_reduction > 0) {
                    ++aspiration_stats_.accepted_reduced_depth_iterations;
                    aspiration_stats_.max_accepted_depth_reduction = std::max(
                        aspiration_stats_.max_accepted_depth_reduction,
                        accepted_depth_reduction);
                }
            }
            // The public depth is the completed iterative-deepening
            // iteration.  As in the Plenty-style policy, a fail-high retry
            // may have used depth - fail_high_reductions before producing an
            // exact result inside the widened window.
            current.result.depth = depth;
            record_completed_iteration(current);
            inherit_ponder_move(current.result, best);
            best = current.result;
            continue;
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
            if (current.bound
                != V44SingleBoundTranspositionTable::Bound::Exact) {
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
            if (current.bound
                != V44SingleBoundTranspositionTable::Bound::Exact) {
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
