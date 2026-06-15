#include "nnue_searcher_v10.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"
#include "zobrist.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

namespace chess {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int MaxQuiescenceDepth = 16;
constexpr int LmrMinDepth = 3;
constexpr int LmrMoveIndex = 4;
constexpr int LmrReduction = 1;
constexpr int NullMoveMinDepth = 3;
constexpr int NullMoveReduction = 2;
constexpr int KillerMoveReward = 100000;

bool is_valid_move(Move move) {
    return move.value != 0;
}

bool is_promotion(Move move) {
    return promotion_piece(move) != PieceType::None;
}

bool is_noisy_move(Move move) {
    return is_capture(move) || is_promotion(move);
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

void assert_accumulator_matches_full_recompute(
    const NnueValueModel& model,
    const NnueAccumulator& accumulator,
    const Position& pos
) {
#ifndef NDEBUG
    const float incremental = accumulator.predict_normalized(pos);
    const float full = model.predict_normalized(encode_position(pos));
    assert(std::fabs(incremental - full) <= 1.0e-4F);
#else
    static_cast<void>(model);
    static_cast<void>(accumulator);
    static_cast<void>(pos);
#endif
}

class ScopedNnueUpdate {
public:
    ScopedNnueUpdate(
        NnueAccumulator& accumulator,
        const Position& before,
        Move move,
        const Position& after
    )
        : accumulator_(&accumulator),
          undo_(accumulator.make_move_with_undo(before, move, after)) {
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
    NnueAccumulator* accumulator_ = nullptr;
    NnueAccumulatorUndo undo_{};
};

} // namespace

void NnueSearcherV10::make_null_move(Position& pos) const {
    if (pos.en_passant_square != NoSquare) {
        pos.zobrist_key ^= zobrist::en_passant_file_key(file_of(pos.en_passant_square));
        pos.en_passant_square = NoSquare;
    }

    pos.zobrist_key ^= zobrist::side_key();
    pos.side_to_move = opposite(pos.side_to_move);
}

bool NnueSearcherV10::should_stop(SearchContext& context) const {
    if (!context.has_deadline) {
        return false;
    }
    if ((context.nodes & 1023ULL) != 0) {
        return false;
    }
    if (Clock::now() >= context.deadline) {
        context.stopped = true;
        return true;
    }
    return false;
}

NnueSearcherV10::MoveOrderStage NnueSearcherV10::move_order_stage(
    Move move,
    Move tt_move,
    int ply,
    bool gives_check,
    bool capture,
    bool promotion,
    int see_score,
    ScoringMode scoring_mode
) const {
    if (is_valid_move(tt_move) && move == tt_move) {
        return MoveOrderStage::TtMove;
    }
    if (promotion) {
        return MoveOrderStage::Promotion;
    }
    if (capture) {
        if (see_score >= 0) {
            return MoveOrderStage::GoodCapture;
        }
        return gives_check ? MoveOrderStage::Check : MoveOrderStage::BadCapture;
    }
    if (gives_check) {
        return MoveOrderStage::Check;
    }
    if (scoring_mode == ScoringMode::MainSearch && killer_table_.score(ply, move) > 0) {
        return MoveOrderStage::Killer;
    }
    return MoveOrderStage::HistoryQuiet;
}

int NnueSearcherV10::move_tie_break_score(
    Move move,
    const Position& pos,
    int ply,
    bool gives_check,
    bool capture,
    bool promotion,
    int see_score,
    ScoringMode stage
) const {
    if (stage == ScoringMode::Quiescence) {
        return promotion ? Infinity : see_score;
    }

    int score = history_table_.get_score(pos, move);
    if (!capture && !promotion && !gives_check) {
        score += KillerMoveReward * killer_table_.score(ply, move);
    }

    return score;
}

NnueSearcherV10::ScoredMove NnueSearcherV10::make_scored_move(
    const Position& pos,
    Move move,
    int ply,
    Move tt_move,
    ScoringMode stage
) const {
    Position next = pos;
    next.make_move(move);
    const bool gives_check = in_check(next, next.side_to_move);
    const bool capture = is_capture(move);
    const bool promotion = is_promotion(move);
    const int see_score = capture ? static_exchange_eval(pos, move) : 0;

    if (stage == ScoringMode::Quiescence) {
        return ScoredMove{
            move,
            move_order_stage(
                move,
                tt_move,
                ply,
                gives_check,
                capture,
                promotion,
                see_score,
                stage),
            move_tie_break_score(
                move,
                pos,
                ply,
                gives_check,
                capture,
                promotion,
                see_score,
                stage),
            gives_check,
            capture,
            promotion
        };
    }

    return ScoredMove{
        move,
        move_order_stage(
            move,
            tt_move,
            ply,
            gives_check,
            capture,
            promotion,
            see_score,
            stage),
        move_tie_break_score(
            move,
            pos,
            ply,
            gives_check,
            capture,
            promotion,
            see_score,
            stage),
        gives_check,
        capture,
        promotion
    };
}

bool NnueSearcherV10::better_scored_move(
    const ScoredMove& lhs,
    const ScoredMove& rhs
) const {
    if (lhs.stage != rhs.stage) {
        return stage_rank(lhs.stage) < stage_rank(rhs.stage);
    }
    return lhs.tie_break_score > rhs.tie_break_score;
}

int NnueSearcherV10::stage_rank(MoveOrderStage stage) const {
    switch (profile_) {
    case V10MoveOrderingProfile::Baseline:
        switch (stage) {
        case MoveOrderStage::TtMove: return 0;
        case MoveOrderStage::Promotion: return 1;
        case MoveOrderStage::GoodCapture: return 2;
        case MoveOrderStage::Check: return 3;
        case MoveOrderStage::Killer: return 4;
        case MoveOrderStage::HistoryQuiet: return 5;
        case MoveOrderStage::BadCapture: return 6;
        }
        break;
    case V10MoveOrderingProfile::CheckBeforeGoodCapture:
        switch (stage) {
        case MoveOrderStage::TtMove: return 0;
        case MoveOrderStage::Promotion: return 1;
        case MoveOrderStage::Check: return 2;
        case MoveOrderStage::GoodCapture: return 3;
        case MoveOrderStage::Killer: return 4;
        case MoveOrderStage::HistoryQuiet: return 5;
        case MoveOrderStage::BadCapture: return 6;
        }
        break;
    case V10MoveOrderingProfile::KillerBeforeCheck:
        switch (stage) {
        case MoveOrderStage::TtMove: return 0;
        case MoveOrderStage::Promotion: return 1;
        case MoveOrderStage::GoodCapture: return 2;
        case MoveOrderStage::Killer: return 3;
        case MoveOrderStage::Check: return 4;
        case MoveOrderStage::HistoryQuiet: return 5;
        case MoveOrderStage::BadCapture: return 6;
        }
        break;
    case V10MoveOrderingProfile::HistoryBeforeKiller:
        switch (stage) {
        case MoveOrderStage::TtMove: return 0;
        case MoveOrderStage::Promotion: return 1;
        case MoveOrderStage::GoodCapture: return 2;
        case MoveOrderStage::Check: return 3;
        case MoveOrderStage::HistoryQuiet: return 4;
        case MoveOrderStage::Killer: return 5;
        case MoveOrderStage::BadCapture: return 6;
        }
        break;
    case V10MoveOrderingProfile::BadCaptureBeforeQuiet:
        switch (stage) {
        case MoveOrderStage::TtMove: return 0;
        case MoveOrderStage::Promotion: return 1;
        case MoveOrderStage::GoodCapture: return 2;
        case MoveOrderStage::Check: return 3;
        case MoveOrderStage::Killer: return 4;
        case MoveOrderStage::BadCapture: return 5;
        case MoveOrderStage::HistoryQuiet: return 6;
        }
        break;
    }
    return 100;
}

std::vector<NnueSearcherV10::ScoredMove> NnueSearcherV10::ordered_moves(
    const Position& pos,
    int ply,
    Move tt_move
) const {
    std::vector<ScoredMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        ordered.push_back(make_scored_move(pos, move, ply, tt_move));
    }

    std::stable_sort(ordered.begin(), ordered.end(), [this](const ScoredMove& lhs, const ScoredMove& rhs) {
        return better_scored_move(lhs, rhs);
    });

    return ordered;
}

std::vector<NnueSearcherV10::ScoredMove> NnueSearcherV10::ordered_noisy_moves(
    const Position& pos
) const {
    std::vector<ScoredMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        if (!is_noisy_move(move)) {
            continue;
        }

        ordered.push_back(make_scored_move(pos, move, 0, Move{}, ScoringMode::Quiescence));
    }

    std::stable_sort(ordered.begin(), ordered.end(), [this](const ScoredMove& lhs, const ScoredMove& rhs) {
        return better_scored_move(lhs, rhs);
    });

    return ordered;
}

bool NnueSearcherV10::should_reduce_late_move(
    const Position& pos,
    const ScoredMove& scored_move,
    int depth,
    int move_index
) const {
    return depth >= LmrMinDepth
        && move_index >= LmrMoveIndex
        && is_quiet_move(scored_move)
        && !in_check(pos, pos.side_to_move);
}

bool NnueSearcherV10::can_null_move_prune(
    const Position& pos,
    int depth,
    const SearchContext& context
) const {
    return depth >= NullMoveMinDepth
        && !context.in_null_move
        && !in_check(pos, pos.side_to_move)
        && has_non_pawn_material(pos, pos.side_to_move);
}

int NnueSearcherV10::quiescence(
    Position pos,
    int alpha,
    int beta,
    int ply,
    int q_depth,
    SearchContext& context
) {
    assert(alpha < beta);
    ++context.nodes;

    if (should_stop(context)) {
        return 0;
    }

    const std::vector<Move> legal_moves = generate_legal_moves(pos);
    if (legal_moves.empty()) {
        return in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0;
    }

    assert_accumulator_matches_full_recompute(model_, context.accumulator, pos);
    const int stand_pat = context.accumulator.evaluate_cp_rounded(pos);
    if (stand_pat >= beta) {
        return beta;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    if (q_depth >= MaxQuiescenceDepth) {
        return alpha;
    }

    const std::vector<ScoredMove> moves = ordered_noisy_moves(pos);
    for (const ScoredMove& scored_move : moves) {
        Position next = pos;
        next.make_move(scored_move.move);
        const ScopedNnueUpdate nnue_update(context.accumulator, pos, scored_move.move, next);

        const int score = -quiescence(next, -beta, -alpha, ply + 1, q_depth + 1, context);
        if (context.stopped) {
            return 0;
        }

        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    return alpha;
}

bool NnueSearcherV10::is_quiet_move(const ScoredMove& scored_move) const {
    return !scored_move.capture
        && !scored_move.promotion
        && !scored_move.gives_check;
}

int NnueSearcherV10::negamax(
    Position pos,
    int depth,
    int ply,
    int alpha,
    int beta,
    SearchContext& context
) {
    assert(depth >= 0);
    assert(alpha < beta);
    ++context.nodes;

    if (should_stop(context)) {
        return 0;
    }

    int tt_score = 0;
    Move tt_best_move{};
    if (tt_.probe(pos.zobrist_key, depth, alpha, beta, tt_score, tt_best_move)) {
        return tt_score;
    }

    const int alpha_before_search = alpha;
    const int beta_before_search = beta;

    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply, 0, context);
    }

    if (can_null_move_prune(pos, depth, context)) {
        Position null_pos = pos;
        make_null_move(null_pos);

        const bool previous_in_null_move = context.in_null_move;
        context.in_null_move = true;
        const int null_depth = std::max(0, depth - 1 - NullMoveReduction);
        const int null_score = -negamax(
            null_pos,
            null_depth,
            ply + 1,
            -beta,
            -beta + 1,
            context);
        context.in_null_move = previous_in_null_move;

        if (context.stopped) {
            return 0;
        }
        if (null_score >= beta) {
            return beta;
        }
    }

    const std::vector<ScoredMove> moves = ordered_moves(pos, ply, tt_best_move);

    if (moves.empty()) {
        return in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0;
    }

    int best_score = -Infinity;
    Move best_move{};
    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];

        Position next = pos;
        next.make_move(scored_move.move);
        const ScopedNnueUpdate nnue_update(context.accumulator, pos, scored_move.move, next);

        int score = -Infinity;
        if (should_reduce_late_move(pos, scored_move, depth, static_cast<int>(move_index))) {
            const int reduced_depth = std::max(0, depth - 1 - LmrReduction);
            score = -negamax(next, reduced_depth, ply + 1, -alpha - 1, -alpha, context);
            if (context.stopped) {
                return 0;
            }

            if (score > alpha) {
                score = -negamax(next, depth - 1, ply + 1, -beta, -alpha, context);
                if (context.stopped) {
                    return 0;
                }
            }
        } else {
            score = -negamax(next, depth - 1, ply + 1, -beta, -alpha, context);
            if (context.stopped) {
                return 0;
            }
        }

        if (score > best_score) {
            best_score = score;
            best_move = scored_move.move;
        }

        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            if (is_quiet_move(scored_move)) {
                killer_table_.store(ply, scored_move.move);
                history_table_.store(pos, scored_move.move, depth);
            }
            break;
        }
    }

    TTBound bound = TTBound::Exact;
    if (best_score <= alpha_before_search) {
        bound = TTBound::Upper;
    } else if (best_score >= beta_before_search) {
        bound = TTBound::Lower;
    }
    tt_.store(pos.zobrist_key, depth, best_score, bound, best_move);

    return best_score;
}

SearchResult NnueSearcherV10::make_fallback_result(const Position& pos) const {
    SearchResult result;
    const std::vector<ScoredMove> moves = ordered_moves(pos, 0);
    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }
    result.best_move = moves.front().move;
    result.score = model_.evaluate_cp_rounded(pos);
    result.nodes = 1;
    return result;
}

SearchResult NnueSearcherV10::search_fixed_depth(
    const Position& pos,
    int depth,
    SearchContext& context
) {
    assert(depth >= 0);
    context.accumulator.reset(model_, pos);

    SearchResult result;
    result.depth = depth;

    if (depth == 0) {
        result.score = quiescence(pos, -Infinity, Infinity, 0, 0, context);
        result.nodes = context.nodes;
        return result;
    }

    int alpha = -Infinity;
    int beta = Infinity;
    int tt_score = 0;
    Move tt_best_move{};
    if (tt_.probe(pos.zobrist_key, depth, alpha, beta, tt_score, tt_best_move)) {
        result.score = tt_score;
        result.best_move = tt_best_move;
        result.nodes = 1;
        return result;
    }

    const int alpha_before_search = alpha;
    const int beta_before_search = beta;

    const std::vector<ScoredMove> moves = ordered_moves(pos, 0, tt_best_move);

    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }

    result.score = -Infinity;
    for (const ScoredMove& scored_move : moves) {
        Position next = pos;
        next.make_move(scored_move.move);
        const ScopedNnueUpdate nnue_update(context.accumulator, pos, scored_move.move, next);

        const int score = -negamax(next, depth - 1, 1, -beta, -alpha, context);
        if (context.stopped) {
            result.stopped = true;
            result.nodes = context.nodes;
            return result;
        }
        if (score > result.score) {
            result.score = score;
            result.best_move = scored_move.move;
        }
        alpha = std::max(alpha, score);
    }

    TTBound bound = TTBound::Exact;
    if (result.score <= alpha_before_search) {
        bound = TTBound::Upper;
    } else if (result.score >= beta_before_search) {
        bound = TTBound::Lower;
    }
    tt_.store(pos.zobrist_key, depth, result.score, bound, result.best_move);

    result.nodes = context.nodes;
    return result;
}

NnueSearcherV10::NnueSearcherV10(
    const NnueValueModel& model,
    std::size_t tt_mb,
    V10MoveOrderingProfile profile
)
    : tt_(tt_mb),
      model_(model),
      profile_(profile) {
}

SearchResult NnueSearcherV10::search_best_move(const Position& pos, int depth) {
    killer_table_.clear();
    SearchContext context;
    context.accumulator.reset(model_, pos);
    return search_fixed_depth(pos, depth, context);
}

SearchResult NnueSearcherV10::search_best_move(const Position& pos, const SearchLimits& limits) {
    assert(limits.max_depth >= 0);

    killer_table_.clear();
    SearchResult best = make_fallback_result(pos);
    best.depth = 0;

    SearchContext context;
    context.accumulator.reset(model_, pos);
    context.has_deadline = limits.move_time.count() > 0;
    if (context.has_deadline) {
        context.deadline = Clock::now() + limits.move_time;
    }

    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        SearchResult current = search_fixed_depth(pos, depth, context);
        if (current.stopped || context.stopped) {
            best.stopped = true;
            best.nodes = context.nodes;
            return best;
        }
        best = current;
    }

    best.nodes = context.nodes;
    return best;
}

std::string_view NnueSearcherV10::name() const {
    return "nnue_v10";
}

void NnueSearcherV10::clear_tt() {
    tt_.clear();
}

std::size_t NnueSearcherV10::tt_entry_count() const {
    return tt_.entry_count();
}

void NnueSearcherV10::clear_tt_stats() {
    tt_.clear_stats();
}

const TranspositionTableStats& NnueSearcherV10::tt_stats() const {
    return tt_.stats();
}

} // namespace chess
