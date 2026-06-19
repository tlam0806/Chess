#include "heuristic_searcher_v26.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace chess {

namespace {

using Clock = std::chrono::steady_clock;

struct ProfileTimer {
    bool enabled = false;
    std::uint64_t& total_ns;
    std::uint64_t& calls;
    Clock::time_point start{};

    ProfileTimer(bool enabled_, std::uint64_t& total_ns_, std::uint64_t& calls_)
        : enabled(enabled_),
          total_ns(total_ns_),
          calls(calls_) {
        if (enabled) {
            ++calls;
            start = Clock::now();
        }
    }

    ~ProfileTimer() {
        if (!enabled) {
            return;
        }
        total_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
        );
    }
};

constexpr int MaxQuiescenceDepth = 16;
constexpr int MaxCheckEvasionQuiescenceDepth = 16;
constexpr int MateScoreThreshold = CheckmateScore - 1024;

constexpr bool EnableAspirationWindow = true;
constexpr bool EnableTt = true;
constexpr bool EnableTtExactStore = true;

bool is_valid_move(Move move) {
    return move.value != 0;
}

bool is_promotion(Move move) {
    return promotion_piece(move) != PieceType::None;
}

int ordering_piece_value(PieceType piece) {
    switch (piece) {
    case PieceType::Pawn:
        return 100;
    case PieceType::Knight:
        return 320;
    case PieceType::Bishop:
        return 330;
    case PieceType::Rook:
        return 500;
    case PieceType::Queen:
        return 900;
    default:
        return 0;
    }
}

ScoreRange exact_range(int score) {
    return ScoreRange{score, score};
}

ScoreRange lower_range(int score) {
    return ScoreRange{score, Infinity};
}

ScoreRange negate_range(ScoreRange range) {
    return ScoreRange{-range.upper, -range.lower};
}

ScoreRange intersect_ranges(ScoreRange lhs, ScoreRange rhs) {
    return ScoreRange{
        std::max(lhs.lower, rhs.lower),
        std::min(lhs.upper, rhs.upper)
    };
}

int representative_score(ScoreRange range) {
    return range.lower != -Infinity ? range.lower : range.upper;
}

Move preferred_tt_move(MoveRange moves) {
    return moves.lower.value != 0 ? moves.lower : moves.upper;
}

PieceType piece_type_on_square_for_color(const Position& pos, Color color, Square square) {
    if ((pos.occupancy(color) & bit(square)) == EmptyBB) {
        return PieceType::None;
    }
    return pos.piece_type_on_occupied(color, square);
}

bool is_pseudo_move_shape_valid(const Position& pos, Move move, PieceType moved_piece) {
    if (!is_valid_move(move) || moved_piece == PieceType::None) {
        return false;
    }

    const Square from = from_square(move);
    const Square to = to_square(move);
    if (from == to) {
        return false;
    }

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Bitboard to_mask = bit(to);
    const Bitboard occupancy = pos.occupancy();
    const bool target_is_ours = (pos.occupancy(us) & to_mask) != EmptyBB;
    const bool target_is_theirs = (pos.occupancy(them) & to_mask) != EmptyBB;
    const bool target_is_empty = !target_is_ours && !target_is_theirs;

    if (target_is_ours) {
        return false;
    }

    const MoveFlag flag = move_flag(move);
    const bool capture = is_capture(move);
    const bool promotion = promotion_piece(move) != PieceType::None;
    if (promotion && moved_piece != PieceType::Pawn) {
        return false;
    }

    if (moved_piece != PieceType::Pawn
        && flag != MoveFlag::Quiet
        && flag != MoveFlag::Capture
        && flag != MoveFlag::KingCastle
        && flag != MoveFlag::QueenCastle) {
        return false;
    }

    if (capture) {
        if (flag == MoveFlag::EnPassant) {
            if (moved_piece != PieceType::Pawn || to != pos.en_passant_square) {
                return false;
            }
        } else if (!target_is_theirs) {
            return false;
        }
    } else if (!target_is_empty) {
        return false;
    }
    switch (moved_piece) {
    case PieceType::Knight:
        return (knight_attacks(from) & to_mask) != EmptyBB;
    case PieceType::Bishop:
        return (bishop_attacks(from, occupancy) & to_mask) != EmptyBB;
    case PieceType::Rook:
        return (rook_attacks(from, occupancy) & to_mask) != EmptyBB;
    case PieceType::Queen:
        return (queen_attacks(from, occupancy) & to_mask) != EmptyBB;
    case PieceType::King:
        if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
            return false;
        }
        return (king_attacks(from) & to_mask) != EmptyBB;
    case PieceType::Pawn: {
        const int direction = us == Color::White ? 1 : -1;
        const int from_rank = rank_of(from);
        const int to_rank = rank_of(to);
        const int file_delta = file_of(to) - file_of(from);
        const int rank_delta = to_rank - from_rank;
        const int promotion_rank = us == Color::White
            ? static_cast<int>(Rank::R8)
            : static_cast<int>(Rank::R1);

        if (promotion != (to_rank == promotion_rank)) {
            return false;
        }
        if (capture) {
            return rank_delta == direction && (file_delta == -1 || file_delta == 1);
        }
        if (file_delta != 0) {
            return false;
        }
        if (rank_delta == direction) {
            return true;
        }
        if (flag != MoveFlag::DoublePawnPush || rank_delta != 2 * direction) {
            return false;
        }
        const int start_rank = us == Color::White
            ? static_cast<int>(Rank::R2)
            : static_cast<int>(Rank::R7);
        const Square middle = make_square(file_of(from), from_rank + direction);
        return from_rank == start_rank && pos.is_empty(middle);
    }
    case PieceType::None:
        return false;
    }

    return false;
}

bool has_rook_on(const Position& pos, Color color, Square square) {
    return (pos.pieces[static_cast<int>(color)][static_cast<int>(PieceType::Rook)] & bit(square)) != EmptyBB;
}

bool castle_shape_valid(const Position& pos, Move move, PieceType moved_piece) {
    if (moved_piece != PieceType::King) {
        return false;
    }
    const Color us = pos.side_to_move;
    const Square from = from_square(move);
    const Square to = to_square(move);
    const MoveFlag flag = move_flag(move);
    const Bitboard occupancy = pos.occupancy();

    if (us == Color::White) {
        const Square e1 = make_square(4, 0);
        if (from != e1) {
            return false;
        }
        if (flag == MoveFlag::KingCastle) {
            const Square f1 = make_square(5, 0);
            const Square g1 = make_square(6, 0);
            const Square h1 = make_square(7, 0);
            return to == g1
                && pos.white_can_castle_kingside
                && has_rook_on(pos, us, h1)
                && (occupancy & (bit(f1) | bit(g1))) == EmptyBB;
        }
        if (flag == MoveFlag::QueenCastle) {
            const Square d1 = make_square(3, 0);
            const Square c1 = make_square(2, 0);
            const Square b1 = make_square(1, 0);
            const Square a1 = make_square(0, 0);
            return to == c1
                && pos.white_can_castle_queenside
                && has_rook_on(pos, us, a1)
                && (occupancy & (bit(d1) | bit(c1) | bit(b1))) == EmptyBB;
        }
        return false;
    }

    const Square e8 = make_square(4, 7);
    if (from != e8) {
        return false;
    }
    if (flag == MoveFlag::KingCastle) {
        const Square f8 = make_square(5, 7);
        const Square g8 = make_square(6, 7);
        const Square h8 = make_square(7, 7);
        return to == g8
            && pos.black_can_castle_kingside
            && has_rook_on(pos, us, h8)
            && (occupancy & (bit(f8) | bit(g8))) == EmptyBB;
    }
    if (flag == MoveFlag::QueenCastle) {
        const Square d8 = make_square(3, 7);
        const Square c8 = make_square(2, 7);
        const Square b8 = make_square(1, 7);
        const Square a8 = make_square(0, 7);
        return to == c8
            && pos.black_can_castle_queenside
            && has_rook_on(pos, us, a8)
            && (occupancy & (bit(d8) | bit(c8) | bit(b8))) == EmptyBB;
    }
    return false;
}

bool is_move_legal_by_attack_check(
    const Position& pos,
    Move move,
    PieceType moved_piece
) {
    const MoveFlag flag = move_flag(move);
    if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        if (!castle_shape_valid(pos, move, moved_piece)) {
            return false;
        }
    } else if (!is_pseudo_move_shape_valid(pos, move, moved_piece)) {
        return false;
    }

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Square from = from_square(move);
    const Square to = to_square(move);
    Bitboard occupancy = pos.occupancy();
    occupancy &= ~bit(from);

    Square excluded_attacker_square = NoSquare;
    if (flag == MoveFlag::EnPassant) {
        excluded_attacker_square = us == Color::White ? to - 8 : to + 8;
        occupancy &= ~bit(excluded_attacker_square);
    } else if (is_capture(move)) {
        excluded_attacker_square = to;
        occupancy &= ~bit(to);
    }

    if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        const int direction = flag == MoveFlag::KingCastle ? 1 : -1;
        const Square transit = make_square(file_of(from) + direction, rank_of(from));
        if (is_square_attacked(pos, from, them)
            || is_square_attacked(pos, transit, them)) {
            return false;
        }
        const Square rook_from = flag == MoveFlag::KingCastle
            ? make_square(7, rank_of(from))
            : make_square(0, rank_of(from));
        const Square rook_to = flag == MoveFlag::KingCastle
            ? make_square(5, rank_of(from))
            : make_square(3, rank_of(from));
        occupancy &= ~bit(rook_from);
        occupancy |= bit(rook_to);
    }

    occupancy |= bit(to);
    const Square king = moved_piece == PieceType::King ? to : king_square(pos, us);
    return !is_square_attacked(pos, king, them, occupancy, excluded_attacker_square);
}

bool matches_any(Move move, Move a, Move b = Move{}, Move c = Move{}) {
    return (is_valid_move(a) && move == a)
        || (is_valid_move(b) && move == b)
        || (is_valid_move(c) && move == c);
}

} // namespace

struct HeuristicSearcherV26::SearchState {
    std::uint64_t nodes = 0;
    Clock::time_point deadline{};
    bool has_deadline = false;
    bool stopped = false;
};

enum class HeuristicSearcherV26::ScoringMode {
    MainSearch,
    Quiescence
};

enum class HeuristicSearcherV26::MoveGenerationStage {
    Priority,
    QuietNonPromotion
};

struct HeuristicSearcherV26::SearchValue {
    ScoreRange range{};
};

struct HeuristicSearcherV26::RootSearchResult {
    SearchResult result{};
    ScoreRange range{};
};

struct HeuristicSearcherV26::TTProbeResult {
    ScoreRange range{};
    MoveRange moves{};
    bool hit = false;
    bool has_score = false;
};

int HeuristicSearcherV26::score_to_tt(int score, int ply) const {
    if (score >= MateScoreThreshold) {
        return score + ply;
    }
    if (score <= -MateScoreThreshold) {
        return score - ply;
    }
    return score;
}

HeuristicSearcherV26::TTProbeResult HeuristicSearcherV26::probe_tt(
    HashKey key,
    int depth,
    int& alpha,
    int& beta,
    int ply,
    bool allow_probe
) const {
    ProfileTimer timer(
        timing_stats_enabled_,
        timing_stats_.tt_probe_ns,
        timing_stats_.tt_probe_calls);
    TTProbeResult result;
    if (!allow_probe || !EnableTt) {
        return result;
    }

    ScoreRange search_range{alpha, beta};
    result.hit = tt_.probe(
        key,
        depth,
        search_range,
        result.range,
        result.moves,
        ply,
        result.has_score,
        TTDepthPolicy::Exact
    );
    alpha = search_range.lower;
    beta = search_range.upper;
    return result;
}

bool HeuristicSearcherV26::should_store_tt(ScoreRange range) const {
    return EnableTt
        && (range.lower != range.upper || EnableTtExactStore);
}

void HeuristicSearcherV26::update_best_range(
    ScoreRange& node_range,
    Move& best_lower_move,
    Move& best_upper_move,
    int& alpha,
    Move move,
    ScoreRange move_range
) const {
    if (!is_valid_move(best_lower_move) && move_range.lower != -Infinity) {
        best_lower_move = move;
    }
    if (!is_valid_move(best_upper_move) && move_range.upper != -Infinity) {
        best_upper_move = move;
    }
    if (move_range.lower > node_range.lower) {
        node_range.lower = move_range.lower;
        best_lower_move = move;
    }
    if (move_range.upper > node_range.upper) {
        node_range.upper = move_range.upper;
        best_upper_move = move;
    }
    if (move_range.lower != -Infinity && move_range.lower > alpha) {
        alpha = move_range.lower;
    }
}

void HeuristicSearcherV26::store_tt_if_needed(
    HashKey key,
    int depth,
    int ply,
    ScoreRange range,
    Move best_lower_move,
    Move best_upper_move,
    Move fallback_best_move
) {
    if (!should_store_tt(range)) {
        return;
    }
    if (!is_valid_move(best_lower_move)) {
        best_lower_move = fallback_best_move;
    }
    if (!is_valid_move(best_upper_move)) {
        best_upper_move = fallback_best_move;
    }

    ScoreRange tt_range = range;
    if (tt_range.lower != -Infinity) {
        tt_range.lower = score_to_tt(tt_range.lower, ply);
    }
    if (tt_range.upper != Infinity) {
        tt_range.upper = score_to_tt(tt_range.upper, ply);
    }
    tt_.store(key, depth, tt_range, MoveRange{best_lower_move, best_upper_move});
}

void HeuristicSearcherV26::reward_quiet_cutoff(
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

void HeuristicSearcherV26::penalize_failed_quiets(
    Color side_to_move,
    int depth,
    Move prev_move,
    PieceType prev_moved_piece,
    const HeuristicSearcherV26::ScoredMoveList& failed_quiet_moves
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

bool HeuristicSearcherV26::should_stop(SearchState& state) const {
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

KingSafetyContext HeuristicSearcherV26::profiled_king_safety_context(const Position& pos) const {
    ProfileTimer timer(
        timing_stats_enabled_,
        timing_stats_.king_safety_ns,
        timing_stats_.king_safety_calls);
    return make_king_safety_context(pos);
}

int HeuristicSearcherV26::profiled_evaluate_for_side_to_move(const Position& pos) const {
    ProfileTimer timer(
        timing_stats_enabled_,
        timing_stats_.evaluate_ns,
        timing_stats_.evaluate_calls);
    return evaluate_for_side_to_move(pos);
}

void HeuristicSearcherV26::profiled_make_move(
    Position& pos,
    const ScoredMove& scored_move,
    UndoState& undo
) const {
    ProfileTimer timer(
        timing_stats_enabled_,
        timing_stats_.make_move_ns,
        timing_stats_.make_move_calls);
    pos.make_move(scored_move.move, scored_move.moved_piece, scored_move.captured_piece, undo);
}

void HeuristicSearcherV26::profiled_unmake_move(
    Position& pos,
    Move move,
    const UndoState& undo
) const {
    ProfileTimer timer(
        timing_stats_enabled_,
        timing_stats_.make_move_ns,
        timing_stats_.make_move_calls);
    pos.unmake_move(move, undo);
}

int HeuristicSearcherV26::move_order_score(
    Move move,
    const Position& pos,
    PieceType moved_piece,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    bool gives_check,
    bool capture,
    bool promotion,
    PieceType captured_piece,
    int see_score,
    ScoringMode stage
) const {
    ProfileTimer order_timer(
        timing_stats_enabled_,
        timing_stats_.order_score_ns,
        timing_stats_.order_score_calls);
    const MoveOrderingWeights& weights = move_ordering_weights_;
    const int weighted_see = weights.see_weight * see_score;
    if (stage == ScoringMode::Quiescence) {
        int score = weights.qsearch_see_weight * see_score;
        if (promotion) {
            score += weights.qsearch_promotion_bonus;
        }
        if (capture) {
            score += weights.qsearch_captured_value_weight * ordering_piece_value(captured_piece);
            score += see_score >= 0
                ? weights.qsearch_good_capture_bonus
                : weights.qsearch_bad_capture_bonus;
        }
        return score;
    }

    if (is_valid_move(tt_moves.lower) && move == tt_moves.lower) {
        return weights.tt_lower_bonus;
    }

    int score = 0;
    {
        ProfileTimer history_timer(
            timing_stats_enabled_,
            timing_stats_.history_score_ns,
            timing_stats_.history_score_calls);
        score = history_table_.get_score(pos.side_to_move, moved_piece, move);
    }
    if (promotion) {
        return weights.promotion_bonus + score + weighted_see;
    }
    if (capture) {
        if (see_score >= 0) {
            return weights.good_capture_bonus + score + weighted_see;
        }
        return weights.bad_capture_bonus + score + weighted_see;
    }
    if (gives_check) {
        score += weights.check_bonus;
    }

    if (!gives_check
        && is_valid_move(prev_move)
        && prev_moved_piece != PieceType::None) {
        int counter_history_score = 0;
        {
            ProfileTimer counter_history_timer(
                timing_stats_enabled_,
                timing_stats_.counter_history_score_ns,
                timing_stats_.counter_history_score_calls);
            counter_history_score = counter_history_table_.get_score(
                pos.side_to_move,
                prev_moved_piece,
                prev_move,
                moved_piece,
                move);
        }
        score += static_cast<int>(
            static_cast<long long>(weights.counter_history_bonus) * counter_history_score
                / CounterHistoryTable::MaxScore
        );
    }

    int killer_score = 0;
    {
        ProfileTimer killer_timer(
            timing_stats_enabled_,
            timing_stats_.killer_score_ns,
            timing_stats_.killer_score_calls);
        killer_score = killer_table_.score(ply, move);
    }
    if (killer_score == 2) {
        score += weights.killer1_bonus;
    } else if (killer_score == 1) {
        score += weights.killer2_bonus;
    }

    return score;
}

HeuristicSearcherV26::ScoredMove HeuristicSearcherV26::make_scored_move(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Move move,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    ScoringMode stage
) const {
    const PieceType moved_piece =
        piece_type_on_square_for_color(pos, pos.side_to_move, from_square(move));
    if (moved_piece == PieceType::None) {
        return ScoredMove{};
    }
    const bool capture = is_capture(move);
    PieceType captured_piece = PieceType::None;
    if (capture
        && move_flag(move) != MoveFlag::EnPassant) {
        captured_piece = pos.piece_type_on_occupied(opposite(pos.side_to_move), to_square(move));
    } else if (move_flag(move) == MoveFlag::EnPassant) {
        captured_piece = PieceType::Pawn;
    }
    if (!is_pseudo_move_legal(pos, king_safety, move, moved_piece, captured_piece)) {
        return ScoredMove{};
    }

    return make_scored_legal_move(
        pos,
        move,
        moved_piece,
        captured_piece,
        ply,
        tt_moves,
        prev_move,
        prev_moved_piece,
        stage);
}

HeuristicSearcherV26::ScoredMove HeuristicSearcherV26::make_scored_move_by_attack_check(
    const Position& pos,
    Move move,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    ScoringMode stage
) const {
    const PieceType moved_piece = pos.piece_type_on_occupied(pos.side_to_move, from_square(move));
    const bool capture = is_capture(move);
    PieceType captured_piece = PieceType::None;
    if (capture && move_flag(move) != MoveFlag::EnPassant) {
        captured_piece = pos.piece_type_on_occupied(opposite(pos.side_to_move), to_square(move));
    } else if (move_flag(move) == MoveFlag::EnPassant) {
        captured_piece = PieceType::Pawn;
    }
    if (!is_move_legal_by_attack_check(pos, move, moved_piece)) {
        return ScoredMove{};
    }
    return make_scored_legal_move(
        pos,
        move,
        moved_piece,
        captured_piece,
        ply,
        tt_moves,
        prev_move,
        prev_moved_piece,
        stage);
}

HeuristicSearcherV26::ScoredMove HeuristicSearcherV26::make_scored_legal_move(
    const Position& pos,
    Move move,
    PieceType moved_piece,
    PieceType captured_piece,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    ScoringMode stage
) const {
    ProfileTimer score_timer(
        timing_stats_enabled_,
        timing_stats_.score_move_ns,
        timing_stats_.score_move_calls);
    ProfileTimer mode_score_timer(
        timing_stats_enabled_,
        stage == ScoringMode::Quiescence ? timing_stats_.qsearch_score_move_ns : timing_stats_.main_score_move_ns,
        stage == ScoringMode::Quiescence ? timing_stats_.qsearch_score_move_calls : timing_stats_.main_score_move_calls);
    const bool capture = is_capture(move);
    bool gives_check = false;
    if (stage == ScoringMode::MainSearch) {
        ProfileTimer gives_check_timer(
            timing_stats_enabled_,
            timing_stats_.gives_check_ns,
            timing_stats_.gives_check_calls);
        gives_check = gives_check_fast(pos, move, moved_piece, captured_piece);
    }
    const bool promotion = is_promotion(move);
    int see_score = 0;
    if (capture) {
        ProfileTimer see_timer(
            timing_stats_enabled_,
            timing_stats_.see_ns,
            timing_stats_.see_calls);
        ProfileTimer mode_see_timer(
            timing_stats_enabled_,
            stage == ScoringMode::Quiescence ? timing_stats_.qsearch_see_ns : timing_stats_.main_see_ns,
            stage == ScoringMode::Quiescence ? timing_stats_.qsearch_see_calls : timing_stats_.main_see_calls);
        see_score = static_exchange_eval(pos, move, moved_piece, captured_piece);
    }

    if (stage == ScoringMode::Quiescence) {
        return ScoredMove{
            move_order_score(
                move,
                pos,
                moved_piece,
                ply,
                tt_moves,
                prev_move,
                prev_moved_piece,
                gives_check,
                capture,
                promotion,
                captured_piece,
                see_score,
                stage),
            see_score,
            move,
            moved_piece,
            captured_piece,
            gives_check,
            capture,
            promotion
        };
    }

    return ScoredMove{
        move_order_score(
            move,
            pos,
            moved_piece,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            gives_check,
            capture,
            promotion,
            captured_piece,
            see_score,
            stage),
        see_score,
        move,
        moved_piece,
        captured_piece,
        gives_check,
        capture,
        promotion
    };
}

bool HeuristicSearcherV26::better_scored_move(
    const ScoredMove& lhs,
    const ScoredMove& rhs
) const {
    return lhs.order_score > rhs.order_score;
}

void HeuristicSearcherV26::sort_scored_moves(ScoredMoveList& moves) const {
    ProfileTimer timer(
        timing_stats_enabled_,
        timing_stats_.sort_ns,
        timing_stats_.sort_calls);
    for (std::size_t i = 1; i < moves.size(); ++i) {
        ScoredMove current = moves[i];
        std::size_t j = i;
        while (j > 0 && better_scored_move(current, moves[j - 1])) {
            moves[j] = moves[j - 1];
            --j;
        }
        moves[j] = current;
    }
}

HeuristicSearcherV26::ScoredMoveList HeuristicSearcherV26::ordered_moves(
    const Position& pos,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    Move skip_move
) const {
    ScoredMoveList ordered;
    const KingSafetyContext king_safety = profiled_king_safety_context(pos);
    MoveList moves;
    {
        ProfileTimer timer(
            timing_stats_enabled_,
            timing_stats_.move_generation_ns,
            timing_stats_.move_generation_calls);
        generate_pseudo_legal_moves(pos, moves);
    }

    for (Move move : moves) {
        if (is_valid_move(skip_move) && move == skip_move) {
            continue;
        }
        ScoredMove scored_move =
            make_scored_move(
                pos,
                king_safety,
                move,
                ply,
                tt_moves,
                prev_move,
                prev_moved_piece,
                ScoringMode::MainSearch);
        if (is_valid_move(scored_move.move)) {
            ordered.push_back(scored_move);
        }
    }

    sort_scored_moves(ordered);

    return ordered;
}

Move HeuristicSearcherV26::valid_priority_killer(const Position& pos, int ply, int slot) const {
    const Move move = killer_table_.move(ply, slot);
    if (!is_valid_move(move)) {
        return Move{};
    }
    const MoveFlag flag = move_flag(move);
    if (flag != MoveFlag::Quiet && flag != MoveFlag::DoublePawnPush) {
        return Move{};
    }
    const PieceType moved_piece =
        piece_type_on_square_for_color(pos, pos.side_to_move, from_square(move));
    if (!is_pseudo_move_shape_valid(pos, move, moved_piece)) {
        return Move{};
    }
    return move;
}

HeuristicSearcherV26::ScoredMoveList
HeuristicSearcherV26::generate_legal_noisy_scored_moves_for_searcher(
    const Position& pos,
    const KingSafetyContext& king_safety,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    Move skip_tt_move,
    std::uint64_t& legal_checks
) {
    MoveList legal_noisy_moves;
    {
        ProfileTimer timer(
            timing_stats_enabled_,
            timing_stats_.move_generation_ns,
            timing_stats_.move_generation_calls);
        generate_legal_noisy_moves(pos, king_safety, legal_noisy_moves);
    }

    ScoredMoveList ordered;
    legal_checks += legal_noisy_moves.size();

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    for (Move move : legal_noisy_moves) {
        if (matches_any(move, skip_tt_move)) {
            continue;
        }

        const PieceType moved_piece = pos.piece_type_on_occupied(us, from_square(move));
        PieceType captured_piece = PieceType::None;
        if (is_capture(move) && move_flag(move) != MoveFlag::EnPassant) {
            captured_piece = pos.piece_type_on_occupied(them, to_square(move));
        } else if (move_flag(move) == MoveFlag::EnPassant) {
            captured_piece = PieceType::Pawn;
        }

        ScoredMove scored_move = make_scored_legal_move(
            pos,
            move,
            moved_piece,
            captured_piece,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            ScoringMode::MainSearch);
        const bool deferred_bad_capture =
            scored_move.capture
            && !scored_move.promotion
            && scored_move.see_score < move_ordering_weights_.bad_capture_stage_threshold;
        if (deferred_bad_capture) {
            continue;
        }

        record_generated_stage_move(
            scored_move,
            pos.side_to_move,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            MoveGenerationStage::Priority);
        ordered.push_back(scored_move);
    }

    return ordered;
}

HeuristicSearcherV26::ScoredMoveList HeuristicSearcherV26::ordered_moves_for_stage(
    const Position& pos,
    const KingSafetyContext& king_safety,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    Move skip_tt_move,
    Move skip_killer1,
    Move skip_killer2,
    MoveGenerationStage generation_stage
) {
    ScoredMoveList ordered;
    std::uint64_t legal_checks = 0;
    if (move_ordering_stats_enabled_) {
        if (generation_stage == MoveGenerationStage::Priority) {
            ++move_ordering_stats_.priority_king_safety_contexts;
        } else {
            ++move_ordering_stats_.quiet_king_safety_contexts;
        }
    }
    auto add_scored_move = [&](Move move) {
        if (!is_valid_move(move)) {
            return;
        }
        ++legal_checks;
        ScoredMove scored_move =
            make_scored_move(
                pos,
                king_safety,
                move,
                ply,
                tt_moves,
                prev_move,
                prev_moved_piece,
                ScoringMode::MainSearch);
        if (is_valid_move(scored_move.move)) {
            const bool deferred_bad_capture =
                scored_move.capture
                && !scored_move.promotion
                && scored_move.see_score < move_ordering_weights_.bad_capture_stage_threshold;
            if (generation_stage == MoveGenerationStage::Priority && deferred_bad_capture) {
                return;
            }
            if (generation_stage == MoveGenerationStage::QuietNonPromotion
                && scored_move.capture
                && !deferred_bad_capture) {
                return;
            }
            record_generated_stage_move(
                scored_move,
                pos.side_to_move,
                ply,
                tt_moves,
                prev_move,
                prev_moved_piece,
                generation_stage);
            ordered.push_back(scored_move);
        }
    };
    auto add_killer_move = [&](Move move, Move other_killer) {
        if (!is_valid_move(move) || matches_any(move, skip_tt_move, other_killer)) {
            return;
        }
        add_scored_move(move);
    };

    if (generation_stage == MoveGenerationStage::Priority) {
        ordered = generate_legal_noisy_scored_moves_for_searcher(
            pos,
            king_safety,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            skip_tt_move,
            legal_checks);
        add_killer_move(skip_killer1, skip_killer2);
        add_killer_move(skip_killer2, skip_killer1);
    } else {
        MoveList moves;
        {
            ProfileTimer timer(
                timing_stats_enabled_,
                timing_stats_.move_generation_ns,
                timing_stats_.move_generation_calls);
            generate_legal_non_capture_moves(pos, king_safety, moves);
        }
        MoveList bad_captures;
        {
            ProfileTimer timer(
                timing_stats_enabled_,
                timing_stats_.move_generation_ns,
                timing_stats_.move_generation_calls);
            generate_pseudo_noisy_moves(pos, bad_captures);
        }
        legal_checks += moves.size();
        const Color us = pos.side_to_move;
        for (Move move : moves) {
            if (!is_promotion(move)
                && !matches_any(move, skip_tt_move, skip_killer1, skip_killer2)) {
                const PieceType moved_piece = pos.piece_type_on_occupied(us, from_square(move));
                ScoredMove scored_move = make_scored_legal_move(
                    pos,
                    move,
                    moved_piece,
                    PieceType::None,
                    ply,
                    tt_moves,
                    prev_move,
                    prev_moved_piece,
                    ScoringMode::MainSearch);
                record_generated_stage_move(
                    scored_move,
                    pos.side_to_move,
                    ply,
                    tt_moves,
                    prev_move,
                    prev_moved_piece,
                    generation_stage);
                ordered.push_back(scored_move);
            }
        }
        for (Move move : bad_captures) {
            if (is_capture(move)
                && !is_promotion(move)
                && !matches_any(move, skip_tt_move, skip_killer1, skip_killer2)) {
                add_scored_move(move);
            }
        }
    }

    if (move_ordering_stats_enabled_) {
        if (generation_stage == MoveGenerationStage::Priority) {
            move_ordering_stats_.priority_legal_checks += legal_checks;
        } else {
            move_ordering_stats_.quiet_legal_checks += legal_checks;
        }
    }
    sort_scored_moves(ordered);
    if (generation_stage == MoveGenerationStage::Priority) {
        record_priority_stage_top_moves(
            ordered,
            pos.side_to_move,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece);
    }
    return ordered;
}

HeuristicSearcherV26::SearchValue HeuristicSearcherV26::quiescence(
    Position& pos,
    int alpha,
    int beta,
    int ply,
    int q_depth,
    SearchState& state
) {
    ProfileTimer qsearch_timer(
        timing_stats_enabled_,
        timing_stats_.qsearch_ns,
        timing_stats_.qsearch_calls);
    assert(alpha < beta);
    ++state.nodes;

    if (should_stop(state)) {
        return SearchValue{exact_range(0)};
    }

    const KingSafetyContext king_safety = profiled_king_safety_context(pos);
    if (move_ordering_stats_enabled_) {
        ++move_ordering_stats_.qsearch_king_safety_contexts;
    }
    const bool side_in_check = king_safety.checkers != EmptyBB;
    ScoredMoveList moves;
    bool has_legal_move = false;
    MoveList legal_moves;
    {
        ProfileTimer timer(
            timing_stats_enabled_,
            timing_stats_.move_generation_ns,
            timing_stats_.move_generation_calls);
        if (side_in_check) {
            generate_legal_evasion_moves(pos, king_safety, legal_moves);
        } else {
            generate_legal_noisy_moves(pos, king_safety, legal_moves);
        }
    }

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    for (Move move : legal_moves) {
        const PieceType moved_piece = pos.piece_type_on_occupied(us, from_square(move));
        PieceType captured_piece = PieceType::None;
        if (is_capture(move) && move_flag(move) != MoveFlag::EnPassant) {
            captured_piece = pos.piece_type_on_occupied(them, to_square(move));
        } else if (move_flag(move) == MoveFlag::EnPassant) {
            captured_piece = PieceType::Pawn;
        }
        if (move_ordering_stats_enabled_) {
            ++move_ordering_stats_.qsearch_legal_checks;
        }
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
    }

    if (!has_legal_move && side_in_check) {
        return SearchValue{exact_range(-CheckmateScore + ply)};
    }
    // Non-check quiescence intentionally does not detect stalemate.
    // Root/negamax handle terminal positions when depth remains.
 
    if (side_in_check) {
        if (q_depth >= MaxCheckEvasionQuiescenceDepth) {
            return SearchValue{exact_range(profiled_evaluate_for_side_to_move(pos))};
        }

        ScoreRange node_range;
        node_range.lower = -Infinity;
        node_range.upper = -Infinity;
        sort_scored_moves(moves);
        if (!moves.empty()) {
            record_node_with_moves(ScoringMode::Quiescence);
        }

        for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
            const ScoredMove& scored_move = moves[move_index];
            UndoState undo;
            pos.make_move(scored_move.move, scored_move.moved_piece, scored_move.captured_piece, undo);

            SearchValue child = quiescence(pos, -beta, -alpha, ply + 1, q_depth + 1, state);
            pos.unmake_move(scored_move.move, undo);
            const ScoreRange move_range = negate_range(child.range);
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }

            node_range.lower = std::max(node_range.lower, move_range.lower);
            node_range.upper = std::max(node_range.upper, move_range.upper);
            if (node_range.lower >= beta) {
                if (move_ordering_stats_enabled_) {
                    record_beta_cutoff(
                        moves,
                        move_index,
                        pos.side_to_move,
                        ply,
                        MoveRange{},
                        Move{},
                        PieceType::None,
                        ScoringMode::Quiescence);
                }
                return SearchValue{ScoreRange{node_range.lower, Infinity}};
            }
            if (move_range.lower != -Infinity && move_range.lower > alpha) {
                alpha = move_range.lower;
            }
        }

        return SearchValue{node_range};
    }

    int best_score = profiled_evaluate_for_side_to_move(pos);
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
    if (!moves.empty()) {
        record_node_with_moves(ScoringMode::Quiescence);
    }
    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];
        UndoState undo;
        pos.make_move(scored_move.move, scored_move.moved_piece, scored_move.captured_piece, undo);

        SearchValue child = quiescence(pos, -beta, -alpha, ply + 1, q_depth + 1, state);
        pos.unmake_move(scored_move.move, undo);
        const ScoreRange move_range = negate_range(child.range);
        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }

        node_range.lower = std::max(node_range.lower, move_range.lower);
        node_range.upper = std::max(node_range.upper, move_range.upper);
        if (node_range.lower >= beta) {
            if (move_ordering_stats_enabled_) {
                record_beta_cutoff(
                    moves,
                    move_index,
                    pos.side_to_move,
                    ply,
                    MoveRange{},
                    Move{},
                    PieceType::None,
                    ScoringMode::Quiescence);
            }
            return SearchValue{ScoreRange{node_range.lower, Infinity}};
        }
        if (move_range.lower != -Infinity && move_range.lower > alpha) {
            alpha = move_range.lower;
        }
    }

    return SearchValue{node_range};
}

bool HeuristicSearcherV26::is_quiet_move(const ScoredMove& scored_move) const {
    return !scored_move.capture
        && !scored_move.promotion
        && !scored_move.gives_check;
}

void HeuristicSearcherV26::add_move_type_stats(
    MoveTypeStats& stats,
    const ScoredMove& move,
    Color side_to_move,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    ScoringMode stage
) const {
    ++stats.total;
    if (stage == ScoringMode::MainSearch) {
        if (is_valid_move(tt_moves.lower) && move.move == tt_moves.lower) {
            ++stats.tt_lower;
        }
        if (is_valid_move(tt_moves.upper)
            && tt_moves.upper != tt_moves.lower
            && move.move == tt_moves.upper) {
            ++stats.tt_upper;
        }
    }
    if (move.promotion) {
        ++stats.promotion;
    }
    if (move.capture) {
        ++stats.capture;
    }
    if (is_quiet_move(move)) {
        ++stats.quiet;
    }
    if (move.gives_check) {
        ++stats.check;
    }
    if (stage == ScoringMode::MainSearch) {
        const int killer_score = killer_table_.score(ply, move.move);
        if (killer_score == 2) {
            ++stats.killer1;
        } else if (killer_score == 1) {
            ++stats.killer2;
        }
        if (history_table_.get_score(side_to_move, move.moved_piece, move.move) > 0) {
            ++stats.history_positive;
        }
        if (prev_moved_piece != PieceType::None && is_valid_move(prev_move)) {
            const int counter_history_score = counter_history_table_.get_score(
                side_to_move,
                prev_moved_piece,
                prev_move,
                move.moved_piece,
                move.move);
            if (counter_history_score > 0) {
                ++stats.counter_history_positive;
            }
        }
    }
}

void HeuristicSearcherV26::record_node_with_moves(ScoringMode stage) {
    if (!move_ordering_stats_enabled_) {
        return;
    }
    MoveCutoffStats& stats = stage == ScoringMode::MainSearch
        ? move_ordering_stats_.main
        : move_ordering_stats_.qsearch;
    ++stats.nodes_with_moves;
}

void HeuristicSearcherV26::record_beta_cutoff(
    const HeuristicSearcherV26::ScoredMoveList& moves,
    std::size_t cutoff_index,
    Color side_to_move,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    ScoringMode stage
) {
    if (!move_ordering_stats_enabled_) {
        return;
    }
    MoveCutoffStats& stats = stage == ScoringMode::MainSearch
        ? move_ordering_stats_.main
        : move_ordering_stats_.qsearch;
    ++stats.beta_cutoffs;
    stats.cutoff_index_sum += cutoff_index;
    const std::size_t bucket = std::min<std::size_t>(cutoff_index, stats.cutoff_index_buckets.size() - 1);
    ++stats.cutoff_index_buckets[bucket];

    for (std::size_t i = 0; i < cutoff_index; ++i) {
        add_move_type_stats(stats.before_cutoff, moves[i], side_to_move, ply, tt_moves, prev_move, prev_moved_piece, stage);
    }
    add_move_type_stats(stats.cutoff_move, moves[cutoff_index], side_to_move, ply, tt_moves, prev_move, prev_moved_piece, stage);
    for (std::size_t i = cutoff_index + 1; i < moves.size(); ++i) {
        add_move_type_stats(stats.after_cutoff, moves[i], side_to_move, ply, tt_moves, prev_move, prev_moved_piece, stage);
    }
}

void HeuristicSearcherV26::record_generated_stage_move(
    const ScoredMove& move,
    Color side_to_move,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    MoveGenerationStage generation_stage
) {
    if (!move_ordering_stats_enabled_) {
        return;
    }
    MoveTypeStats& stats = generation_stage == MoveGenerationStage::Priority
        ? move_ordering_stats_.noisy_stage
        : move_ordering_stats_.quiet_stage;
    add_move_type_stats(
        stats,
        move,
        side_to_move,
        ply,
        tt_moves,
        prev_move,
        prev_moved_piece,
        ScoringMode::MainSearch);
}

void HeuristicSearcherV26::record_priority_stage_top_moves(
    const HeuristicSearcherV26::ScoredMoveList& moves,
    Color side_to_move,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece
) {
    if (!move_ordering_stats_enabled_) {
        return;
    }
    const std::size_t limit = std::min<std::size_t>(moves.size(), move_ordering_stats_.priority_stage_top.size());
    for (std::size_t i = 0; i < limit; ++i) {
        add_move_type_stats(
            move_ordering_stats_.priority_stage_top[i],
            moves[i],
            side_to_move,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            ScoringMode::MainSearch);
    }
}

HeuristicSearcherV26::SearchValue HeuristicSearcherV26::negamax(
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

    ScoreRange node_range;
    node_range.lower = -Infinity;
    node_range.upper = -Infinity;
    Move best_lower_move{};
    Move best_upper_move{};
    Move fallback_best_move{};
    ScoredMoveList failed_quiet_moves;
    bool searched_any_move = false;
    Move searched_tt_move = tt_probe.moves.lower;
    ScoredMoveList searched_moves;
    std::optional<KingSafetyContext> king_safety_cache;
    auto get_king_safety = [&]() -> const KingSafetyContext& {
        if (!king_safety_cache) {
            king_safety_cache = profiled_king_safety_context(pos);
        }
        return *king_safety_cache;
    };

    const Move tt_lower_move = tt_probe.moves.lower;
    if (is_valid_move(tt_lower_move)) {
        if (move_ordering_stats_enabled_) {
            ++move_ordering_stats_.tt_lower_legal_checks;
        }
        ScoredMove scored_move = make_scored_move_by_attack_check(
            pos,
            tt_lower_move,
            ply,
            tt_probe.moves,
            prev_move,
            prev_moved_piece,
            ScoringMode::MainSearch);
        if (is_valid_move(scored_move.move)) {
            if (move_ordering_stats_enabled_) {
                ++move_ordering_stats_.tt_lower_stage0_attempts;
            }
            searched_any_move = true;
            searched_tt_move = scored_move.move;
            fallback_best_move = scored_move.move;
            if (move_ordering_stats_enabled_) {
                searched_moves.push_back(scored_move);
            }

            UndoState undo;
            pos.make_move(scored_move.move, scored_move.moved_piece, scored_move.captured_piece, undo);

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
            pos.unmake_move(scored_move.move, undo);
            const ScoreRange move_range = negate_range(child.range);
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }

            update_best_range(node_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (node_range.lower >= beta) {
                if (move_ordering_stats_enabled_) {
                    ++move_ordering_stats_.tt_lower_stage0_cutoffs;
                }
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
    const Move priority_killer1{};
    const Move priority_killer2{};
    for (MoveGenerationStage generation_stage : {
             MoveGenerationStage::Priority,
             MoveGenerationStage::QuietNonPromotion
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
            priority_killer1,
            priority_killer2,
            generation_stage);
        if (!searched_any_move && !moves.empty()) {
            record_node_with_moves(ScoringMode::MainSearch);
        }

        for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
            const ScoredMove& scored_move = moves[move_index];
            const PieceType moved_piece = scored_move.moved_piece;
            if (!is_valid_move(fallback_best_move)) {
                fallback_best_move = scored_move.move;
            }

            UndoState undo;
            pos.make_move(scored_move.move, scored_move.moved_piece, scored_move.captured_piece, undo);

            ScoreRange move_range;
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
                    pos.unmake_move(scored_move.move, undo);
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
                        pos.unmake_move(scored_move.move, undo);
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
                    pos.unmake_move(scored_move.move, undo);
                    return SearchValue{exact_range(0)};
                }
            }
            pos.unmake_move(scored_move.move, undo);
            searched_any_move = true;
            if (move_ordering_stats_enabled_) {
                searched_moves.push_back(scored_move);
            }
            update_best_range(node_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (node_range.lower >= beta) {
                if (move_ordering_stats_enabled_) {
                    if (generation_stage == MoveGenerationStage::Priority) {
                        ++move_ordering_stats_.priority_stage_cutoffs;
                    } else {
                        ++move_ordering_stats_.quiet_stage_cutoffs;
                    }
                    record_beta_cutoff(
                        searched_moves,
                        searched_moves.size() - 1,
                        pos.side_to_move,
                        ply,
                        tt_probe.moves,
                        prev_move,
                        prev_moved_piece,
                        ScoringMode::MainSearch);
                }
                reward_quiet_cutoff(pos.side_to_move, depth, ply, scored_move, prev_move, prev_moved_piece);
                penalize_failed_quiets(pos.side_to_move, depth, prev_move, prev_moved_piece, failed_quiet_moves);
                const bool has_unsearched_stage =
                    generation_stage == MoveGenerationStage::Priority
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

SearchResult HeuristicSearcherV26::make_fallback_result(const Position& pos) const {
    SearchResult result;
    const ScoredMoveList moves = ordered_moves(pos, 0);
    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }
    result.best_move = moves[0].move;
    result.score = profiled_evaluate_for_side_to_move(pos);
    result.nodes = 1;
    return result;
}

HeuristicSearcherV26::RootSearchResult HeuristicSearcherV26::search_fixed_depth(
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

    ScoreRange root_range;
    root_range.lower = -Infinity;
    root_range.upper = -Infinity;
    Move best_lower_move{};
    Move best_upper_move{};
    Move fallback_best_move{};
    bool searched_any_move = false;
    Move searched_tt_move = tt_probe.moves.lower;
    ScoredMoveList searched_moves;
    std::optional<KingSafetyContext> king_safety_cache;
    auto get_king_safety = [&]() -> const KingSafetyContext& {
        if (!king_safety_cache) {
            king_safety_cache = profiled_king_safety_context(pos);
        }
        return *king_safety_cache;
    };

    const Move tt_lower_move = tt_probe.moves.lower;
    if (is_valid_move(tt_lower_move)) {
        if (move_ordering_stats_enabled_) {
            ++move_ordering_stats_.tt_lower_legal_checks;
        }
        ScoredMove scored_move = make_scored_move_by_attack_check(
            pos,
            tt_lower_move,
            0,
            tt_probe.moves,
            Move{},
            PieceType::None,
            ScoringMode::MainSearch);
        if (is_valid_move(scored_move.move)) {
            if (move_ordering_stats_enabled_) {
                ++move_ordering_stats_.tt_lower_stage0_attempts;
            }
            searched_any_move = true;
            searched_tt_move = scored_move.move;
            fallback_best_move = scored_move.move;
            if (move_ordering_stats_enabled_) {
                searched_moves.push_back(scored_move);
            }

            UndoState undo;
            pos.make_move(scored_move.move, scored_move.moved_piece, scored_move.captured_piece, undo);

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
            pos.unmake_move(scored_move.move, undo);
            const ScoreRange move_range = negate_range(child.range);
            if (state.stopped) {
                result.stopped = true;
                result.nodes = state.nodes;
                return root_result;
            }

            update_best_range(root_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (root_range.lower >= beta) {
                if (move_ordering_stats_enabled_) {
                    ++move_ordering_stats_.tt_lower_stage0_cutoffs;
                }
                root_range.upper = Infinity;
            }
        }
    }

    bool cutoff = root_range.lower >= beta;
    const Move priority_killer1{};
    const Move priority_killer2{};
    for (MoveGenerationStage generation_stage : {
             MoveGenerationStage::Priority,
             MoveGenerationStage::QuietNonPromotion
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
            priority_killer1,
            priority_killer2,
            generation_stage);
        if (!searched_any_move && !moves.empty()) {
            record_node_with_moves(ScoringMode::MainSearch);
        }

        for (std::size_t move_index = 0; root_range.lower < beta && move_index < moves.size(); ++move_index) {
            const ScoredMove& scored_move = moves[move_index];
            const PieceType moved_piece = scored_move.moved_piece;
            if (!is_valid_move(fallback_best_move)) {
                fallback_best_move = scored_move.move;
            }

            UndoState undo;
            pos.make_move(scored_move.move, scored_move.moved_piece, scored_move.captured_piece, undo);

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
            pos.unmake_move(scored_move.move, undo);
            const ScoreRange move_range = negate_range(child.range);
            if (state.stopped) {
                result.stopped = true;
                result.nodes = state.nodes;
                return root_result;
            }

            searched_any_move = true;
            if (move_ordering_stats_enabled_) {
                searched_moves.push_back(scored_move);
            }

            update_best_range(root_range, best_lower_move, best_upper_move, alpha, scored_move.move, move_range);
            if (root_range.lower >= beta) {
                if (move_ordering_stats_enabled_) {
                    if (generation_stage == MoveGenerationStage::Priority) {
                        ++move_ordering_stats_.priority_stage_cutoffs;
                    } else {
                        ++move_ordering_stats_.quiet_stage_cutoffs;
                    }
                    record_beta_cutoff(
                        searched_moves,
                        searched_moves.size() - 1,
                        pos.side_to_move,
                        0,
                        tt_probe.moves,
                        Move{},
                        PieceType::None,
                        ScoringMode::MainSearch);
                }
                const bool has_unsearched_stage =
                    generation_stage == MoveGenerationStage::Priority
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

SearchResult HeuristicSearcherV26::search_root_without_tt_probe(
    const Position& pos,
    int depth,
    SearchState& state
) {
    return search_fixed_depth(pos, depth, state, -Infinity, Infinity, false).result;
}

SearchResult HeuristicSearcherV26::search_best_move(const Position& pos, int depth) {
    killer_table_.clear();
    SearchState state;
    return search_fixed_depth(pos, depth, state).result;
}

SearchResult HeuristicSearcherV26::search_best_move(const Position& pos, const SearchLimits& limits) {
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
