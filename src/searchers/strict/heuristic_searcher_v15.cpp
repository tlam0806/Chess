#include "heuristic_searcher_v15.hpp"

#include "attacks.hpp"
#include "evaluate.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <vector>

namespace chess {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int MaxQuiescenceDepth = 16;
constexpr int MateScoreThreshold = CheckmateScore - 1024;

constexpr bool EnableAspirationWindow = true;
constexpr bool EnableTt = true;
constexpr bool EnableTtExactStore = true;

constexpr int TtBonus = 1'000'000'000;
constexpr int PromotionBonus = 900'000'000;
constexpr int GoodCaptureBonus = 50'000'000;
constexpr int BadCaptureBonus = 30'000;
constexpr int SeeWeight = 10;
constexpr int Killer1Bonus = 40'000;
constexpr int Killer2Bonus = 10'000;
constexpr int CheckBonus = 160'000;

bool is_valid_move(Move move) {
    return move.value != 0;
}

bool is_promotion(Move move) {
    return promotion_piece(move) != PieceType::None;
}

bool is_noisy_move(Move move) {
    return is_capture(move) || is_promotion(move);
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

enum class WindowStatus {
    Inside,
    Below,
    Above
};

int representative_score_for_status(ScoreRange range, WindowStatus status) {
    if (status == WindowStatus::Below) {
        return range.upper;
    }
    return representative_score(range);
}

WindowStatus status_from_range(ScoreRange range) {
    if (range.lower == range.upper) {
        return WindowStatus::Inside;
    }
    if (range.lower != -Infinity) {
        return WindowStatus::Above;
    }
    return WindowStatus::Below;
}

WindowStatus status_for_window(ScoreRange range, int alpha, int beta) {
    if (range.lower == range.upper) {
        return WindowStatus::Inside;
    }
    if (range.upper <= alpha) {
        return WindowStatus::Below;
    }
    if (range.lower >= beta) {
        return WindowStatus::Above;
    }
    return status_from_range(range);
}

Move preferred_tt_move(MoveRange moves) {
    return moves.lower.value != 0 ? moves.lower : moves.upper;
}

} // namespace

struct HeuristicSearcherV15::SearchState {
    std::uint64_t nodes = 0;
    Clock::time_point deadline{};
    bool has_deadline = false;
    bool stopped = false;
    WindowStatus root_status = WindowStatus::Inside;
};

enum class HeuristicSearcherV15::ScoringMode {
    MainSearch,
    Quiescence
};

struct HeuristicSearcherV15::ScoredMove {
    Move move{};
    int order_score = 0;
    bool gives_check = false;
    bool capture = false;
    bool promotion = false;
};

struct HeuristicSearcherV15::SearchValue {
    ScoreRange range{};
};

struct HeuristicSearcherV15::TTProbeResult {
    ScoreRange range{};
    MoveRange moves{};
    bool hit = false;
    bool has_score = false;
};

int HeuristicSearcherV15::score_to_tt(int score, int ply) const {
    if (score >= MateScoreThreshold) {
        return score + ply;
    }
    if (score <= -MateScoreThreshold) {
        return score - ply;
    }
    return score;
}

HeuristicSearcherV15::TTProbeResult HeuristicSearcherV15::probe_tt(
    HashKey key,
    int depth,
    int& alpha,
    int& beta,
    int ply,
    bool allow_probe
) const {
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

bool HeuristicSearcherV15::should_store_tt(ScoreRange range) const {
    return EnableTt
        && (range.lower != range.upper || EnableTtExactStore);
}

bool HeuristicSearcherV15::should_stop(SearchState& state) const {
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

int HeuristicSearcherV15::move_order_score(
    Move move,
    const Position& pos,
    int ply,
    Move tt_move,
    bool gives_check,
    bool capture,
    bool promotion,
    int see_score,
    ScoringMode stage
) const {
    const int weighted_see = SeeWeight * see_score;
    if (stage == ScoringMode::Quiescence) {
        return promotion ? PromotionBonus : weighted_see;
    }

    if (is_valid_move(tt_move) && move == tt_move) {
        return TtBonus;
    }

    int score = history_table_.get_score(pos, move);
    if (promotion) {
        return PromotionBonus + score + weighted_see;
    }
    if (capture) {
        if (see_score >= 0) {
            return GoodCaptureBonus + score + weighted_see;
        }
        return BadCaptureBonus + score + weighted_see;
    }
    if (gives_check) {
        score += CheckBonus;
    }

    const int killer_score = killer_table_.score(ply, move);
    if (killer_score == 2) {
        score += Killer1Bonus;
    } else if (killer_score == 1) {
        score += Killer2Bonus;
    }

    return score;
}

HeuristicSearcherV15::ScoredMove HeuristicSearcherV15::make_scored_move(
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
            move_order_score(
                move,
                pos,
                ply,
                tt_move,
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
        move_order_score(
            move,
            pos,
            ply,
            tt_move,
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

bool HeuristicSearcherV15::better_scored_move(
    const ScoredMove& lhs,
    const ScoredMove& rhs
) const {
    return lhs.order_score > rhs.order_score;
}

std::vector<HeuristicSearcherV15::ScoredMove> HeuristicSearcherV15::ordered_moves(
    const Position& pos,
    int ply,
    Move tt_move
) const {
    std::vector<ScoredMove> ordered;
    const std::vector<Move> moves = generate_legal_moves(pos);
    ordered.reserve(moves.size());

    for (Move move : moves) {
        ordered.push_back(make_scored_move(pos, move, ply, tt_move, ScoringMode::MainSearch));
    }

    std::stable_sort(ordered.begin(), ordered.end(), [this](const ScoredMove& lhs, const ScoredMove& rhs) {
        return better_scored_move(lhs, rhs);
    });

    return ordered;
}

std::vector<HeuristicSearcherV15::ScoredMove> HeuristicSearcherV15::ordered_noisy_moves(
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

HeuristicSearcherV15::SearchValue HeuristicSearcherV15::quiescence(
    Position pos,
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

    const std::vector<Move> legal_moves = generate_legal_moves(pos);
    if (legal_moves.empty()) {
        return SearchValue{exact_range(in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0)};
    }

    if (in_check(pos, pos.side_to_move)) {
        if (q_depth >= MaxQuiescenceDepth) {
            return SearchValue{exact_range(evaluate_for_side_to_move(pos))};
        }

        ScoreRange node_range;
        node_range.lower = -Infinity;
        node_range.upper = -Infinity;
        std::vector<ScoredMove> moves;
        moves.reserve(legal_moves.size());
        for (Move move : legal_moves) {
            moves.push_back(make_scored_move(pos, move, ply, Move{}, ScoringMode::Quiescence));
        }
        std::stable_sort(moves.begin(), moves.end(), [this](const ScoredMove& lhs, const ScoredMove& rhs) {
            return better_scored_move(lhs, rhs);
        });

        for (const ScoredMove& scored_move : moves) {
            Position next = pos;
            next.make_move(scored_move.move);

            SearchValue child = quiescence(next, -beta, -alpha, ply + 1, q_depth + 1, state);
            const ScoreRange move_range = negate_range(child.range);
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

    int best_score = evaluate_for_side_to_move(pos);
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
    const std::vector<ScoredMove> moves = ordered_noisy_moves(pos);
    for (const ScoredMove& scored_move : moves) {
        Position next = pos;
        next.make_move(scored_move.move);

        SearchValue child = quiescence(next, -beta, -alpha, ply + 1, q_depth + 1, state);
        const ScoreRange move_range = negate_range(child.range);
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

bool HeuristicSearcherV15::is_quiet_move(const ScoredMove& scored_move) const {
    return !scored_move.capture
        && !scored_move.promotion
        && !scored_move.gives_check;
}

HeuristicSearcherV15::SearchValue HeuristicSearcherV15::negamax(
    Position pos,
    int depth,
    int ply,
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

    const std::vector<ScoredMove> moves = ordered_moves(pos, ply, preferred_tt_move(tt_probe.moves));

    if (moves.empty()) {
        return SearchValue{exact_range(in_check(pos, pos.side_to_move) ? -CheckmateScore + ply : 0)};
    }

    ScoreRange node_range;
    node_range.lower = -Infinity;
    node_range.upper = -Infinity;
    Move best_lower_move{};
    Move best_upper_move{};

    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];

        Position next = pos;
        next.make_move(scored_move.move);

        ScoreRange move_range;
        if (move_index > 0 && beta > alpha + 1) {
            SearchValue scout = negamax(next, depth - 1, ply + 1, -alpha - 1, -alpha, state);
            move_range = negate_range(scout.range);
            if (state.stopped) {
                return SearchValue{exact_range(0)};
            }

            const bool scout_proves_fail_low = move_range.upper <= alpha;
            const bool scout_proves_fail_high = move_range.lower >= beta;
            if (!scout_proves_fail_low && !scout_proves_fail_high) {
                SearchValue full = negamax(next, depth - 1, ply + 1, -beta, -alpha, state);
                move_range = negate_range(full.range);
            }
        } else {
            SearchValue child = negamax(next, depth - 1, ply + 1, -beta, -alpha, state);
            move_range = negate_range(child.range);
        }
        if (state.stopped) {
            return SearchValue{exact_range(0)};
        }

        if (move_range.lower > node_range.lower) {
            node_range.lower = move_range.lower;
            best_lower_move = scored_move.move;
        }
        if (move_range.upper > node_range.upper) {
            node_range.upper = move_range.upper;
            best_upper_move = scored_move.move;
        }

        if (move_range.lower != -Infinity && move_range.lower > alpha) {
            alpha = move_range.lower;
        }
        if (node_range.lower >= beta) {
            if (is_quiet_move(scored_move)) {
                killer_table_.store(ply, scored_move.move);
                history_table_.store(pos, scored_move.move, depth);
            }
            node_range.upper = move_index == moves.size() - 1 ? node_range.upper : Infinity;
            break;
        }
    }
    assert(node_range.lower <= node_range.upper);
    if (tt_probe.has_score) {
        node_range = intersect_ranges(node_range, tt_probe.range);
    }
    assert(node_range.lower <= node_range.upper);

    if (should_store_tt(node_range)) {
        ScoreRange tt_range = node_range;
        if (tt_range.lower != -Infinity) {
            tt_range.lower = score_to_tt(tt_range.lower, ply);
        }
        if (tt_range.upper != Infinity) {
            tt_range.upper = score_to_tt(tt_range.upper, ply);
        }
        tt_.store(pos.zobrist_key, depth, tt_range, MoveRange{best_lower_move, best_upper_move});
    }

    return SearchValue{node_range};
}

SearchResult HeuristicSearcherV15::make_fallback_result(const Position& pos) const {
    SearchResult result;
    const std::vector<ScoredMove> moves = ordered_moves(pos, 0);
    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        return result;
    }
    result.best_move = moves.front().move;
    result.score = evaluate_for_side_to_move(pos);
    result.nodes = 1;
    return result;
}

SearchResult HeuristicSearcherV15::search_fixed_depth(
    const Position& pos,
    int depth,
    SearchState& state,
    int alpha,
    int beta,
    bool allow_root_tt_probe
) {
    assert(depth >= 0);

    SearchResult result;
    result.depth = depth;

    if (depth == 0) {
        result.score = representative_score(quiescence(pos, -Infinity, Infinity, 0, 0, state).range);
        result.nodes = state.nodes;
        state.root_status = WindowStatus::Inside;
        return result;
    }

    const int alpha_original = alpha;
    const int beta_original = beta;
    TTProbeResult tt_probe = probe_tt(pos.zobrist_key, depth, alpha, beta, 0, allow_root_tt_probe);
    if (tt_probe.hit) {
        state.root_status = status_for_window(tt_probe.range, alpha_original, beta_original);
        result.score = representative_score_for_status(tt_probe.range, state.root_status);
        result.best_move = preferred_tt_move(tt_probe.moves);
        result.nodes = 1;
        return result;
    }

    const std::vector<ScoredMove> moves = ordered_moves(pos, 0, preferred_tt_move(tt_probe.moves));

    if (moves.empty()) {
        result.score = in_check(pos, pos.side_to_move) ? -CheckmateScore : 0;
        result.nodes = 1;
        state.root_status = WindowStatus::Inside;
        return result;
    }

    ScoreRange root_range;
    root_range.lower = -Infinity;
    root_range.upper = -Infinity;
    Move best_lower_move{};
    Move best_upper_move{};

    for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
        const ScoredMove& scored_move = moves[move_index];

        Position next = pos;
        next.make_move(scored_move.move);

        SearchValue child = negamax(next, depth - 1, 1, -beta, -alpha, state);
        const ScoreRange move_range = negate_range(child.range);
        if (state.stopped) {
            result.stopped = true;
            result.nodes = state.nodes;
            return result;
        }

        if (move_range.lower > root_range.lower) {
            root_range.lower = move_range.lower;
            best_lower_move = scored_move.move;
        }
        if (move_range.upper > root_range.upper) {
            root_range.upper = move_range.upper;
            best_upper_move = scored_move.move;
        }

        if (move_range.lower != -Infinity && move_range.lower > alpha) {
            alpha = move_range.lower;
        }
        if (root_range.lower >= beta) {
            root_range.upper = move_index == moves.size() - 1 ? root_range.upper : Infinity;
            break;
        }
    }

    if (root_range.lower != -Infinity && root_range.upper <= root_range.lower) {
        root_range.upper = root_range.lower;
    }
    if (tt_probe.has_score) {
        root_range = intersect_ranges(root_range, tt_probe.range);
    }
    assert(root_range.lower <= root_range.upper);

    state.root_status = status_for_window(root_range, alpha_original, beta_original);
    result.score = representative_score_for_status(root_range, state.root_status);
    result.best_move = root_range.lower != -Infinity ? best_lower_move : best_upper_move;

    if (should_store_tt(root_range)) {
        ScoreRange tt_range = root_range;
        if (tt_range.lower != -Infinity) {
            tt_range.lower = score_to_tt(tt_range.lower, 0);
        }
        if (tt_range.upper != Infinity) {
            tt_range.upper = score_to_tt(tt_range.upper, 0);
        }
        tt_.store(pos.zobrist_key, depth, tt_range, MoveRange{best_lower_move, best_upper_move});
    }

    result.nodes = state.nodes;
    return result;
}

SearchResult HeuristicSearcherV15::search_root_without_tt_probe(
    const Position& pos,
    int depth,
    SearchState& state
) {
    return search_fixed_depth(pos, depth, state, -Infinity, Infinity, false);
}

HeuristicSearcherV15::HeuristicSearcherV15(std::size_t tt_mb)
    : tt_(tt_mb) {
}

SearchResult HeuristicSearcherV15::search_best_move(const Position& pos, int depth) {
    killer_table_.clear();
    SearchState state;
    return search_fixed_depth(pos, depth, state);
}

SearchResult HeuristicSearcherV15::search_best_move(const Position& pos, const SearchLimits& limits) {
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
            SearchResult current = search_fixed_depth(pos, depth, state);
            if (current.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return best;
            }
            best = current;
        } else if (EnableAspirationWindow) {
            int alpha = best.score - aspiration_window_cp;
            int beta = best.score + aspiration_window_cp;
            SearchResult current;
            for (;;) {
                current = search_fixed_depth(pos, depth, state, alpha, beta);
                if (current.stopped || state.stopped) {
                    best.stopped = true;
                    best.nodes = state.nodes;
                    return best;
                }
                if (state.root_status == WindowStatus::Inside || (alpha == -Infinity && beta == Infinity)) {
                    break;
                }
                if (state.root_status == WindowStatus::Below) {
                    beta = std::min(beta, current.score);
                    alpha = -Infinity;
                } else {
                    alpha = std::max(alpha, current.score);
                    beta = Infinity;
                }
            }
            best = current;
        } else {
            SearchResult current = search_fixed_depth(pos, depth, state);
            if (current.stopped || state.stopped) {
                best.stopped = true;
                best.nodes = state.nodes;
                return best;
            }
            best = current;
        }
    }

    best.nodes = state.nodes;
    return best;
}

std::string_view HeuristicSearcherV15::name() const {
    return "heuristic_v15";
}

void HeuristicSearcherV15::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherV15::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherV15::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& HeuristicSearcherV15::tt_stats() const {
    return tt_.stats();
}

} // namespace chess
