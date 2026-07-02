#include "heuristic_searcher_v30.hpp"

#include "evaluate.hpp"
#include "heuristic_searcher_v30_detail.hpp"
#include "legal_noisy_generator.hpp"
#include "legal_non_capture_generator.hpp"

namespace chess {

 int HeuristicSearcherV30::qsearch_order_score(
    bool capture,
    bool promotion,
    PieceType moved_piece,
    PieceType captured_piece
) const {
    const MoveOrderingWeights& weights = move_ordering_weights_;
    int score = 0;
    if (promotion) {
        score += weights.qsearch_promotion_bonus;
    }
    if (capture) {
        score += weights.qsearch_captured_value_weight * ordering_piece_value(captured_piece);
        score += weights.qsearch_capture_metric_weight
            * (ordering_piece_value(captured_piece) * 10 - ordering_piece_value(moved_piece));
        score += weights.qsearch_good_capture_bonus;
    }
    return score;
}

bool HeuristicSearcherV30::is_quiet_move(const ScoredMove& scored_move) const {
    return !scored_move.capture && !scored_move.promotion && !scored_move.gives_check;
}

int HeuristicSearcherV30::main_order_score(
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
    int see_score
) const {
    const MoveOrderingWeights& weights = move_ordering_weights_;
    const int weighted_see = weights.see_weight * see_score;

    if (is_valid_move(tt_moves.lower) && move == tt_moves.lower) {
        return weights.tt_lower_bonus;
    }

    int score = 0;
    {
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
        killer_score = killer_table_.score(ply, move);
    }
    if (killer_score == 2) {
        score += weights.killer1_bonus;
    } else if (killer_score == 1) {
        score += weights.killer2_bonus;
    }

    return score;
}

HeuristicSearcherV30::ScoredMove HeuristicSearcherV30::make_scored_move(
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

HeuristicSearcherV30::ScoredMove HeuristicSearcherV30::make_tt_lower_scored_move(
    const Position& pos,
    Move move
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

    const bool promotion = is_promotion(move);
    const bool gives_check = !capture && !promotion
        ? gives_check_fast(pos, move, moved_piece, captured_piece)
        : false;

    return ScoredMove{
        0,
        0,
        move,
        moved_piece,
        captured_piece,
        gives_check,
        capture,
        promotion
    };
}

HeuristicSearcherV30::ScoredMove HeuristicSearcherV30::make_scored_legal_move(
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
    const bool capture = is_capture(move);
    bool gives_check = false;
    if (stage == ScoringMode::MainSearch) {
        gives_check = gives_check_fast(pos, move, moved_piece, captured_piece);
    }
    const bool promotion = is_promotion(move);
    int see_score = 0;
    if (capture && stage != ScoringMode::Quiescence) {
        see_score = static_exchange_eval(pos, move, moved_piece, captured_piece);
    }

    if (stage == ScoringMode::Quiescence) {
        return ScoredMove{
            qsearch_order_score(
                capture,
                promotion,
                moved_piece,
                captured_piece),
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
        main_order_score(
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
            see_score),
        see_score,
        move,
        moved_piece,
        captured_piece,
        gives_check,
        capture,
        promotion
    };
}

bool HeuristicSearcherV30::better_scored_move(
    const ScoredMove& lhs,
    const ScoredMove& rhs
) const {
    return lhs.order_score > rhs.order_score;
}

void HeuristicSearcherV30::sort_scored_moves(ScoredMoveList& moves) const {
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

HeuristicSearcherV30::ScoredMoveList HeuristicSearcherV30::ordered_moves(
    const Position& pos,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    Move skip_move
) const {
    ScoredMoveList ordered;
    const KingSafetyContext king_safety = current_king_safety_context(pos);
    MoveList moves;
    {
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

HeuristicSearcherV30::ScoredMoveList
HeuristicSearcherV30::generate_legal_promotion_scored_moves_for_searcher(
    const Position& pos,
    const KingSafetyContext& king_safety,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    Move skip_tt_move
) {
    ScoredMoveList ordered;
    auto score_promotion_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
        if (matches_any(move, skip_tt_move)) {
            return;
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
        assert(scored_move.promotion);
        ordered.push_back(scored_move);
    };
    {
        generate_legal_promotion_moves_with_info(pos, king_safety, score_promotion_move);
    }

    return ordered;
}

HeuristicSearcherV30::ScoredMoveList
HeuristicSearcherV30::generate_legal_capture_scored_moves_for_searcher(
    const Position& pos,
    const KingSafetyContext& king_safety,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    Move skip_tt_move,
    bool good_captures
) {
    ScoredMoveList ordered;
    auto score_capture_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
        if (matches_any(move, skip_tt_move)) {
            return;
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
        assert(scored_move.capture);
        assert(!scored_move.promotion);
        const bool is_good_capture =
            scored_move.see_score >= move_ordering_weights_.bad_capture_stage_threshold;
        if (is_good_capture != good_captures) {
            return;
        }

        ordered.push_back(scored_move);
    };
    {
        generate_legal_non_promotion_capture_moves_with_info(pos, king_safety, score_capture_move);
    }

    return ordered;
}

HeuristicSearcherV30::ScoredMoveList HeuristicSearcherV30::ordered_moves_for_stage(
    const Position& pos,
    const KingSafetyContext& king_safety,
    int ply,
    MoveRange tt_moves,
    Move prev_move,  
    PieceType prev_moved_piece,
    Move skip_tt_move,
    MoveGenerationStage generation_stage
) {
    ScoredMoveList ordered;

    if (generation_stage == MoveGenerationStage::Promotion) {
        ordered = generate_legal_promotion_scored_moves_for_searcher(
            pos,
            king_safety,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            skip_tt_move);
    } else if (generation_stage == MoveGenerationStage::GoodCapture) {
        ordered = generate_legal_capture_scored_moves_for_searcher(
            pos,
            king_safety,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            skip_tt_move,
            true);
    } else if (generation_stage == MoveGenerationStage::QuietNonPromotion) {
        auto score_quiet_move = [&](Move move, PieceType moved_piece) {
            if (matches_any(move, skip_tt_move)) {
                return;
            }
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
            ordered.push_back(scored_move);
        };
        {
            generate_legal_quiet_non_promotion_moves_with_info(
                pos,
                king_safety,
                score_quiet_move);
        }
    } else {
        assert(generation_stage == MoveGenerationStage::BadCapture);
        ordered = generate_legal_capture_scored_moves_for_searcher(
            pos,
            king_safety,
            ply,
            tt_moves,
            prev_move,
            prev_moved_piece,
            skip_tt_move,
            false);
    }

    sort_scored_moves(ordered);
    return ordered;
}

} // namespace chess
