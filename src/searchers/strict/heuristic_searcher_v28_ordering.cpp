#include "heuristic_searcher_v28.hpp"

#include "evaluate.hpp"
#include "heuristic_searcher_v28_detail.hpp"
#include "legal_noisy_generator.hpp"
#include "legal_non_capture_generator.hpp"

namespace chess {

int HeuristicSearcherV28::qsearch_order_score(
    bool capture,
    bool promotion,
    PieceType captured_piece,
    int see_score
) const {
    const MoveOrderingWeights& weights = move_ordering_weights_;
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

bool HeuristicSearcherV28::is_quiet_move(const ScoredMove& scored_move) const {
    return !scored_move.capture && !scored_move.promotion && !scored_move.gives_check;
}

int HeuristicSearcherV28::main_order_score(
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

HeuristicSearcherV28::ScoredMove HeuristicSearcherV28::make_scored_move(
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

HeuristicSearcherV28::ScoredMove HeuristicSearcherV28::make_scored_move_by_attack_check(
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

HeuristicSearcherV28::ScoredMove HeuristicSearcherV28::make_scored_legal_move(
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
    if (capture) {
        see_score = static_exchange_eval(pos, move, moved_piece, captured_piece);
    }

    if (stage == ScoringMode::Quiescence) {
        return ScoredMove{
            qsearch_order_score(
                capture,
                promotion,
                captured_piece,
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

bool HeuristicSearcherV28::better_scored_move(
    const ScoredMove& lhs,
    const ScoredMove& rhs
) const {
    return lhs.order_score > rhs.order_score;
}

void HeuristicSearcherV28::sort_scored_moves(ScoredMoveList& moves) const {
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

HeuristicSearcherV28::ScoredMoveList HeuristicSearcherV28::ordered_moves(
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

Move HeuristicSearcherV28::valid_priority_killer(const Position& pos, int ply, int slot) const {
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

HeuristicSearcherV28::ScoredMoveList
HeuristicSearcherV28::generate_legal_noisy_scored_moves_for_searcher(
    const Position& pos,
    const KingSafetyContext& king_safety,
    int ply,
    MoveRange tt_moves,
    Move prev_move,
    PieceType prev_moved_piece,
    Move skip_tt_move
) {
    ScoredMoveList ordered;
    auto score_noisy_move = [&](Move move, PieceType moved_piece, PieceType captured_piece) {
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
        const bool deferred_bad_capture =
            scored_move.capture
            && !scored_move.promotion
            && scored_move.see_score < move_ordering_weights_.bad_capture_stage_threshold;
        if (deferred_bad_capture) {
            return;
        }

        ordered.push_back(scored_move);
    };
    {
        generate_legal_noisy_moves_with_info(pos, king_safety, score_noisy_move);
    }

    return ordered;
}

HeuristicSearcherV28::ScoredMoveList HeuristicSearcherV28::ordered_moves_for_stage(
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
    auto add_scored_move = [&](Move move) {
        if (!is_valid_move(move)) {
            return;
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
            skip_tt_move);
        add_killer_move(skip_killer1, skip_killer2);
        add_killer_move(skip_killer2, skip_killer1);
    } else {
        auto score_quiet_move = [&](Move move, PieceType moved_piece) {
            if (matches_any(move, skip_tt_move, skip_killer1, skip_killer2)) {
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
        MoveList bad_captures;
        {
            generate_pseudo_noisy_moves(pos, bad_captures);
        }
        for (Move move : bad_captures) {
            if (is_capture(move)
                && !is_promotion(move)
                && !matches_any(move, skip_tt_move, skip_killer1, skip_killer2)) {
                add_scored_move(move);
            }
        }
    }

    sort_scored_moves(ordered);
    return ordered;
}

} // namespace chess
