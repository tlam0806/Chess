#include "history_table_v16.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>

namespace chess {

namespace {

int history_bonus(int depth) {
    return depth * depth;
}

int history_penalty(int depth, int divisor_numerator, int divisor_denominator) {
    assert(divisor_numerator > 0);
    assert(divisor_denominator > 0);
    const int bonus = history_bonus(depth);
    return std::max(
        1,
        static_cast<int>(
            static_cast<long long>(bonus) * divisor_denominator / divisor_numerator
        )
    );
}

int gravity_update(int current, int bonus) {
    const int clamped_bonus = std::clamp(
        bonus,
        -HistoryTableV16::MaxScore,
        HistoryTableV16::MaxScore
    );
    const int gravity = static_cast<int>(
        static_cast<long long>(current) * std::abs(clamped_bonus)
            / HistoryTableV16::MaxScore
    );
    return current + clamped_bonus - gravity;
}

int piece_to_square_index(Color color, PieceType piece, Square square) {
    assert(piece != PieceType::None);
    return (static_cast<int>(color) * 6 * 64)
        + (static_cast<int>(piece) * 64)
        + square;
}

} // namespace

HistoryTableV16::HistoryTableV16(int penalty_divisor_numerator, int penalty_divisor_denominator)
    : penalty_divisor_numerator_(penalty_divisor_numerator),
      penalty_divisor_denominator_(penalty_divisor_denominator) {
    assert(penalty_divisor_numerator_ > 0);
    assert(penalty_divisor_denominator_ > 0);
}

void HistoryTableV16::reset() {
    piece_to_square_.fill(0);
}

void HistoryTableV16::store(const Position& pos, Move move, int depth) {
    const Square from = from_square(move);
    const PieceType piece = pos.piece_type_on_occupied(from);
    assert(piece != PieceType::None);

    Score& score = piece_to_square_[piece_to_square_index(pos.side_to_move, piece, to_square(move))];
    score = gravity_update(score, history_bonus(depth));
}

void HistoryTableV16::penalize(const Position& pos, Move move, int depth) {
    const Square from = from_square(move);
    const PieceType piece = pos.piece_type_on_occupied(from);
    assert(piece != PieceType::None);

    Score& score = piece_to_square_[piece_to_square_index(pos.side_to_move, piece, to_square(move))];
    score = gravity_update(
        score,
        -history_penalty(depth, penalty_divisor_numerator_, penalty_divisor_denominator_)
    );
}

HistoryTableV16::Score HistoryTableV16::get_score(const Position& pos, Move move) const {
    const Square from = from_square(move);
    const PieceType piece = pos.piece_type_on_occupied(from);
    assert(piece != PieceType::None);

    return piece_to_square_[piece_to_square_index(pos.side_to_move, piece, to_square(move))];
}

} // namespace chess
