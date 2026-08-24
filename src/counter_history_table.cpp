#include "counter_history_table.hpp"

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
        -CounterHistoryTable::MaxScore,
        CounterHistoryTable::MaxScore
    );
    const int gravity = static_cast<int>(
        static_cast<long long>(current) * std::abs(clamped_bonus)
            / CounterHistoryTable::MaxScore
    );
    return current + clamped_bonus - gravity;
}

} // namespace

CounterHistoryTable::CounterHistoryTable(int penalty_divisor_numerator, int penalty_divisor_denominator)
    : penalty_divisor_numerator_(penalty_divisor_numerator),
      penalty_divisor_denominator_(penalty_divisor_denominator) {
    assert(penalty_divisor_numerator_ > 0);
    assert(penalty_divisor_denominator_ > 0);
}

void CounterHistoryTable::reset() {
    scores.fill(0);
}

std::size_t CounterHistoryTable::get_index(
    Color color,
    PieceType previous_piece,
    Move previous_move,
    PieceType counter_piece,
    Move counter_move
) const {
    assert(previous_piece != PieceType::None);
    assert(counter_piece != PieceType::None);

    std::size_t index = static_cast<std::size_t>(color);
    index = index * PieceTypeEncoder + static_cast<std::size_t>(previous_piece);
    index = index * SquareEncoder + static_cast<std::size_t>(previous_move.to());
    index = index * PieceTypeEncoder + static_cast<std::size_t>(counter_piece);
    index = index * SquareEncoder + static_cast<std::size_t>(counter_move.to());
    return index;
}

void CounterHistoryTable::store(
    Color color,
    PieceType previous_piece,
    Move previous_move,
    PieceType counter_piece,
    Move counter_move,
    int depth
) {
    Score& score = scores[get_index(color, previous_piece, previous_move, counter_piece, counter_move)];
    score = gravity_update(score, history_bonus(depth));
}

void CounterHistoryTable::penalize(
    Color color,
    PieceType previous_piece,
    Move previous_move,
    PieceType counter_piece,
    Move counter_move,
    int depth
) {
    Score& score = scores[get_index(color, previous_piece, previous_move, counter_piece, counter_move)];
    score = gravity_update(
        score,
        -history_penalty(depth, penalty_divisor_numerator_, penalty_divisor_denominator_)
    );
}

CounterHistoryTable::Score CounterHistoryTable::get_score(
    Color color,
    PieceType previous_piece,
    Move previous_move,
    PieceType counter_piece,
    Move counter_move
) const {
    return scores[get_index(color, previous_piece, previous_move, counter_piece, counter_move)];
}

} // namespace chess
