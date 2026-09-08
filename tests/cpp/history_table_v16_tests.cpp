#include "counter_history_table.hpp"
#include "history_table_v16.hpp"
#include "move.hpp"
#include "position.hpp"

#include <cassert>

namespace {

void test_history_generalizes_same_piece_to_square() {
    chess::Position pos;
    assert(pos.set_fen("8/8/8/8/8/8/8/1N3N1K w - - 0 1"));

    const chess::Square b1 = chess::make_square(1, 0);
    const chess::Square d2 = chess::make_square(3, 1);
    const chess::Square f1 = chess::make_square(5, 0);
    const chess::Move left_knight = chess::make_move(b1, d2);
    const chess::Move right_knight = chess::make_move(f1, d2);

    chess::HistoryTableV16 table;
    assert(table.get_score(pos, left_knight) == 0);
    assert(table.get_score(pos, right_knight) == 0);

    table.store(pos, left_knight, 4);

    assert(table.get_score(pos, left_knight) > 0);
    assert(table.get_score(pos, right_knight) == table.get_score(pos, left_knight));
}

void test_gravity_update_is_bounded() {
    chess::Position pos;
    assert(pos.set_fen("8/8/8/8/8/8/8/1N5K w - - 0 1"));

    const chess::Move move = chess::make_move(chess::make_square(1, 0), chess::make_square(3, 1));
    chess::HistoryTableV16 table;

    table.store(pos, move, 8);
    const int first_score = table.get_score(pos, move);
    assert(first_score > 0);

    for (int i = 0; i < 1000; ++i) {
        table.store(pos, move, 8);
    }

    const int saturated_score = table.get_score(pos, move);
    assert(saturated_score > first_score);
    assert(saturated_score <= chess::HistoryTableV16::MaxScore);
}

void test_penalty_is_bounded() {
    chess::Position pos;
    assert(pos.set_fen("8/8/8/8/8/8/8/1N5K w - - 0 1"));

    const chess::Move move = chess::make_move(chess::make_square(1, 0), chess::make_square(3, 1));
    chess::HistoryTableV16 table;

    table.penalize(pos, move, 8);
    const int first_score = table.get_score(pos, move);
    assert(first_score < 0);

    for (int i = 0; i < 1000; ++i) {
        table.penalize(pos, move, 8);
    }

    const int saturated_score = table.get_score(pos, move);
    assert(saturated_score < first_score);
    assert(saturated_score >= -chess::HistoryTableV16::MaxScore);
}

void test_penalty_divisor_reduces_malus() {
    chess::Position pos;
    assert(pos.set_fen("8/8/8/8/8/8/8/1N5K w - - 0 1"));

    const chess::Move move = chess::make_move(chess::make_square(1, 0), chess::make_square(3, 1));
    chess::HistoryTableV16 full_penalty(1);
    chess::HistoryTableV16 half_penalty(2);

    full_penalty.penalize(pos, move, 8);
    half_penalty.penalize(pos, move, 8);

    assert(full_penalty.get_score(pos, move) < half_penalty.get_score(pos, move));
}

void test_bonus_can_recover_penalty() {
    chess::Position pos;
    assert(pos.set_fen("8/8/8/8/8/8/8/1N5K w - - 0 1"));

    const chess::Move move = chess::make_move(chess::make_square(1, 0), chess::make_square(3, 1));
    chess::HistoryTableV16 table;

    table.penalize(pos, move, 8);
    const int penalty_score = table.get_score(pos, move);
    table.store(pos, move, 8);

    assert(table.get_score(pos, move) > penalty_score);
}

void test_reset_clears_scores() {
    chess::Position pos;
    assert(pos.set_fen("8/8/8/8/8/8/8/1N5K w - - 0 1"));

    const chess::Move move = chess::make_move(chess::make_square(1, 0), chess::make_square(3, 1));
    chess::HistoryTableV16 table;
    table.store(pos, move, 3);
    assert(table.get_score(pos, move) > 0);

    table.reset();
    assert(table.get_score(pos, move) == 0);
}

void test_counter_history_reset_clears_scores() {
    const chess::Move previous = chess::make_move(
        chess::make_square(1, 0), chess::make_square(2, 2));
    const chess::Move counter = chess::make_move(
        chess::make_square(6, 7), chess::make_square(5, 5));
    chess::CounterHistoryTable table;
    table.store(
        chess::Color::White,
        chess::PieceType::Knight,
        previous,
        chess::PieceType::Knight,
        counter,
        4);
    assert(table.get_score(
        chess::Color::White,
        chess::PieceType::Knight,
        previous,
        chess::PieceType::Knight,
        counter) > 0);

    table.reset();
    assert(table.get_score(
        chess::Color::White,
        chess::PieceType::Knight,
        previous,
        chess::PieceType::Knight,
        counter) == 0);
}

} // namespace

int main() {
    test_history_generalizes_same_piece_to_square();
    test_gravity_update_is_bounded();
    test_penalty_is_bounded();
    test_penalty_divisor_reduces_malus();
    test_bonus_can_recover_penalty();
    test_reset_clears_scores();
    test_counter_history_reset_clears_scores();
}
