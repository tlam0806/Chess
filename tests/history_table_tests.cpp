#include "history_table.hpp"
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

    chess::HistoryTable table;
    assert(table.get_score(pos, left_knight) == 0);
    assert(table.get_score(pos, right_knight) == 0);

    table.store(pos, left_knight, 4);

    assert(table.get_score(pos, left_knight) > 0);
    assert(table.get_score(pos, right_knight) == table.get_score(pos, left_knight));
}

void test_reset_clears_scores() {
    chess::Position pos;
    assert(pos.set_fen("8/8/8/8/8/8/8/1N5K w - - 0 1"));

    const chess::Move move = chess::make_move(chess::make_square(1, 0), chess::make_square(3, 1));
    chess::HistoryTable table;
    table.store(pos, move, 3);
    assert(table.get_score(pos, move) > 0);

    table.reset();
    assert(table.get_score(pos, move) == 0);
}

} // namespace

int main() {
    test_history_generalizes_same_piece_to_square();
    test_reset_clears_scores();
}
