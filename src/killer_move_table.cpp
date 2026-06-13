#include "killer_move_table.hpp"

namespace chess {

void KillerMoveTable::clear() {
    for (auto& row : killers_) {
        row = {};
    }
}

void KillerMoveTable::store(int ply, Move move) {
    if (ply < 0 || ply >= MaxPly || move.value == 0) {
        return;
    }

    auto& row = killers_[ply];
    if (row[0] == move) {
        return;
    }

    row[1] = row[0];
    row[0] = move;
}

int KillerMoveTable::score(int ply, Move move) const {
    if (ply < 0 || ply >= MaxPly || move.value == 0) {
        return 0;
    }

    const auto& row = killers_[ply];
    if (row[0] == move) {
        return 2;
    }
    if (row[1] == move) {
        return 1;
    }
    return 0;
}

} // namespace chess
