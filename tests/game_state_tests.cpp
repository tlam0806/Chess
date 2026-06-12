#include "game_state.hpp"

#include "move.hpp"

#include <cassert>
#include <vector>

using namespace chess;

namespace {

Move find_move(const Position& pos, const char* uci) {
    for (Move move : generate_legal_moves(pos)) {
        if (move_to_string(move) == uci) {
            return move;
        }
    }
    assert(false);
    return Move{};
}

void play(Position& pos, std::vector<HashKey>& hashes, const char* uci) {
    pos.make_move(find_move(pos, uci));
    hashes.push_back(pos.zobrist_key);
}

void play(Position& pos, std::vector<Position>& history, const char* uci) {
    history.push_back(pos);
    pos.make_move(find_move(pos, uci));
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_startpos();
        std::vector<HashKey> hashes{pos.zobrist_key};

        assert(repetition_count(pos.zobrist_key, hashes) == 1);
        assert(!is_threefold_repetition(pos.zobrist_key, hashes));

        play(pos, hashes, "g1f3");
        play(pos, hashes, "g8f6");
        play(pos, hashes, "f3g1");
        play(pos, hashes, "f6g8");

        assert(repetition_count(pos.zobrist_key, hashes) == 2);
        assert(!is_threefold_repetition(pos.zobrist_key, hashes));

        play(pos, hashes, "g1f3");
        play(pos, hashes, "g8f6");
        play(pos, hashes, "f3g1");
        play(pos, hashes, "f6g8");

        assert(repetition_count(pos.zobrist_key, hashes) == 3);
        assert(is_threefold_repetition(pos.zobrist_key, hashes));
    }

    {
        Position pos;
        pos.set_startpos();
        std::vector<Position> history;

        play(pos, history, "g1f3");
        play(pos, history, "g8f6");
        play(pos, history, "f3g1");
        play(pos, history, "f6g8");
        assert(repetition_count(pos, history) == 2);
        assert(!is_threefold_repetition(pos, history));

        play(pos, history, "g1f3");
        play(pos, history, "g8f6");
        play(pos, history, "f3g1");
        play(pos, history, "f6g8");
        assert(repetition_count(pos, history) == 3);
        assert(is_threefold_repetition(pos, history));
    }
}
