#include "perft.hpp"

#include "move.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace chess;

namespace {

std::uint64_t divide_nodes_for(const std::vector<PerftDivideEntry>& divide, Move move) {
    const auto it = std::find_if(divide.begin(), divide.end(), [move](const PerftDivideEntry& entry) {
        return entry.move == move;
    });
    assert(it != divide.end());
    return it->nodes;
}

std::uint64_t divide_total(const std::vector<PerftDivideEntry>& divide) {
    std::uint64_t total = 0;
    for (const PerftDivideEntry& entry : divide) {
        total += entry.nodes;
    }
    return total;
}

} // namespace

int main() {
    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.side_to_move = Color::White;

        assert(perft(pos, 0) == 1);
        assert(perft(pos, 1) == 5);
        assert(perft(pos, 2) == 25);
    }

    {
        Position pos;
        pos.set_startpos();

        assert(perft(pos, 1) == 20);
        assert(perft(pos, 2) == 400);
        assert(perft(pos, 3) == 8902);

        const std::vector<PerftDivideEntry> divide = perft_divide(pos, 2);
        assert(divide.size() == 20);
        assert(divide_total(divide) == perft(pos, 2));
        assert(divide_nodes_for(divide, make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush)) == 20);
        assert(divide_nodes_for(divide, make_move(make_square(6, 0), make_square(5, 2))) == 20);
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));

        assert(perft(pos, 1) == 48);
        assert(perft(pos, 2) == 2039);

        const std::vector<PerftDivideEntry> divide = perft_divide(pos, 2);
        assert(divide.size() == 48);
        assert(divide_total(divide) == perft(pos, 2));
    }

    {
        Position pos;
        assert(pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"));

        assert(perft(pos, 1) == 14);
        assert(perft(pos, 2) == 191);
        assert(perft(pos, 3) == 2812);
        assert(perft(pos, 4) == 43238);
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"));

        assert(perft(pos, 1) == 6);
        assert(perft(pos, 2) == 264);
        assert(perft(pos, 3) == 9467);
        assert(perft(pos, 4) == 422333);
    }

    {
        Position pos;
        assert(pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"));

        assert(perft(pos, 1) == 44);
        assert(perft(pos, 2) == 1486);
        assert(perft(pos, 3) == 62379);
    }

    {
        Position pos;
        assert(pos.set_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"));

        assert(perft(pos, 1) == 46);
        assert(perft(pos, 2) == 2079);
        assert(perft(pos, 3) == 89890);
    }
}
