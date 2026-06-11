#include "position.hpp"

#include <cassert>
#include <sstream>
#include <string>

using namespace chess;

int main() {
    Position pos;

    pos.pieces[static_cast<int>(Color::White)][static_cast<int>(PieceType::King)] =
        bit(make_square(4, 0));
    pos.pieces[static_cast<int>(Color::White)][static_cast<int>(PieceType::Pawn)] =
        bit(make_square(0, 1));
    pos.pieces[static_cast<int>(Color::Black)][static_cast<int>(PieceType::King)] =
        bit(make_square(4, 7));
    pos.pieces[static_cast<int>(Color::Black)][static_cast<int>(PieceType::Rook)] =
        bit(make_square(7, 7));

    std::ostringstream os;
    pos.print(os);

    const std::string expected =
        ". . . . k . . r\n"
        ". . . . . . . .\n"
        ". . . . . . . .\n"
        ". . . . . . . .\n"
        ". . . . . . . .\n"
        ". . . . . . . .\n"
        "P . . . . . . .\n"
        ". . . . K . . .\n";

    assert(os.str() == expected);
}
