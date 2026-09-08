#include "position.hpp"

#include <cassert>
#include <sstream>
#include <string>

using namespace chess;

int main() {
    Position pos;

    assert(pos.set_fen("4k2r/8/8/8/8/8/P7/4K3 w - - 0 1"));

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
