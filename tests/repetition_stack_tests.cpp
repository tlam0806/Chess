#include "repetition_stack.hpp"

#include <array>
#include <cassert>
#include <iostream>

int main() {
    constexpr chess::HashKey A = 0xA1;
    constexpr chess::HashKey B = 0xB2;
    constexpr chess::HashKey C = 0xC3;
    constexpr chess::HashKey D = 0xD4;
    constexpr chess::HashKey E = 0xE5;
    constexpr chess::HashKey F = 0xF6;
    constexpr chess::HashKey G = 0x17;

    chess::RepetitionStack stack;

    const std::array<chess::HashKey, 9> threefold{
        A, B, C, D, A, E, F, G, A};
    stack.reset(threefold, A, 8);
    assert(stack.current_is_threefold());
    assert(stack.has_twofold_position());

    const std::array<chess::HashKey, 8> before_third{
        A, B, C, D, A, E, F, G};
    stack.reset(before_third, G, 7);
    assert(!stack.current_is_threefold());
    assert(stack.has_twofold_position());
    stack.push(A, false, 8);
    assert(stack.current_is_threefold());
    stack.pop();
    assert(!stack.current_is_threefold());
    assert(stack.has_twofold_position());

    // An irreversible move starts a new repetition segment, even when its
    // resulting hash happens to equal an older synthetic test hash.
    stack.push(A, true, 0);
    assert(!stack.current_is_threefold());
    assert(!stack.has_twofold_position());
    stack.pop();
    assert(stack.has_twofold_position());

    // Losing castling rights is irreversible for repetition but does not
    // reset halfmove_clock. Descendants must still not scan past that marker.
    stack.push(0x1818, true, 8);
    stack.push(B, false, 9);
    stack.push(C, false, 10);
    stack.push(D, false, 11);
    stack.push(A, false, 12);
    assert(!stack.current_is_threefold());
    assert(!stack.has_twofold_position());
    stack.pop();
    stack.pop();
    stack.pop();
    stack.pop();
    stack.pop();
    assert(stack.has_twofold_position());

    // halfmove_clock=0 retains only the current position from supplied game
    // history, so positions before a pawn move/capture cannot leak through.
    stack.reset(threefold, A, 0);
    assert(stack.size() == 1);
    assert(!stack.current_is_threefold());
    assert(!stack.has_twofold_position());

    std::cout << "repetition stack checks passed\n";
}
