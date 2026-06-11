#pragma once

#include "position.hpp"

namespace chess {

int evaluate(const Position& pos);
int evaluate_for_side_to_move(const Position& pos);

} // namespace chess
