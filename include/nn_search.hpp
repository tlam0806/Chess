#pragma once

#include "nn_value.hpp"
#include "position.hpp"
#include "search.hpp"

namespace chess {

int negamax_nn(Position pos, int depth, const NnValueModel& model);
int negamax_nn(Position pos, int depth, int alpha, int beta, const NnValueModel& model);
SearchResult search_best_move_nn(Position pos, int depth, const NnValueModel& model);

} // namespace chess
