#include "heuristic_searcher_fast_v32.hpp"

namespace chess {

HeuristicSearcherFastV32::HeuristicSearcherFastV32(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus
)
    : HeuristicSearcherFastV32(
        tt_mb,
        bucket_size,
        history_penalty_divisor_numerator,
        history_penalty_divisor_denominator,
        counter_history_bonus,
        MoveOrderingWeights{}
    ) {
}

HeuristicSearcherFastV32::HeuristicSearcherFastV32(
    std::size_t tt_mb,
    std::size_t bucket_size,
    int history_penalty_divisor_numerator,
    int history_penalty_divisor_denominator,
    int counter_history_bonus,
    MoveOrderingWeights weights
)
    : tt_(tt_mb, bucket_size),
      history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      counter_history_table_(history_penalty_divisor_numerator, history_penalty_divisor_denominator),
      move_ordering_weights_(weights) {
    move_ordering_weights_.counter_history_bonus = counter_history_bonus;
}

std::string_view HeuristicSearcherFastV32::name() const {
    return "heuristic_fast_v32";
}

void HeuristicSearcherFastV32::clear_tt() {
    tt_.clear();
}

std::size_t HeuristicSearcherFastV32::tt_entry_count() const {
    return tt_.entry_count();
}

void HeuristicSearcherFastV32::clear_tt_stats() {
    tt_.clear_stats();
}

const RangeTranspositionTableStats& HeuristicSearcherFastV32::tt_stats() const {
    return tt_.stats();
}

void HeuristicSearcherFastV32::set_lmr_config(LmrConfig config) {
    lmr_config_ = config;
}

const HeuristicSearcherFastV32::LmrConfig& HeuristicSearcherFastV32::lmr_config() const {
    return lmr_config_;
}

#ifdef CHESS_PROFILE_TT_TIMING
void HeuristicSearcherFastV32::clear_tt_timing_stats() {
    tt_.clear_timing_stats();
}

const TTFunctionTimingStats& HeuristicSearcherFastV32::tt_timing_stats() const {
    return tt_.timing_stats();
}
#endif

#ifdef CHESS_PROFILE_TT_PATH_TIMING
void HeuristicSearcherFastV32::clear_tt_probe_path_timing_stats() {
    tt_.clear_probe_path_timing_stats();
}

const TTProbePathTimingStats& HeuristicSearcherFastV32::tt_probe_path_timing_stats() const {
    return tt_.probe_path_timing_stats();
}
#endif

} // namespace chess
