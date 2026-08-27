#pragma once

#include <cstddef>
#include <cstdint>

namespace chess::phase_nnue_detail {

struct AccumulatorRowView {
    const std::int8_t* positional;
    const std::int32_t* psqt;
};

void update_accumulator_avx2_1_1(
    std::int32_t* accumulator,
    std::int32_t* psqt,
    AccumulatorRowView added,
    AccumulatorRowView removed
);

void update_accumulator_avx2_1_2(
    std::int32_t* accumulator,
    std::int32_t* psqt,
    AccumulatorRowView added,
    AccumulatorRowView removed0,
    AccumulatorRowView removed1
);

void update_accumulator_avx2_2_2(
    std::int32_t* accumulator,
    std::int32_t* psqt,
    AccumulatorRowView added0,
    AccumulatorRowView added1,
    AccumulatorRowView removed0,
    AccumulatorRowView removed1
);

void rebuild_accumulator_avx2(
    std::int32_t* accumulator,
    std::int32_t* psqt,
    const std::int32_t* accumulator_bias,
    const AccumulatorRowView* active_rows,
    std::size_t active_count
);

} // namespace chess::phase_nnue_detail
