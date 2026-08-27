#include "phase_quantized_nnue.hpp"
#include "phase_quantized_nnue_accumulator_backend.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>

namespace {

using Accumulator = std::array<
    std::int32_t,
    chess::PhaseQuantizedNnueModel::PerspectiveAccumulatorSize>;
using Psqt = std::array<
    std::int32_t,
    chess::PhaseQuantizedNnueModel::PsqtBucketCount>;

struct Row {
    std::array<
        std::int8_t,
        chess::PhaseQuantizedNnueModel::PerspectiveAccumulatorSize> positional;
    Psqt psqt;
};

chess::phase_nnue_detail::AccumulatorRowView view(const Row& row) {
    return {row.positional.data(), row.psqt.data()};
}

void fill_row(Row& row, std::mt19937& random) {
    std::uniform_int_distribution<int> positional(-127, 127);
    std::uniform_int_distribution<std::int32_t> psqt(-4096, 4096);
    for (std::int8_t& value : row.positional) {
        value = static_cast<std::int8_t>(positional(random));
    }
    for (std::int32_t& value : row.psqt) {
        value = psqt(random);
    }
}

template<std::size_t AddedCount, std::size_t RemovedCount>
void update_reference(
    Accumulator& accumulator,
    Psqt& psqt,
    const std::array<const Row*, AddedCount>& added,
    const std::array<const Row*, RemovedCount>& removed
) {
    for (std::size_t lane = 0; lane < accumulator.size(); ++lane) {
        for (const Row* row : added) {
            accumulator[lane] += row->positional[lane];
        }
        for (const Row* row : removed) {
            accumulator[lane] -= row->positional[lane];
        }
    }
    for (std::size_t bucket = 0; bucket < psqt.size(); ++bucket) {
        for (const Row* row : added) {
            psqt[bucket] += row->psqt[bucket];
        }
        for (const Row* row : removed) {
            psqt[bucket] -= row->psqt[bucket];
        }
    }
}

void test_incremental_updates() {
    std::mt19937 random(0x41acc212U);
    std::uniform_int_distribution<std::int32_t> initial(-100000, 100000);

    for (std::size_t iteration = 0; iteration < 4096; ++iteration) {
        std::array<Row, 4> rows;
        for (Row& row : rows) {
            fill_row(row, random);
        }
        Accumulator initial_accumulator;
        Psqt initial_psqt;
        for (std::int32_t& value : initial_accumulator) {
            value = initial(random);
        }
        for (std::int32_t& value : initial_psqt) {
            value = initial(random);
        }

        {
            Accumulator expected = initial_accumulator;
            Psqt expected_psqt = initial_psqt;
            update_reference<1, 1>(
                expected, expected_psqt, {&rows[0]}, {&rows[1]});
            Accumulator actual = initial_accumulator;
            Psqt actual_psqt = initial_psqt;
            chess::phase_nnue_detail::update_accumulator_avx2_1_1(
                actual.data(), actual_psqt.data(), view(rows[0]), view(rows[1]));
            assert(actual == expected);
            assert(actual_psqt == expected_psqt);
        }
        {
            Accumulator expected = initial_accumulator;
            Psqt expected_psqt = initial_psqt;
            update_reference<1, 2>(
                expected,
                expected_psqt,
                {&rows[0]},
                {&rows[1], &rows[2]});
            Accumulator actual = initial_accumulator;
            Psqt actual_psqt = initial_psqt;
            chess::phase_nnue_detail::update_accumulator_avx2_1_2(
                actual.data(),
                actual_psqt.data(),
                view(rows[0]),
                view(rows[1]),
                view(rows[2]));
            assert(actual == expected);
            assert(actual_psqt == expected_psqt);
        }
        {
            Accumulator expected = initial_accumulator;
            Psqt expected_psqt = initial_psqt;
            update_reference<2, 2>(
                expected,
                expected_psqt,
                {&rows[0], &rows[1]},
                {&rows[2], &rows[3]});
            Accumulator actual = initial_accumulator;
            Psqt actual_psqt = initial_psqt;
            chess::phase_nnue_detail::update_accumulator_avx2_2_2(
                actual.data(),
                actual_psqt.data(),
                view(rows[0]),
                view(rows[1]),
                view(rows[2]),
                view(rows[3]));
            assert(actual == expected);
            assert(actual_psqt == expected_psqt);
        }
    }
}

void test_rebuild() {
    std::mt19937 random(0x41acc2ebU);
    std::uniform_int_distribution<std::int32_t> bias(-100000, 100000);

    for (std::size_t iteration = 0; iteration < 512; ++iteration) {
        std::array<Row, 64> rows;
        std::array<chess::phase_nnue_detail::AccumulatorRowView, 64> views;
        for (std::size_t index = 0; index < rows.size(); ++index) {
            fill_row(rows[index], random);
            views[index] = view(rows[index]);
        }
        Accumulator accumulator_bias;
        for (std::int32_t& value : accumulator_bias) {
            value = bias(random);
        }
        const std::size_t active_count = iteration % (rows.size() + 1);

        Accumulator expected = accumulator_bias;
        Psqt expected_psqt{};
        for (std::size_t index = 0; index < active_count; ++index) {
            for (std::size_t lane = 0; lane < expected.size(); ++lane) {
                expected[lane] += rows[index].positional[lane];
            }
            for (std::size_t bucket = 0;
                 bucket < expected_psqt.size();
                 ++bucket) {
                expected_psqt[bucket] += rows[index].psqt[bucket];
            }
        }

        Accumulator actual;
        Psqt actual_psqt;
        chess::phase_nnue_detail::rebuild_accumulator_avx2(
            actual.data(),
            actual_psqt.data(),
            accumulator_bias.data(),
            views.data(),
            active_count);
        assert(actual == expected);
        assert(actual_psqt == expected_psqt);
    }
}

} // namespace

int main() {
    test_incremental_updates();
    test_rebuild();
    std::cout << "phase NNUE AVX2 accumulator parity passed\n";
}
