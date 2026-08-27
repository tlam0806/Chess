#include "phase_quantized_nnue_accumulator_backend.hpp"

#include "phase_quantized_nnue.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace chess::phase_nnue_detail {

namespace {

constexpr std::size_t AccumulatorSize =
    PhaseQuantizedNnueModel::PerspectiveAccumulatorSize;
constexpr std::size_t PsqtSize =
    PhaseQuantizedNnueModel::PsqtBucketCount;
constexpr std::size_t Avx2Int32Lanes = sizeof(__m256i) / sizeof(std::int32_t);

static_assert(AccumulatorSize % Avx2Int32Lanes == 0);
static_assert(PsqtSize == Avx2Int32Lanes);

[[gnu::always_inline]] inline __m256i load_signed_i8_as_i32(
    const std::int8_t* values
) {
    const __m128i packed = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(values));
    return _mm256_cvtepi8_epi32(packed);
}

template<std::size_t AddedCount, std::size_t RemovedCount>
void update_accumulator_avx2(
    std::int32_t* __restrict accumulator,
    std::int32_t* __restrict psqt,
    const std::array<AccumulatorRowView, AddedCount>& added,
    const std::array<AccumulatorRowView, RemovedCount>& removed
) {
    for (std::size_t lane = 0;
         lane < AccumulatorSize;
         lane += Avx2Int32Lanes) {
        __m256i delta = _mm256_setzero_si256();
        for (const AccumulatorRowView row : added) {
            assert(row.positional != nullptr);
            delta = _mm256_add_epi32(
                delta,
                load_signed_i8_as_i32(row.positional + lane));
        }
        for (const AccumulatorRowView row : removed) {
            assert(row.positional != nullptr);
            delta = _mm256_sub_epi32(
                delta,
                load_signed_i8_as_i32(row.positional + lane));
        }
        const __m256i value = _mm256_add_epi32(
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(accumulator + lane)),
            delta);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(accumulator + lane),
            value);
    }

    __m256i psqt_value = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(psqt));
    for (const AccumulatorRowView row : added) {
        assert(row.psqt != nullptr);
        psqt_value = _mm256_add_epi32(
            psqt_value,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row.psqt)));
    }
    for (const AccumulatorRowView row : removed) {
        assert(row.psqt != nullptr);
        psqt_value = _mm256_sub_epi32(
            psqt_value,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row.psqt)));
    }
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(psqt),
        psqt_value);
}

} // namespace

void update_accumulator_avx2_1_1(
    std::int32_t* accumulator,
    std::int32_t* psqt,
    AccumulatorRowView added,
    AccumulatorRowView removed
) {
    update_accumulator_avx2<1, 1>(
        accumulator,
        psqt,
        std::array{added},
        std::array{removed});
}

void update_accumulator_avx2_1_2(
    std::int32_t* accumulator,
    std::int32_t* psqt,
    AccumulatorRowView added,
    AccumulatorRowView removed0,
    AccumulatorRowView removed1
) {
    update_accumulator_avx2<1, 2>(
        accumulator,
        psqt,
        std::array{added},
        std::array{removed0, removed1});
}

void update_accumulator_avx2_2_2(
    std::int32_t* accumulator,
    std::int32_t* psqt,
    AccumulatorRowView added0,
    AccumulatorRowView added1,
    AccumulatorRowView removed0,
    AccumulatorRowView removed1
) {
    update_accumulator_avx2<2, 2>(
        accumulator,
        psqt,
        std::array{added0, added1},
        std::array{removed0, removed1});
}

void rebuild_accumulator_avx2(
    std::int32_t* __restrict accumulator,
    std::int32_t* __restrict psqt,
    const std::int32_t* __restrict accumulator_bias,
    const AccumulatorRowView* __restrict active_rows,
    std::size_t active_count
) {
    for (std::size_t lane = 0;
         lane < AccumulatorSize;
         lane += Avx2Int32Lanes) {
        __m256i value = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(accumulator_bias + lane));
        for (std::size_t row_index = 0;
             row_index < active_count;
             ++row_index) {
            assert(active_rows[row_index].positional != nullptr);
            value = _mm256_add_epi32(
                value,
                load_signed_i8_as_i32(
                    active_rows[row_index].positional + lane));
        }
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(accumulator + lane),
            value);
    }

    __m256i psqt_value = _mm256_setzero_si256();
    for (std::size_t row_index = 0;
         row_index < active_count;
         ++row_index) {
        assert(active_rows[row_index].psqt != nullptr);
        psqt_value = _mm256_add_epi32(
            psqt_value,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    active_rows[row_index].psqt)));
    }
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(psqt),
        psqt_value);
}

} // namespace chess::phase_nnue_detail
