#pragma once

#include "phase_quantized_nnue_forward_backend.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <string_view>

namespace chess::phase_nnue_detail {

inline constexpr std::uint32_t X86HiddenClip = 181;
inline constexpr std::uint32_t X86ScreluDivisor = 128;

[[gnu::always_inline]] inline __m128i screlu_16(
    const std::int32_t* input,
    const std::int16_t* combined_aux_row
) {
    const __m128i zero = _mm_setzero_si128();
    const __m128i clip = _mm_set1_epi32(
        static_cast<int>(X86HiddenClip));
    std::array<__m128i, 4> values{};

    for (std::size_t block = 0; block < values.size(); ++block) {
        __m128i value = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(input + block * 4));
        if (combined_aux_row != nullptr) {
            const __m128i aux16 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(
                    combined_aux_row + block * 4));
            value = _mm_add_epi32(value, _mm_cvtepi16_epi32(aux16));
        }
        value = _mm_max_epi32(zero, _mm_min_epi32(value, clip));
        values[block] = _mm_srli_epi32(_mm_mullo_epi32(value, value), 7);
    }

    // The 128-bit packs preserve the logical lane order.  All values are in
    // [0, 255], so neither pack can change a value through saturation.
    const __m128i low = _mm_packs_epi32(values[0], values[1]);
    const __m128i high = _mm_packs_epi32(values[2], values[3]);
    return _mm_packus_epi16(low, high);
}

[[gnu::always_inline]] inline void build_first_activation(
    const std::int32_t* stm_accumulator,
    const std::int32_t* opponent_accumulator,
    const std::int16_t* combined_aux_row,
    std::uint8_t* destination
) {
    constexpr std::size_t Half =
        PhaseQuantizedNnueModel::PerspectiveAccumulatorSize;
    for (std::size_t input = 0; input < Half; input += 16) {
        const std::int16_t* aux = combined_aux_row == nullptr
            ? nullptr
            : combined_aux_row + input;
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(destination + input),
            screlu_16(stm_accumulator + input, aux));
    }
    for (std::size_t input = 0; input < Half; input += 16) {
        const std::int16_t* aux = combined_aux_row == nullptr
            ? nullptr
            : combined_aux_row + Half + input;
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(destination + Half + input),
            screlu_16(opponent_accumulator + input, aux));
    }
}

[[nodiscard]] inline std::uint16_t scaled_clipped_relu(
    std::int64_t sum,
    unsigned scale_shift
) {
    if (sum <= 0) {
        return 0;
    }
    const std::uint64_t scaled =
        static_cast<std::uint64_t>(sum) >> scale_shift;
    return static_cast<std::uint16_t>(std::min<std::uint64_t>(
        scaled, std::numeric_limits<std::uint16_t>::max()));
}

[[nodiscard]] inline unsigned power_of_two_shift(std::uint32_t scale) {
    assert(scale != 0 && (scale & (scale - 1)) == 0);
    return std::countr_zero(scale);
}

template<class Network>
class X86PhaseKernel final : public PhaseCandidateKernelBase {
private:
    std::array<Network, PhaseQuantizedNnueModel::PhaseCount> phases_{};
    alignas(64) std::array<
        std::int16_t,
        CombinedAuxStateCount * PhaseQuantizedNnueModel::DenseInputSize>
        combined_aux_rows_{};
    std::string_view name_;

public:
    X86PhaseKernel(
        const PhaseKernelSourceView& source,
        std::string_view name
    ) : name_(name) {
        assert(source.aux_weights != nullptr);
        for (std::size_t phase = 0; phase < phases_.size(); ++phase) {
            phases_[phase].load(
                source.phases[phase],
                source.hidden2_scale,
                source.hidden3_scale,
                source.output_scale);
        }

        for (std::size_t castling_state = 0;
             castling_state < AuxCastlingStateCount;
             ++castling_state) {
            for (std::size_t en_passant_state = 0;
                 en_passant_state < AuxEnPassantStateCount;
                 ++en_passant_state) {
                const std::size_t state =
                    castling_state * AuxEnPassantStateCount
                    + en_passant_state;
                for (std::size_t input = 0;
                     input < PhaseQuantizedNnueModel::DenseInputSize;
                     ++input) {
                    std::int32_t sum = 0;
                    for (std::size_t feature = 0;
                         feature < AuxCastlingFeatureCount;
                         ++feature) {
                        if ((castling_state & (1U << feature)) != 0) {
                            sum += source.aux_weights[
                                feature
                                * PhaseQuantizedNnueModel::DenseInputSize
                                + input];
                        }
                    }
                    if (en_passant_state != 0) {
                        sum += source.aux_weights[
                            AuxHasEnPassantFeature
                            * PhaseQuantizedNnueModel::DenseInputSize
                            + input];
                        const std::size_t file_feature =
                            AuxEnPassantFileFirstFeature
                            + en_passant_state - 1;
                        sum += source.aux_weights[
                            file_feature
                            * PhaseQuantizedNnueModel::DenseInputSize
                            + input];
                    }
                    assert(
                        sum >= std::numeric_limits<std::int16_t>::min()
                        && sum <= std::numeric_limits<std::int16_t>::max());
                    combined_aux_rows_[
                        state * PhaseQuantizedNnueModel::DenseInputSize
                        + input] = static_cast<std::int16_t>(sum);
                }
            }
        }
    }

    [[nodiscard]] std::int64_t evaluate(
        const std::int32_t* stm_accumulator,
        const std::int32_t* opponent_accumulator,
        std::size_t aux_state,
        std::size_t phase_index
    ) const override {
        assert(aux_state < CombinedAuxStateCount);
        assert(phase_index < phases_.size());
        const std::int16_t* combined_aux_row = aux_state == 0
            ? nullptr
            : combined_aux_rows_.data()
                + aux_state * PhaseQuantizedNnueModel::DenseInputSize;
        return phases_[phase_index].evaluate(
            stm_accumulator,
            opponent_accumulator,
            combined_aux_row);
    }

    [[nodiscard]] std::string_view name() const override {
        return name_;
    }

#if defined(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)
    [[nodiscard]] PhaseForwardBenchmarkBatch benchmark_stage(
        PhaseForwardBenchmarkStage stage,
        const std::vector<PhaseForwardBenchmarkSample>& samples,
        std::size_t evaluations
    ) const override {
        if constexpr (!requires(
            const Network& network,
            const std::int32_t* input,
            const std::int16_t* aux,
            typename Network::Hidden2& hidden2,
            typename Network::Hidden3& hidden3
        ) {
            network.input_to_hidden2(input, input, aux, hidden2);
            network.hidden2_to_hidden3(hidden2, hidden3);
            network.hidden3_to_output(hidden3);
        }) {
            return {};
        } else {
            if (samples.empty() || evaluations == 0) {
                return {};
            }
            using Hidden2 = typename Network::Hidden2;
            using Hidden3 = typename Network::Hidden3;
            struct Prepared {
                alignas(32) Hidden2 hidden2{};
                alignas(32) Hidden3 hidden3{};
            };
            std::vector<Prepared> prepared(samples.size());
            const auto aux_row = [&](const PhaseForwardBenchmarkSample& sample) {
                return sample.aux_state == 0
                    ? static_cast<const std::int16_t*>(nullptr)
                    : combined_aux_rows_.data()
                        + sample.aux_state
                            * PhaseQuantizedNnueModel::DenseInputSize;
            };
            for (std::size_t index = 0; index < samples.size(); ++index) {
                const auto& sample = samples[index];
                const Network& network = phases_[sample.phase_index];
                network.input_to_hidden2(
                    sample.stm.data(),
                    sample.opponent.data(),
                    aux_row(sample),
                    prepared[index].hidden2);
                network.hidden2_to_hidden3(
                    prepared[index].hidden2,
                    prepared[index].hidden3);
            }

            std::uint64_t checksum = 0xcbf29ce484222325ULL;
            for (std::size_t index = 0; index < samples.size(); ++index) {
                const Network& network = phases_[samples[index].phase_index];
                if (stage == PhaseForwardBenchmarkStage::InputToHidden2) {
                    for (std::uint16_t value : prepared[index].hidden2) {
                        checksum = phase_benchmark_mix(checksum, value);
                    }
                } else if (stage == PhaseForwardBenchmarkStage::Hidden2ToHidden3) {
                    for (std::uint16_t value : prepared[index].hidden3) {
                        checksum = phase_benchmark_mix(checksum, value);
                    }
                } else if (
                    stage == PhaseForwardBenchmarkStage::Hidden3ToOutput) {
                    const std::int64_t value = network.hidden3_to_output(
                        prepared[index].hidden3);
                    checksum = phase_benchmark_mix(
                        checksum, static_cast<std::uint64_t>(value));
                } else {
                    const auto& sample = samples[index];
                    const std::int64_t value = network.evaluate(
                        sample.stm.data(),
                        sample.opponent.data(),
                        aux_row(sample));
                    checksum = phase_benchmark_mix(
                        checksum, static_cast<std::uint64_t>(value));
                }
            }

            std::uint64_t sink = 0;
            const std::clock_t cpu_start = std::clock();
            const auto start = std::chrono::steady_clock::now();
            if (stage == PhaseForwardBenchmarkStage::InputToHidden2) {
                alignas(32) Hidden2 output{};
                std::size_t sample_index = 0;
                for (std::size_t iteration = 0;
                     iteration < evaluations;
                     ++iteration) {
                    const auto& sample = samples[sample_index];
                    phases_[sample.phase_index].input_to_hidden2(
                        sample.stm.data(),
                        sample.opponent.data(),
                        aux_row(sample),
                        output);
                    phase_benchmark_consume_all(output);
                    sink += output[iteration & (output.size() - 1)];
                    if (++sample_index == samples.size()) {
                        sample_index = 0;
                    }
                }
            } else if (stage == PhaseForwardBenchmarkStage::Hidden2ToHidden3) {
                alignas(32) Hidden3 output{};
                std::size_t sample_index = 0;
                for (std::size_t iteration = 0;
                     iteration < evaluations;
                     ++iteration) {
                    phases_[samples[sample_index].phase_index]
                        .hidden2_to_hidden3(
                            prepared[sample_index].hidden2, output);
                    phase_benchmark_consume_all(output);
                    sink += output[iteration & (output.size() - 1)];
                    if (++sample_index == samples.size()) {
                        sample_index = 0;
                    }
                }
            } else if (stage == PhaseForwardBenchmarkStage::Hidden3ToOutput) {
                std::size_t sample_index = 0;
                for (std::size_t iteration = 0;
                     iteration < evaluations;
                     ++iteration) {
                    const std::int64_t value =
                        phases_[samples[sample_index].phase_index]
                            .hidden3_to_output(prepared[sample_index].hidden3);
                    sink += static_cast<std::uint64_t>(value);
                    if (++sample_index == samples.size()) {
                        sample_index = 0;
                    }
                }
            } else {
                std::size_t sample_index = 0;
                for (std::size_t iteration = 0;
                     iteration < evaluations;
                     ++iteration) {
                    const auto& sample = samples[sample_index];
                    const std::int64_t value = phases_[sample.phase_index].evaluate(
                        sample.stm.data(),
                        sample.opponent.data(),
                        aux_row(sample));
                    sink += static_cast<std::uint64_t>(value);
                    if (++sample_index == samples.size()) {
                        sample_index = 0;
                    }
                }
            }
            const auto stop = std::chrono::steady_clock::now();
            const std::clock_t cpu_stop = std::clock();
            return {
                true,
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        stop - start).count()),
                static_cast<std::uint64_t>(
                    static_cast<long double>(cpu_stop - cpu_start)
                    * 1'000'000'000.0L / CLOCKS_PER_SEC),
                checksum,
                sink,
            };
        }
    }
#endif
};

} // namespace chess::phase_nnue_detail
