#include "phase_quantized_nnue_forward_x86_impl.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace chess::phase_nnue_detail {

namespace {

class VnniNetwork {
private:
    static constexpr std::size_t H1 =
        PhaseQuantizedNnueModel::DenseInputSize;
    static constexpr std::size_t H2 =
        PhaseQuantizedNnueModel::Hidden2Size;
    static constexpr std::size_t H3 =
        PhaseQuantizedNnueModel::Hidden3Size;
    static constexpr std::size_t OutputsPerVector = 8;
    static constexpr std::size_t InputsPerDot = 4;

    std::int64_t output_bias_ = 0;
    unsigned hidden2_scale_shift_ = 0;
    unsigned hidden3_scale_shift_ = 0;
    std::uint32_t output_scale_ = 0;
    alignas(64) std::array<std::int32_t, H2> hidden2_bias_{};
    alignas(64) std::array<std::int8_t, H1 * H2> hidden2_weight_{};
    alignas(64) std::array<std::int32_t, H3> hidden3_bias_{};
    alignas(64) std::array<std::int8_t, H2 * H3> hidden3_weight_{};
    alignas(64) std::array<std::int8_t, H3> output_weight_{};

    template<std::size_t InputSize, std::size_t OutputSize>
    static void pack_weights(
        const std::int8_t* source,
        std::array<std::int8_t, InputSize * OutputSize>& packed
    ) {
        static_assert(InputSize % InputsPerDot == 0);
        static_assert(OutputSize % OutputsPerVector == 0);
        std::int8_t* destination = packed.data();
        for (std::size_t input = 0;
             input < InputSize;
             input += InputsPerDot) {
            for (std::size_t output_group = 0;
                 output_group < OutputSize;
                 output_group += OutputsPerVector) {
                for (std::size_t output = 0;
                     output < OutputsPerVector;
                     ++output) {
                    for (std::size_t offset = 0;
                         offset < InputsPerDot;
                         ++offset) {
                        *destination++ = source[
                            (output_group + output) * InputSize
                            + input + offset];
                    }
                }
            }
        }
        assert(destination == packed.data() + packed.size());
    }

    template<std::size_t InputSize, std::size_t OutputSize>
    [[gnu::always_inline]] static void dot_packed(
        const std::uint8_t* activation,
        const std::array<std::int8_t, InputSize * OutputSize>& weight,
        std::int32_t* output
    ) {
        static_assert(InputSize % InputsPerDot == 0);
        static_assert(OutputSize == 32);
        std::array<__m256i, OutputSize / OutputsPerVector> accumulators{};
        for (__m256i& accumulator : accumulators) {
            accumulator = _mm256_setzero_si256();
        }

        const std::int8_t* packed = weight.data();
        for (std::size_t input = 0;
             input < InputSize;
             input += InputsPerDot) {
            std::uint32_t activation_word = 0;
            std::memcpy(
                &activation_word,
                activation + input,
                sizeof(activation_word));
            const __m256i broadcast = _mm256_set1_epi32(
                static_cast<std::int32_t>(activation_word));
            for (std::size_t group = 0;
                 group < accumulators.size();
                 ++group) {
                const __m256i weights = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(packed));
                // VPDPBUSD: unsigned bytes from the activation are multiplied
                // by signed bytes from the weights, with non-saturating i32
                // accumulation.  Do not replace this with VPDPBUSDS.
                accumulators[group] = _mm256_dpbusd_epi32(
                    accumulators[group], broadcast, weights);
                packed += 32;
            }
        }
        assert(packed == weight.data() + weight.size());
        for (std::size_t group = 0;
             group < accumulators.size();
             ++group) {
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(
                    output + group * OutputsPerVector),
                accumulators[group]);
        }
    }

public:
    using Hidden2 = std::array<std::uint16_t, H2>;
    using Hidden3 = std::array<std::uint16_t, H3>;

    void load(
        const PhaseKernelPhaseView& source,
        std::uint32_t hidden2_scale,
        std::uint32_t hidden3_scale,
        std::uint32_t output_scale
    ) {
        assert(source.hidden2_bias != nullptr);
        assert(source.hidden2_weight != nullptr);
        assert(source.hidden3_bias != nullptr);
        assert(source.hidden3_weight != nullptr);
        assert(source.output_weight != nullptr);
        output_bias_ = source.output_bias;
        hidden2_scale_shift_ = power_of_two_shift(hidden2_scale);
        hidden3_scale_shift_ = power_of_two_shift(hidden3_scale);
        output_scale_ = output_scale;
        std::copy_n(source.hidden2_bias, H2, hidden2_bias_.begin());
        std::copy_n(source.hidden3_bias, H3, hidden3_bias_.begin());
        std::copy_n(source.output_weight, H3, output_weight_.begin());
        pack_weights<H1, H2>(source.hidden2_weight, hidden2_weight_);
        pack_weights<H2, H3>(source.hidden3_weight, hidden3_weight_);
    }

    [[gnu::always_inline]] void input_to_hidden2(
        const std::int32_t* stm_accumulator,
        const std::int32_t* opponent_accumulator,
        const std::int16_t* combined_aux_row,
        Hidden2& hidden2
    ) const {
        alignas(32) std::array<std::uint8_t, H1> hidden1{};
        build_first_activation(
            stm_accumulator,
            opponent_accumulator,
            combined_aux_row,
            hidden1.data());

        alignas(32) std::array<std::int32_t, H2> hidden2_dot{};
        dot_packed<H1, H2>(
            hidden1.data(), hidden2_weight_, hidden2_dot.data());
        for (std::size_t output = 0; output < H2; ++output) {
            const std::int64_t sum = hidden2_bias_[output]
                + static_cast<std::int64_t>(hidden2_dot[output]);
            hidden2[output] = scaled_clipped_relu(
                sum, hidden2_scale_shift_);
        }
    }

    [[gnu::always_inline]] void hidden2_to_hidden3(
        const Hidden2& hidden2,
        Hidden3& hidden3
    ) const {
        // VPDPBUSD consumes u8 activations, while hidden2 is u16.  Splitting
        // into low and high bytes is exact because
        // u16 = low + 256 * high.  Keep the two dot products separate until
        // widening so neither a signed reinterpretation nor a narrow shift
        // can corrupt values above 32767.
        alignas(32) std::array<std::uint8_t, H2> hidden2_low{};
        alignas(32) std::array<std::uint8_t, H2> hidden2_high{};
        for (std::size_t input = 0; input < H2; ++input) {
            hidden2_low[input] = static_cast<std::uint8_t>(hidden2[input]);
            hidden2_high[input] = static_cast<std::uint8_t>(
                hidden2[input] >> 8);
        }
        alignas(32) std::array<std::int32_t, H3> hidden3_dot_low{};
        alignas(32) std::array<std::int32_t, H3> hidden3_dot_high{};
        dot_packed<H2, H3>(
            hidden2_low.data(),
            hidden3_weight_,
            hidden3_dot_low.data());
        dot_packed<H2, H3>(
            hidden2_high.data(),
            hidden3_weight_,
            hidden3_dot_high.data());

        for (std::size_t output = 0; output < H3; ++output) {
            const std::int64_t sum = hidden3_bias_[output]
                + static_cast<std::int64_t>(hidden3_dot_low[output])
                + 256LL * static_cast<std::int64_t>(
                    hidden3_dot_high[output]);
            hidden3[output] = scaled_clipped_relu(
                sum, hidden3_scale_shift_);
        }
    }

    [[gnu::always_inline, nodiscard]] std::int64_t hidden3_to_output(
        const Hidden3& hidden3
    ) const {
        std::int64_t raw = output_bias_;
        for (std::size_t input = 0; input < H3; ++input) {
            raw += static_cast<std::int64_t>(hidden3[input])
                * output_weight_[input];
        }
        // Signed division is deliberate: right shift rounds negative values
        // differently from C++ truncation toward zero.
        return raw / static_cast<std::int64_t>(output_scale_);
    }

    [[nodiscard]] std::int64_t evaluate(
        const std::int32_t* stm_accumulator,
        const std::int32_t* opponent_accumulator,
        const std::int16_t* combined_aux_row
    ) const {
        alignas(32) Hidden2 hidden2{};
        input_to_hidden2(
            stm_accumulator,
            opponent_accumulator,
            combined_aux_row,
            hidden2);
        alignas(32) Hidden3 hidden3{};
        hidden2_to_hidden3(hidden2, hidden3);
        return hidden3_to_output(hidden3);
    }
};

} // namespace

std::unique_ptr<PhaseCandidateKernelBase> make_phase_nnue_vnni_kernel(
    const PhaseKernelSourceView& source
) {
    return std::make_unique<X86PhaseKernel<VnniNetwork>>(
        source, "x86_avx512vnni_256");
}

} // namespace chess::phase_nnue_detail
