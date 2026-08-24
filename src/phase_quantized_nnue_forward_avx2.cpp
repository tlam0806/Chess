#include "phase_quantized_nnue_forward_x86_impl.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace chess::phase_nnue_detail {

namespace {

class Avx2Network {
private:
    static constexpr std::size_t H1 =
        PhaseQuantizedNnueModel::DenseInputSize;
    static constexpr std::size_t H2 =
        PhaseQuantizedNnueModel::Hidden2Size;
    static constexpr std::size_t H3 =
        PhaseQuantizedNnueModel::Hidden3Size;

    std::int64_t output_bias_ = 0;
    unsigned hidden2_scale_shift_ = 0;
    unsigned hidden3_scale_shift_ = 0;
    std::uint32_t output_scale_ = 0;
    alignas(64) std::array<std::int32_t, H2> hidden2_bias_{};
    alignas(64) std::array<std::int8_t, H1 * H2> hidden2_weight_{};
    alignas(64) std::array<std::int32_t, H3> hidden3_bias_{};
    alignas(64) std::array<std::int8_t, H2 * H3> hidden3_weight_{};
    alignas(64) std::array<std::int8_t, H3> output_weight_{};

    template<std::size_t InputSize>
    [[nodiscard, gnu::always_inline]] static std::int64_t dot_u8_s8(
        const std::uint8_t* activation,
        const std::int8_t* weight
    ) {
        static_assert(InputSize % 16 == 0);
        __m256i sums = _mm256_setzero_si256();
        for (std::size_t input = 0; input < InputSize; input += 16) {
            const __m128i activation8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(activation + input));
            const __m128i weight8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(weight + input));
            const __m256i activation16 =
                _mm256_cvtepu8_epi16(activation8);
            const __m256i weight16 = _mm256_cvtepi8_epi16(weight8);
            sums = _mm256_add_epi32(
                sums,
                _mm256_madd_epi16(activation16, weight16));
        }
        alignas(32) std::array<std::int32_t, 8> lanes{};
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(lanes.data()), sums);
        std::int64_t result = 0;
        for (std::int32_t lane : lanes) {
            result += lane;
        }
        return result;
    }

public:
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
        std::copy_n(source.hidden2_weight, H1 * H2, hidden2_weight_.begin());
        std::copy_n(source.hidden3_bias, H3, hidden3_bias_.begin());
        std::copy_n(source.hidden3_weight, H2 * H3, hidden3_weight_.begin());
        std::copy_n(source.output_weight, H3, output_weight_.begin());
    }

    [[nodiscard]] std::int64_t evaluate(
        const std::int32_t* stm_accumulator,
        const std::int32_t* opponent_accumulator,
        const std::int16_t* combined_aux_row
    ) const {
        alignas(32) std::array<std::uint8_t, H1> hidden1{};
        build_first_activation(
            stm_accumulator,
            opponent_accumulator,
            combined_aux_row,
            hidden1.data());

        alignas(32) std::array<std::uint16_t, H2> hidden2{};
        for (std::size_t output = 0; output < H2; ++output) {
            const std::int64_t sum = hidden2_bias_[output]
                + dot_u8_s8<H1>(
                    hidden1.data(),
                    hidden2_weight_.data() + output * H1);
            hidden2[output] = scaled_clipped_relu(
                sum, hidden2_scale_shift_);
        }

        // _mm256_madd_epi16 treats both operands as signed.  Split the u16
        // activation instead of reinterpreting values >= 32768 as negative.
        alignas(32) std::array<std::uint8_t, H2> hidden2_low{};
        alignas(32) std::array<std::uint8_t, H2> hidden2_high{};
        for (std::size_t input = 0; input < H2; ++input) {
            hidden2_low[input] = static_cast<std::uint8_t>(hidden2[input]);
            hidden2_high[input] = static_cast<std::uint8_t>(
                hidden2[input] >> 8);
        }

        alignas(32) std::array<std::uint16_t, H3> hidden3{};
        for (std::size_t output = 0; output < H3; ++output) {
            const std::int64_t sum = hidden3_bias_[output]
                + dot_u8_s8<H2>(
                    hidden2_low.data(),
                    hidden3_weight_.data() + output * H2)
                + 256LL * dot_u8_s8<H2>(
                    hidden2_high.data(),
                    hidden3_weight_.data() + output * H2);
            hidden3[output] = scaled_clipped_relu(
                sum, hidden3_scale_shift_);
        }

        std::int64_t raw = output_bias_;
        for (std::size_t input = 0; input < H3; ++input) {
            raw += static_cast<std::int64_t>(hidden3[input])
                * output_weight_[input];
        }
        return raw / static_cast<std::int64_t>(output_scale_);
    }
};

} // namespace

std::unique_ptr<PhaseCandidateKernelBase> make_phase_nnue_avx2_kernel(
    const PhaseKernelSourceView& source
) {
    return std::make_unique<X86PhaseKernel<Avx2Network>>(
        source, "x86_avx2_exact");
}

} // namespace chess::phase_nnue_detail
