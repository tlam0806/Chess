#include "phase_quantized_nnue.hpp"
#include "phase_quantized_nnue_forward_backend.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>

#if defined(__ARM_NEON) \
    && defined(__ARM_FEATURE_DOTPROD) \
    && defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>
#define CHESS_PHASE_NNUE_CANDIDATE 1
#else
#define CHESS_PHASE_NNUE_CANDIDATE 0
#endif

namespace chess {

namespace {

constexpr std::int64_t Relu16Max = (1LL << 16) - 1;
constexpr std::uint32_t CandidateHiddenClip = 181;
constexpr std::uint32_t CandidateScreluDivisor = 128;
constexpr std::size_t AuxCastlingFeatureCount = 4;
[[maybe_unused]] constexpr std::size_t AuxHasEnPassantFeature = 4;
constexpr std::size_t AuxEnPassantFileFirstFeature = 5;
constexpr std::size_t AuxEnPassantStateCount = 9;
constexpr std::size_t AuxCastlingStateCount =
    1U << AuxCastlingFeatureCount;
[[maybe_unused]] constexpr std::size_t CombinedAuxStateCount =
    AuxCastlingStateCount * AuxEnPassantStateCount;

std::size_t combined_aux_state_index(
    const std::array<
        std::uint8_t,
        PhaseQuantizedNnueModel::AuxFeatureCount>& aux
) {
    std::size_t castling_state = 0;
    for (std::size_t feature = 0;
         feature < AuxCastlingFeatureCount;
         ++feature) {
        assert(aux[feature] <= 1);
        castling_state |=
            static_cast<std::size_t>(aux[feature]) << feature;
    }

    assert(aux[AuxHasEnPassantFeature] <= 1);
    std::size_t en_passant_state = 0;
    for (std::size_t file = 0; file < 8; ++file) {
        const std::size_t feature =
            AuxEnPassantFileFirstFeature + file;
        assert(aux[feature] <= 1);
        if (aux[feature] != 0) {
            assert(en_passant_state == 0);
            en_passant_state = file + 1;
        }
    }
    assert(
        (aux[AuxHasEnPassantFeature] != 0)
        == (en_passant_state != 0));

    return castling_state * AuxEnPassantStateCount
        + en_passant_state;
}

constexpr bool supported_scale_tuple(
    std::uint32_t hidden2_scale,
    std::uint32_t hidden3_scale,
    std::uint32_t output_scale
) {
    return (hidden2_scale == 2
            && hidden3_scale == 8
            && output_scale == 128)
        || (hidden2_scale == 4
            && hidden3_scale == 8
            && output_scale == 64)
        || (hidden2_scale == 8
            && hidden3_scale == 16
            && output_scale == 128);
}

std::string_view requested_kernel_backend() {
    const char* value = std::getenv("CHESS_NNUE_BACKEND");
    return value == nullptr || *value == '\0'
        ? std::string_view{"auto"}
        : std::string_view{value};
}

#if CHESS_PHASE_NNUE_CANDIDATE

template<
    std::size_t H1,
    std::size_t H2,
    std::size_t H3,
    int Hidden2Scale,
    int Hidden3Scale,
    int OutputScale
>
class Candidate {
private:
    static constexpr int FirstActivationClip = 181;
    static constexpr int FirstActivationScale = 128;
    static constexpr int FirstActivationShift = 7;
    static_assert(
        FirstActivationClip * FirstActivationClip / FirstActivationScale
        == UINT8_MAX);
    static_assert(H1 > 0 && H1 % 16 == 0);
    static_assert(H1 % 2 == 0 && (H1 / 2) % 16 == 0);
    static_assert(H2 == 32);
    static_assert(H3 == 32);
    static_assert(Hidden2Scale > 0);
    static_assert((Hidden2Scale & (Hidden2Scale - 1)) == 0);
    static_assert(Hidden3Scale > 0);
    static_assert((Hidden3Scale & (Hidden3Scale - 1)) == 0);
    static_assert(OutputScale > 0);

    alignas(64) std::array<std::int8_t, H2 * H3> hidden3_weight_{};
    std::int64_t output_bias_ = 0;
    alignas(64) std::array<std::int32_t, H2> hidden2_bias_{};
    alignas(64) std::array<std::int8_t, H1 * H2> hidden2_weight_{};
    alignas(64) std::array<std::int32_t, H3> hidden3_bias_{};
    alignas(64) std::array<std::int16_t, H3> output_weight_{};

    [[gnu::always_inline]] static uint16x8_t square8(
        int32x4_t input0,
        int32x4_t input1
    ) {
        const uint16x4_t clipped_low = vqmovun_s32(input0);
        const uint16x8_t clipped_wide =
            vqmovun_high_s32(clipped_low, input1);
        const uint8x8_t clipped = vqmovn_u16(clipped_wide);
        return vmull_u8(clipped, clipped);
    }

    [[gnu::always_inline]] static uint8x16_t screlu16(
        int32x4_t input0,
        int32x4_t input1,
        int32x4_t input2,
        int32x4_t input3
    ) {
        const uint16x8_t squared_low = square8(
            input0, input1);
        const uint16x8_t squared_high = square8(
            input2, input3);
        // floor(clamp(x, 0, 181)^2 / 128). Values at and above 181
        // saturate to 255 during the narrowing shift.
        return vqshrn_high_n_u16(
            vqshrn_n_u16(squared_low, FirstActivationShift),
            squared_high,
            FirstActivationShift);
    }

    [[gnu::always_inline]] static uint8x16_t load_screlu16(
        const std::int32_t* input
    ) {
        return screlu16(
            vld1q_s32(input),
            vld1q_s32(input + 4),
            vld1q_s32(input + 8),
            vld1q_s32(input + 12));
    }

    [[gnu::always_inline]] static uint8x16_t load_screlu16_with_aux(
        const std::int32_t* input,
        const std::int16_t* combined_aux_row,
        std::size_t logical_input
    ) {
        int32x4_t value0 = vld1q_s32(input);
        int32x4_t value1 = vld1q_s32(input + 4);
        int32x4_t value2 = vld1q_s32(input + 8);
        int32x4_t value3 = vld1q_s32(input + 12);

        const int16x8_t weight_low =
            vld1q_s16(combined_aux_row + logical_input);
        const int16x8_t weight_high =
            vld1q_s16(combined_aux_row + logical_input + 8);
        value0 = vaddw_s16(value0, vget_low_s16(weight_low));
        value1 = vaddw_high_s16(value1, weight_low);
        value2 = vaddw_s16(value2, vget_low_s16(weight_high));
        value3 = vaddw_high_s16(value3, weight_high);

        return screlu16(value0, value1, value2, value3);
    }

    template<int Scale>
    [[gnu::always_inline]] static uint16x8_t clipped_relu16(
        int32x4_t input0,
        int32x4_t input1
    ) {
        static_assert(Scale > 0 && (Scale & (Scale - 1)) == 0);
        constexpr int shift = __builtin_ctz(static_cast<unsigned>(Scale));
        if constexpr (shift != 0) {
            input0 = vshrq_n_s32(input0, shift);
            input1 = vshrq_n_s32(input1, shift);
        }
        return vqmovun_high_s32(vqmovun_s32(input0), input1);
    }

    template<std::size_t Lane>
    [[gnu::always_inline]] static void add_to_accumulator(
        std::array<int32x4_t, 8>& accumulators,
        const std::int8_t*& __restrict weights,
        uint8x16_t activation
    ) {
#pragma clang loop unroll(enable)
        for (std::size_t index = 0; index < accumulators.size(); ++index) {
            accumulators[index] = vsudotq_laneq_s32(
                accumulators[index],
                vld1q_s8(weights),
                activation,
                Lane);
            weights += 16;
        }
    }

    [[gnu::always_inline]] static void dot_block(
        std::array<int32x4_t, 8>& accumulators,
        const std::int8_t*& __restrict weights,
        uint8x16_t activation
    ) {
        add_to_accumulator<0>(accumulators, weights, activation);
        add_to_accumulator<1>(accumulators, weights, activation);
        add_to_accumulator<2>(accumulators, weights, activation);
        add_to_accumulator<3>(accumulators, weights, activation);
    }

    template<bool HasAux>
    [[gnu::always_inline]] static void accumulate_input_half(
        std::array<int32x4_t, 8>& hidden2_accumulators,
        const std::int8_t*& __restrict weights,
        const std::int32_t* accumulator,
        const std::int16_t* combined_aux_row,
        std::size_t logical_offset
    ) {
        constexpr std::size_t HalfInputSize = H1 / 2;
        for (std::size_t input = 0; input < HalfInputSize; input += 16) {
            uint8x16_t activation;
            if constexpr (HasAux) {
                activation = load_screlu16_with_aux(
                    accumulator + input,
                    combined_aux_row,
                    logical_offset + input);
            } else {
                activation = load_screlu16(accumulator + input);
            }
            dot_block(hidden2_accumulators, weights, activation);
        }
    }

    template<std::size_t Lane>
    [[gnu::always_inline]] static void add_low_and_high(
        std::array<int32x4_t, 8>& low,
        std::array<int32x4_t, 8>& high,
        const std::int8_t*& __restrict weights,
        uint8x16_t activation_low,
        uint8x16_t activation_high
    ) {
#pragma clang loop unroll(enable)
        for (std::size_t index = 0; index < low.size(); index += 2) {
            const int8x16_t weight0 = vld1q_s8(weights);
            const int8x16_t weight1 = vld1q_s8(weights + 16);
            low[index] = vsudotq_laneq_s32(
                low[index], weight0, activation_low, Lane);
            high[index] = vsudotq_laneq_s32(
                high[index], weight0, activation_high, Lane);
            low[index + 1] = vsudotq_laneq_s32(
                low[index + 1], weight1, activation_low, Lane);
            high[index + 1] = vsudotq_laneq_s32(
                high[index + 1], weight1, activation_high, Lane);
            weights += 32;
        }
    }

    [[gnu::always_inline]] static void dot_block_low_and_high(
        std::array<int32x4_t, 8>& low,
        std::array<int32x4_t, 8>& high,
        const std::int8_t*& __restrict weights,
        uint8x16_t activation_low,
        uint8x16_t activation_high
    ) {
        add_low_and_high<0>(
            low, high, weights, activation_low, activation_high);
        add_low_and_high<1>(
            low, high, weights, activation_low, activation_high);
        add_low_and_high<2>(
            low, high, weights, activation_low, activation_high);
        add_low_and_high<3>(
            low, high, weights, activation_low, activation_high);
    }

    template<std::size_t InputSize, std::size_t OutputSize>
    static void pack_weights(
        const std::array<std::int8_t, InputSize * OutputSize>& source,
        std::array<std::int8_t, InputSize * OutputSize>& packed
    ) {
        static_assert(InputSize % 16 == 0);
        static_assert(OutputSize == 32);
        std::int8_t* destination = packed.data();
        for (std::size_t block = 0; block < InputSize; block += 16) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
                for (std::size_t accumulator = 0;
                     accumulator < 8;
                     ++accumulator) {
                    for (std::size_t offset = 0; offset < 16; ++offset) {
                        const std::size_t output =
                            accumulator * 4 + offset / 4;
                        const std::size_t input =
                            block + lane * 4 + offset % 4;
                        *destination++ = source[output * InputSize + input];
                    }
                }
            }
        }
        assert(destination == packed.data() + packed.size());
    }

public:
    Candidate() = default;

    void load(
        std::int64_t output_bias,
        const std::array<std::int32_t, H2>& hidden2_bias,
        const std::array<std::int8_t, H1 * H2>& hidden2_weight,
        const std::array<std::int32_t, H3>& hidden3_bias,
        const std::array<std::int8_t, H2 * H3>& hidden3_weight,
        const std::array<std::int8_t, H3>& output_weight
    ) {
        output_bias_ = output_bias;
        hidden2_bias_ = hidden2_bias;
        hidden3_bias_ = hidden3_bias;
        std::transform(
            output_weight.begin(),
            output_weight.end(),
            output_weight_.begin(),
            [](std::int8_t weight) {
                return static_cast<std::int16_t>(weight);
            });
        pack_weights<H1, H2>(hidden2_weight, hidden2_weight_);
        pack_weights<H2, H3>(hidden3_weight, hidden3_weight_);
    }

    [[gnu::always_inline]] std::int64_t evaluate(
        const std::int32_t* stm_accumulator,
        const std::int32_t* opponent_accumulator,
        const std::int16_t* combined_aux_row
    ) const {
        std::array<int32x4_t, 8> hidden2_accumulators{};
        for (std::size_t output = 0; output < H2; output += 4) {
            hidden2_accumulators[output / 4] =
                vld1q_s32(hidden2_bias_.data() + output);
        }

        const std::int8_t* weights = hidden2_weight_.data();
        if (combined_aux_row == nullptr) {
            accumulate_input_half<false>(
                hidden2_accumulators,
                weights,
                stm_accumulator,
                combined_aux_row,
                0);
            accumulate_input_half<false>(
                hidden2_accumulators,
                weights,
                opponent_accumulator,
                combined_aux_row,
                H1 / 2);
        } else {
            accumulate_input_half<true>(
                hidden2_accumulators,
                weights,
                stm_accumulator,
                combined_aux_row,
                0);
            accumulate_input_half<true>(
                hidden2_accumulators,
                weights,
                opponent_accumulator,
                combined_aux_row,
                H1 / 2);
        }
        assert(weights == hidden2_weight_.data() + hidden2_weight_.size());

        alignas(64) std::array<std::uint16_t, H2> hidden2{};
        for (std::size_t block = 0;
             block < hidden2_accumulators.size();
             block += 2) {
            vst1q_u16(
                hidden2.data() + block * 4,
                clipped_relu16<Hidden2Scale>(
                    hidden2_accumulators[block],
                    hidden2_accumulators[block + 1]));
        }

        std::array<int32x4_t, 8> hidden3_low{};
        std::array<int32x4_t, 8> hidden3_high{};
        for (std::size_t output = 0; output < H3; output += 4) {
            hidden3_low[output / 4] =
                vld1q_s32(hidden3_bias_.data() + output);
        }

        weights = hidden3_weight_.data();
        for (std::size_t input = 0; input < H2; input += 16) {
            const uint16x8_t activation0 =
                vld1q_u16(hidden2.data() + input);
            const uint16x8_t activation1 =
                vld1q_u16(hidden2.data() + input + 8);
            const uint8x16_t activation_low = vcombine_u8(
                vmovn_u16(activation0), vmovn_u16(activation1));
            const uint8x16_t activation_high = vcombine_u8(
                vshrn_n_u16(activation0, 8),
                vshrn_n_u16(activation1, 8));
            dot_block_low_and_high(
                hidden3_low,
                hidden3_high,
                weights,
                activation_low,
                activation_high);
        }
        assert(weights == hidden3_weight_.data() + hidden3_weight_.size());

        alignas(64) std::array<std::uint16_t, H3> hidden3{};
        for (std::size_t block = 0; block < hidden3_low.size(); block += 2) {
            const int32x4_t first = vaddq_s32(
                hidden3_low[block],
                vshlq_n_s32(hidden3_high[block], 8));
            const int32x4_t second = vaddq_s32(
                hidden3_low[block + 1],
                vshlq_n_s32(hidden3_high[block + 1], 8));
            vst1q_u16(
                hidden3.data() + block * 4,
                clipped_relu16<Hidden3Scale>(first, second));
        }

        int64x2_t output_sum0 = vdupq_n_s64(0);
        int64x2_t output_sum1 = vdupq_n_s64(0);
        for (std::size_t input = 0; input < H3; input += 8) {
            const int32x4_t activation0 = vreinterpretq_s32_u32(
                vmovl_u16(vld1_u16(hidden3.data() + input)));
            const int32x4_t output_weights0 = vmovl_s16(
                vld1_s16(output_weight_.data() + input));
            output_sum0 = vpadalq_s32(
                output_sum0,
                vmulq_s32(activation0, output_weights0));

            const int32x4_t activation1 = vreinterpretq_s32_u32(
                vmovl_u16(vld1_u16(hidden3.data() + input + 4)));
            const int32x4_t output_weights1 = vmovl_s16(
                vld1_s16(output_weight_.data() + input + 4));
            output_sum1 = vpadalq_s32(
                output_sum1,
                vmulq_s32(activation1, output_weights1));
        }
        const int64x2_t output_sum = vaddq_s64(
            output_sum0, output_sum1);
        const std::int64_t raw = output_bias_ + vaddvq_s64(output_sum);
        return raw / OutputScale;
    }
};

#endif

} // namespace

#if CHESS_PHASE_NNUE_CANDIDATE

template<int Hidden2Scale, int Hidden3Scale, int OutputScale>
class PhaseCandidateKernel final : public PhaseCandidateKernelBase {
private:
    using Network = Candidate<
        PhaseQuantizedNnueModel::DenseInputSize,
        PhaseQuantizedNnueModel::Hidden2Size,
        PhaseQuantizedNnueModel::Hidden3Size,
        Hidden2Scale,
        Hidden3Scale,
        OutputScale>;

    std::array<Network, PhaseQuantizedNnueModel::PhaseCount> phases_{};
    alignas(64) std::array<
        std::int16_t,
        CombinedAuxStateCount * PhaseQuantizedNnueModel::DenseInputSize>
        combined_aux_rows_{};

public:
    explicit PhaseCandidateKernel(const PhaseQuantizedNnueModel& model) {
        for (std::size_t phase_index = 0;
             phase_index < phases_.size();
             ++phase_index) {
            const auto& source = model.phases_[phase_index];
            phases_[phase_index].load(
                source.output_bias,
                source.hidden2_bias,
                source.hidden2_weight,
                source.hidden3_bias,
                source.hidden3_weight,
                source.output_weight);
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
                            sum += model.aux_weights_[
                                feature
                                * PhaseQuantizedNnueModel::DenseInputSize
                                + input];
                        }
                    }
                    if (en_passant_state != 0) {
                        sum += model.aux_weights_[
                            AuxHasEnPassantFeature
                            * PhaseQuantizedNnueModel::DenseInputSize
                            + input];
                        const std::size_t file_feature =
                            AuxEnPassantFileFirstFeature
                            + en_passant_state - 1;
                        sum += model.aux_weights_[
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
        assert(phase_index < phases_.size());
        assert(aux_state < CombinedAuxStateCount);
        const std::int16_t* combined_aux_row =
            aux_state == 0
            ? nullptr
            : combined_aux_rows_.data()
                + aux_state * PhaseQuantizedNnueModel::DenseInputSize;
        return phases_[phase_index].evaluate(
            stm_accumulator,
            opponent_accumulator,
            combined_aux_row);
    }

    [[nodiscard]] std::string_view name() const override {
        return "arm_neon_dotprod_i8mm";
    }
};

#endif

PhaseQuantizedNnueModel::PhaseQuantizedNnueModel() = default;
PhaseQuantizedNnueModel::~PhaseQuantizedNnueModel() = default;

bool PhaseQuantizedNnueModel::supports_candidate_configuration(
    std::uint32_t hidden_clip,
    std::uint32_t screlu_divisor,
    std::uint32_t hidden2_scale,
    std::uint32_t hidden3_scale,
    std::uint32_t output_scale
) {
#if CHESS_PHASE_NNUE_CANDIDATE \
    || defined(CHESS_PHASE_NNUE_X86_BACKENDS)
    return hidden_clip == CandidateHiddenClip
        && screlu_divisor == CandidateScreluDivisor
        && supported_scale_tuple(
            hidden2_scale, hidden3_scale, output_scale);
#else
    static_cast<void>(hidden_clip);
    static_cast<void>(screlu_divisor);
    static_cast<void>(hidden2_scale);
    static_cast<void>(hidden3_scale);
    static_cast<void>(output_scale);
    return false;
#endif
}

bool PhaseQuantizedNnueModel::initialize_candidate_kernel() {
    candidate_kernel_.reset();
    const std::string_view requested = requested_kernel_backend();
    const bool valid_request = requested == "auto"
        || requested == "scalar"
        || requested == "avx2"
        || requested == "vnni"
        || requested == "neon";
    if (!valid_request) {
        return false;
    }
    if (hidden_clip_ != CandidateHiddenClip
        || screlu_divisor_ != CandidateScreluDivisor
        || !supported_scale_tuple(
            hidden2_scale_, hidden3_scale_, output_scale_)) {
        return requested == "auto" || requested == "scalar";
    }
    if (requested == "scalar") {
        return true;
    }

#if defined(CHESS_PHASE_NNUE_X86_BACKENDS)
    phase_nnue_detail::PhaseKernelSourceView source{};
    source.hidden2_scale = hidden2_scale_;
    source.hidden3_scale = hidden3_scale_;
    source.output_scale = output_scale_;
    source.aux_weights = aux_weights_.data();
    for (std::size_t phase = 0; phase < PhaseCount; ++phase) {
        source.phases[phase] = {
            phases_[phase].output_bias,
            phases_[phase].hidden2_bias.data(),
            phases_[phase].hidden2_weight.data(),
            phases_[phase].hidden3_bias.data(),
            phases_[phase].hidden3_weight.data(),
            phases_[phase].output_weight.data(),
        };
    }

    // These builtins include the OSXSAVE/XCR0 checks needed before entering an
    // AVX/AVX-512 translation unit.  No ISA-specific constructor is called
    // until the corresponding gate succeeds.
    __builtin_cpu_init();
    if ((requested == "auto" || requested == "vnni")
        && __builtin_cpu_supports("avx2")
        && __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl")
        && __builtin_cpu_supports("avx512vnni")) {
        candidate_kernel_ =
            phase_nnue_detail::make_phase_nnue_vnni_kernel(source);
        return true;
    }
    if ((requested == "auto" || requested == "avx2")
        && __builtin_cpu_supports("avx2")) {
        candidate_kernel_ =
            phase_nnue_detail::make_phase_nnue_avx2_kernel(source);
        return true;
    }
#endif

#if CHESS_PHASE_NNUE_CANDIDATE
    if ((requested == "auto" || requested == "neon")
        && hidden2_scale_ == 2
        && hidden3_scale_ == 8
        && output_scale_ == 128) {
        candidate_kernel_ =
            std::make_unique<PhaseCandidateKernel<2, 8, 128>>(*this);
    } else if (
        (requested == "auto" || requested == "neon")
        && hidden2_scale_ == 4
        && hidden3_scale_ == 8
        && output_scale_ == 64) {
        candidate_kernel_ =
            std::make_unique<PhaseCandidateKernel<4, 8, 64>>(*this);
    } else if (
        (requested == "auto" || requested == "neon")
        && hidden2_scale_ == 8
        && hidden3_scale_ == 16
        && output_scale_ == 128) {
        candidate_kernel_ =
            std::make_unique<PhaseCandidateKernel<8, 16, 128>>(*this);
    }
#endif
    if (candidate_kernel_ != nullptr) {
        return true;
    }
    // "auto" deliberately permits scalar fallback on an older CPU.  An
    // explicit backend request is a reproducibility control, so never hide an
    // unavailable/misspelled backend by silently selecting scalar.
    return requested == "auto";
}

bool PhaseQuantizedNnueModel::uses_accelerated_kernel() const {
    return accelerated_kernel_enabled_ && candidate_kernel_ != nullptr;
}

bool PhaseQuantizedNnueModel::uses_neon_dotprod_kernel() const {
    return uses_accelerated_kernel();
}

std::string_view PhaseQuantizedNnueModel::forward_kernel_name() const {
    return uses_accelerated_kernel()
        ? candidate_kernel_->name()
        : std::string_view{"scalar"};
}

std::int64_t PhaseQuantizedNnueModel::forward_positional_scalar(
    const std::array<std::int32_t, DenseInputSize>& dense_input,
    std::size_t phase_index
) const {
    assert(loaded());
    assert(phase_index < PhaseCount);

    const DensePhase& phase = phases_[phase_index];
    std::array<std::int32_t, DenseInputSize> hidden1{};
    for (std::size_t lane = 0; lane < DenseInputSize; ++lane) {
        const std::int64_t clipped = std::clamp<std::int64_t>(
            dense_input[lane],
            0,
            static_cast<std::int64_t>(hidden_clip_));
        hidden1[lane] = static_cast<std::int32_t>(
            (clipped * clipped)
            / static_cast<std::int64_t>(screlu_divisor_));
    }

    std::array<std::int32_t, Hidden2Size> hidden2{};
    for (std::size_t output = 0; output < Hidden2Size; ++output) {
        std::int64_t sum = phase.hidden2_bias[output];
        const std::size_t offset = output * DenseInputSize;
        for (std::size_t input = 0; input < DenseInputSize; ++input) {
            sum += static_cast<std::int64_t>(hidden1[input])
                * phase.hidden2_weight[offset + input];
        }
        hidden2[output] = static_cast<std::int32_t>(
            std::clamp<std::int64_t>(
                sum / static_cast<std::int64_t>(hidden2_scale_),
                0,
                Relu16Max));
    }

    std::array<std::int32_t, Hidden3Size> hidden3{};
    for (std::size_t output = 0; output < Hidden3Size; ++output) {
        std::int64_t sum = phase.hidden3_bias[output];
        const std::size_t offset = output * Hidden2Size;
        for (std::size_t input = 0; input < Hidden2Size; ++input) {
            sum += static_cast<std::int64_t>(hidden2[input])
                * phase.hidden3_weight[offset + input];
        }
        hidden3[output] = static_cast<std::int32_t>(
            std::clamp<std::int64_t>(
                sum / static_cast<std::int64_t>(hidden3_scale_),
                0,
                Relu16Max));
    }

    std::int64_t raw = phase.output_bias;
    for (std::size_t input = 0; input < Hidden3Size; ++input) {
        raw += static_cast<std::int64_t>(hidden3[input])
            * phase.output_weight[input];
    }
    return raw / static_cast<std::int64_t>(output_scale_);
}

std::int64_t PhaseQuantizedNnueModel::forward_positional_candidate(
    const std::array<std::int32_t, PerspectiveAccumulatorSize>&
        stm_accumulator,
    const std::array<std::int32_t, PerspectiveAccumulatorSize>&
        opponent_accumulator,
    const std::array<std::uint8_t, AuxFeatureCount>& aux,
    std::size_t phase_index
) const {
    assert(loaded());
    assert(phase_index < PhaseCount);
    assert(uses_accelerated_kernel());

    return candidate_kernel_->evaluate(
        stm_accumulator.data(),
        opponent_accumulator.data(),
        combined_aux_state_index(aux),
        phase_index);
}

} // namespace chess
