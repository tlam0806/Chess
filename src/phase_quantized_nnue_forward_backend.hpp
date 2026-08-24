#pragma once

#include "phase_quantized_nnue.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace chess {

// The ISA-specific implementations live in their own translation units.  This
// small interface is intentionally private to chess_core: the public model
// owns one selected implementation and pays one virtual call per evaluation,
// not once per layer or dot product.
class PhaseCandidateKernelBase {
public:
    virtual ~PhaseCandidateKernelBase() = default;

    [[nodiscard]] virtual std::int64_t evaluate(
        const std::int32_t* stm_accumulator,
        const std::int32_t* opponent_accumulator,
        std::size_t aux_state,
        std::size_t phase_index
    ) const = 0;

    [[nodiscard]] virtual std::string_view name() const = 0;
};

namespace phase_nnue_detail {

inline constexpr std::size_t AuxCastlingFeatureCount = 4;
inline constexpr std::size_t AuxHasEnPassantFeature = 4;
inline constexpr std::size_t AuxEnPassantFileFirstFeature = 5;
inline constexpr std::size_t AuxEnPassantStateCount = 9;
inline constexpr std::size_t AuxCastlingStateCount =
    1U << AuxCastlingFeatureCount;
inline constexpr std::size_t CombinedAuxStateCount =
    AuxCastlingStateCount * AuxEnPassantStateCount;

struct PhaseKernelPhaseView {
    std::int64_t output_bias = 0;
    const std::int32_t* hidden2_bias = nullptr;
    const std::int8_t* hidden2_weight = nullptr;
    const std::int32_t* hidden3_bias = nullptr;
    const std::int8_t* hidden3_weight = nullptr;
    const std::int8_t* output_weight = nullptr;
};

struct PhaseKernelSourceView {
    std::uint32_t hidden2_scale = 0;
    std::uint32_t hidden3_scale = 0;
    std::uint32_t output_scale = 0;
    const std::int8_t* aux_weights = nullptr;
    std::array<
        PhaseKernelPhaseView,
        PhaseQuantizedNnueModel::PhaseCount> phases{};
};

#if defined(CHESS_PHASE_NNUE_X86_BACKENDS)

[[nodiscard]] std::unique_ptr<PhaseCandidateKernelBase>
make_phase_nnue_avx2_kernel(const PhaseKernelSourceView& source);

[[nodiscard]] std::unique_ptr<PhaseCandidateKernelBase>
make_phase_nnue_vnni_kernel(const PhaseKernelSourceView& source);

#endif

} // namespace phase_nnue_detail
} // namespace chess
