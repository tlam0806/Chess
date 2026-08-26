#pragma once

#include "phase_quantized_nnue.hpp"

#include <array>
#if defined(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)
#include <chrono>
#include <ctime>
#endif
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#if defined(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)
#include <vector>
#endif

namespace chess {

#if defined(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)

enum class PhaseForwardBenchmarkStage {
    InputToHidden2,
    Hidden2ToHidden3,
    Hidden3ToOutput,
    Full,
};

struct PhaseForwardBenchmarkSample {
    alignas(64) std::array<
        std::int32_t,
        PhaseQuantizedNnueModel::PerspectiveAccumulatorSize> stm{};
    alignas(64) std::array<
        std::int32_t,
        PhaseQuantizedNnueModel::PerspectiveAccumulatorSize> opponent{};
    std::size_t aux_state = 0;
    std::size_t phase_index = 0;
};

struct PhaseForwardBenchmarkBatch {
    bool supported = false;
    std::uint64_t elapsed_ns = 0;
    std::uint64_t cpu_ns = 0;
    std::uint64_t checksum = 0;
    std::uint64_t timed_sink = 0;
};

#endif

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

#if defined(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)
    [[nodiscard]] virtual PhaseForwardBenchmarkBatch benchmark_stage(
        PhaseForwardBenchmarkStage,
        const std::vector<PhaseForwardBenchmarkSample>&,
        std::size_t
    ) const {
        return {};
    }
#endif
};

namespace phase_nnue_detail {

#if defined(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)

[[nodiscard]] inline std::uint64_t phase_benchmark_mix(
    std::uint64_t state,
    std::uint64_t value
) {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
    return state;
}

template<typename T>
[[gnu::always_inline]] inline void phase_benchmark_consume_all(
    const T& value
) {
    // The benchmark consumes one selected lane in its cheap timed sink.  This
    // opaque memory operand additionally makes every output byte observable to
    // the compiler, preventing dead-lane elimination without executing a
    // per-lane checksum in the measured loop.
    asm volatile("" : : "m"(value) : "memory");
}

#endif

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
