#pragma once

#include "position.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chess {

class PhaseQuantizedNnueModel;

struct PhaseNnueStageTiming {
    std::string name;
    std::size_t evaluations_per_repeat = 0;
    std::vector<std::uint64_t> elapsed_ns;
    std::vector<std::uint64_t> cpu_ns;
    std::uint64_t checksum = 0;
    std::uint64_t timed_sink = 0;
};

struct PhaseNnueStageBatchTiming {
    std::size_t repeat = 0;
    std::size_t order = 0;
    std::string stage;
    std::size_t evaluations = 0;
    std::uint64_t elapsed_ns = 0;
    std::uint64_t cpu_ns = 0;
};

struct PhaseNnueStageBenchmarkResult {
    std::string kernel;
    std::size_t sample_count = 0;
    std::uint64_t corpus_checksum = 0;
    std::array<std::size_t, 8> phase_counts{};
    std::size_t nonzero_aux_samples = 0;
    bool scalar_parity = false;
    bool stage_parity = false;
    bool timed_sink_parity = false;
    std::vector<PhaseNnueStageTiming> stages;
    std::vector<PhaseNnueStageBatchTiming> batches;
};

[[nodiscard]] PhaseNnueStageBenchmarkResult benchmark_phase_nnue_forward_stages(
    const PhaseQuantizedNnueModel& model,
    const std::vector<Position>& positions,
    std::size_t warmup_evaluations,
    const std::array<std::size_t, 4>& evaluations_per_stage,
    std::size_t repeats
);

} // namespace chess
