#pragma once

#include "move.hpp"
#include "position.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace chess {

class PhaseCandidateKernelBase;
template<int Hidden2Scale, int Hidden3Scale, int OutputScale>
class PhaseCandidateKernel;

inline constexpr std::string_view DefaultPhaseQuantizedNnueModelPath =
    "models/quantized_scale_grid/"
    "old_score_huber200_lr_sweep_then_5ep_20260724_142758/"
    "best/phase_quantized_nnue.bin";

class PhaseQuantizedNnueAccumulator;

class PhaseQuantizedNnueModel {
public:
    static constexpr std::size_t FeatureRowCount = 6 * 2 * 64 * 64;
    static constexpr std::size_t PerspectiveAccumulatorSize = 128;
    static constexpr std::size_t DenseInputSize = 2 * PerspectiveAccumulatorSize;
    static constexpr std::size_t Hidden2Size = 32;
    static constexpr std::size_t Hidden3Size = 32;
    static constexpr std::size_t PhaseCount = 8;
    static constexpr std::size_t PsqtBucketCount = 8;
    static constexpr std::size_t AuxFeatureCount = 13;

    PhaseQuantizedNnueModel();
    ~PhaseQuantizedNnueModel();
    PhaseQuantizedNnueModel(const PhaseQuantizedNnueModel&) = delete;
    PhaseQuantizedNnueModel& operator=(const PhaseQuantizedNnueModel&) = delete;

    bool load(std::string_view path);

    [[nodiscard]] bool loaded() const {
        return feature_rows_.size() == FeatureRowCount;
    }

    [[nodiscard]] int evaluate_cp_rounded(const Position& pos) const;

    [[nodiscard]] std::uint32_t hidden_clip() const { return hidden_clip_; }
    [[nodiscard]] std::uint32_t screlu_divisor() const { return screlu_divisor_; }
    [[nodiscard]] std::uint32_t hidden2_scale() const { return hidden2_scale_; }
    [[nodiscard]] std::uint32_t hidden3_scale() const { return hidden3_scale_; }
    [[nodiscard]] std::uint32_t output_scale() const { return output_scale_; }
    [[nodiscard]] std::uint32_t psqt_scale() const { return psqt_scale_; }
    [[nodiscard]] static bool supports_candidate_configuration(
        std::uint32_t hidden_clip,
        std::uint32_t screlu_divisor,
        std::uint32_t hidden2_scale,
        std::uint32_t hidden3_scale,
        std::uint32_t output_scale
    );
    [[nodiscard]] bool has_candidate_kernel() const {
        return candidate_kernel_ != nullptr;
    }
    [[nodiscard]] bool uses_neon_dotprod_kernel() const;
    void set_neon_dotprod_enabled(bool enabled) {
        neon_dotprod_enabled_ = enabled;
    }

private:
    friend class PhaseQuantizedNnueAccumulator;
    template<int Hidden2Scale, int Hidden3Scale, int OutputScale>
    friend class PhaseCandidateKernel;

    struct alignas(64) DensePhase {
        std::array<std::int32_t, Hidden2Size> hidden2_bias{};
        std::array<std::int8_t, Hidden2Size * DenseInputSize> hidden2_weight{};
        std::array<std::int32_t, Hidden3Size> hidden3_bias{};
        std::array<std::int8_t, Hidden3Size * Hidden2Size> hidden3_weight{};
        std::int64_t output_bias = 0;
        std::array<std::int8_t, Hidden3Size> output_weight{};
    };

    // Keep every sparse feature's positional and PSQT payload together.
    // The 32-byte alignment and 160-byte stride also keep every row naturally
    // aligned for the NEON loads used by the incremental accumulator.
    struct alignas(32) FeatureRow {
        std::array<std::int8_t, PerspectiveAccumulatorSize> positional{};
        std::array<std::int32_t, PsqtBucketCount> psqt{};
    };
    static_assert(sizeof(FeatureRow) == 160);

    [[nodiscard]] std::int64_t forward_positional_scalar(
        const std::array<std::int32_t, DenseInputSize>& dense_input,
        std::size_t phase_index
    ) const;
    [[nodiscard]] std::int64_t forward_positional_candidate(
        const std::array<std::int32_t, PerspectiveAccumulatorSize>&
            stm_accumulator,
        const std::array<std::int32_t, PerspectiveAccumulatorSize>&
            opponent_accumulator,
        const std::array<std::uint8_t, AuxFeatureCount>& aux,
        std::size_t phase_index
    ) const;

    void initialize_candidate_kernel();

    [[nodiscard]] int evaluate(
        const Position& pos,
        const std::array<
            std::array<std::int32_t, PerspectiveAccumulatorSize>, 2>& accumulators,
        const std::array<
            std::array<std::int32_t, PsqtBucketCount>, 2>& psqt_accumulators,
        std::size_t piece_count
    ) const;

    std::uint32_t hidden_clip_ = 0;
    std::uint32_t screlu_divisor_ = 0;
    std::uint32_t hidden2_scale_ = 0;
    std::uint32_t hidden3_scale_ = 0;
    std::uint32_t output_scale_ = 0;
    std::uint32_t feature_weight_scale_ = 0;
    std::uint32_t linear_weight_scale_ = 0;
    std::uint32_t output_weight_scale_ = 0;
    std::uint32_t psqt_scale_ = 0;
    bool neon_dotprod_enabled_ = true;

    std::array<std::int32_t, PerspectiveAccumulatorSize> accumulator_bias_{};
    std::vector<FeatureRow> feature_rows_;
    std::array<std::int8_t, AuxFeatureCount * DenseInputSize> aux_weights_{};
    std::array<DensePhase, PhaseCount> phases_{};
    std::unique_ptr<PhaseCandidateKernelBase> candidate_kernel_;
};

struct PhaseQuantizedNnueUndo {
    std::size_t ply_before;
};

class PhaseQuantizedNnueAccumulator {
public:
    using PerspectiveAccumulator = std::array<
        std::int32_t, PhaseQuantizedNnueModel::PerspectiveAccumulatorSize>;
    using PsqtAccumulator = std::array<
        std::int32_t, PhaseQuantizedNnueModel::PsqtBucketCount>;

    void reset(const PhaseQuantizedNnueModel& model, const Position& pos);

    [[nodiscard]] PhaseQuantizedNnueUndo make_move_with_undo(
        Move move,
        Color moving_color,
        PieceType moved_piece,
        PieceType captured_piece,
        const Position& after
    );

    void undo(const PhaseQuantizedNnueUndo& undo);

    [[nodiscard]] int evaluate_cp_rounded(const Position& pos) const;
    [[nodiscard]] bool matches_full_recompute(const Position& pos) const;

    [[nodiscard]] bool initialized() const {
        return model_ != nullptr && !states_.empty();
    }
    [[nodiscard]] std::size_t piece_count() const {
        return current_state().piece_count;
    }

    [[nodiscard]] const std::array<PerspectiveAccumulator, 2>&
    positional_accumulators() const {
        return current_state().accumulators;
    }

    [[nodiscard]] const std::array<PsqtAccumulator, 2>& psqt_accumulators() const {
        return current_state().psqt;
    }

private:
    struct NnueState {
        std::array<PerspectiveAccumulator, 2> accumulators;
        std::array<PsqtAccumulator, 2> psqt;
        std::array<Square, 2> king_squares;
        std::size_t piece_count;
    };

    static constexpr std::size_t PreallocatedStateCount = 256;

    [[nodiscard]] NnueState& current_state() {
        assert(current_ply_ < states_.size());
        return states_[current_ply_];
    }
    [[nodiscard]] const NnueState& current_state() const {
        assert(current_ply_ < states_.size());
        return states_[current_ply_];
    }
    void push_state();

    void rebuild_perspective(const Position& pos, Color perspective);
    template<std::size_t AddedCount, std::size_t RemovedCount>
    void update_features(
        Color perspective,
        const std::array<std::size_t, AddedCount>& added,
        const std::array<std::size_t, RemovedCount>& removed
    );
    [[nodiscard]] std::size_t feature_row(
        Color perspective,
        Color piece_color,
        PieceType piece,
        Square square
    ) const;

    const PhaseQuantizedNnueModel* model_ = nullptr;
    std::vector<NnueState> states_;
    std::size_t current_ply_ = 0;
};

} // namespace chess
