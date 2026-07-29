#pragma once

#include "quantized_position_network.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

enum class NnuePerspective : std::uint8_t {
    White = 0,
    Black = 1,
};

// Chess-facing wrapper around QuantizedPositionNetwork.
//
// The caller updates each normalized perspective independently through the
// compile-time activate/deactivate<Perspective>() API. The wrapper owns both
// incremental accumulators, both PSQT accumulators, feature state, phase, and
// undo history. Only evaluate() concatenates [STM, opponent] and calls the
// feature-agnostic dense network.
template<
    std::size_t PerspectiveFeatureCount,
    std::size_t PerspectiveAccumulatorSize,
    std::size_t HiddenSize,
    int AccumulatorClipMax,
    int HiddenScale,
    int HiddenClipMax,
    int OutputScale
>
class IncrementalNnue {
public:
    static constexpr std::size_t PerspectiveCount = 2;
    static constexpr std::size_t PsqtBucketCount = 8;
    static constexpr std::size_t NetworkAccumulatorSize = 2 * PerspectiveAccumulatorSize;

    using PerspectiveAccumulator =
        std::array<std::int32_t, PerspectiveAccumulatorSize>;
    using PsqtAccumulator = std::array<std::int32_t, PsqtBucketCount>;
    using Network = QuantizedPositionNetwork<
        NetworkAccumulatorSize,
        HiddenSize,
        AccumulatorClipMax,
        HiddenScale,
        HiddenClipMax,
        OutputScale>;

private:
    static constexpr std::int32_t PsqtWeightAbsMax = 33'554'428;

    struct HistoryEntry {
        NnuePerspective perspective = NnuePerspective::White;
        std::size_t feature = 0;
    };

    Network network_;
    std::int32_t psqt_scale_ = 1;

    alignas(64) std::array<std::array<std::uint8_t, PerspectiveFeatureCount>,
                           PerspectiveCount> feature_state_{};
    alignas(64) std::array<PerspectiveAccumulator, PerspectiveCount> accumulator_{};

    // Both perspectives use the same normalized feature table.
    alignas(64) std::array<std::int16_t,
                           PerspectiveFeatureCount * PerspectiveAccumulatorSize>
        feature_weight_{};
    alignas(64) std::array<std::int32_t,
                           PerspectiveFeatureCount * PsqtBucketCount>
        psqt_weight_{};
    alignas(64) std::array<PsqtAccumulator, PerspectiveCount> psqt_accumulator_{};

    std::array<std::size_t, PerspectiveCount> active_feature_count_{};
    std::vector<HistoryEntry> feature_history_;
    std::vector<std::size_t> checkpoints_;

    [[nodiscard]] static constexpr std::size_t index(NnuePerspective perspective) {
        return static_cast<std::size_t>(perspective);
    }

    template<NnuePerspective Perspective>
    [[nodiscard]] static consteval std::size_t perspective_index() {
        if constexpr (Perspective == NnuePerspective::White) {
            return 0;
        } else {
            static_assert(Perspective == NnuePerspective::Black);
            return 1;
        }
    }

    [[nodiscard]] static constexpr std::size_t psqt_bucket(std::size_t piece_count) {
        return piece_count == 0
            ? 0
            : std::min<std::size_t>((piece_count - 1) / 4, PsqtBucketCount - 1);
    }

    template<NnuePerspective Perspective>
    void update_perspective(std::size_t feature, int sign) {
        assert(feature < PerspectiveFeatureCount);
        constexpr std::size_t perspective_index =
            IncrementalNnue::perspective_index<Perspective>();
        const std::size_t feature_offset = feature * PerspectiveAccumulatorSize;
        for (std::size_t lane = 0; lane < PerspectiveAccumulatorSize; ++lane) {
            const std::int64_t next =
                static_cast<std::int64_t>(accumulator_[perspective_index][lane])
                + static_cast<std::int64_t>(sign) * feature_weight_[feature_offset + lane];
            assert(next >= INT32_MIN && next <= INT32_MAX);
            accumulator_[perspective_index][lane] = static_cast<std::int32_t>(next);
        }

        const std::size_t psqt_offset = feature * PsqtBucketCount;
        for (std::size_t bucket = 0; bucket < PsqtBucketCount; ++bucket) {
            const std::int64_t next =
                static_cast<std::int64_t>(psqt_accumulator_[perspective_index][bucket])
                + static_cast<std::int64_t>(sign) * psqt_weight_[psqt_offset + bucket];
            assert(next >= INT32_MIN && next <= INT32_MAX);
            psqt_accumulator_[perspective_index][bucket] = static_cast<std::int32_t>(next);
        }
    }

    template<NnuePerspective Perspective>
    void toggle_without_history(std::size_t feature) {
        constexpr std::size_t perspective_index =
            IncrementalNnue::perspective_index<Perspective>();
        const bool active = feature_state_[perspective_index][feature] != 0;
        const int sign = active ? -1 : 1;
        assert(sign > 0 || active_feature_count_[perspective_index] > 0);

        feature_state_[perspective_index][feature] = !active;
        update_perspective<Perspective>(feature, sign);
        active_feature_count_[perspective_index] = static_cast<std::size_t>(
            static_cast<std::int64_t>(active_feature_count_[perspective_index]) + sign);
    }

public:
    IncrementalNnue(
        std::int64_t output_bias,
        const std::vector<std::int32_t>& accumulator_bias,
        const std::vector<std::int16_t>& feature_weight,
        const std::vector<std::int32_t>& hidden_bias,
        const std::vector<std::int16_t>& hidden_weight,
        const std::vector<std::int16_t>& output_weight,
        std::int32_t psqt_scale,
        const std::vector<std::int32_t>& psqt_weight)
        : network_(output_bias, hidden_bias, hidden_weight, output_weight),
          psqt_scale_(psqt_scale) {
        static_assert(PerspectiveFeatureCount > 0);
        static_assert(PerspectiveAccumulatorSize > 0);
        assert(accumulator_bias.size() == PerspectiveAccumulatorSize);
        assert(
            feature_weight.size()
            == PerspectiveFeatureCount * PerspectiveAccumulatorSize);
        assert(psqt_scale > 0);
        assert(psqt_weight.size() == PerspectiveFeatureCount * PsqtBucketCount);

        std::copy(accumulator_bias.begin(), accumulator_bias.end(), accumulator_[0].begin());
        std::copy(accumulator_bias.begin(), accumulator_bias.end(), accumulator_[1].begin());
        std::copy(feature_weight.begin(), feature_weight.end(), feature_weight_.begin());
        for (std::int32_t weight : psqt_weight) {
            // At most 64 physical pieces can contribute to either perspective.
            assert(weight >= -PsqtWeightAbsMax && weight <= PsqtWeightAbsMax);
        }
        std::copy(psqt_weight.begin(), psqt_weight.end(), psqt_weight_.begin());
    }

    template<NnuePerspective Perspective>
    void activate(std::size_t feature) {
        static_assert(
            Perspective == NnuePerspective::White
                || Perspective == NnuePerspective::Black);
        constexpr std::size_t perspective_index =
            IncrementalNnue::perspective_index<Perspective>();
        assert(feature < PerspectiveFeatureCount);
        if (feature_state_[perspective_index][feature] != 0) {
            return;
        }
        feature_history_.push_back(HistoryEntry{Perspective, feature});
        toggle_without_history<Perspective>(feature);
    }

    template<NnuePerspective Perspective>
    void deactivate(std::size_t feature) {
        static_assert(
            Perspective == NnuePerspective::White
                || Perspective == NnuePerspective::Black);
        constexpr std::size_t perspective_index =
            IncrementalNnue::perspective_index<Perspective>();
        assert(feature < PerspectiveFeatureCount);
        if (feature_state_[perspective_index][feature] == 0) {
            return;
        }
        feature_history_.push_back(HistoryEntry{Perspective, feature});
        toggle_without_history<Perspective>(feature);
    }

    [[nodiscard]] int evaluate(NnuePerspective side_to_move) const {
        assert(active_feature_count_[0] == active_feature_count_[1]);
        typename Network::Accumulator concatenated{};
        const std::size_t stm = index(side_to_move);
        const std::size_t opponent = 1 - stm;
        std::copy(
            accumulator_[stm].begin(),
            accumulator_[stm].end(),
            concatenated.begin());
        std::copy(
            accumulator_[opponent].begin(),
            accumulator_[opponent].end(),
            concatenated.begin() + PerspectiveAccumulatorSize);

        const std::int64_t positional = network_.evaluate(concatenated);
        const std::size_t bucket = psqt_bucket(active_feature_count_[0]);
        const std::int64_t psqt = (
            static_cast<std::int64_t>(psqt_accumulator_[stm][bucket])
            - static_cast<std::int64_t>(psqt_accumulator_[opponent][bucket])
        ) / (2LL * psqt_scale_);
        return static_cast<int>(positional + psqt);
    }

    void push() {
        checkpoints_.push_back(feature_history_.size());
    }

    void pop() {
        assert(!checkpoints_.empty());
        const std::size_t checkpoint = checkpoints_.back();
        checkpoints_.pop_back();
        while (feature_history_.size() > checkpoint) {
            const HistoryEntry entry = feature_history_.back();
            feature_history_.pop_back();
            if (entry.perspective == NnuePerspective::White) {
                toggle_without_history<NnuePerspective::White>(entry.feature);
            } else {
                toggle_without_history<NnuePerspective::Black>(entry.feature);
            }
        }
    }

    [[nodiscard]] std::size_t piece_count() const {
        assert(active_feature_count_[0] == active_feature_count_[1]);
        return active_feature_count_[0];
    }

    [[nodiscard]] const std::array<PerspectiveAccumulator, PerspectiveCount>&
    positional_accumulators() const {
        return accumulator_;
    }

    [[nodiscard]] const std::array<PsqtAccumulator, PerspectiveCount>&
    psqt_accumulators() const {
        return psqt_accumulator_;
    }
};
