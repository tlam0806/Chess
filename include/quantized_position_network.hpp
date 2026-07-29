#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

// Dense, stateless part of the quantized evaluator.
//
// This class intentionally knows nothing about chess features, perspectives,
// incremental updates, phase buckets, or PSQT.  Its only input is the already
// concatenated [side-to-move, opponent] accumulator.
template<
    std::size_t AccumulatorSize,
    std::size_t HiddenSize,
    int AccumulatorClipMax,
    int HiddenScale,
    int HiddenClipMax,
    int OutputScale
>
class QuantizedPositionNetwork {
public:
    using Accumulator = std::array<std::int32_t, AccumulatorSize>;

private:
    std::int64_t output_bias_ = 0;

    alignas(64) std::array<std::int32_t, HiddenSize> hidden_bias_{};
    // Layout: hidden_weight_[hidden_neuron * AccumulatorSize + input_lane].
    alignas(64) std::array<std::int16_t, HiddenSize * AccumulatorSize> hidden_weight_{};
    alignas(64) std::array<std::int16_t, HiddenSize> output_weight_{};

public:
    QuantizedPositionNetwork(
        std::int64_t output_bias,
        const std::vector<std::int32_t>& hidden_bias,
        const std::vector<std::int16_t>& hidden_weight,
        const std::vector<std::int16_t>& output_weight)
        : output_bias_(output_bias) {
        static_assert(AccumulatorSize > 0);
        static_assert(HiddenSize > 0);
        static_assert(AccumulatorClipMax > 0);
        static_assert(AccumulatorClipMax <= INT16_MAX);
        static_assert(HiddenScale > 0);
        static_assert(HiddenClipMax > 0);
        static_assert(OutputScale > 0);

        assert(hidden_bias.size() == HiddenSize);
        assert(hidden_weight.size() == HiddenSize * AccumulatorSize);
        assert(output_weight.size() == HiddenSize);

        std::copy(hidden_bias.begin(), hidden_bias.end(), hidden_bias_.begin());
        std::copy(hidden_weight.begin(), hidden_weight.end(), hidden_weight_.begin());
        std::copy(output_weight.begin(), output_weight.end(), output_weight_.begin());
    }

    [[nodiscard]] int evaluate(const Accumulator& accumulator) const {
        std::array<std::int16_t, AccumulatorSize> activated_input{};
        for (std::size_t lane = 0; lane < AccumulatorSize; ++lane) {
            activated_input[lane] = static_cast<std::int16_t>(
                std::clamp<std::int32_t>(
                    accumulator[lane], 0, AccumulatorClipMax));
        }

        std::int64_t raw = output_bias_;
        for (std::size_t hidden = 0; hidden < HiddenSize; ++hidden) {
            std::int64_t sum = hidden_bias_[hidden];
            const std::size_t offset = hidden * AccumulatorSize;
            for (std::size_t lane = 0; lane < AccumulatorSize; ++lane) {
                sum += static_cast<std::int64_t>(activated_input[lane])
                    * hidden_weight_[offset + lane];
            }
            const std::int64_t activated = std::clamp<std::int64_t>(
                sum / HiddenScale, 0, HiddenClipMax);
            raw += activated * output_weight_[hidden];
        }
        return static_cast<int>(raw / OutputScale);
    }
};
