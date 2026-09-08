#include "incremental_nnue.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    constexpr std::size_t PerspectiveFeatureCount = 16;
    constexpr std::size_t PerspectiveAccumulatorSize = 1;
    constexpr std::size_t NetworkAccumulatorSize = 2;
    constexpr std::size_t HiddenSize = 1;

    using PositionNetwork = QuantizedPositionNetwork<
        NetworkAccumulatorSize, HiddenSize, 255, 1, 255, 1>;
    using TestNnue = IncrementalNnue<
        PerspectiveFeatureCount,
        PerspectiveAccumulatorSize,
        HiddenSize,
        255,
        1,
        255,
        1>;

    // The dense network consumes only a fully assembled accumulator. It has
    // no feature, side-to-move, phase, PSQT, or history API/state.
    const std::vector<std::int32_t> hidden_bias{100};
    const std::vector<std::int16_t> hidden_weight{1, -1};
    const std::vector<std::int16_t> output_weight{1};
    PositionNetwork network(-100, hidden_bias, hidden_weight, output_weight);
    assert(network.evaluate(PositionNetwork::Accumulator{15, 40}) == -25);
    assert(network.evaluate(PositionNetwork::Accumulator{40, 15}) == 25);

    std::vector<std::int32_t> accumulator_bias(PerspectiveAccumulatorSize, 0);
    std::vector<std::int16_t> feature_weight(
        PerspectiveFeatureCount * PerspectiveAccumulatorSize, 0);
    for (std::size_t feature = 0; feature < PerspectiveFeatureCount; ++feature) {
        feature_weight[feature] = static_cast<std::int16_t>(feature + 1);
    }

    constexpr std::int32_t PsqtScale = 4;
    std::vector<std::int32_t> psqt_weight(PerspectiveFeatureCount * 8, 0);
    for (std::size_t feature = 0; feature < PerspectiveFeatureCount; ++feature) {
        for (std::size_t bucket = 0; bucket < 8; ++bucket) {
            psqt_weight[feature * 8 + bucket] = static_cast<std::int32_t>(
                (feature + 1) * (bucket + 1) * PsqtScale);
        }
    }

    TestNnue model(
        -100,
        accumulator_bias,
        feature_weight,
        hidden_bias,
        hidden_weight,
        output_weight,
        PsqtScale,
        psqt_weight);

    // Each physical piece supplies one feature row for each perspective. Each
    // compile-time call touches only the selected accumulator and PSQT vector.
    for (std::size_t piece = 0; piece < 5; ++piece) {
        model.activate<NnuePerspective::White>(piece);
        model.activate<NnuePerspective::Black>(5 + piece);
    }
    assert(model.piece_count() == 5);
    const auto& positional = model.positional_accumulators();
    const auto& psqt = model.psqt_accumulators();
    assert(positional[0][0] == 15);
    assert(positional[1][0] == 40);
    for (std::size_t bucket = 0; bucket < 8; ++bucket) {
        assert(psqt[0][bucket] == static_cast<std::int32_t>(
            15 * (bucket + 1) * PsqtScale));
        assert(psqt[1][bucket] == static_cast<std::int32_t>(
            40 * (bucket + 1) * PsqtScale));
    }

    // With five pieces bucket 1 is selected. Positional and PSQT are both -25
    // from White's perspective, and both reverse when Black is to move.
    assert(model.evaluate(NnuePerspective::White) == -50);
    assert(model.evaluate(NnuePerspective::Black) == 50);

    model.push();
    model.deactivate<NnuePerspective::White>(4);
    model.deactivate<NnuePerspective::Black>(9);
    assert(model.piece_count() == 4);
    assert(model.positional_accumulators()[0][0] == 10);
    assert(model.positional_accumulators()[1][0] == 30);
    assert(model.evaluate(NnuePerspective::White) == -30);
    model.pop();

    assert(model.piece_count() == 5);
    assert(model.positional_accumulators()[0][0] == 15);
    assert(model.positional_accumulators()[1][0] == 40);
    assert(model.evaluate(NnuePerspective::White) == -50);
}
