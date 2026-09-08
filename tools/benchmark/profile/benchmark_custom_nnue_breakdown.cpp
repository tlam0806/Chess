#include "quantized_position_network.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t FeatureCount = 768;
constexpr std::size_t Hidden1Size = 256;
constexpr std::size_t Hidden2Size = 32;
constexpr int Clip1Max = 255;
constexpr int Clip2Max = 255;
constexpr int Hidden2Scale = 64;
constexpr int OutputScale = 16;

using BenchNetwork = QuantizedPositionNetwork<
    Hidden1Size,
    Hidden2Size,
    Clip1Max,
    Hidden2Scale,
    Clip2Max,
    OutputScale>;

struct ModelData {
    std::vector<std::int32_t> hidden1_bias;
    std::vector<std::int16_t> input_weight;
    std::vector<std::int32_t> hidden2_bias;
    std::vector<std::int16_t> hidden2_weight;
    std::vector<std::int16_t> output_weight;
    std::int64_t output_bias = 0;
};

struct BenchResult {
    std::string_view name;
    double ns_per_op = 0.0;
    std::int64_t checksum = 0;
};

ModelData make_model(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> bias_dist(-128, 128);
    std::uniform_int_distribution<int> input_dist(-32, 32);
    std::uniform_int_distribution<int> output_dist(-32, 32);
    std::uniform_int_distribution<int> output_bias_dist(-4096, 4096);

    ModelData model;
    model.hidden1_bias.resize(Hidden1Size);
    model.input_weight.resize(FeatureCount * Hidden1Size);
    model.hidden2_bias.resize(Hidden2Size);
    model.hidden2_weight.resize(Hidden2Size * Hidden1Size);
    model.output_weight.resize(Hidden2Size);
    for (std::int32_t& value : model.hidden1_bias) {
        value = bias_dist(rng);
    }
    for (std::int16_t& value : model.input_weight) {
        value = static_cast<std::int16_t>(input_dist(rng));
    }
    for (std::int32_t& value : model.hidden2_bias) {
        value = bias_dist(rng);
    }
    for (std::int16_t& value : model.hidden2_weight) {
        value = static_cast<std::int16_t>(output_dist(rng));
    }
    for (std::int16_t& value : model.output_weight) {
        value = static_cast<std::int16_t>(output_dist(rng));
    }
    model.output_bias = output_bias_dist(rng);
    return model;
}

BenchNetwork make_network(const ModelData& model) {
    return BenchNetwork(
        model.output_bias,
        model.hidden2_bias,
        model.hidden2_weight,
        model.output_weight);
}

std::array<std::int32_t, Hidden1Size> make_acc1_after_activations(const ModelData& model) {
    std::array<std::int32_t, Hidden1Size> acc1{};
    std::copy(model.hidden1_bias.begin(), model.hidden1_bias.end(), acc1.begin());
    for (std::size_t feature = 0; feature < FeatureCount; feature += 3) {
        const std::size_t offset = feature * Hidden1Size;
        for (std::size_t j = 0; j < Hidden1Size; ++j) {
            acc1[j] += model.input_weight[offset + j];
        }
    }
    return acc1;
}

[[gnu::noinline]] std::int32_t clamp_hidden1_stage(
    const std::array<std::int32_t, Hidden1Size>& acc1,
    std::array<std::int16_t, Hidden1Size>& hidden1) {
    for (std::size_t j = 0; j < Hidden1Size; ++j) {
        hidden1[j] = static_cast<std::int16_t>(
            std::clamp<std::int32_t>(acc1[j], 0, Clip1Max));
    }
    return hidden1[0] + hidden1[Hidden1Size - 1];
}

[[gnu::noinline]] std::int32_t layer2_dot_stage(
    const std::array<std::int16_t, Hidden1Size>& hidden1,
    const ModelData& model,
    std::array<std::int32_t, Hidden2Size>& acc2) {
    for (std::size_t k = 0; k < Hidden2Size; ++k) {
        std::int32_t acc2_k = model.hidden2_bias[k];
        const std::size_t offset = k * Hidden1Size;
        for (std::size_t j = 0; j < Hidden1Size; ++j) {
            acc2_k += hidden1[j] * model.hidden2_weight[offset + j];
        }
        acc2[k] = acc2_k;
    }
    return acc2[0] + acc2[Hidden2Size - 1];
}

[[gnu::noinline]] int output_stage(
    const std::array<std::int32_t, Hidden2Size>& acc2,
    const ModelData& model) {
    std::int64_t raw = model.output_bias;
    for (std::size_t k = 0; k < Hidden2Size; ++k) {
        const std::int32_t hidden2_k =
            std::clamp<std::int32_t>(acc2[k] / Hidden2Scale, 0, Clip2Max);
        raw += hidden2_k * model.output_weight[k];
    }
    return static_cast<int>(raw / OutputScale);
}

[[gnu::noinline]] int reference_evaluate(
    const std::array<std::int32_t, Hidden1Size>& acc1,
    const ModelData& model,
    std::array<std::int16_t, Hidden1Size>& hidden1,
    std::array<std::int32_t, Hidden2Size>& acc2) {
    (void)clamp_hidden1_stage(acc1, hidden1);
    (void)layer2_dot_stage(hidden1, model, acc2);
    return output_stage(acc2, model);
}

[[gnu::noinline]] std::int32_t clamp_and_layer2_stage(
    const std::array<std::int32_t, Hidden1Size>& acc1,
    const ModelData& model,
    std::array<std::int16_t, Hidden1Size>& hidden1,
    std::array<std::int32_t, Hidden2Size>& acc2) {
    for (std::size_t j = 0; j < Hidden1Size; ++j) {
        hidden1[j] = static_cast<std::int16_t>(
            std::clamp<std::int32_t>(acc1[j], 0, Clip1Max));
    }
    for (std::size_t k = 0; k < Hidden2Size; ++k) {
        std::int32_t acc2_k = model.hidden2_bias[k];
        const std::size_t offset = k * Hidden1Size;
        for (std::size_t j = 0; j < Hidden1Size; ++j) {
            acc2_k += hidden1[j] * model.hidden2_weight[offset + j];
        }
        acc2[k] = acc2_k;
    }
    return acc2[0] + acc2[Hidden2Size - 1];
}

template <typename Fn>
BenchResult run_bench(std::string_view name, std::uint64_t iterations, Fn&& fn) {
    std::vector<double> samples;
    std::vector<std::int64_t> checksums;
    samples.reserve(9);
    checksums.reserve(9);

    for (int run = 0; run < 9; ++run) {
        std::int64_t checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::uint64_t i = 0; i < iterations; ++i) {
            checksum += fn(i);
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        samples.push_back(static_cast<double>(elapsed) / static_cast<double>(iterations));
        checksums.push_back(checksum);
    }

    std::sort(samples.begin(), samples.end());
    return BenchResult{name, samples[samples.size() / 2], checksums.back()};
}

void print_result(const BenchResult& result) {
    std::cout << result.name
              << " ns_per_op=" << result.ns_per_op
              << " checksum=" << result.checksum
              << '\n';
}

} // namespace

int main() {
    const ModelData model = make_model(0xBEEFBEEFu);
    BenchNetwork network = make_network(model);

    const std::array<std::int32_t, Hidden1Size> acc1 = make_acc1_after_activations(model);
    std::array<std::int16_t, Hidden1Size> hidden1{};
    std::array<std::int32_t, Hidden2Size> acc2{};

    const int nnue_score = network.evaluate(acc1);
    const int reference_score = reference_evaluate(acc1, model, hidden1, acc2);
    if (nnue_score != reference_score) {
        std::cerr << "correctness mismatch: nnue=" << nnue_score
                  << " reference=" << reference_score << '\n';
        return 1;
    }

    std::cout << "reference_score=" << reference_score << '\n';

    print_result(run_bench("full_nnue_evaluate", 2'000'000, [&](std::uint64_t) {
        return network.evaluate(acc1);
    }));
    std::array<std::int32_t, Hidden1Size> acc1_for_full = acc1;
    print_result(run_bench("reference_full_evaluate", 2'000'000, [&](std::uint64_t i) {
        acc1_for_full[0] += (i & 1U) == 0U ? 1 : -1;
        return reference_evaluate(acc1_for_full, model, hidden1, acc2);
    }));
    std::array<std::int32_t, Hidden1Size> acc1_for_stage1 = acc1;
    print_result(run_bench("stage1_clamp_acc1_to_hidden1", 5'000'000, [&](std::uint64_t i) {
        acc1_for_stage1[0] += (i & 1U) == 0U ? 1 : -1;
        return clamp_hidden1_stage(acc1_for_stage1, hidden1);
    }));
    (void)clamp_hidden1_stage(acc1, hidden1);
    std::array<std::int16_t, Hidden1Size> hidden1_for_stage2 = hidden1;
    print_result(run_bench("stage2_layer2_dot", 2'000'000, [&](std::uint64_t i) {
        hidden1_for_stage2[0] =
            static_cast<std::int16_t>(hidden1_for_stage2[0] + ((i & 1U) == 0U ? 1 : -1));
        return layer2_dot_stage(hidden1_for_stage2, model, acc2);
    }));
    std::array<std::int32_t, Hidden1Size> acc1_for_stage12 = acc1;
    print_result(run_bench("stage1_plus_stage2", 2'000'000, [&](std::uint64_t i) {
        acc1_for_stage12[0] += (i & 1U) == 0U ? 1 : -1;
        return clamp_and_layer2_stage(acc1_for_stage12, model, hidden1, acc2);
    }));
    (void)layer2_dot_stage(hidden1, model, acc2);
    std::array<std::int32_t, Hidden2Size> acc2_for_stage3 = acc2;
    print_result(run_bench("stage3_hidden2_output", 10'000'000, [&](std::uint64_t i) {
        acc2_for_stage3[0] += (i & 1U) == 0U ? Hidden2Scale : -Hidden2Scale;
        return output_stage(acc2_for_stage3, model);
    }));
}
