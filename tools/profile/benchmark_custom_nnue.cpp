#include "incremental_nnue.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t PerspectiveFeatureCount = 768;
constexpr std::size_t PerspectiveAccumulatorSize = 128;
constexpr std::size_t NetworkAccumulatorSize = 2 * PerspectiveAccumulatorSize;
constexpr std::size_t Hidden2Size = 32;

using BenchNnue = IncrementalNnue<
    PerspectiveFeatureCount,
    PerspectiveAccumulatorSize,
    Hidden2Size,
    255,
    64,
    255,
    16>;

struct ModelData {
    std::vector<int32_t> hidden1_bias;
    std::vector<int16_t> input_weight;
    std::vector<int32_t> hidden2_bias;
    std::vector<int16_t> hidden2_weight;
    std::vector<int16_t> output_weight;
    std::vector<int32_t> psqt_weight;
    int64_t output_bias = 0;
};

struct BenchResult {
    std::string_view name;
    std::uint64_t operations = 0;
    std::uint64_t us = 0;
    std::int64_t checksum = 0;
};

enum class OpKind : std::uint8_t {
    Activate,
    Deactivate,
    Evaluate,
    Push,
    Pop
};

struct Op {
    OpKind kind = OpKind::Evaluate;
    std::size_t feature = 0;
};

ModelData make_model(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> bias_dist(-128, 128);
    std::uniform_int_distribution<int> input_dist(-32, 32);
    std::uniform_int_distribution<int> output_dist(-32, 32);
    std::uniform_int_distribution<int> output_bias_dist(-4096, 4096);

    ModelData model;
    model.hidden1_bias.resize(PerspectiveAccumulatorSize);
    model.input_weight.resize(PerspectiveFeatureCount * PerspectiveAccumulatorSize);
    model.hidden2_bias.resize(Hidden2Size);
    model.hidden2_weight.resize(Hidden2Size * NetworkAccumulatorSize);
    model.output_weight.resize(Hidden2Size);
    model.psqt_weight.resize(PerspectiveFeatureCount * 8);
    for (int32_t& value : model.hidden1_bias) {
        value = bias_dist(rng);
    }
    for (int16_t& value : model.input_weight) {
        value = static_cast<int16_t>(input_dist(rng));
    }
    for (int32_t& value : model.hidden2_bias) {
        value = bias_dist(rng);
    }
    for (int16_t& value : model.hidden2_weight) {
        value = static_cast<int16_t>(output_dist(rng));
    }
    for (int16_t& value : model.output_weight) {
        value = static_cast<int16_t>(output_dist(rng));
    }
    model.output_bias = output_bias_dist(rng);
    return model;
}

BenchNnue make_nnue(const ModelData& model) {
    return BenchNnue(
        model.output_bias,
        model.hidden1_bias,
        model.input_weight,
        model.hidden2_bias,
        model.hidden2_weight,
        model.output_weight,
        1,
        model.psqt_weight);
}

std::size_t black_feature(std::size_t feature) {
    return (feature * 37) % PerspectiveFeatureCount;
}

void activate_feature(BenchNnue& nnue, std::size_t feature) {
    nnue.activate<NnuePerspective::White>(feature);
    nnue.activate<NnuePerspective::Black>(black_feature(feature));
}

void deactivate_feature(BenchNnue& nnue, std::size_t feature) {
    nnue.deactivate<NnuePerspective::White>(feature);
    nnue.deactivate<NnuePerspective::Black>(black_feature(feature));
}

std::uint64_t elapsed_us(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count());
}

void print_result(const BenchResult& result) {
    const double seconds = static_cast<double>(result.us) / 1'000'000.0;
    const double mop_per_sec = seconds == 0.0
        ? 0.0
        : static_cast<double>(result.operations) / seconds / 1'000'000.0;
    const double ns_per_op = result.operations == 0
        ? 0.0
        : static_cast<double>(result.us) * 1000.0 / static_cast<double>(result.operations);

    std::cout << result.name
              << " ops=" << result.operations
              << " us=" << result.us
              << " ns_per_op=" << ns_per_op
              << " mop_per_sec=" << mop_per_sec
              << " checksum=" << result.checksum
              << '\n';
}

BenchResult bench_evaluate_only(std::uint64_t iterations) {
    const ModelData model = make_model(0xA17E0001u);
    BenchNnue nnue = make_nnue(model);
    for (std::size_t feature = 0; feature < PerspectiveFeatureCount; feature += 3) {
        activate_feature(nnue, feature);
    }

    std::int64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
        checksum += nnue.evaluate(NnuePerspective::White);
    }

    return BenchResult{"evaluate_only", iterations, elapsed_us(start), checksum};
}

BenchResult bench_activate_deactivate(std::uint64_t pairs) {
    const ModelData model = make_model(0xA17E0002u);
    BenchNnue nnue = make_nnue(model);

    std::int64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < pairs; ++i) {
        const std::size_t feature = static_cast<std::size_t>((i * 37) % PerspectiveFeatureCount);
        activate_feature(nnue, feature);
        deactivate_feature(nnue, feature);
        checksum += nnue.evaluate(NnuePerspective::White);
    }

    return BenchResult{"activate_deactivate_eval", pairs * 3, elapsed_us(start), checksum};
}

BenchResult bench_push_pop(std::uint64_t iterations) {
    const ModelData model = make_model(0xA17E0003u);
    BenchNnue nnue = make_nnue(model);

    std::int64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
        nnue.push();
        for (std::size_t offset = 0; offset < 8; ++offset) {
            const std::size_t feature = static_cast<std::size_t>(
                (i * 17 + offset * 43) % PerspectiveFeatureCount);
            activate_feature(nnue, feature);
        }
        checksum += nnue.evaluate(NnuePerspective::White);
        nnue.pop();
    }

    return BenchResult{"push_8activate_eval_pop", iterations * 11, elapsed_us(start), checksum};
}

std::vector<Op> make_mixed_trace(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> op_dist(0, 99);
    std::uniform_int_distribution<std::size_t> feature_dist(0, PerspectiveFeatureCount - 1);
    std::vector<Op> trace;
    trace.reserve(count);
    int stack_depth = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const int roll = op_dist(rng);
        if (roll < 34) {
            trace.push_back(Op{OpKind::Activate, feature_dist(rng)});
        } else if (roll < 68) {
            trace.push_back(Op{OpKind::Deactivate, feature_dist(rng)});
        } else if (roll < 82) {
            trace.push_back(Op{OpKind::Evaluate, 0});
        } else if (roll < 92) {
            trace.push_back(Op{OpKind::Push, 0});
            ++stack_depth;
        } else if (stack_depth > 0) {
            trace.push_back(Op{OpKind::Pop, 0});
            --stack_depth;
        } else {
            trace.push_back(Op{OpKind::Evaluate, 0});
        }
    }
    while (stack_depth-- > 0) {
        trace.push_back(Op{OpKind::Pop, 0});
    }
    return trace;
}

BenchResult bench_mixed_trace(std::size_t trace_size) {
    const ModelData model = make_model(0xA17E0004u);
    BenchNnue nnue = make_nnue(model);
    const std::vector<Op> trace = make_mixed_trace(trace_size, 0xA17E1000u);

    std::int64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const Op& op : trace) {
        switch (op.kind) {
        case OpKind::Activate:
            activate_feature(nnue, op.feature);
            break;
        case OpKind::Deactivate:
            deactivate_feature(nnue, op.feature);
            break;
        case OpKind::Evaluate:
            checksum += nnue.evaluate(NnuePerspective::White);
            break;
        case OpKind::Push:
            nnue.push();
            break;
        case OpKind::Pop:
            nnue.pop();
            break;
        }
    }
    checksum += nnue.evaluate(NnuePerspective::White);

    return BenchResult{"mixed_trace", trace.size(), elapsed_us(start), checksum};
}

} // namespace

int main() {
    std::cout << "perspective_feature_count=" << PerspectiveFeatureCount
              << " perspective_accumulator_size=" << PerspectiveAccumulatorSize
              << " network_accumulator_size=" << NetworkAccumulatorSize
              << " hidden2_size=" << Hidden2Size
              << " clip1_max=255 clip2_max=255 hidden2_scale=64 output_scale=16\n";

    print_result(bench_evaluate_only(2'000'000));
    print_result(bench_activate_deactivate(300'000));
    print_result(bench_push_pop(120'000));
    print_result(bench_mixed_trace(1'000'000));
}
