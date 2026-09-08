#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

constexpr std::size_t FeatureCount = 768;
constexpr std::size_t Hidden1Size = 256;
constexpr std::size_t Hidden2Size = 32;

template<std::size_t N, std::size_t H1, std::size_t H2>
class PlainArrayNnue {
private:
    int clip1_max_;
    int clip2_max_;
    int hidden2_scale_;
    int output_scale_;
    std::int64_t output_bias_;

    std::array<std::uint8_t, N> feature_state_{};
    std::array<std::int32_t, H1> acc1_{};
    std::array<std::int16_t, H1> hidden1_{};
    std::array<std::int16_t, N * H1> input_weight_{};
    std::array<std::int32_t, H2> hidden2_bias_{};
    std::array<std::int16_t, H1 * H2> hidden2_weight_{};
    std::array<std::int16_t, H2> output_weight_{};

    std::vector<std::size_t> feature_history_;
    std::vector<std::size_t> checkpoints_;

    void toggle_without_history(std::size_t feature) {
        const std::size_t offset = feature * H1;
        if (feature_state_[feature] != 0) {
            feature_state_[feature] = 0;
            for (std::size_t i = 0; i < H1; ++i) {
                acc1_[i] -= input_weight_[offset + i];
            }
        } else {
            feature_state_[feature] = 1;
            for (std::size_t i = 0; i < H1; ++i) {
                acc1_[i] += input_weight_[offset + i];
            }
        }
    }

public:
    PlainArrayNnue(
        int clip1_max,
        int clip2_max,
        int hidden2_scale,
        int output_scale,
        const std::vector<std::int32_t>& hidden1_bias,
        const std::vector<std::int16_t>& input_weight,
        const std::vector<std::int32_t>& hidden2_bias,
        const std::vector<std::int16_t>& hidden2_weight,
        const std::vector<std::int16_t>& output_weight,
        std::int64_t output_bias)
        : clip1_max_(clip1_max),
          clip2_max_(clip2_max),
          hidden2_scale_(hidden2_scale),
          output_scale_(output_scale),
          output_bias_(output_bias) {
        std::copy(hidden1_bias.begin(), hidden1_bias.end(), acc1_.begin());
        std::copy(input_weight.begin(), input_weight.end(), input_weight_.begin());
        std::copy(hidden2_bias.begin(), hidden2_bias.end(), hidden2_bias_.begin());
        std::copy(hidden2_weight.begin(), hidden2_weight.end(), hidden2_weight_.begin());
        std::copy(output_weight.begin(), output_weight.end(), output_weight_.begin());
    }

    void activate(std::size_t feature) {
        if (feature_state_[feature] != 0) {
            return;
        }
        feature_history_.push_back(feature);
        feature_state_[feature] = 1;
        const std::size_t offset = feature * H1;
        for (std::size_t i = 0; i < H1; ++i) {
            acc1_[i] += input_weight_[offset + i];
        }
    }

    void deactivate(std::size_t feature) {
        if (feature_state_[feature] == 0) {
            return;
        }
        feature_history_.push_back(feature);
        feature_state_[feature] = 0;
        const std::size_t offset = feature * H1;
        for (std::size_t i = 0; i < H1; ++i) {
            acc1_[i] -= input_weight_[offset + i];
        }
    }

    int evaluate() {
        for (std::size_t j = 0; j < H1; ++j) {
            hidden1_[j] = static_cast<std::int16_t>(
                std::clamp<std::int32_t>(acc1_[j], 0, static_cast<std::int32_t>(clip1_max_)));
        }

        std::int64_t raw = output_bias_;
        for (std::size_t k = 0; k < H2; ++k) {
            std::int32_t acc2 = hidden2_bias_[k];
            const std::size_t offset = k * H1;
            for (std::size_t j = 0; j < H1; ++j) {
                acc2 += hidden1_[j] * hidden2_weight_[offset + j];
            }
            const std::int32_t hidden2 =
                std::clamp<std::int32_t>(acc2 / hidden2_scale_, 0, clip2_max_);
            raw += hidden2 * output_weight_[k];
        }
        return static_cast<int>(raw / output_scale_);
    }

    void push() {
        checkpoints_.push_back(feature_history_.size());
    }

    void pop() {
        const std::size_t checkpoint = checkpoints_.back();
        checkpoints_.pop_back();
        while (feature_history_.size() > checkpoint) {
            const std::size_t feature = feature_history_.back();
            feature_history_.pop_back();
            toggle_without_history(feature);
        }
    }
};

template <typename T, std::size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(
            ::operator new(count * sizeof(T), std::align_val_t{Alignment}));
    }

    void deallocate(T* pointer, std::size_t) noexcept {
        ::operator delete(pointer, std::align_val_t{Alignment});
    }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) {
    return false;
}

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

template <typename T>
using DefaultVector = std::vector<T>;

template<std::size_t N, std::size_t H1, std::size_t H2, template <typename> typename Storage>
class VectorNnueImpl {
private:
    int clip1_max_;
    int clip2_max_;
    int hidden2_scale_;
    int output_scale_;
    std::int64_t output_bias_;

    Storage<std::uint8_t> feature_state_;
    Storage<std::int32_t> acc1_;
    Storage<std::int16_t> hidden1_;
    Storage<std::int16_t> input_weight_;
    Storage<std::int32_t> hidden2_bias_;
    Storage<std::int16_t> hidden2_weight_;
    Storage<std::int16_t> output_weight_;

    std::vector<std::size_t> feature_history_;
    std::vector<std::size_t> checkpoints_;

    void toggle_without_history(std::size_t feature) {
        const std::size_t offset = feature * H1;
        if (feature_state_[feature] != 0) {
            feature_state_[feature] = 0;
            for (std::size_t i = 0; i < H1; ++i) {
                acc1_[i] -= input_weight_[offset + i];
            }
        } else {
            feature_state_[feature] = 1;
            for (std::size_t i = 0; i < H1; ++i) {
                acc1_[i] += input_weight_[offset + i];
            }
        }
    }

public:
    VectorNnueImpl(
        int clip1_max,
        int clip2_max,
        int hidden2_scale,
        int output_scale,
        const std::vector<std::int32_t>& hidden1_bias,
        const std::vector<std::int16_t>& input_weight,
        const std::vector<std::int32_t>& hidden2_bias,
        const std::vector<std::int16_t>& hidden2_weight,
        const std::vector<std::int16_t>& output_weight,
        std::int64_t output_bias)
        : clip1_max_(clip1_max),
          clip2_max_(clip2_max),
          hidden2_scale_(hidden2_scale),
          output_scale_(output_scale),
          output_bias_(output_bias),
          feature_state_(N),
          acc1_(H1),
          hidden1_(H1),
          input_weight_(N * H1),
          hidden2_bias_(H2),
          hidden2_weight_(H2 * H1),
          output_weight_(H2) {
        std::copy(hidden1_bias.begin(), hidden1_bias.end(), acc1_.begin());
        std::copy(input_weight.begin(), input_weight.end(), input_weight_.begin());
        std::copy(hidden2_bias.begin(), hidden2_bias.end(), hidden2_bias_.begin());
        std::copy(hidden2_weight.begin(), hidden2_weight.end(), hidden2_weight_.begin());
        std::copy(output_weight.begin(), output_weight.end(), output_weight_.begin());
    }

    void activate(std::size_t feature) {
        if (feature_state_[feature] != 0) {
            return;
        }
        feature_history_.push_back(feature);
        feature_state_[feature] = 1;
        const std::size_t offset = feature * H1;
        for (std::size_t i = 0; i < H1; ++i) {
            acc1_[i] += input_weight_[offset + i];
        }
    }

    void deactivate(std::size_t feature) {
        if (feature_state_[feature] == 0) {
            return;
        }
        feature_history_.push_back(feature);
        feature_state_[feature] = 0;
        const std::size_t offset = feature * H1;
        for (std::size_t i = 0; i < H1; ++i) {
            acc1_[i] -= input_weight_[offset + i];
        }
    }

    int evaluate() {
        for (std::size_t j = 0; j < H1; ++j) {
            hidden1_[j] = static_cast<std::int16_t>(
                std::clamp<std::int32_t>(acc1_[j], 0, static_cast<std::int32_t>(clip1_max_)));
        }

        std::int64_t raw = output_bias_;
        for (std::size_t k = 0; k < H2; ++k) {
            std::int32_t acc2 = hidden2_bias_[k];
            const std::size_t offset = k * H1;
            for (std::size_t j = 0; j < H1; ++j) {
                acc2 += hidden1_[j] * hidden2_weight_[offset + j];
            }
            const std::int32_t hidden2 =
                std::clamp<std::int32_t>(acc2 / hidden2_scale_, 0, clip2_max_);
            raw += hidden2 * output_weight_[k];
        }
        return static_cast<int>(raw / output_scale_);
    }

    void push() {
        checkpoints_.push_back(feature_history_.size());
    }

    void pop() {
        const std::size_t checkpoint = checkpoints_.back();
        checkpoints_.pop_back();
        while (feature_history_.size() > checkpoint) {
            const std::size_t feature = feature_history_.back();
            feature_history_.pop_back();
            toggle_without_history(feature);
        }
    }
};

using DefaultVectorNnue = VectorNnueImpl<FeatureCount, Hidden1Size, Hidden2Size, DefaultVector>;
using AlignedVectorNnue = VectorNnueImpl<FeatureCount, Hidden1Size, Hidden2Size, AlignedVector>;
using PlainArrayBenchNnue = PlainArrayNnue<FeatureCount, Hidden1Size, Hidden2Size>;
// This legacy storage benchmark predates the split incremental wrapper. Keep
// it self-contained instead of depending on the production NNUE classes.
using ArrayNnue = PlainArrayBenchNnue;

struct ModelData {
    std::vector<std::int32_t> hidden1_bias;
    std::vector<std::int16_t> input_weight;
    std::vector<std::int32_t> hidden2_bias;
    std::vector<std::int16_t> hidden2_weight;
    std::vector<std::int16_t> output_weight;
    std::int64_t output_bias = 0;
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

template <typename Net>
Net make_nnue(const ModelData& model) {
    return Net(
        255,
        255,
        64,
        16,
        model.hidden1_bias,
        model.input_weight,
        model.hidden2_bias,
        model.hidden2_weight,
        model.output_weight,
        model.output_bias);
}

std::vector<Op> make_trace(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> op_dist(0, 99);
    std::uniform_int_distribution<std::size_t> feature_dist(0, FeatureCount - 1);
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

template <typename Net>
std::int64_t run_trace(Net& nnue, const std::vector<Op>& trace) {
    std::int64_t checksum = 0;
    for (const Op& op : trace) {
        switch (op.kind) {
        case OpKind::Activate:
            nnue.activate(op.feature);
            break;
        case OpKind::Deactivate:
            nnue.deactivate(op.feature);
            break;
        case OpKind::Evaluate:
            checksum += nnue.evaluate();
            break;
        case OpKind::Push:
            nnue.push();
            break;
        case OpKind::Pop:
            nnue.pop();
            break;
        }
    }
    return checksum + nnue.evaluate();
}

template <typename Net>
std::uint64_t bench_evaluate_only(const ModelData& model, std::uint64_t iterations, std::int64_t& checksum) {
    Net nnue = make_nnue<Net>(model);
    for (std::size_t feature = 0; feature < FeatureCount; feature += 3) {
        nnue.activate(feature);
    }

    checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
        checksum += nnue.evaluate();
    }
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
}

template <typename Net>
std::uint64_t bench_mixed_trace(const ModelData& model, const std::vector<Op>& trace, std::int64_t& checksum) {
    Net nnue = make_nnue<Net>(model);
    const auto start = std::chrono::steady_clock::now();
    checksum = run_trace(nnue, trace);
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void print_compare(
    std::string_view name,
    const std::vector<double>& aligned_array_ns,
    const std::vector<double>& plain_array_ns,
    const std::vector<double>& default_vector_ns,
    const std::vector<double>& aligned_vector_ns) {
    const double aligned_array_med = median(aligned_array_ns);
    const double plain_array_med = median(plain_array_ns);
    const double default_vector_med = median(default_vector_ns);
    const double aligned_vector_med = median(aligned_vector_ns);
    std::cout << name
              << " aligned_array_ns=" << aligned_array_med
              << " plain_array_ns=" << plain_array_med
              << " plain/aligned_array=" << (plain_array_med / aligned_array_med)
              << " default_vector_ns=" << default_vector_med
              << " default_vector/aligned_array=" << (default_vector_med / aligned_array_med)
              << " aligned_vector_ns=" << aligned_vector_med
              << " aligned_vector/aligned_array=" << (aligned_vector_med / aligned_array_med)
              << '\n';
}

} // namespace

int main() {
    const ModelData model = make_model(0xC0FFEEu);
    const std::vector<Op> trace = make_trace(1'000'000, 0xBAD5EEDu);

    constexpr int Runs = 15;
    std::vector<double> array_eval;
    std::vector<double> plain_array_eval;
    std::vector<double> default_vector_eval;
    std::vector<double> vector_eval;
    std::vector<double> array_mixed;
    std::vector<double> plain_array_mixed;
    std::vector<double> default_vector_mixed;
    std::vector<double> vector_mixed;
    array_eval.reserve(Runs);
    plain_array_eval.reserve(Runs);
    default_vector_eval.reserve(Runs);
    vector_eval.reserve(Runs);
    array_mixed.reserve(Runs);
    plain_array_mixed.reserve(Runs);
    default_vector_mixed.reserve(Runs);
    vector_mixed.reserve(Runs);

    for (int run = 0; run < Runs; ++run) {
        std::int64_t array_checksum = 0;
        std::int64_t plain_array_checksum = 0;
        std::int64_t default_vector_checksum = 0;
        std::int64_t vector_checksum = 0;
        const std::uint64_t array_eval_ns =
            bench_evaluate_only<ArrayNnue>(model, 1'000'000, array_checksum);
        const std::uint64_t plain_array_eval_ns =
            bench_evaluate_only<PlainArrayBenchNnue>(model, 1'000'000, plain_array_checksum);
        const std::uint64_t default_vector_eval_ns =
            bench_evaluate_only<DefaultVectorNnue>(model, 1'000'000, default_vector_checksum);
        const std::uint64_t vector_eval_ns =
            bench_evaluate_only<AlignedVectorNnue>(model, 1'000'000, vector_checksum);
        if (array_checksum != plain_array_checksum
            || array_checksum != default_vector_checksum
            || array_checksum != vector_checksum) {
            std::cerr << "checksum mismatch evaluate_only: "
                      << array_checksum << " vs " << plain_array_checksum
                      << " vs " << default_vector_checksum
                      << " vs " << vector_checksum << '\n';
            return 1;
        }
        array_eval.push_back(static_cast<double>(array_eval_ns) / 1'000'000.0);
        plain_array_eval.push_back(static_cast<double>(plain_array_eval_ns) / 1'000'000.0);
        default_vector_eval.push_back(static_cast<double>(default_vector_eval_ns) / 1'000'000.0);
        vector_eval.push_back(static_cast<double>(vector_eval_ns) / 1'000'000.0);

        const std::uint64_t array_mixed_ns =
            bench_mixed_trace<ArrayNnue>(model, trace, array_checksum);
        const std::uint64_t plain_array_mixed_ns =
            bench_mixed_trace<PlainArrayBenchNnue>(model, trace, plain_array_checksum);
        const std::uint64_t default_vector_mixed_ns =
            bench_mixed_trace<DefaultVectorNnue>(model, trace, default_vector_checksum);
        const std::uint64_t vector_mixed_ns =
            bench_mixed_trace<AlignedVectorNnue>(model, trace, vector_checksum);
        if (array_checksum != plain_array_checksum
            || array_checksum != default_vector_checksum
            || array_checksum != vector_checksum) {
            std::cerr << "checksum mismatch mixed_trace: "
                      << array_checksum << " vs " << plain_array_checksum
                      << " vs " << default_vector_checksum
                      << " vs " << vector_checksum << '\n';
            return 1;
        }
        array_mixed.push_back(static_cast<double>(array_mixed_ns) / static_cast<double>(trace.size()));
        plain_array_mixed.push_back(static_cast<double>(plain_array_mixed_ns) / static_cast<double>(trace.size()));
        default_vector_mixed.push_back(static_cast<double>(default_vector_mixed_ns) / static_cast<double>(trace.size()));
        vector_mixed.push_back(static_cast<double>(vector_mixed_ns) / static_cast<double>(trace.size()));
    }

    print_compare("evaluate_only", array_eval, plain_array_eval, default_vector_eval, vector_eval);
    print_compare("mixed_trace", array_mixed, plain_array_mixed, default_vector_mixed, vector_mixed);
}
