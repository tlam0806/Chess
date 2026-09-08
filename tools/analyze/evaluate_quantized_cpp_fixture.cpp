#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<typename T>
T read_value(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("unexpected end of fixture");
    }
    return value;
}

template<typename T>
std::vector<T> read_values(std::istream& input, std::size_t count) {
    std::vector<T> values(count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(sizeof(T) * count)
    );
    if (!input) {
        throw std::runtime_error("unexpected end of fixture");
    }
    return values;
}

struct Fixture {
    std::uint32_t version{1};
    std::uint32_t h1{};
    std::uint32_t h2{};
    std::uint32_t h3{};
    std::uint32_t clip{};
    std::uint32_t divisor{};
    std::uint32_t hidden2_scale{};
    std::uint32_t hidden3_scale{};
    std::uint32_t output_scale{};
    std::uint32_t count{};
    bool psqt_enabled{};
    std::uint32_t psqt_scale{1};
    std::vector<std::int32_t> hidden2_bias;
    std::vector<std::int8_t> hidden2_weight;
    std::vector<std::int32_t> hidden3_bias;
    std::vector<std::int8_t> hidden3_weight;
    std::int64_t output_bias{};
    std::vector<std::int16_t> output_weight;
};

Fixture read_fixture_header(std::istream& input) {
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected{'Q', 'N', 'N', 'U', 'E', 'F', '1', '\0'};
    if (!input || magic != expected) {
        throw std::runtime_error("invalid fixture magic");
    }
    const std::uint32_t version = read_value<std::uint32_t>(input);
    if (version != 1 && version != 2) {
        throw std::runtime_error("unsupported fixture version");
    }
    Fixture fixture;
    fixture.version = version;
    fixture.h1 = read_value<std::uint32_t>(input);
    fixture.h2 = read_value<std::uint32_t>(input);
    fixture.h3 = read_value<std::uint32_t>(input);
    fixture.clip = read_value<std::uint32_t>(input);
    fixture.divisor = read_value<std::uint32_t>(input);
    fixture.hidden2_scale = read_value<std::uint32_t>(input);
    fixture.hidden3_scale = read_value<std::uint32_t>(input);
    fixture.output_scale = read_value<std::uint32_t>(input);
    fixture.count = read_value<std::uint32_t>(input);
    if (version >= 2) {
        fixture.psqt_enabled = read_value<std::uint32_t>(input) != 0;
        fixture.psqt_scale = read_value<std::uint32_t>(input);
    }
    if (
        fixture.h1 == 0 || fixture.h2 == 0 || fixture.h3 == 0 ||
        fixture.clip == 0 || fixture.divisor == 0 ||
        fixture.hidden2_scale == 0 || fixture.hidden3_scale == 0 ||
        fixture.output_scale == 0 || fixture.count == 0 || fixture.psqt_scale == 0
    ) {
        throw std::runtime_error("fixture contains a non-positive dimension or scale");
    }
    fixture.hidden2_bias = read_values<std::int32_t>(input, fixture.h2);
    fixture.hidden2_weight = read_values<std::int8_t>(input, fixture.h1 * fixture.h2);
    fixture.hidden3_bias = read_values<std::int32_t>(input, fixture.h3);
    fixture.hidden3_weight = read_values<std::int8_t>(input, fixture.h2 * fixture.h3);
    fixture.output_bias = read_value<std::int64_t>(input);
    fixture.output_weight = read_values<std::int16_t>(input, fixture.h3);
    return fixture;
}

std::int32_t activate(std::int64_t accumulator, std::uint32_t scale, const Fixture& fixture) {
    const std::int64_t rescaled = accumulator / static_cast<std::int64_t>(scale);
    const std::int64_t clipped = std::clamp<std::int64_t>(
        rescaled,
        0,
        static_cast<std::int64_t>(fixture.clip)
    );
    return static_cast<std::int32_t>(
        (clipped * clipped) / static_cast<std::int64_t>(fixture.divisor)
    );
}

constexpr std::size_t psqt_bucket(std::uint32_t piece_count) {
    if (piece_count == 0) {
        return 0;
    }
    return std::min<std::size_t>((piece_count - 1) / 4, 7);
}

std::int32_t evaluate(
    const Fixture& fixture,
    const std::vector<std::int32_t>& first_acc,
    const std::array<std::array<std::int32_t, 8>, 2>& psqt_accumulators,
    std::uint32_t piece_count
) {
    std::vector<std::int32_t> hidden1(fixture.h1);
    for (std::size_t index = 0; index < fixture.h1; ++index) {
        hidden1[index] = activate(first_acc[index], 1, fixture);
    }
    std::vector<std::int32_t> hidden2(fixture.h2);
    for (std::size_t output = 0; output < fixture.h2; ++output) {
        std::int64_t acc = fixture.hidden2_bias[output];
        for (std::size_t input = 0; input < fixture.h1; ++input) {
            acc += static_cast<std::int64_t>(hidden1[input]) *
                fixture.hidden2_weight[output * fixture.h1 + input];
        }
        hidden2[output] = activate(acc, fixture.hidden2_scale, fixture);
    }
    std::vector<std::int32_t> hidden3(fixture.h3);
    for (std::size_t output = 0; output < fixture.h3; ++output) {
        std::int64_t acc = fixture.hidden3_bias[output];
        for (std::size_t input = 0; input < fixture.h2; ++input) {
            acc += static_cast<std::int64_t>(hidden2[input]) *
                fixture.hidden3_weight[output * fixture.h2 + input];
        }
        hidden3[output] = activate(acc, fixture.hidden3_scale, fixture);
    }
    std::int64_t raw = fixture.output_bias;
    for (std::size_t input = 0; input < fixture.h3; ++input) {
        raw += static_cast<std::int64_t>(hidden3[input]) * fixture.output_weight[input];
    }
    const std::int64_t positional = raw / static_cast<std::int64_t>(fixture.output_scale);
    const std::size_t bucket = psqt_bucket(piece_count);
    const std::int64_t psqt = fixture.psqt_enabled
        ? (static_cast<std::int64_t>(psqt_accumulators[0][bucket])
            - static_cast<std::int64_t>(psqt_accumulators[1][bucket]))
            / (2LL * fixture.psqt_scale)
        : 0;
    return static_cast<std::int32_t>(positional + psqt);
}

struct Bucket {
    double lower{};
    double upper{};
    std::uint64_t count{};
    double target_sum{};
    double prediction_sum{};
    double residual_sum{};
    double absolute_error_sum{};
    double prediction_square_sum{};
};

void print_bucket(const Bucket& bucket) {
    std::cout << "{\"target_min\":" << bucket.lower
              << ",\"target_max\":" << bucket.upper
              << ",\"samples\":" << bucket.count;
    if (bucket.count != 0) {
        const double count = static_cast<double>(bucket.count);
        const double mean_prediction = bucket.prediction_sum / count;
        const double variance = std::max(
            0.0,
            bucket.prediction_square_sum / count - mean_prediction * mean_prediction
        );
        std::cout << ",\"mean_target\":" << bucket.target_sum / count
                  << ",\"mean_prediction\":" << mean_prediction
                  << ",\"mean_residual\":" << bucket.residual_sum / count
                  << ",\"cp_mae\":" << bucket.absolute_error_sum / count
                  << ",\"prediction_std\":" << std::sqrt(variance);
    }
    std::cout << '}';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: evaluate_quantized_cpp_fixture FIXTURE\n";
            return 2;
        }
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open fixture");
        }
        const Fixture fixture = read_fixture_header(input);
        constexpr std::array<double, 13> edges{
            -2000, -1600, -1000, -600, -300, -100, 0,
            100, 300, 600, 1000, 1600, 2000
        };
        std::array<Bucket, edges.size() - 1> buckets{};
        for (std::size_t index = 0; index < buckets.size(); ++index) {
            buckets[index].lower = edges[index];
            buckets[index].upper = edges[index + 1];
        }
        double absolute_error_sum = 0.0;
        double target_sum = 0.0;
        double prediction_sum = 0.0;
        double target_square_sum = 0.0;
        double prediction_square_sum = 0.0;
        double target_prediction_sum = 0.0;
        double parity_absolute_sum = 0.0;
        std::int64_t parity_max_abs = 0;
        std::uint64_t parity_mismatches = 0;
        for (std::size_t sample = 0; sample < fixture.count; ++sample) {
            const auto accumulator = read_values<std::int32_t>(input, fixture.h1);
            std::array<std::array<std::int32_t, 8>, 2> psqt_accumulators{};
            std::uint32_t piece_count = 0;
            if (fixture.version >= 2) {
                const auto values = read_values<std::int32_t>(input, 16);
                std::copy(values.begin(), values.begin() + 8, psqt_accumulators[0].begin());
                std::copy(values.begin() + 8, values.end(), psqt_accumulators[1].begin());
                piece_count = read_value<std::uint32_t>(input);
            }
            const float target = read_value<float>(input);
            const std::int32_t python_prediction = read_value<std::int32_t>(input);
            (void)read_value<float>(input); // Float PyTorch prediction is reported by Python.
            const std::int32_t prediction = evaluate(
                fixture,
                accumulator,
                psqt_accumulators,
                piece_count
            );
            const std::int64_t parity_delta =
                static_cast<std::int64_t>(prediction) - python_prediction;
            const std::int64_t parity_abs = std::abs(parity_delta);
            parity_absolute_sum += static_cast<double>(parity_abs);
            parity_max_abs = std::max(parity_max_abs, parity_abs);
            parity_mismatches += parity_delta != 0;
            const double residual = static_cast<double>(prediction) - target;
            absolute_error_sum += std::abs(residual);
            target_sum += target;
            prediction_sum += prediction;
            target_square_sum += static_cast<double>(target) * target;
            prediction_square_sum += static_cast<double>(prediction) * prediction;
            target_prediction_sum += static_cast<double>(target) * prediction;
            for (std::size_t index = 0; index < buckets.size(); ++index) {
                const bool final = index + 1 == buckets.size();
                if (target >= edges[index] && (target < edges[index + 1] || (final && target <= edges[index + 1]))) {
                    Bucket& bucket = buckets[index];
                    ++bucket.count;
                    bucket.target_sum += target;
                    bucket.prediction_sum += prediction;
                    bucket.residual_sum += residual;
                    bucket.absolute_error_sum += std::abs(residual);
                    bucket.prediction_square_sum += static_cast<double>(prediction) * prediction;
                    break;
                }
            }
        }
        if (input.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("fixture has trailing bytes");
        }
        const double count = static_cast<double>(fixture.count);
        const double mean_target = target_sum / count;
        const double mean_prediction = prediction_sum / count;
        const double target_variance_sum = target_square_sum - count * mean_target * mean_target;
        const double covariance_sum = target_prediction_sum - count * mean_target * mean_prediction;
        const double regression_slope = covariance_sum / target_variance_sum;
        const double regression_intercept = mean_prediction - regression_slope * mean_target;
        const double prediction_variance = std::max(
            0.0,
            prediction_square_sum / count - mean_prediction * mean_prediction
        );
        std::cout << std::setprecision(10)
                  << "{\"samples\":" << fixture.count
                  << ",\"cpp_cp_mae\":" << absolute_error_sum / fixture.count
                  << ",\"overall\":{\"mean_target\":" << mean_target
                  << ",\"mean_prediction\":" << mean_prediction
                  << ",\"prediction_std\":" << std::sqrt(prediction_variance)
                  << ",\"regression_slope\":" << regression_slope
                  << ",\"regression_intercept\":" << regression_intercept << "}"
                  << ",\"python_cpp_parity\":{\"mismatches\":" << parity_mismatches
                  << ",\"mae\":" << parity_absolute_sum / fixture.count
                  << ",\"max_abs\":" << parity_max_abs << "}"
                  << ",\"signed_buckets\":[";
        for (std::size_t index = 0; index < buckets.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            print_bucket(buckets[index]);
        }
        std::cout << "]}\n";
        return parity_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "evaluate_quantized_cpp_fixture: " << error.what() << '\n';
        return 2;
    }
}
