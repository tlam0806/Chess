#include "nn_value.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

namespace chess {

namespace {

constexpr std::array<char, 4> Magic{'C', 'V', 'N', '1'};
constexpr std::uint32_t Version = 1;

template <typename T>
bool read_exact(std::istream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(input);
}

bool read_floats(std::istream& input, std::vector<float>& values, std::size_t count) {
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(input);
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

} // namespace

bool NnValueModel::load(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        return false;
    }

    std::array<char, 4> magic{};
    std::uint32_t version = 0;
    std::uint32_t feature_count = 0;
    std::uint32_t aux_feature_count = 0;
    std::uint32_t hidden_size = 0;
    float target_scale = 0.0F;
    float target_clip = 0.0F;

    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != Magic) {
        return false;
    }
    if (!read_exact(input, version) || version != Version) {
        return false;
    }
    if (!read_exact(input, feature_count)
        || !read_exact(input, aux_feature_count)
        || !read_exact(input, hidden_size)
        || !read_exact(input, target_scale)
        || !read_exact(input, target_clip)) {
        return false;
    }
    if (feature_count == 0 || aux_feature_count != AuxFeatureCount || hidden_size == 0) {
        return false;
    }

    std::size_t sparse_count = 0;
    std::size_t hidden_input_size = 0;
    std::size_t hidden_weight_count = 0;
    if (!checked_mul(feature_count, hidden_size, sparse_count)) {
        return false;
    }
    hidden_input_size = static_cast<std::size_t>(hidden_size) + aux_feature_count;
    if (!checked_mul(hidden_size, hidden_input_size, hidden_weight_count)) {
        return false;
    }

    std::vector<float> sparse_weights;
    std::vector<float> hidden_weights;
    std::vector<float> hidden_bias;
    std::vector<float> output_weights;
    if (!read_floats(input, sparse_weights, sparse_count)
        || !read_floats(input, hidden_weights, hidden_weight_count)
        || !read_floats(input, hidden_bias, hidden_size)
        || !read_floats(input, output_weights, hidden_size)
        || !read_exact(input, output_bias_)) {
        return false;
    }

    char trailing = 0;
    if (input.read(&trailing, 1)) {
        return false;
    }

    feature_count_ = feature_count;
    aux_feature_count_ = aux_feature_count;
    hidden_size_ = hidden_size;
    target_scale_ = target_scale;
    target_clip_ = target_clip;
    sparse_weights_ = std::move(sparse_weights);
    hidden_weights_ = std::move(hidden_weights);
    hidden_bias_ = std::move(hidden_bias);
    output_weights_ = std::move(output_weights);
    return true;
}

float NnValueModel::predict_normalized(const EncodedPosition& encoded) const {
    assert(loaded());
    assert(aux_feature_count_ == AuxFeatureCount);

    std::vector<float> sparse(hidden_size_, 0.0F);
    for (FeatureIndex feature : encoded.features) {
        assert(feature < feature_count_);
        const std::size_t base = static_cast<std::size_t>(feature) * hidden_size_;
        for (std::uint32_t i = 0; i < hidden_size_; ++i) {
            sparse[i] += sparse_weights_[base + i];
        }
    }

    float output = output_bias_;
    const std::size_t hidden_input_size = static_cast<std::size_t>(hidden_size_) + aux_feature_count_;
    for (std::uint32_t row = 0; row < hidden_size_; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row) * hidden_input_size;
        float activation = hidden_bias_[row];

        for (std::uint32_t col = 0; col < hidden_size_; ++col) {
            activation += hidden_weights_[row_base + col] * sparse[col];
        }
        for (std::uint32_t col = 0; col < aux_feature_count_; ++col) {
            activation += hidden_weights_[row_base + hidden_size_ + col]
                * static_cast<float>(encoded.aux[col]);
        }

        activation = std::max(activation, 0.0F);
        output += output_weights_[row] * activation;
    }

    return output;
}

float NnValueModel::evaluate_cp(const Position& pos) const {
    return predict_normalized(encode_position(pos)) * target_scale_;
}

int NnValueModel::evaluate_cp_rounded(const Position& pos) const {
    return static_cast<int>(std::lround(evaluate_cp(pos)));
}

} // namespace chess
