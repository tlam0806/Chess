#pragma once

#include "board_encoder.hpp"
#include "position.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace chess {

class NnValueModel {
public:
    bool load(std::string_view path);

    float predict_normalized(const EncodedPosition& encoded) const;
    float evaluate_cp(const Position& pos) const;
    int evaluate_cp_rounded(const Position& pos) const;

    std::uint32_t feature_count() const {
        return feature_count_;
    }

    std::uint32_t aux_feature_count() const {
        return aux_feature_count_;
    }

    std::uint32_t hidden_size() const {
        return hidden_size_;
    }

    float target_scale() const {
        return target_scale_;
    }

    bool loaded() const {
        return !sparse_weights_.empty();
    }

private:
    std::uint32_t feature_count_ = 0;
    std::uint32_t aux_feature_count_ = 0;
    std::uint32_t hidden_size_ = 0;
    float target_scale_ = 1.0F;
    float target_clip_ = 0.0F;

    std::vector<float> sparse_weights_;
    std::vector<float> hidden_weights_;
    std::vector<float> hidden_bias_;
    std::vector<float> output_weights_;
    float output_bias_ = 0.0F;
};

} // namespace chess
