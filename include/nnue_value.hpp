#pragma once

#include "board_encoder.hpp"
#include "move.hpp"
#include "position.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <string_view>
#include <vector>

namespace chess {

class NnueAccumulator;

struct NnueAccumulatorUndo {
    Move move{};
    Color moving_color = Color::White;
    PieceType moved_piece_before = PieceType::None;
    PieceType moved_piece_after = PieceType::None;
    PieceType captured_piece = PieceType::None;
    Square captured_square = NoSquare;
    bool used_snapshot = false;
    std::array<std::vector<float>, 2> accumulator_snapshot{};
    std::array<Square, 2> friendly_king_snapshot{NoSquare, NoSquare};
    std::array<Square, 2> enemy_king_snapshot{NoSquare, NoSquare};
};

class NnueValueModel {
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

    float target_clip() const {
        return target_clip_;
    }

    bool loaded() const {
        return !feature_weights_.empty();
    }

private:
    friend class NnueAccumulator;

    std::uint32_t feature_count_ = 0;
    std::uint32_t aux_feature_count_ = 0;
    std::uint32_t hidden_size_ = 0;
    float target_scale_ = 1.0F;
    float target_clip_ = 0.0F;

    std::vector<float> feature_weights_;
    std::vector<float> aux_projection_weights_;
    std::vector<float> accumulator_bias_;
    std::vector<float> output_weights_;
    float output_bias_ = 0.0F;
};

class NnueAccumulator {
public:
    void reset(const NnueValueModel& model, const Position& pos);
    void make_move(const Position& before, Move move, const Position& after);
    NnueAccumulatorUndo make_move_with_undo(const Position& before, Move move, const Position& after);
    void undo(const NnueAccumulatorUndo& undo);

    float predict_normalized(const Position& pos) const;
    float evaluate_cp(const Position& pos) const;
    int evaluate_cp_rounded(const Position& pos) const;

    bool initialized() const {
        return model_ != nullptr;
    }

private:
    void rebuild_perspective(const Position& pos, Color perspective);
    void add_piece_features(Color perspective, Color piece_color, PieceType piece, Square square, float sign);
    void add_feature(Color perspective, FeatureIndex feature, float sign);

    const NnueValueModel* model_ = nullptr;
    std::array<std::vector<float>, 2> accumulators_{};
    std::array<Square, 2> friendly_king_squares_{NoSquare, NoSquare};
    std::array<Square, 2> enemy_king_squares_{NoSquare, NoSquare};
};

} // namespace chess
