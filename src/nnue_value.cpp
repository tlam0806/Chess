#include "nnue_value.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "move.hpp"

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

constexpr std::array<char, 4> Magic{'N', 'N', 'U', 'E'};
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

constexpr std::array<PieceType, EncoderPieceTypes> EncodedPieces{
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King
};

bool can_castle_kingside(const Position& pos, Color color) {
    return color == Color::White ? pos.white_can_castle_kingside : pos.black_can_castle_kingside;
}

bool can_castle_queenside(const Position& pos, Color color) {
    return color == Color::White ? pos.white_can_castle_queenside : pos.black_can_castle_queenside;
}

std::array<std::uint8_t, AuxFeatureCount> aux_features_for_perspective(const Position& pos, Color perspective) {
    std::array<std::uint8_t, AuxFeatureCount> aux{};
    const Color enemy = opposite(perspective);

    aux[FriendlyCanCastleKingside] = can_castle_kingside(pos, perspective);
    aux[FriendlyCanCastleQueenside] = can_castle_queenside(pos, perspective);
    aux[EnemyCanCastleKingside] = can_castle_kingside(pos, enemy);
    aux[EnemyCanCastleQueenside] = can_castle_queenside(pos, enemy);

    if (pos.en_passant_square != NoSquare) {
        const Square ep_square = relative_square(perspective, pos.en_passant_square);
        aux[HasEnPassant] = 1;
        aux[EnPassantFileA + file_of(ep_square)] = 1;
    }

    return aux;
}

int color_index(Color color) {
    return static_cast<int>(color);
}

} // namespace

bool NnueValueModel::load(std::string_view path) {
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

    std::size_t feature_weight_count = 0;
    std::size_t aux_projection_weight_count = 0;
    if (!checked_mul(feature_count, hidden_size, feature_weight_count)
        || !checked_mul(hidden_size, aux_feature_count, aux_projection_weight_count)) {
        return false;
    }

    std::vector<float> feature_weights;
    std::vector<float> aux_projection_weights;
    std::vector<float> accumulator_bias;
    std::vector<float> output_weights;
    if (!read_floats(input, feature_weights, feature_weight_count)
        || !read_floats(input, aux_projection_weights, aux_projection_weight_count)
        || !read_floats(input, accumulator_bias, hidden_size)
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
    feature_weights_ = std::move(feature_weights);
    aux_projection_weights_ = std::move(aux_projection_weights);
    accumulator_bias_ = std::move(accumulator_bias);
    output_weights_ = std::move(output_weights);
    return true;
}

float NnueValueModel::predict_normalized(const EncodedPosition& encoded) const {
    assert(loaded());
    assert(aux_feature_count_ == AuxFeatureCount);

    std::vector<float> accumulator(accumulator_bias_);
    for (FeatureIndex feature : encoded.features) {
        assert(feature < feature_count_);
        const std::size_t base = static_cast<std::size_t>(feature) * hidden_size_;
        for (std::uint32_t i = 0; i < hidden_size_; ++i) {
            accumulator[i] += feature_weights_[base + i];
        }
    }

    float output = output_bias_;
    for (std::uint32_t row = 0; row < hidden_size_; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row) * aux_feature_count_;
        for (std::uint32_t col = 0; col < aux_feature_count_; ++col) {
            accumulator[row] += aux_projection_weights_[row_base + col]
                * static_cast<float>(encoded.aux[col]);
        }
        const float activation = std::clamp(accumulator[row], 0.0F, 1.0F);
        output += output_weights_[row] * activation;
    }

    return output;
}

float NnueValueModel::evaluate_cp(const Position& pos) const {
    return predict_normalized(encode_position(pos)) * target_scale_;
}

int NnueValueModel::evaluate_cp_rounded(const Position& pos) const {
    return static_cast<int>(std::lround(evaluate_cp(pos)));
}

void NnueAccumulator::reset(const NnueValueModel& model, const Position& pos) {
    assert(model.loaded());
    model_ = &model;
    rebuild_perspective(pos, Color::White);
    rebuild_perspective(pos, Color::Black);
}

void NnueAccumulator::make_move(const Position& before, Move move, const Position& after) {
    static_cast<void>(make_move_with_undo(before, move, after));
}

NnueAccumulatorUndo NnueAccumulator::make_move_with_undo(
    const Position& before,
    Move move,
    const Position& after
) {
    assert(initialized());

    NnueAccumulatorUndo undo;
    undo.move = move;

    const Square from = move.from();
    const Square to = move.to();
    const Color moving_color = before.color_on_occupied(from);
    const PieceType piece_before_move = before.piece_type_on_occupied(from);
    const PieceType promoted_piece = promotion_piece(move);
    const PieceType piece_after_move =
        promoted_piece == PieceType::None ? piece_before_move : promoted_piece;

    undo.moving_color = moving_color;
    undo.moved_piece_before = piece_before_move;
    undo.moved_piece_after = piece_after_move;

    if (piece_before_move == PieceType::King) {
        undo.used_snapshot = true;
        undo.accumulator_snapshot = accumulators_;
        undo.friendly_king_snapshot = friendly_king_squares_;
        undo.enemy_king_snapshot = enemy_king_squares_;
        rebuild_perspective(after, Color::White);
        rebuild_perspective(after, Color::Black);
        return undo;
    }

    Square captured_square = to;
    if (move.flag() == MoveFlag::EnPassant) {
        captured_square = moving_color == Color::White ? to - 8 : to + 8;
    }
    const bool has_capture = is_capture(move);
    const Color captured_color = opposite(moving_color);
    const PieceType captured_piece =
        has_capture ? before.piece_type_on_occupied(captured_square) : PieceType::None;
    undo.captured_piece = captured_piece;
    undo.captured_square = has_capture ? captured_square : NoSquare;

    for (Color perspective : {Color::White, Color::Black}) {
        add_piece_features(perspective, moving_color, piece_before_move, from, -1.0F);
        add_piece_features(perspective, moving_color, piece_after_move, to, 1.0F);
        if (has_capture) {
            add_piece_features(perspective, captured_color, captured_piece, captured_square, -1.0F);
        }
    }

    return undo;
}

NnueAccumulatorUndo NnueAccumulator::make_move_with_undo(
    Move move,
    Color moving_color,
    PieceType moved_piece,
    PieceType captured_piece,
    Square captured_square,
    const Position& after
) {
    assert(initialized());
    assert(moved_piece != PieceType::None);

    NnueAccumulatorUndo undo;
    undo.move = move;
    undo.moving_color = moving_color;
    undo.moved_piece_before = moved_piece;

    const PieceType promoted_piece = promotion_piece(move);
    const PieceType moved_piece_after =
        promoted_piece == PieceType::None ? moved_piece : promoted_piece;
    undo.moved_piece_after = moved_piece_after;

    if (moved_piece == PieceType::King) {
        undo.used_snapshot = true;
        undo.accumulator_snapshot = accumulators_;
        undo.friendly_king_snapshot = friendly_king_squares_;
        undo.enemy_king_snapshot = enemy_king_squares_;
        rebuild_perspective(after, Color::White);
        rebuild_perspective(after, Color::Black);
        return undo;
    }

    const bool has_capture = captured_piece != PieceType::None;
    undo.captured_piece = captured_piece;
    undo.captured_square = has_capture ? captured_square : NoSquare;

    const Square from = move.from();
    const Square to = move.to();
    const Color captured_color = opposite(moving_color);
    for (Color perspective : {Color::White, Color::Black}) {
        add_piece_features(perspective, moving_color, moved_piece, from, -1.0F);
        add_piece_features(perspective, moving_color, moved_piece_after, to, 1.0F);
        if (has_capture) {
            add_piece_features(perspective, captured_color, captured_piece, captured_square, -1.0F);
        }
    }

    return undo;
}

void NnueAccumulator::undo(const NnueAccumulatorUndo& undo) {
    assert(initialized());
    if (undo.used_snapshot) {
        accumulators_ = undo.accumulator_snapshot;
        friendly_king_squares_ = undo.friendly_king_snapshot;
        enemy_king_squares_ = undo.enemy_king_snapshot;
        return;
    }

    const Square from = undo.move.from();
    const Square to = undo.move.to();
    const Color captured_color = opposite(undo.moving_color);

    for (Color perspective : {Color::White, Color::Black}) {
        add_piece_features(perspective, undo.moving_color, undo.moved_piece_after, to, -1.0F);
        add_piece_features(perspective, undo.moving_color, undo.moved_piece_before, from, 1.0F);
        if (undo.captured_piece != PieceType::None) {
            add_piece_features(
                perspective,
                captured_color,
                undo.captured_piece,
                undo.captured_square,
                1.0F);
        }
    }
}

float NnueAccumulator::predict_normalized(const Position& pos) const {
    assert(initialized());
    const Color perspective = pos.side_to_move;
    const auto& accumulator = accumulators_[color_index(perspective)];
    assert(accumulator.size() == model_->hidden_size_);

    const auto aux = aux_features_for_perspective(pos, perspective);
    float output = model_->output_bias_;
    for (std::uint32_t row = 0; row < model_->hidden_size_; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row) * model_->aux_feature_count_;
        float value = accumulator[row];
        for (std::uint32_t col = 0; col < model_->aux_feature_count_; ++col) {
            value += model_->aux_projection_weights_[row_base + col] * static_cast<float>(aux[col]);
        }

        const float activation = std::clamp(value, 0.0F, 1.0F);
        output += model_->output_weights_[row] * activation;
    }

    return output;
}

float NnueAccumulator::evaluate_cp(const Position& pos) const {
    return predict_normalized(pos) * model_->target_scale_;
}

int NnueAccumulator::evaluate_cp_rounded(const Position& pos) const {
    return static_cast<int>(std::lround(evaluate_cp(pos)));
}

void NnueAccumulator::rebuild_perspective(const Position& pos, Color perspective) {
    assert(initialized());
    auto& accumulator = accumulators_[color_index(perspective)];
    accumulator = model_->accumulator_bias_;
    friendly_king_squares_[color_index(perspective)] =
        relative_square(perspective, king_square(pos, perspective));
    enemy_king_squares_[color_index(perspective)] =
        relative_square(perspective, king_square(pos, opposite(perspective)));

    for (Color piece_color : {perspective, opposite(perspective)}) {
        for (PieceType piece : EncodedPieces) {
            Bitboard bb = pos.pieces[color_index(piece_color)][static_cast<int>(piece)];
            while (bb) {
                const Square square = pop_lsb(bb);
                add_piece_features(perspective, piece_color, piece, square, 1.0F);
            }
        }
    }
}

void NnueAccumulator::add_piece_features(
    Color perspective,
    Color piece_color,
    PieceType piece,
    Square square,
    float sign
) {
    assert(piece != PieceType::None);
    const EncodedPieceSide encoded_side =
        piece_color == perspective ? EncodedPieceSide::Friendly : EncodedPieceSide::Enemy;
    const Square piece_square = relative_square(perspective, square);
    const int perspective_index = color_index(perspective);

    add_feature(
        perspective,
        feature_index(
            piece,
            encoded_side,
            EncodedKingContext::FriendlyKing,
            friendly_king_squares_[perspective_index],
            piece_square),
        sign);
    add_feature(
        perspective,
        feature_index(
            piece,
            encoded_side,
            EncodedKingContext::EnemyKing,
            enemy_king_squares_[perspective_index],
            piece_square),
        sign);
}

void NnueAccumulator::add_feature(Color perspective, FeatureIndex feature, float sign) {
    assert(initialized());
    assert(feature < model_->feature_count_);
    auto& accumulator = accumulators_[color_index(perspective)];
    const std::size_t base = static_cast<std::size_t>(feature) * model_->hidden_size_;
    for (std::uint32_t i = 0; i < model_->hidden_size_; ++i) {
        accumulator[i] += sign * model_->feature_weights_[base + i];
    }
}

} // namespace chess
