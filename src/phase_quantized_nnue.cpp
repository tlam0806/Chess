#include "phase_quantized_nnue.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "board_encoder.hpp"
#if defined(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)
#include "phase_quantized_nnue_forward_backend.hpp"
#include "phase_quantized_nnue_stage_benchmark.hpp"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace chess {

namespace {

constexpr std::array<char, 8> Magic{'Q', 'P', 'H', 'N', 'U', 'E', '1', '\0'};
constexpr std::uint32_t LegacyVersion = 1;
constexpr std::uint32_t HorizontalMirrorVersion = 2;
constexpr std::int64_t MaximumPositionPieces = 64;
constexpr std::int64_t MaximumAbsInt8 = 128;
constexpr std::int64_t PositionalAccumulatorHeadroom =
    MaximumPositionPieces * MaximumAbsInt8;
constexpr std::int64_t AuxAccumulatorHeadroom =
    PhaseQuantizedNnueModel::AuxFeatureCount * MaximumAbsInt8;
constexpr std::int64_t TotalAccumulatorHeadroom =
    PositionalAccumulatorHeadroom + AuxAccumulatorHeadroom;

constexpr std::array<PieceType, 6> PieceTypes{
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
};

template<typename T>
bool read_exact(std::istream& input, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(input);
}

template<typename T, std::size_t Size>
bool read_array(std::istream& input, std::array<T, Size>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(sizeof(T) * Size));
    return static_cast<bool>(input);
}

constexpr std::size_t color_index(Color color) {
    return static_cast<std::size_t>(color);
}

bool can_castle_kingside(const Position& pos, Color color) {
    return color == Color::White
        ? pos.white_can_castle_kingside
        : pos.black_can_castle_kingside;
}

bool can_castle_queenside(const Position& pos, Color color) {
    return color == Color::White
        ? pos.white_can_castle_queenside
        : pos.black_can_castle_queenside;
}

std::array<std::uint8_t, PhaseQuantizedNnueModel::AuxFeatureCount>
aux_features(
    const Position& pos,
    Color perspective,
    std::uint8_t horizontal_mirror_mask
) {
    std::array<std::uint8_t, PhaseQuantizedNnueModel::AuxFeatureCount> aux{};
    const Color enemy = opposite(perspective);
    const bool mirrored = horizontal_mirror_mask != 0;
    aux[mirrored ? FriendlyCanCastleQueenside : FriendlyCanCastleKingside] =
        can_castle_kingside(pos, perspective);
    aux[mirrored ? FriendlyCanCastleKingside : FriendlyCanCastleQueenside] =
        can_castle_queenside(pos, perspective);
    aux[mirrored ? EnemyCanCastleQueenside : EnemyCanCastleKingside] =
        can_castle_kingside(pos, enemy);
    aux[mirrored ? EnemyCanCastleKingside : EnemyCanCastleQueenside] =
        can_castle_queenside(pos, enemy);
    if (pos.en_passant_square != NoSquare) {
        const Square square = relative_square(perspective, pos.en_passant_square);
        aux[HasEnPassant] = 1;
        aux[EnPassantFileA + (file_of(square) ^ horizontal_mirror_mask)] = 1;
    }
    return aux;
}

std::size_t count_pieces(const Position& pos) {
    std::size_t count = 0;
    for (Color color : {Color::White, Color::Black}) {
        for (PieceType piece : PieceTypes) {
            count += static_cast<std::size_t>(popcount(
                pos.pieces[static_cast<int>(color)][static_cast<int>(piece)]));
        }
    }
    return count;
}

bool has_runtime_accumulator_headroom(std::int32_t bias) {
    const std::int64_t value = bias;
    return value >= static_cast<std::int64_t>(
                        std::numeric_limits<std::int32_t>::min())
            + TotalAccumulatorHeadroom
        && value <= static_cast<std::int64_t>(
                        std::numeric_limits<std::int32_t>::max())
            - TotalAccumulatorHeadroom;
}

[[maybe_unused]] inline void add_bounded_int32(
    std::int32_t& value,
    std::int32_t delta
) {
#ifndef NDEBUG
    const std::int64_t next =
        static_cast<std::int64_t>(value) + static_cast<std::int64_t>(delta);
    assert(next >= std::numeric_limits<std::int32_t>::min());
    assert(next <= std::numeric_limits<std::int32_t>::max());
#endif
    value += delta;
}

void add_aux_weight_row(
    std::array<
        std::int32_t,
        PhaseQuantizedNnueModel::DenseInputSize>& dense_input,
    const std::int8_t* weights
) {
#if defined(__ARM_NEON)
    for (std::size_t lane = 0;
         lane < PhaseQuantizedNnueModel::DenseInputSize;
         lane += 16) {
        const int8x16_t weight8 = vld1q_s8(weights + lane);
        const int16x8_t weight16_low = vmovl_s8(vget_low_s8(weight8));
        const int16x8_t weight16_high = vmovl_high_s8(weight8);

        const int32x4_t value0 = vaddq_s32(
            vld1q_s32(dense_input.data() + lane),
            vmovl_s16(vget_low_s16(weight16_low)));
        const int32x4_t value1 = vaddq_s32(
            vld1q_s32(dense_input.data() + lane + 4),
            vmovl_high_s16(weight16_low));
        const int32x4_t value2 = vaddq_s32(
            vld1q_s32(dense_input.data() + lane + 8),
            vmovl_s16(vget_low_s16(weight16_high)));
        const int32x4_t value3 = vaddq_s32(
            vld1q_s32(dense_input.data() + lane + 12),
            vmovl_high_s16(weight16_high));

        vst1q_s32(dense_input.data() + lane, value0);
        vst1q_s32(dense_input.data() + lane + 4, value1);
        vst1q_s32(dense_input.data() + lane + 8, value2);
        vst1q_s32(dense_input.data() + lane + 12, value3);
    }
#else
    for (std::size_t lane = 0; lane < dense_input.size(); ++lane) {
        add_bounded_int32(
            dense_input[lane],
            static_cast<std::int32_t>(weights[lane]));
    }
#endif
}

} // namespace

bool PhaseQuantizedNnueModel::load(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        return false;
    }

    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != Magic) {
        return false;
    }

    std::array<std::uint32_t, 18> header{};
    if (!read_array(input, header)) {
        return false;
    }
    const bool legacy =
        header[0] == LegacyVersion
        && header[1] == static_cast<std::uint32_t>(FeatureRowCount);
    const bool horizontal_mirror =
        header[0] == HorizontalMirrorVersion
        && header[1]
            == static_cast<std::uint32_t>(HorizontalMirrorFeatureRowCount);
    if (!legacy && !horizontal_mirror) {
        return false;
    }
    const std::array<std::uint32_t, 7> expected_shape{
        static_cast<std::uint32_t>(PerspectiveAccumulatorSize),
        static_cast<std::uint32_t>(DenseInputSize),
        static_cast<std::uint32_t>(Hidden2Size),
        static_cast<std::uint32_t>(Hidden3Size),
        static_cast<std::uint32_t>(PhaseCount),
        static_cast<std::uint32_t>(PsqtBucketCount),
        static_cast<std::uint32_t>(AuxFeatureCount),
    };
    if (!std::equal(
            expected_shape.begin(), expected_shape.end(), header.begin() + 2)) {
        return false;
    }
    for (std::size_t index = 9; index < header.size(); ++index) {
        if (header[index] == 0) {
            return false;
        }
    }

    const std::size_t feature_row_count = horizontal_mirror
        ? HorizontalMirrorFeatureRowCount
        : FeatureRowCount;
    const std::size_t king_square_count = horizontal_mirror ? 32 : 64;
    std::array<std::int32_t, PerspectiveAccumulatorSize> accumulator_bias{};
    std::vector<FeatureRow> feature_rows(feature_row_count);
    std::array<std::int8_t, AuxFeatureCount * DenseInputSize> aux_weights{};
    std::array<DensePhase, PhaseCount> phases{};
    if (!read_array(input, accumulator_bias)) {
        return false;
    }

    const auto runtime_feature_row = [king_square_count](std::size_t disk_row) {
        const std::size_t square = disk_row % 64;
        disk_row /= 64;
        const std::size_t king = disk_row % king_square_count;
        disk_row /= king_square_count;
        const std::size_t piece_side = disk_row % 2;
        const std::size_t piece = disk_row / 2;
        return (((king * 6 + piece) * 2 + piece_side) * 64) + square;
    };
    for (std::size_t disk_row = 0; disk_row < feature_row_count; ++disk_row) {
        if (!read_array(
                input,
                feature_rows[runtime_feature_row(disk_row)].positional)) {
            return false;
        }
    }
    if (!read_array(input, aux_weights)) {
        return false;
    }
    for (std::size_t disk_row = 0; disk_row < feature_row_count; ++disk_row) {
        if (!read_array(
                input,
                feature_rows[runtime_feature_row(disk_row)].psqt)) {
            return false;
        }
    }
    for (DensePhase& phase : phases) {
        if (!read_array(input, phase.hidden2_bias)
            || !read_array(input, phase.hidden2_weight)
            || !read_array(input, phase.hidden3_bias)
            || !read_array(input, phase.hidden3_weight)
            || !read_exact(input, phase.output_bias)
            || !read_array(input, phase.output_weight)) {
            return false;
        }
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        return false;
    }

    constexpr std::int32_t MaximumPsqtWeight =
        std::numeric_limits<std::int32_t>::max() / 64;
    for (const FeatureRow& row : feature_rows) {
        for (std::int32_t weight : row.psqt) {
            if (weight < -MaximumPsqtWeight || weight > MaximumPsqtWeight) {
                return false;
            }
        }
    }
    for (std::int32_t bias : accumulator_bias) {
        if (!has_runtime_accumulator_headroom(bias)) {
            return false;
        }
    }

    hidden_clip_ = header[9];
    screlu_divisor_ = header[10];
    hidden2_scale_ = header[11];
    hidden3_scale_ = header[12];
    output_scale_ = header[13];
    feature_weight_scale_ = header[14];
    linear_weight_scale_ = header[15];
    output_weight_scale_ = header[16];
    psqt_scale_ = header[17];
    horizontal_mirror_ = horizontal_mirror;
    feature_row_count_ = feature_row_count;
    accumulator_bias_ = accumulator_bias;
    feature_rows_ = std::move(feature_rows);
    aux_weights_ = aux_weights;
    phases_ = phases;
    if (!initialize_candidate_kernel()) {
        // Backend selection is part of loading when the caller explicitly
        // requests one.  Do not leave an object that reports loaded()==true
        // after load() has returned false.
        feature_rows_.clear();
        feature_row_count_ = 0;
        horizontal_mirror_ = false;
        return false;
    }
    return true;
}

int PhaseQuantizedNnueModel::evaluate(
    const Position& pos,
    const std::array<
        std::array<std::int32_t, PerspectiveAccumulatorSize>, 2>& accumulators,
    const std::array<
        std::array<std::int32_t, PsqtBucketCount>, 2>& psqt_accumulators,
    const std::array<std::uint8_t, 2>& square_xor_masks,
    std::size_t piece_count
) const {
    assert(loaded());
    assert(piece_count > 0 && piece_count <= MaximumPositionPieces);
    const std::size_t stm = color_index(pos.side_to_move);
    const std::size_t opponent = 1 - stm;
    const auto aux = aux_features(
        pos,
        pos.side_to_move,
        static_cast<std::uint8_t>(square_xor_masks[stm] & 7U));

    const auto fill_dense_input = [&](auto& dense_input) {
        std::copy(
            accumulators[stm].begin(),
            accumulators[stm].end(),
            dense_input.begin());
        std::copy(
            accumulators[opponent].begin(),
            accumulators[opponent].end(),
            dense_input.begin() + PerspectiveAccumulatorSize);

        for (std::size_t feature = 0; feature < AuxFeatureCount; ++feature) {
            if (aux[feature] == 0) {
                continue;
            }
            add_aux_weight_row(
                dense_input,
                aux_weights_.data() + feature * DenseInputSize);
        }
    };

    const std::size_t phase_index = std::min<std::size_t>(
        (piece_count - 1) / 4, PhaseCount - 1);
    std::int64_t positional = 0;
    if (uses_accelerated_kernel()) {
        positional = forward_positional_candidate(
            accumulators[stm],
            accumulators[opponent],
            aux,
            phase_index);
#ifndef NDEBUG
        alignas(64) std::array<std::int32_t, DenseInputSize> dense_input;
        fill_dense_input(dense_input);
        assert(positional == forward_positional_scalar(
            dense_input, phase_index));
#endif
    } else {
        alignas(64) std::array<std::int32_t, DenseInputSize> dense_input;
        fill_dense_input(dense_input);
        positional = forward_positional_scalar(dense_input, phase_index);
    }
    const std::int64_t psqt = (
        static_cast<std::int64_t>(psqt_accumulators[stm][phase_index])
        - static_cast<std::int64_t>(psqt_accumulators[opponent][phase_index])
    ) / (2LL * static_cast<std::int64_t>(psqt_scale_));
    const std::int64_t total = positional + psqt;
    if (total < std::numeric_limits<int>::min()
        || total > std::numeric_limits<int>::max()) {
        throw std::overflow_error("NNUE output overflowed int");
    }
    return static_cast<int>(total);
}

int PhaseQuantizedNnueModel::evaluate_cp_rounded(const Position& pos) const {
    PhaseQuantizedNnueAccumulator accumulator;
    accumulator.reset(*this, pos);
    return accumulator.evaluate_cp_rounded(pos);
}

void PhaseQuantizedNnueAccumulator::reset(
    const PhaseQuantizedNnueModel& model,
    const Position& pos
) {
    assert(model.loaded());
    model_ = &model;
    states_.clear();
    if (states_.capacity() < PreallocatedStateCount) {
        states_.reserve(PreallocatedStateCount);
    }
    states_.push_back(NnueState{});
    current_ply_ = 0;

    NnueState& state = current_state();
    state.piece_count = count_pieces(pos);
    if (state.piece_count == 0 || state.piece_count > 64) {
        throw std::runtime_error("invalid physical piece count for NNUE");
    }
    rebuild_perspective(pos, Color::White);
    rebuild_perspective(pos, Color::Black);
}

void PhaseQuantizedNnueAccumulator::push_state() {
    assert(initialized());
    const std::size_t next_ply = current_ply_ + 1;
    if (next_ply == states_.size()) {
        if (states_.size() == states_.capacity()) {
            const std::size_t next_capacity = std::max<std::size_t>(
                states_.capacity() * 2,
                PreallocatedStateCount);
            states_.reserve(next_capacity);
        }
        states_.push_back(current_state());
    } else {
        assert(next_ply < states_.size());
        states_[next_ply] = current_state();
    }
    current_ply_ = next_ply;
}


void PhaseQuantizedNnueAccumulator::rebuild_perspective(
    const Position& pos,
    Color perspective
) {
    assert(model_ != nullptr);
    NnueState& state = current_state();
    const std::size_t index = color_index(perspective);
    const std::uint8_t vertical_mask =
        perspective == Color::Black ? 56U : 0U;
    const Square relative_king = static_cast<Square>(
        king_square(pos, perspective) ^ vertical_mask);
    const std::uint8_t horizontal_mask =
        model_->horizontal_mirror_ && file_of(relative_king) >= 4 ? 7U : 0U;
    const std::uint8_t square_xor_mask = vertical_mask ^ horizontal_mask;
    state.square_xor_masks[index] = square_xor_mask;
    state.king_squares[index] = static_cast<Square>(
        king_square(pos, perspective) ^ square_xor_mask);
    const std::size_t canonical_king =
        static_cast<std::size_t>(state.king_squares[index]);
    const std::size_t king = model_->horizontal_mirror_
        ? static_cast<std::size_t>(rank_of(state.king_squares[index]) * 4
            + file_of(state.king_squares[index]))
        : canonical_king;
    state.king_row_bases[index] = king * PieceTypes.size() * 2 * BoardSize;

    std::array<std::size_t, 64> active_rows{};
    std::size_t active_count = 0;
    for (Color piece_color : {Color::White, Color::Black}) {
        const std::size_t piece_side = piece_color == perspective ? 0 : 1;
        for (PieceType piece : PieceTypes) {
            const std::size_t piece_index = static_cast<std::size_t>(piece);
            const std::size_t row_base = state.king_row_bases[index]
                + (piece_index * 2 + piece_side) * BoardSize;
            Bitboard pieces =
                pos.pieces[static_cast<int>(piece_color)][static_cast<int>(piece)];
            while (pieces != EmptyBB) {
                assert(active_count < active_rows.size());
                const Square square =
                    static_cast<Square>(std::countr_zero(pieces));
                pieces &= pieces - 1;
                const std::size_t piece_square = static_cast<std::size_t>(
                    square ^ square_xor_mask);
                const std::size_t row = row_base + piece_square;
                assert(row < model_->feature_row_count_);
                active_rows[active_count++] = row;
            }
        }
    }

    constexpr std::size_t LaneBlock = 16;
    PerspectiveAccumulator& accumulator = state.accumulators[index];
    for (std::size_t block = 0;
         block < PhaseQuantizedNnueModel::PerspectiveAccumulatorSize;
         block += LaneBlock) {
        std::array<std::int32_t, LaneBlock> values{};
        for (std::size_t lane = 0; lane < LaneBlock; ++lane) {
            values[lane] = model_->accumulator_bias_[block + lane];
        }

        for (std::size_t row_index = 0;
             row_index < active_count;
             ++row_index) {
            const auto& weights =
                model_->feature_rows_[active_rows[row_index]].positional;
            for (std::size_t lane = 0; lane < LaneBlock; ++lane) {
                values[lane] += static_cast<std::int32_t>(
                    weights[block + lane]);
            }
        }

        for (std::size_t lane = 0; lane < LaneBlock; ++lane) {
            accumulator[block + lane] = values[lane];
        }
    }

    PsqtAccumulator& psqt = state.psqt[index];
    psqt.fill(0);
    for (std::size_t row_index = 0;
         row_index < active_count;
         ++row_index) {
        const auto& weights =
            model_->feature_rows_[active_rows[row_index]].psqt;
        for (std::size_t bucket = 0;
             bucket < PhaseQuantizedNnueModel::PsqtBucketCount;
             ++bucket) {
            psqt[bucket] += weights[bucket];
        }
    }
}

std::size_t PhaseQuantizedNnueAccumulator::feature_row(
    Color perspective,
    Color piece_color,
    PieceType piece,
    Square square
) const {
    assert(piece != PieceType::None);
    assert(is_valid_square(square));
    const std::size_t piece_index = static_cast<std::size_t>(piece);
    const std::size_t piece_side = piece_color == perspective ? 0 : 1;
    const std::size_t perspective_index = color_index(perspective);
    const std::size_t piece_square = static_cast<std::size_t>(
        square ^ current_state().square_xor_masks[perspective_index]);
    const std::size_t row = current_state().king_row_bases[perspective_index]
        + (piece_index * 2 + piece_side) * 64
        + piece_square;
    assert(row < model_->feature_row_count_);
    return row;
}

template<std::size_t AddedCount, std::size_t RemovedCount>
void PhaseQuantizedNnueAccumulator::update_features(
    Color perspective,
    const std::array<std::size_t, AddedCount>& added,
    const std::array<std::size_t, RemovedCount>& removed
) {
    assert(model_ != nullptr);
    const std::size_t perspective_index = color_index(perspective);
    const PhaseQuantizedNnueModel::FeatureRow* feature_rows =
        model_->feature_rows_.data();
    PerspectiveAccumulator& accumulator =
        current_state().accumulators[perspective_index];

    for (std::size_t lane = 0;
         lane < PhaseQuantizedNnueModel::PerspectiveAccumulatorSize;
         ++lane) {
        std::int32_t delta = 0;
        for (std::size_t row : added) {
            delta += static_cast<std::int32_t>(
                feature_rows[row].positional[lane]);
        }
        for (std::size_t row : removed) {
            delta -= static_cast<std::int32_t>(
                feature_rows[row].positional[lane]);
        }
        accumulator[lane] += delta;
    }

    PsqtAccumulator psqt_value = current_state().psqt[perspective_index];

    for (std::size_t row : added) {
        for (std::size_t bucket = 0;
             bucket < PhaseQuantizedNnueModel::PsqtBucketCount;
             ++bucket) {
            psqt_value[bucket] += feature_rows[row].psqt[bucket];
        }
    }

    for (std::size_t row : removed) {
        for (std::size_t bucket = 0;
             bucket < PhaseQuantizedNnueModel::PsqtBucketCount;
             ++bucket) {
            psqt_value[bucket] -= feature_rows[row].psqt[bucket];
        }
    }

    current_state().psqt[perspective_index] = psqt_value;
}

PhaseQuantizedNnueUndo PhaseQuantizedNnueAccumulator::make_move_with_undo(
    Move move,
    Color moving_color,
    PieceType moved_piece,
    PieceType captured_piece,
    const Position& after
) {
    assert(initialized());
    assert(moved_piece != PieceType::None);
    const PhaseQuantizedNnueUndo undo{current_ply_};
    push_state();

    const PieceType moved_piece_after = promotion_piece(move) == PieceType::None
        ? moved_piece
        : promotion_piece(move);
    Square captured_square = NoSquare;
    if (captured_piece != PieceType::None) {
        captured_square = move.flag() == MoveFlag::EnPassant
            ? (moving_color == Color::White ? move.to() - 8 : move.to() + 8)
            : move.to();
    }

    if (moved_piece == PieceType::King) {
        // Moving the king changes the king-square dimension of every feature
        // only for that king's own perspective. The opposite perspective is
        // still anchored to the opposite king and can be updated incrementally.
        const Color opposite_perspective = opposite(moving_color);
        if (move.flag() == MoveFlag::KingCastle
            || move.flag() == MoveFlag::QueenCastle) {
            assert(captured_piece == PieceType::None);
            const int home_rank =
                moving_color == Color::White ? 0 : 7;
            const bool king_side = move.flag() == MoveFlag::KingCastle;
            const Square rook_from = make_square(
                king_side ? 7 : 0, home_rank);
            const Square rook_to = make_square(
                king_side ? 5 : 3, home_rank);
            update_features(
                opposite_perspective,
                std::array<std::size_t, 2>{
                    feature_row(
                        opposite_perspective,
                        moving_color,
                        PieceType::King,
                        move.to()),
                    feature_row(
                        opposite_perspective,
                        moving_color,
                        PieceType::Rook,
                        rook_to),
                },
                std::array<std::size_t, 2>{
                    feature_row(
                        opposite_perspective,
                        moving_color,
                        PieceType::King,
                        move.from()),
                    feature_row(
                        opposite_perspective,
                        moving_color,
                        PieceType::Rook,
                        rook_from),
                });
        } else if (captured_piece != PieceType::None) {
            update_features(
                opposite_perspective,
                std::array<std::size_t, 1>{
                    feature_row(
                        opposite_perspective,
                        moving_color,
                        PieceType::King,
                        move.to()),
                },
                std::array<std::size_t, 2>{
                    feature_row(
                        opposite_perspective,
                        moving_color,
                        PieceType::King,
                        move.from()),
                    feature_row(
                        opposite_perspective,
                        opposite(moving_color),
                        captured_piece,
                        captured_square),
                });
        } else {
            update_features(
                opposite_perspective,
                std::array<std::size_t, 1>{
                    feature_row(
                        opposite_perspective,
                        moving_color,
                        PieceType::King,
                        move.to()),
                },
                std::array<std::size_t, 1>{
                    feature_row(
                        opposite_perspective,
                        moving_color,
                        PieceType::King,
                        move.from()),
                });
        }

        if (captured_piece != PieceType::None) {
            assert(current_state().piece_count > 0);
            --current_state().piece_count;
        }
        rebuild_perspective(after, moving_color);
        return undo;
    }

    for (Color perspective : {Color::White, Color::Black}) {
        if (captured_piece != PieceType::None) {
            update_features(
                perspective,
                std::array<std::size_t, 1>{
                    feature_row(
                        perspective,
                        moving_color,
                        moved_piece_after,
                        move.to()),
                },
                std::array<std::size_t, 2>{
                    feature_row(
                        perspective,
                        moving_color,
                        moved_piece,
                        move.from()),
                    feature_row(
                        perspective,
                        opposite(moving_color),
                        captured_piece,
                        captured_square),
                });
        } else {
            update_features(
                perspective,
                std::array<std::size_t, 1>{
                    feature_row(
                        perspective,
                        moving_color,
                        moved_piece_after,
                        move.to()),
                },
                std::array<std::size_t, 1>{
                    feature_row(
                        perspective,
                        moving_color,
                        moved_piece,
                        move.from()),
                });
        }
    }
    if (captured_piece != PieceType::None) {
        assert(current_state().piece_count > 0);
        --current_state().piece_count;
    }
    return undo;
}

void PhaseQuantizedNnueAccumulator::undo(const PhaseQuantizedNnueUndo& undo) {
    assert(initialized());
    assert(current_ply_ == undo.ply_before + 1);
    current_ply_ = undo.ply_before;
}

int PhaseQuantizedNnueAccumulator::evaluate_cp_rounded(const Position& pos) const {
    assert(initialized());
    const NnueState& state = current_state();
    return model_->evaluate(
        pos,
        state.accumulators,
        state.psqt,
        state.square_xor_masks,
        state.piece_count);
}

bool PhaseQuantizedNnueAccumulator::matches_full_recompute(
    const Position& pos
) const {
    assert(initialized());
    PhaseQuantizedNnueAccumulator rebuilt;
    rebuilt.reset(*model_, pos);
    const NnueState& state = current_state();
    const NnueState& rebuilt_state = rebuilt.current_state();
    return state.accumulators == rebuilt_state.accumulators
        && state.psqt == rebuilt_state.psqt
        && state.king_squares == rebuilt_state.king_squares
        && state.square_xor_masks == rebuilt_state.square_xor_masks
        && state.king_row_bases == rebuilt_state.king_row_bases
        && state.piece_count == rebuilt_state.piece_count;
}

#if defined(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)

PhaseNnueStageBenchmarkResult benchmark_phase_nnue_forward_stages(
    const PhaseQuantizedNnueModel& model,
    const std::vector<Position>& positions,
    std::size_t warmup_evaluations,
    const std::array<std::size_t, 4>& evaluations_per_stage,
    std::size_t repeats
) {
    if (!model.loaded() || !model.uses_accelerated_kernel()) {
        throw std::runtime_error(
            "stage benchmark requires a loaded accelerated NNUE kernel");
    }
    if (positions.empty()
        || std::ranges::any_of(evaluations_per_stage, [](std::size_t value) {
            return value == 0;
        })
        || repeats == 0) {
        throw std::runtime_error("stage benchmark counts must be positive");
    }

    constexpr std::uint64_t ChecksumSeed = 0xcbf29ce484222325ULL;
    std::array<std::uint64_t, 4> expected{
        ChecksumSeed, ChecksumSeed, ChecksumSeed, ChecksumSeed};
    struct ReferenceSample {
        std::array<std::uint16_t, PhaseQuantizedNnueModel::Hidden2Size>
            hidden2{};
        std::array<std::uint16_t, PhaseQuantizedNnueModel::Hidden3Size>
            hidden3{};
        std::int64_t output = 0;
    };
    std::vector<PhaseForwardBenchmarkSample> samples;
    samples.reserve(positions.size());
    std::vector<ReferenceSample> reference_samples;
    reference_samples.reserve(positions.size());
    std::uint64_t corpus_checksum = ChecksumSeed;

    for (const Position& pos : positions) {
        PhaseQuantizedNnueAccumulator accumulator;
        accumulator.reset(model, pos);
        const auto& positional = accumulator.positional_accumulators();
        const std::size_t stm = color_index(pos.side_to_move);
        const std::size_t opponent = 1 - stm;
        const std::uint8_t horizontal_mask =
            model.horizontal_mirror_
                && file_of(pos.king_squares[stm]) >= 4
            ? 7U
            : 0U;
        const auto aux = aux_features(
            pos, pos.side_to_move, horizontal_mask);

        PhaseForwardBenchmarkSample sample;
        sample.stm = positional[stm];
        sample.opponent = positional[opponent];
        std::size_t castling_state = 0;
        for (std::size_t feature = 0; feature < 4; ++feature) {
            castling_state |= static_cast<std::size_t>(aux[feature]) << feature;
        }
        std::size_t en_passant_state = 0;
        for (std::size_t file = 0; file < 8; ++file) {
            if (aux[EnPassantFileA + file] != 0) {
                en_passant_state = file + 1;
            }
        }
        sample.aux_state = castling_state * 9 + en_passant_state;
        sample.phase_index = std::min<std::size_t>(
            (accumulator.piece_count() - 1) / 4,
            PhaseQuantizedNnueModel::PhaseCount - 1);
        samples.push_back(sample);

        for (std::int32_t value : sample.stm) {
            corpus_checksum = phase_nnue_detail::phase_benchmark_mix(
                corpus_checksum, static_cast<std::uint32_t>(value));
        }
        for (std::int32_t value : sample.opponent) {
            corpus_checksum = phase_nnue_detail::phase_benchmark_mix(
                corpus_checksum, static_cast<std::uint32_t>(value));
        }
        corpus_checksum = phase_nnue_detail::phase_benchmark_mix(
            corpus_checksum, sample.aux_state);
        corpus_checksum = phase_nnue_detail::phase_benchmark_mix(
            corpus_checksum, sample.phase_index);

        alignas(64) std::array<
            std::int32_t,
            PhaseQuantizedNnueModel::DenseInputSize> dense{};
        std::copy(sample.stm.begin(), sample.stm.end(), dense.begin());
        std::copy(
            sample.opponent.begin(),
            sample.opponent.end(),
            dense.begin() + PhaseQuantizedNnueModel::PerspectiveAccumulatorSize);
        for (std::size_t feature = 0;
             feature < PhaseQuantizedNnueModel::AuxFeatureCount;
             ++feature) {
            if (aux[feature] != 0) {
                add_aux_weight_row(
                    dense,
                    model.aux_weights_.data()
                        + feature * PhaseQuantizedNnueModel::DenseInputSize);
            }
        }

        const auto& phase = model.phases_[sample.phase_index];
        std::array<std::int32_t, PhaseQuantizedNnueModel::DenseInputSize>
            hidden1{};
        for (std::size_t input = 0; input < hidden1.size(); ++input) {
            const std::int64_t clipped = std::clamp<std::int64_t>(
                dense[input], 0, model.hidden_clip_);
            hidden1[input] = static_cast<std::int32_t>(
                clipped * clipped / model.screlu_divisor_);
        }
        std::array<std::uint16_t, PhaseQuantizedNnueModel::Hidden2Size>
            hidden2{};
        for (std::size_t output = 0; output < hidden2.size(); ++output) {
            std::int64_t sum = phase.hidden2_bias[output];
            for (std::size_t input = 0; input < hidden1.size(); ++input) {
                sum += static_cast<std::int64_t>(hidden1[input])
                    * phase.hidden2_weight[output * hidden1.size() + input];
            }
            hidden2[output] = static_cast<std::uint16_t>(
                std::clamp<std::int64_t>(
                    sum / model.hidden2_scale_, 0, 65535));
            expected[0] = phase_nnue_detail::phase_benchmark_mix(
                expected[0], hidden2[output]);
        }
        std::array<std::uint16_t, PhaseQuantizedNnueModel::Hidden3Size>
            hidden3{};
        for (std::size_t output = 0; output < hidden3.size(); ++output) {
            std::int64_t sum = phase.hidden3_bias[output];
            for (std::size_t input = 0; input < hidden2.size(); ++input) {
                sum += static_cast<std::int64_t>(hidden2[input])
                    * phase.hidden3_weight[output * hidden2.size() + input];
            }
            hidden3[output] = static_cast<std::uint16_t>(
                std::clamp<std::int64_t>(
                    sum / model.hidden3_scale_, 0, 65535));
            expected[1] = phase_nnue_detail::phase_benchmark_mix(
                expected[1], hidden3[output]);
        }
        std::int64_t output = phase.output_bias;
        for (std::size_t input = 0; input < hidden3.size(); ++input) {
            output += static_cast<std::int64_t>(hidden3[input])
                * phase.output_weight[input];
        }
        output /= model.output_scale_;
        expected[2] = phase_nnue_detail::phase_benchmark_mix(
            expected[2], static_cast<std::uint64_t>(output));
        expected[3] = phase_nnue_detail::phase_benchmark_mix(
            expected[3], static_cast<std::uint64_t>(output));
        reference_samples.push_back({hidden2, hidden3, output});
    }

    const auto expected_timed_sink = [&reference_samples](
        PhaseForwardBenchmarkStage stage,
        std::size_t evaluations
    ) {
        std::uint64_t sink = 0;
        std::size_t sample_index = 0;
        for (std::size_t iteration = 0; iteration < evaluations; ++iteration) {
            const auto& sample = reference_samples[sample_index];
            if (stage == PhaseForwardBenchmarkStage::InputToHidden2) {
                sink += sample.hidden2[
                    iteration & (sample.hidden2.size() - 1)];
            } else if (
                stage == PhaseForwardBenchmarkStage::Hidden2ToHidden3) {
                sink += sample.hidden3[
                    iteration & (sample.hidden3.size() - 1)];
            } else {
                sink += static_cast<std::uint64_t>(sample.output);
            }
            if (++sample_index == reference_samples.size()) {
                sample_index = 0;
            }
        }
        return sink;
    };

    constexpr std::array<PhaseForwardBenchmarkStage, 4> StageIds{
        PhaseForwardBenchmarkStage::InputToHidden2,
        PhaseForwardBenchmarkStage::Hidden2ToHidden3,
        PhaseForwardBenchmarkStage::Hidden3ToOutput,
        PhaseForwardBenchmarkStage::Full,
    };
    std::array<std::uint64_t, StageIds.size()> expected_warmup_sinks{};
    std::array<std::uint64_t, StageIds.size()> expected_repeat_sinks{};
    for (std::size_t index = 0; index < StageIds.size(); ++index) {
        expected_warmup_sinks[index] = expected_timed_sink(
            StageIds[index], warmup_evaluations);
        expected_repeat_sinks[index] = expected_timed_sink(
            StageIds[index], evaluations_per_stage[index]);
    }
    PhaseNnueStageBenchmarkResult result;
    result.kernel = std::string(model.forward_kernel_name());
    result.sample_count = samples.size();
    result.corpus_checksum = corpus_checksum;
    for (const auto& sample : samples) {
        ++result.phase_counts[sample.phase_index];
        result.nonzero_aux_samples += sample.aux_state != 0;
    }
    result.scalar_parity = true;
    result.stage_parity = true;
    result.timed_sink_parity = true;
    result.stages = {
        {"s1_input_to_hidden2", evaluations_per_stage[0], {}, {}, 0, 0},
        {"s2_hidden2_to_hidden3", evaluations_per_stage[1], {}, {}, 0, 0},
        {"s3_hidden3_to_output", evaluations_per_stage[2], {}, {}, 0, 0},
        {"full", evaluations_per_stage[3], {}, {}, 0, 0},
    };

    for (std::size_t index = 0; index < StageIds.size(); ++index) {
        const PhaseForwardBenchmarkStage stage = StageIds[index];
        if (warmup_evaluations != 0) {
            const auto warmup = model.candidate_kernel_->benchmark_stage(
                stage, samples, warmup_evaluations);
            if (!warmup.supported) {
                throw std::runtime_error(
                    "selected NNUE kernel does not expose stage benchmarking");
            }
            result.timed_sink_parity = result.timed_sink_parity
                && warmup.timed_sink
                    == expected_warmup_sinks[index];
        }
    }
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        // Rotate the four stages so no stage consistently receives the same
        // thermal/order position across repeats.
        for (std::size_t offset = 0; offset < StageIds.size(); ++offset) {
            const std::size_t index = (repeat + offset) % StageIds.size();
            const auto batch = model.candidate_kernel_->benchmark_stage(
                StageIds[index], samples, evaluations_per_stage[index]);
            if (!batch.supported) {
                throw std::runtime_error(
                    "selected NNUE kernel does not expose stage benchmarking");
            }
            auto& stage = result.stages[index];
            const bool first_batch = stage.elapsed_ns.empty();
            stage.elapsed_ns.push_back(batch.elapsed_ns);
            stage.cpu_ns.push_back(batch.cpu_ns);
            result.batches.push_back({
                repeat,
                offset,
                stage.name,
                evaluations_per_stage[index],
                batch.elapsed_ns,
                batch.cpu_ns,
            });
            if (first_batch) {
                stage.checksum = batch.checksum;
                stage.timed_sink = batch.timed_sink;
            }
            result.stage_parity = result.stage_parity
                && batch.checksum == expected[index]
                && stage.checksum == batch.checksum;
            result.timed_sink_parity = result.timed_sink_parity
                && batch.timed_sink
                    == expected_repeat_sinks[index]
                && stage.timed_sink == batch.timed_sink;
            if (index == 3) {
                result.scalar_parity = result.scalar_parity
                    && batch.checksum == expected[3];
            }
        }
    }
    return result;
}

#endif

} // namespace chess
