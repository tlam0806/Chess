#include "phase_quantized_nnue.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "board_encoder.hpp"

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
constexpr std::uint32_t Version = 1;
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
aux_features(const Position& pos, Color perspective) {
    std::array<std::uint8_t, PhaseQuantizedNnueModel::AuxFeatureCount> aux{};
    const Color enemy = opposite(perspective);
    aux[FriendlyCanCastleKingside] = can_castle_kingside(pos, perspective);
    aux[FriendlyCanCastleQueenside] = can_castle_queenside(pos, perspective);
    aux[EnemyCanCastleKingside] = can_castle_kingside(pos, enemy);
    aux[EnemyCanCastleQueenside] = can_castle_queenside(pos, enemy);
    if (pos.en_passant_square != NoSquare) {
        const Square square = relative_square(perspective, pos.en_passant_square);
        aux[HasEnPassant] = 1;
        aux[EnPassantFileA + file_of(square)] = 1;
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
    const std::array<std::uint32_t, 9> expected_prefix{
        Version,
        static_cast<std::uint32_t>(FeatureRowCount),
        static_cast<std::uint32_t>(PerspectiveAccumulatorSize),
        static_cast<std::uint32_t>(DenseInputSize),
        static_cast<std::uint32_t>(Hidden2Size),
        static_cast<std::uint32_t>(Hidden3Size),
        static_cast<std::uint32_t>(PhaseCount),
        static_cast<std::uint32_t>(PsqtBucketCount),
        static_cast<std::uint32_t>(AuxFeatureCount),
    };
    if (!std::equal(expected_prefix.begin(), expected_prefix.end(), header.begin())) {
        return false;
    }
    for (std::size_t index = expected_prefix.size(); index < header.size(); ++index) {
        if (header[index] == 0) {
            return false;
        }
    }

    std::array<std::int32_t, PerspectiveAccumulatorSize> accumulator_bias{};
    std::vector<FeatureRow> feature_rows(FeatureRowCount);
    std::array<std::int8_t, AuxFeatureCount * DenseInputSize> aux_weights{};
    std::array<DensePhase, PhaseCount> phases{};
    if (!read_array(input, accumulator_bias)) {
        return false;
    }

    const auto runtime_feature_row = [](std::size_t disk_row) {
        const std::size_t square = disk_row % 64;
        disk_row /= 64;
        const std::size_t king = disk_row % 64;
        disk_row /= 64;
        const std::size_t piece_side = disk_row % 2;
        const std::size_t piece = disk_row / 2;
        return (((king * 6 + piece) * 2 + piece_side) * 64) + square;
    };
    for (std::size_t disk_row = 0; disk_row < FeatureRowCount; ++disk_row) {
        if (!read_array(
                input,
                feature_rows[runtime_feature_row(disk_row)].positional)) {
            return false;
        }
    }
    if (!read_array(input, aux_weights)) {
        return false;
    }
    for (std::size_t disk_row = 0; disk_row < FeatureRowCount; ++disk_row) {
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
    accumulator_bias_ = accumulator_bias;
    feature_rows_ = std::move(feature_rows);
    aux_weights_ = aux_weights;
    phases_ = phases;
    initialize_candidate_kernel();
    return true;
}

int PhaseQuantizedNnueModel::evaluate(
    const Position& pos,
    const std::array<
        std::array<std::int32_t, PerspectiveAccumulatorSize>, 2>& accumulators,
    const std::array<
        std::array<std::int32_t, PsqtBucketCount>, 2>& psqt_accumulators,
    std::size_t piece_count
) const {
    assert(loaded());
    assert(piece_count > 0 && piece_count <= MaximumPositionPieces);
    const std::size_t stm = color_index(pos.side_to_move);
    const std::size_t opponent = 1 - stm;
    const auto aux = aux_features(pos, pos.side_to_move);

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
    if (uses_neon_dotprod_kernel()) {
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
    state.king_squares[index] = relative_square(
        perspective, king_square(pos, perspective));
    const std::size_t king =
        static_cast<std::size_t>(state.king_squares[index]);

    std::array<std::size_t, 64> active_rows{};
    std::size_t active_count = 0;
    for (Color piece_color : {Color::White, Color::Black}) {
        const std::size_t piece_side = piece_color == perspective ? 0 : 1;
        for (PieceType piece : PieceTypes) {
            const std::size_t piece_index = static_cast<std::size_t>(piece);
            const std::size_t row_base =
                ((king * PieceTypes.size() + piece_index) * 2 + piece_side)
                * BoardSize;
            Bitboard pieces =
                pos.pieces[static_cast<int>(piece_color)][static_cast<int>(piece)];
            while (pieces != EmptyBB) {
                assert(active_count < active_rows.size());
                const Square square =
                    static_cast<Square>(std::countr_zero(pieces));
                pieces &= pieces - 1;
                const std::size_t piece_square = static_cast<std::size_t>(
                    relative_square(perspective, square));
                const std::size_t row = row_base + piece_square;
                assert(row < PhaseQuantizedNnueModel::FeatureRowCount);
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
    const std::size_t king = static_cast<std::size_t>(
        current_state().king_squares[color_index(perspective)]);
    const std::size_t piece_square = static_cast<std::size_t>(
        relative_square(perspective, square));
    const std::size_t row = (((king * 6 + piece_index) * 2 + piece_side) * 64)
        + piece_square;
    assert(row < PhaseQuantizedNnueModel::FeatureRowCount);
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
        && state.piece_count == rebuilt_state.piece_count;
}

} // namespace chess
