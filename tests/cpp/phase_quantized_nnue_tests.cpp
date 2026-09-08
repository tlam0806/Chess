#include "move.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <array>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

chess::Move legal_move(const chess::Position& pos, std::string_view uci) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    throw std::runtime_error("missing legal move: " + std::string(uci));
}

chess::PieceType captured_piece(const chess::Position& pos, chess::Move move) {
    if (!chess::is_capture(move)) {
        return chess::PieceType::None;
    }
    const chess::Square square = move.flag() == chess::MoveFlag::EnPassant
        ? (pos.side_to_move == chess::Color::White ? move.to() - 8 : move.to() + 8)
        : move.to();
    return pos.piece_type_on_occupied(square);
}

void assert_state(
    const chess::PhaseQuantizedNnueModel& model,
    const chess::PhaseQuantizedNnueAccumulator& accumulator,
    const chess::Position& pos
) {
    assert(accumulator.matches_full_recompute(pos));
    assert(accumulator.evaluate_cp_rounded(pos) == model.evaluate_cp_rounded(pos));
}

void play_and_check(
    const chess::PhaseQuantizedNnueModel& model,
    chess::PhaseQuantizedNnueAccumulator& accumulator,
    chess::Position& pos,
    std::string_view uci
) {
    const chess::Position before = pos;
    const chess::Move move = legal_move(pos, uci);
    const chess::Color moving_color = pos.side_to_move;
    const chess::PieceType moved_piece = pos.piece_type_on_occupied(move.from());
    const chess::PieceType captured = captured_piece(pos, move);
    pos.make_move(move, moved_piece, captured);

    const chess::PhaseQuantizedNnueUndo undo = accumulator.make_move_with_undo(
        move, moving_color, moved_piece, captured, pos);
    assert_state(model, accumulator, pos);

    // Undo must restore both positional and PSQT accumulators exactly, then
    // applying the same move again must return to the exact rebuilt state.
    accumulator.undo(undo);
    assert_state(model, accumulator, before);
    static_cast<void>(accumulator.make_move_with_undo(
        move, moving_color, moved_piece, captured, pos));
    assert_state(model, accumulator, pos);
}

void run_sequence(
    const chess::PhaseQuantizedNnueModel& model,
    std::string_view fen,
    const std::vector<std::string_view>& moves
) {
    chess::Position pos;
    assert(pos.set_fen(fen));
    chess::PhaseQuantizedNnueAccumulator accumulator;
    accumulator.reset(model, pos);
    assert_state(model, accumulator, pos);
    for (std::string_view move : moves) {
        play_and_check(model, accumulator, pos, move);
    }
}

void assert_unsafe_accumulator_bias_is_rejected(
    const std::filesystem::path& model_path
) {
    const std::filesystem::path unsafe_path =
        std::filesystem::temp_directory_path()
        / "chess_phase_nnue_unsafe_accumulator_bias.bin";
    std::filesystem::copy_file(
        model_path,
        unsafe_path,
        std::filesystem::copy_options::overwrite_existing);

    {
        std::fstream output(
            unsafe_path,
            std::ios::binary | std::ios::in | std::ios::out);
        assert(output);
        constexpr std::streamoff AccumulatorBiasOffset =
            8 + 18 * sizeof(std::uint32_t);
        const std::int32_t unsafe_bias =
            std::numeric_limits<std::int32_t>::max();
        output.seekp(AccumulatorBiasOffset);
        output.write(
            reinterpret_cast<const char*>(&unsafe_bias),
            sizeof(unsafe_bias));
        assert(output);
    }

    chess::PhaseQuantizedNnueModel unsafe_model;
    assert(!unsafe_model.load(unsafe_path.string()));
    assert(std::filesystem::remove(unsafe_path));
}

void assert_failed_backend_request_leaves_model_unloaded(
    const std::filesystem::path& model_path
) {
    const char* previous_value = std::getenv("CHESS_NNUE_BACKEND");
    const bool had_previous_value = previous_value != nullptr;
    const std::string previous = had_previous_value
        ? std::string(previous_value)
        : std::string{};

#if defined(_WIN32)
    assert(_putenv_s("CHESS_NNUE_BACKEND", "not-a-kernel") == 0);
#else
    assert(setenv("CHESS_NNUE_BACKEND", "not-a-kernel", 1) == 0);
#endif

    chess::PhaseQuantizedNnueModel failed_model;
    assert(!failed_model.load(model_path.string()));
    assert(!failed_model.loaded());
    assert(!failed_model.has_candidate_kernel());

#if defined(_WIN32)
    assert(_putenv_s(
        "CHESS_NNUE_BACKEND",
        had_previous_value ? previous.c_str() : "") == 0);
#else
    if (had_previous_value) {
        assert(setenv("CHESS_NNUE_BACKEND", previous.c_str(), 1) == 0);
    } else {
        assert(unsetenv("CHESS_NNUE_BACKEND") == 0);
    }
#endif
}

void set_accumulator_backend_environment(const char* value) {
#if defined(_WIN32)
    assert(_putenv_s(
        "CHESS_NNUE_ACCUMULATOR_BACKEND",
        value == nullptr ? "" : value) == 0);
#else
    if (value == nullptr) {
        assert(unsetenv("CHESS_NNUE_ACCUMULATOR_BACKEND") == 0);
    } else {
        assert(setenv("CHESS_NNUE_ACCUMULATOR_BACKEND", value, 1) == 0);
    }
#endif
}

void assert_accumulator_backend_dispatch(
    const std::filesystem::path& model_path
) {
    const char* previous_value =
        std::getenv("CHESS_NNUE_ACCUMULATOR_BACKEND");
    const bool had_previous_value = previous_value != nullptr;
    const std::string previous = had_previous_value
        ? std::string(previous_value)
        : std::string{};

    set_accumulator_backend_environment("portable");
    chess::PhaseQuantizedNnueModel portable_model;
    assert(portable_model.load(model_path.string()));
    assert(portable_model.accumulator_kernel_name() == "portable");
    run_sequence(
        portable_model,
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        {"e5d6", "e8c8"});

    set_accumulator_backend_environment("not-a-kernel");
    chess::PhaseQuantizedNnueModel failed_model;
    assert(!failed_model.load(model_path.string()));
    assert(!failed_model.loaded());
    assert(failed_model.accumulator_kernel_name() == "portable");

    set_accumulator_backend_environment(
        had_previous_value ? previous.c_str() : nullptr);
}

void overwrite_header_value(
    const std::filesystem::path& model_path,
    std::size_t header_index,
    std::uint32_t value
) {
    std::fstream output(
        model_path,
        std::ios::binary | std::ios::in | std::ios::out);
    assert(output);
    constexpr std::streamoff MagicSize = 8;
    output.seekp(
        MagicSize
        + static_cast<std::streamoff>(header_index * sizeof(std::uint32_t)));
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    assert(output);
}

void assert_neon_scale_dispatch_parity(
    const std::filesystem::path& model_path
) {
    chess::PhaseQuantizedNnueModel base_model;
    assert(base_model.load(model_path.string()));
    if (!base_model.uses_neon_dotprod_kernel()) {
        return;
    }

    constexpr std::array<std::array<std::uint32_t, 3>, 3> ScaleTuples{{
        {2, 8, 128},
        {4, 8, 64},
        {8, 16, 128},
    }};
    constexpr std::array<std::string_view, 3> Fens{
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        "2r3k1/5ppp/1p2p3/p2pP3/P2P1P2/1P1B2P1/5K1P/2R5 b - - 0 28",
    };

    for (const auto& scales : ScaleTuples) {
        const std::filesystem::path scale_path =
            std::filesystem::temp_directory_path()
            / (
                "chess_phase_nnue_scale_"
                + std::to_string(scales[0])
                + "_"
                + std::to_string(scales[1])
                + "_"
                + std::to_string(scales[2])
                + ".bin");
        std::filesystem::copy_file(
            model_path,
            scale_path,
            std::filesystem::copy_options::overwrite_existing);
        overwrite_header_value(scale_path, 11, scales[0]);
        overwrite_header_value(scale_path, 12, scales[1]);
        overwrite_header_value(scale_path, 13, scales[2]);

        chess::PhaseQuantizedNnueModel model;
        assert(model.load(scale_path.string()));
        assert(model.has_candidate_kernel());
        assert(model.uses_neon_dotprod_kernel());
        for (std::string_view fen : Fens) {
            chess::Position pos;
            assert(pos.set_fen(fen));
            model.set_neon_dotprod_enabled(true);
            const int accelerated = model.evaluate_cp_rounded(pos);
            model.set_neon_dotprod_enabled(false);
            const int scalar = model.evaluate_cp_rounded(pos);
            assert(accelerated == scalar);
        }
        assert(std::filesystem::remove(scale_path));
    }

    const std::filesystem::path fallback_path =
        std::filesystem::temp_directory_path()
        / "chess_phase_nnue_unsupported_scale.bin";
    std::filesystem::copy_file(
        model_path,
        fallback_path,
        std::filesystem::copy_options::overwrite_existing);
    overwrite_header_value(fallback_path, 11, 3);
    chess::PhaseQuantizedNnueModel fallback_model;
    const char* requested_backend = std::getenv("CHESS_NNUE_BACKEND");
    const bool explicitly_forced_accelerated = requested_backend != nullptr
        && std::string_view(requested_backend) != ""
        && std::string_view(requested_backend) != "auto"
        && std::string_view(requested_backend) != "scalar";
    if (explicitly_forced_accelerated) {
        assert(!fallback_model.load(fallback_path.string()));
        assert(!fallback_model.loaded());
        assert(!fallback_model.has_candidate_kernel());
    } else {
        assert(fallback_model.load(fallback_path.string()));
        assert(!fallback_model.has_candidate_kernel());
        assert(!fallback_model.uses_neon_dotprod_kernel());
    }
    assert(std::filesystem::remove(fallback_path));
}

void assert_all_aux_state_parity(chess::PhaseQuantizedNnueModel& model) {
    if (!model.uses_neon_dotprod_kernel()) {
        return;
    }

    constexpr std::string_view CastlingSymbols = "KQkq";
    for (char side_to_move : {'w', 'b'}) {
        for (std::size_t castling_state = 0;
             castling_state < 16;
             ++castling_state) {
            std::string castling;
            for (std::size_t right = 0;
                 right < CastlingSymbols.size();
                 ++right) {
                if ((castling_state & (1U << right)) != 0) {
                    castling.push_back(CastlingSymbols[right]);
                }
            }
            if (castling.empty()) {
                castling = "-";
            }

            for (std::size_t en_passant_state = 0;
                 en_passant_state < 9;
                 ++en_passant_state) {
                std::string en_passant = "-";
                if (en_passant_state != 0) {
                    en_passant.clear();
                    en_passant.push_back(static_cast<char>(
                        'a' + en_passant_state - 1));
                    en_passant.push_back('3');
                }
                const std::string fen =
                    "r3k2r/8/8/8/8/8/8/R3K2R "
                    + std::string(1, side_to_move)
                    + " "
                    + castling
                    + " "
                    + en_passant
                    + " 0 1";

                chess::Position pos;
                assert(pos.set_fen(fen));
                model.set_neon_dotprod_enabled(true);
                const int accelerated = model.evaluate_cp_rounded(pos);
                model.set_neon_dotprod_enabled(false);
                const int scalar = model.evaluate_cp_rounded(pos);
                assert(accelerated == scalar);
            }
        }
    }
    model.set_neon_dotprod_enabled(true);
}

void assert_horizontal_mirror_invariance(
    const chess::PhaseQuantizedNnueModel& model
) {
    if (!model.uses_horizontal_mirror()) {
        return;
    }
    constexpr std::array<std::array<std::string_view, 2>, 3> Pairs{{
        {{
            "7k/8/8/3pP3/8/2N5/8/4K3 w - d6 0 1",
            "k7/8/8/3Pp3/8/5N2/8/3K4 w - e6 0 1",
        }},
        {{
            "2r3k1/5ppp/1p2p3/p2pP3/P2P1P2/1P1B2P1/5K1P/2R5 b - - 0 28",
            "1k3r2/ppp5/3p2p1/3Pp2p/2P1P2P/1P2B1P1/P1K5/5R2 b - - 0 28",
        }},
        {{
            "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",
            "r2k3r/8/8/8/8/8/8/R2K3R w Qk - 0 1",
        }},
    }};
    for (const auto& pair : Pairs) {
        chess::Position original;
        chess::Position mirrored;
        assert(original.set_fen(pair[0]));
        assert(mirrored.set_fen(pair[1]));
        assert(model.evaluate_cp_rounded(original)
            == model.evaluate_cp_rounded(mirrored));
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: phase_quantized_nnue_tests MODEL.bin [PARITY.tsv]\n";
        return 2;
    }
    chess::PhaseQuantizedNnueModel model;
    assert(model.load(argv[1]));
    assert(model.feature_row_count() == (
        model.uses_horizontal_mirror()
            ? chess::PhaseQuantizedNnueModel::HorizontalMirrorFeatureRowCount
            : chess::PhaseQuantizedNnueModel::FeatureRowCount));
    assert_unsafe_accumulator_bias_is_rejected(argv[1]);
    assert_failed_backend_request_leaves_model_unloaded(argv[1]);
    assert_accumulator_backend_dispatch(argv[1]);
    assert_neon_scale_dispatch_parity(argv[1]);
    assert_all_aux_state_parity(model);
    assert_horizontal_mirror_invariance(model);
    assert(model.hidden_clip() == 181);
    assert(model.screlu_divisor() == 128);
    assert(model.hidden2_scale() > 0);
    assert(model.hidden3_scale() > 0);
    assert(model.output_scale() > 0);
    assert(model.psqt_scale() == 16);
    assert(model.has_candidate_kernel() == model.uses_accelerated_kernel());
    assert((model.forward_kernel_name() == "scalar")
        == !model.uses_accelerated_kernel());

    run_sequence(
        model,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"});
    run_sequence(
        model,
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        {"e5d6", "e8c8"});
    run_sequence(
        model,
        "7k/P7/8/8/8/8/8/7K w - - 0 1",
        {"a7a8q"});
    run_sequence(
        model,
        "1r5k/P7/8/8/8/8/8/7K w - - 0 1",
        {"a7b8q"});
    run_sequence(
        model,
        "7k/8/8/8/8/8/p7/7K b - - 0 1",
        {"a2a1q"});
    run_sequence(
        model,
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        {"e1d1"});
    run_sequence(
        model,
        "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1",
        {"e1d2"});

    if (argc == 3) {
        std::ifstream input(argv[2]);
        assert(input);
        std::string line;
        std::size_t checked = 0;
        while (std::getline(input, line)) {
            const std::size_t separator = line.rfind('\t');
            assert(separator != std::string::npos);
            const std::string fen = line.substr(0, separator);
            const int expected = std::stoi(line.substr(separator + 1));
            chess::Position pos;
            assert(pos.set_fen(fen));
            chess::PhaseQuantizedNnueAccumulator accumulator;
            accumulator.reset(model, pos);
            assert(accumulator.matches_full_recompute(pos));
            const int accelerated = accumulator.evaluate_cp_rounded(pos);
            model.set_neon_dotprod_enabled(false);
            const int scalar = accumulator.evaluate_cp_rounded(pos);
            model.set_neon_dotprod_enabled(true);
            assert(accelerated == expected);
            assert(scalar == expected);
            ++checked;
        }
        assert(checked > 0);
        std::cout << "python_cpp_parity_positions=" << checked << '\n';
    }

    std::cout << "accumulator_kernel="
              << model.accumulator_kernel_name() << '\n'
              << "phase quantized NNUE incremental parity passed\n";
}
