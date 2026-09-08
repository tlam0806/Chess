# Shared engine library and platform-specific compilation settings.

add_library(chess_core
    src/attacks.cpp
    src/bitboard.cpp
    src/board_encoder.cpp
    src/counter_history_table.cpp
    src/counter_move_table.cpp
    src/evaluate.cpp
    src/game_state.cpp
    src/history_table.cpp
    src/history_table_v16.cpp
    src/king_safety.cpp
    src/killer_move_table.cpp
    src/move.cpp
    src/nn_search.cpp
    src/nn_value.cpp
    src/nnue_value.cpp
    src/phase_quantized_nnue.cpp
    src/phase_quantized_nnue_forward.cpp
    src/perft.cpp
    src/position.cpp
    src/lower_bucket_transposition_table.cpp
    src/lower_move_range_bucket_transposition_table.cpp
    src/single_bound_bucket_transposition_table.cpp
    src/nnue_v37_transposition_table.cpp
    src/range_transposition_table.cpp
    src/range_bucket_transposition_table.cpp
    src/searchers/fast/heuristic_searcher.cpp
    src/searchers/fast/heuristic_searcher_v2.cpp
    src/searchers/fast/heuristic_searcher_v3.cpp
    src/searchers/fast/heuristic_searcher_v4.cpp
    src/searchers/fast/heuristic_searcher_v5.cpp
    src/searchers/fast/heuristic_searcher_v6.cpp
    src/searchers/fast/heuristic_searcher_v7.cpp
    src/searchers/fast/heuristic_searcher_v8.cpp
    src/searchers/fast/heuristic_searcher_v9.cpp
    src/searchers/fast/heuristic_searcher_v10.cpp
    src/searchers/fast/heuristic_searcher_v11.cpp
    src/searchers/fast/heuristic_searcher_v12.cpp
    src/searchers/fast/heuristic_searcher_v13.cpp
    src/searchers/fast/heuristic_searcher_v14.cpp
    src/searchers/fast/heuristic_searcher_v14_experimental.cpp
    src/searchers/fast/heuristic_searcher_fast_v32.cpp
    src/searchers/fast/heuristic_searcher_fast_v32_api.cpp
    src/searchers/fast/heuristic_searcher_fast_v32_ordering.cpp
    src/searchers/fast/heuristic_searcher_fast_v32_tt.cpp
    src/searchers/fast/heuristic_searcher_fast_v35.cpp
    src/searchers/fast/heuristic_searcher_fast_v35_api.cpp
    src/searchers/fast/heuristic_searcher_fast_v35_ordering.cpp
    src/searchers/fast/heuristic_searcher_fast_v35_tt.cpp
    src/searchers/fast/heuristic_searcher_v34.cpp
    src/searchers/fast/heuristic_searcher_v34_api.cpp
    src/searchers/fast/heuristic_searcher_v34_ordering.cpp
    src/searchers/fast/heuristic_searcher_v34_tt.cpp
    src/searchers/strict/heuristic_searcher_v15.cpp
    src/searchers/strict/heuristic_searcher_v16.cpp
    src/searchers/strict/heuristic_searcher_v17.cpp
    src/searchers/strict/heuristic_searcher_v18.cpp
    src/searchers/strict/heuristic_searcher_v19.cpp
    src/searchers/strict/heuristic_searcher_v20.cpp
    src/searchers/strict/heuristic_searcher_v21.cpp
    src/searchers/strict/heuristic_searcher_v22.cpp
    src/searchers/strict/heuristic_searcher_v23.cpp
    src/searchers/strict/heuristic_searcher_v23_api.cpp
    src/searchers/strict/heuristic_searcher_v24.cpp
    src/searchers/strict/heuristic_searcher_v24_api.cpp
    src/searchers/strict/heuristic_searcher_v25.cpp
    src/searchers/strict/heuristic_searcher_v25_api.cpp
    src/searchers/strict/heuristic_searcher_v26.cpp
    src/searchers/strict/heuristic_searcher_v26_api.cpp
    src/searchers/strict/heuristic_searcher_v27.cpp
    src/searchers/strict/heuristic_searcher_v27_api.cpp
    src/searchers/strict/heuristic_searcher_v28.cpp
    src/searchers/strict/heuristic_searcher_v28_api.cpp
    src/searchers/strict/heuristic_searcher_v28_ordering.cpp
    src/searchers/strict/heuristic_searcher_v28_tt.cpp
    src/searchers/strict/heuristic_searcher_v29.cpp
    src/searchers/strict/heuristic_searcher_v29_api.cpp
    src/searchers/strict/heuristic_searcher_v29_ordering.cpp
    src/searchers/strict/heuristic_searcher_v29_tt.cpp
    src/searchers/strict/heuristic_searcher_v30.cpp
    src/searchers/strict/heuristic_searcher_v30_api.cpp
    src/searchers/strict/heuristic_searcher_v30_ordering.cpp
    src/searchers/strict/heuristic_searcher_v30_tt.cpp
    src/searchers/strict/heuristic_searcher_v31.cpp
    src/searchers/strict/heuristic_searcher_v31_api.cpp
    src/searchers/strict/heuristic_searcher_v31_ordering.cpp
    src/searchers/strict/heuristic_searcher_v31_tt.cpp
    src/searchers/strict/heuristic_searcher_v32.cpp
    src/searchers/strict/heuristic_searcher_v32_api.cpp
    src/searchers/strict/heuristic_searcher_v32_ordering.cpp
    src/searchers/strict/heuristic_searcher_v32_tt.cpp
    src/searchers/strict/heuristic_searcher_v33.cpp
    src/searchers/strict/heuristic_searcher_v33_api.cpp
    src/searchers/strict/heuristic_searcher_v33_ordering.cpp
    src/searchers/strict/heuristic_searcher_v33_tt.cpp
    src/searchers/strict/heuristic_searcher_v35.cpp
    src/searchers/strict/heuristic_searcher_v35_api.cpp
    src/searchers/strict/heuristic_searcher_v35_ordering.cpp
    src/searchers/strict/heuristic_searcher_v35_tt.cpp
    src/searchers/strict/nnue_searcher_v36.cpp
    src/searchers/strict/nnue_searcher_v36_api.cpp
    src/searchers/strict/nnue_searcher_v36_ordering.cpp
    src/searchers/strict/nnue_searcher_v36_tt.cpp
    src/searchers/strict/nnue_searcher_v37.cpp
    src/searchers/strict/nnue_searcher_v37_api.cpp
    src/searchers/strict/nnue_searcher_v37_ordering.cpp
    src/searchers/strict/nnue_searcher_v37_tt.cpp
    src/searchers/strict/nnue_searcher_v38.cpp
    src/searchers/strict/nnue_searcher_v38_api.cpp
    src/searchers/strict/nnue_searcher_v38_ordering.cpp
    src/searchers/strict/nnue_searcher_v38_tt.cpp
    src/searchers/strict/nnue_searcher_v39.cpp
    src/searchers/strict/nnue_searcher_v40.cpp
    src/searchers/strict/nnue_searcher_v41.cpp
    src/searchers/strict/nnue_searcher_v42.cpp
    src/searchers/strict/nnue_searcher_v43.cpp
    src/searchers/strict/nnue_searcher_v43_api.cpp
    src/searchers/strict/nnue_searcher_v43_ordering.cpp
    src/searchers/strict/nnue_searcher_v43_tt.cpp
    src/v43_single_bound_transposition_table.cpp
    src/searchers/strict/nnue_searcher_v44.cpp
    src/searchers/strict/nnue_searcher_v44_api.cpp
    src/searchers/strict/nnue_searcher_v44_ordering.cpp
    src/searchers/strict/nnue_searcher_v44_tt.cpp
    src/v44_single_bound_transposition_table.cpp
    src/searchers/fast/heuristic_searcher_v15_pvs.cpp
    src/searchers/fast/nn_searcher.cpp
    src/searchers/fast/nn_searcher_v6.cpp
    src/searchers/fast/nn_searcher_v7.cpp
    src/searchers/fast/nn_searcher_v10.cpp
    src/searchers/fast/nnue_searcher_v10.cpp
    src/searcher.cpp
    src/transposition_table.cpp
    src/zobrist.cpp
)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CHESS_SYSTEM_PROCESSOR_LOWER)
set(CHESS_TARGETS_X86_64 FALSE)
if (CHESS_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64)$")
    set(CHESS_TARGETS_X86_64 TRUE)
elseif (APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    # Allows an Apple Silicon developer machine to cross-build the exact x86
    # translation units for syntax/parity checks under Rosetta.
    set(CHESS_TARGETS_X86_64 TRUE)
endif()
if (
    CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"
    AND CHESS_TARGETS_X86_64
)
    target_sources(chess_core PRIVATE
        src/phase_quantized_nnue_accumulator_avx2.cpp
        src/phase_quantized_nnue_forward_avx2.cpp
        src/phase_quantized_nnue_forward_vnni.cpp
    )
    target_compile_definitions(
        chess_core PRIVATE CHESS_PHASE_NNUE_X86_BACKENDS=1)
    set_source_files_properties(
        src/phase_quantized_nnue_accumulator_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2"
    )
    set_source_files_properties(
        src/phase_quantized_nnue_forward_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2"
    )
    set_source_files_properties(
        src/phase_quantized_nnue_forward_vnni.cpp
        PROPERTIES COMPILE_OPTIONS
        "-mavx2;-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mprefer-vector-width=256"
    )
endif()

if (
    CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64"
    AND NOT CHESS_TARGETS_X86_64
)
    # Match the experimental candidate build so the mixed unsigned/signed
    # int8 matrix-multiply intrinsics are available to the NNUE forward only.
    set_source_files_properties(
        src/phase_quantized_nnue_forward.cpp
        PROPERTIES COMPILE_OPTIONS "-mcpu=native"
    )
endif()

target_include_directories(chess_core PUBLIC
    include
    include/searchers/fast
    include/searchers/strict
)

if(CHESS_ENABLE_NNUE_STAGE_BENCHMARK)
    target_compile_definitions(
        chess_core PRIVATE CHESS_ENABLE_NNUE_STAGE_BENCHMARK=1)
endif()

if (CHESS_ENABLE_TT_STATS)
    target_compile_definitions(chess_core PRIVATE CHESS_ENABLE_TT_STATS=1)
endif()

if (MSVC)
    target_compile_options(chess_core PRIVATE /W4)
else()
    target_compile_options(chess_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
