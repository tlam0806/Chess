# C++ test executables and CTest registrations.

enable_testing()

function(add_chess_test name)
    add_executable(${name} tests/cpp/${name}.cpp)
    target_link_libraries(${name} PRIVATE chess_core)
    if(MSVC)
        target_compile_options(${name} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${name} PRIVATE -UNDEBUG)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_chess_test(attack_detection_tests)
add_chess_test(bishop_attack_tests)
add_chess_test(bitboard_tests)
add_chess_test(board_encoder_tests)
add_chess_test(castling_tests)
add_chess_test(check_tests)
add_chess_test(en_passant_tests)
add_chess_test(evaluate_tests)
add_chess_test(fen_tests)
add_chess_test(game_state_tests)
add_chess_test(history_table_tests)
add_chess_test(history_table_v16_tests)
if(CHESS_BUILD_EXPERIMENTS)
    add_chess_test(heuristic_searcher_v2_tests)
    add_chess_test(heuristic_searcher_v3_tests)
    add_chess_test(heuristic_searcher_v4_tests)
    add_chess_test(heuristic_searcher_v5_tests)
    add_chess_test(heuristic_searcher_v6_tests)
    add_chess_test(heuristic_searcher_v7_tests)
    add_chess_test(heuristic_searcher_v8_tests)
    add_chess_test(heuristic_searcher_v9_tests)
    add_chess_test(heuristic_searcher_v10_tests)
    add_chess_test(heuristic_searcher_v11_tests)
    add_chess_test(heuristic_searcher_v12_tests)
    add_chess_test(heuristic_searcher_v13_tests)
    add_chess_test(heuristic_searcher_v14_tests)
    add_chess_test(heuristic_searcher_v15_tests)
    add_chess_test(heuristic_searcher_v16_tests)
    add_chess_test(heuristic_searcher_v17_tests)
    add_chess_test(heuristic_searcher_v18_tests)
    add_chess_test(heuristic_searcher_v19_tests)
    add_chess_test(heuristic_searcher_v20_tests)
    add_chess_test(heuristic_searcher_v21_tests)
    add_chess_test(heuristic_searcher_v22_tests)
    add_chess_test(heuristic_searcher_v23_tests)
    add_chess_test(heuristic_searcher_v24_tests)
    add_chess_test(heuristic_searcher_v25_tests)
endif()
add_chess_test(king_attack_tests)
add_chess_test(king_movegen_tests)
add_chess_test(knight_attack_tests)
add_chess_test(knight_movegen_tests)
add_chess_test(legal_movegen_tests)
add_chess_test(make_move_tests)
add_chess_test(magic_bitboard_tests)
add_chess_test(move_tests)
add_chess_test(movegen_regression_tests)
add_chess_test(nnue_wdl_calibration_tests)
add_chess_test(pawn_attack_tests)
add_chess_test(pawn_movegen_tests)
add_chess_test(perft_tests)
add_chess_test(position_print_tests)
add_chess_test(position_state_tests)
add_chess_test(pseudo_legal_movegen_tests)
add_chess_test(queen_attack_tests)
add_chess_test(repetition_stack_tests)
add_chess_test(rook_attack_tests)
if(CHESS_BUILD_EXPERIMENTS)
    add_chess_test(search_tests)
    add_chess_test(search_engine_tests)
endif()
add_chess_test(see_tests)
add_chess_test(slider_movegen_tests)
add_chess_test(startpos_tests)
add_executable(
    v43_single_bound_transposition_table_tests
    tests/cpp/v43_single_bound_transposition_table_tests.cpp
    src/v43_single_bound_transposition_table.cpp
)
target_include_directories(v43_single_bound_transposition_table_tests PRIVATE include)
if(MSVC)
    target_compile_options(v43_single_bound_transposition_table_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(v43_single_bound_transposition_table_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME v43_single_bound_transposition_table_tests
    COMMAND v43_single_bound_transposition_table_tests
)

if(CHESS_BUILD_EXPERIMENTS)
add_executable(nn_search_tests tests/cpp/nn_search_tests.cpp)
target_link_libraries(nn_search_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nn_search_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nn_search_tests PRIVATE -UNDEBUG)
endif()
add_test(NAME nn_search_tests COMMAND nn_search_tests ${CMAKE_SOURCE_DIR}/models/value_net_stream_100k.bin)

add_executable(nn_value_tests tests/cpp/nn_value_tests.cpp)
target_link_libraries(nn_value_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nn_value_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nn_value_tests PRIVATE -UNDEBUG)
endif()
add_test(NAME nn_value_tests
    COMMAND nn_value_tests
        ${CMAKE_SOURCE_DIR}/models/value_net_stream_100k.bin
        ${CMAKE_SOURCE_DIR}/data/nn_value_compare_10000.jsonl
)

add_executable(nnue_value_tests tests/cpp/nnue_value_tests.cpp)
target_link_libraries(nnue_value_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_value_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_value_tests PRIVATE -UNDEBUG)
endif()

add_executable(incremental_nnue_tests tests/cpp/incremental_nnue_tests.cpp)
target_include_directories(incremental_nnue_tests PRIVATE include)
if(MSVC)
    target_compile_options(incremental_nnue_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(incremental_nnue_tests PRIVATE -UNDEBUG)
endif()
add_test(NAME incremental_nnue_tests COMMAND incremental_nnue_tests)

add_executable(nnue_searcher_v10_tests tests/cpp/nnue_searcher_v10_tests.cpp)
target_link_libraries(nnue_searcher_v10_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v10_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v10_tests PRIVATE -UNDEBUG)
endif()

add_executable(nnue_searcher_v36_strict_tests tests/cpp/nnue_searcher_v36_strict_tests.cpp)
target_link_libraries(nnue_searcher_v36_strict_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v36_strict_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v36_strict_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_searcher_v36_strict_tests
    COMMAND nnue_searcher_v36_strict_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(nnue_searcher_v37_strict_tests tests/cpp/nnue_searcher_v37_strict_tests.cpp)
target_link_libraries(nnue_searcher_v37_strict_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v37_strict_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v37_strict_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_searcher_v37_strict_tests
    COMMAND nnue_searcher_v37_strict_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(nnue_searcher_v39_selective_tests tests/cpp/nnue_searcher_v39_selective_tests.cpp)
target_link_libraries(nnue_searcher_v39_selective_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v39_selective_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v39_selective_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_searcher_v39_selective_tests
    COMMAND nnue_searcher_v39_selective_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(nnue_searcher_v40_qsee_tests tests/cpp/nnue_searcher_v40_qsee_tests.cpp)
target_link_libraries(nnue_searcher_v40_qsee_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v40_qsee_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v40_qsee_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_searcher_v40_qsee_tests
    COMMAND nnue_searcher_v40_qsee_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(
    nnue_searcher_v41_repetition_tests
    tests/cpp/nnue_searcher_v41_repetition_tests.cpp
)
target_link_libraries(nnue_searcher_v41_repetition_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v41_repetition_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v41_repetition_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_searcher_v41_repetition_tests
    COMMAND nnue_searcher_v41_repetition_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(
    nnue_searcher_v42_aspiration_tests
    tests/cpp/nnue_searcher_v42_aspiration_tests.cpp
)
target_link_libraries(nnue_searcher_v42_aspiration_tests PRIVATE chess_core)
add_test(
    NAME nnue_searcher_v42_aspiration_tests
    COMMAND nnue_searcher_v42_aspiration_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(
    nnue_searcher_v43_single_bound_tests
    tests/cpp/nnue_searcher_v43_single_bound_tests.cpp
)
target_link_libraries(nnue_searcher_v43_single_bound_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v43_single_bound_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v43_single_bound_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_searcher_v43_single_bound_tests
    COMMAND nnue_searcher_v43_single_bound_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(
    nnue_searcher_v44_lifecycle_tests
    tests/cpp/nnue_searcher_v44_lifecycle_tests.cpp
)
target_link_libraries(nnue_searcher_v44_lifecycle_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v44_lifecycle_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v44_lifecycle_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_searcher_v44_lifecycle_tests
    COMMAND nnue_searcher_v44_lifecycle_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(
    nnue_searcher_v44_sable_tt_regression_tests
    tests/cpp/nnue_searcher_v44_sable_tt_regression_tests.cpp
)
# Exact production-depth replay: keep it as an opt-in regression target rather
# than adding its substantially slower assertion build to the default CTest.
target_link_libraries(
    nnue_searcher_v44_sable_tt_regression_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(
        nnue_searcher_v44_sable_tt_regression_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(
        nnue_searcher_v44_sable_tt_regression_tests PRIVATE -UNDEBUG)
endif()

add_chess_test(v44_single_bound_transposition_table_tests)

add_executable(
    uci_nnue_v44_cli_tests
    tests/cpp/uci_nnue_v44_cli_tests.cpp
)
target_link_libraries(uci_nnue_v44_cli_tests PRIVATE chess_core Threads::Threads)
if(MSVC)
    target_compile_options(uci_nnue_v44_cli_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(uci_nnue_v44_cli_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME uci_nnue_v44_cli_tests
    COMMAND uci_nnue_v44_cli_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(
    nnue_v42_control_cache_tests
    tests/cpp/nnue_v42_control_cache_tests.cpp
)
target_compile_definitions(
    nnue_v42_control_cache_tests PRIVATE CHESS_EVALUATE_NNUE_V42)
target_link_libraries(nnue_v42_control_cache_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_v42_control_cache_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_v42_control_cache_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_v42_control_cache_tests
    COMMAND nnue_v42_control_cache_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(
    nnue_v43_control_cache_tests
    tests/cpp/nnue_v42_control_cache_tests.cpp
)
target_compile_definitions(
    nnue_v43_control_cache_tests PRIVATE CHESS_EVALUATE_NNUE_V43)
target_link_libraries(nnue_v43_control_cache_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_v43_control_cache_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_v43_control_cache_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_v43_control_cache_tests
    COMMAND nnue_v43_control_cache_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)

add_executable(
    nnue_v43_final_tune_evaluator_tests
    tests/cpp/nnue_v43_final_tune_evaluator_tests.cpp
)
target_compile_definitions(
    nnue_v43_final_tune_evaluator_tests PRIVATE CHESS_EVALUATE_NNUE_V43)
target_link_libraries(nnue_v43_final_tune_evaluator_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_v43_final_tune_evaluator_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_v43_final_tune_evaluator_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME nnue_v43_final_tune_evaluator_tests
    COMMAND nnue_v43_final_tune_evaluator_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
)
endif()

add_executable(phase_quantized_nnue_tests tests/cpp/phase_quantized_nnue_tests.cpp)
target_link_libraries(phase_quantized_nnue_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(phase_quantized_nnue_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(phase_quantized_nnue_tests PRIVATE -UNDEBUG)
endif()
add_test(
    NAME phase_quantized_nnue_tests
    COMMAND phase_quantized_nnue_tests
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
        ${CMAKE_SOURCE_DIR}/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue_parity.tsv
)

if (
    CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"
    AND CHESS_TARGETS_X86_64
)
    add_executable(
        phase_quantized_nnue_accumulator_avx2_tests
        tests/cpp/phase_quantized_nnue_accumulator_avx2_tests.cpp
    )
    target_include_directories(
        phase_quantized_nnue_accumulator_avx2_tests PRIVATE src)
    target_link_libraries(
        phase_quantized_nnue_accumulator_avx2_tests PRIVATE chess_core)
    target_compile_options(
        phase_quantized_nnue_accumulator_avx2_tests PRIVATE -UNDEBUG)
    add_test(
        NAME phase_quantized_nnue_accumulator_avx2_tests
        COMMAND phase_quantized_nnue_accumulator_avx2_tests
    )
endif()
