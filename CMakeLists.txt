cmake_minimum_required(VERSION 3.20)

project(ChessEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHESS_ENABLE_TT_STATS "Enable transposition-table counters in hot paths" OFF)

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
    src/searchers/strict/heuristic_searcher_v34.cpp
    src/searchers/strict/heuristic_searcher_v34_api.cpp
    src/searchers/strict/heuristic_searcher_v34_ordering.cpp
    src/searchers/strict/heuristic_searcher_v34_tt.cpp
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

if (
    CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64"
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

if (CHESS_ENABLE_TT_STATS)
    target_compile_definitions(chess_core PRIVATE CHESS_ENABLE_TT_STATS=1)
endif()

if (MSVC)
    target_compile_options(chess_core PRIVATE /W4)
else()
    target_compile_options(chess_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(chess src/main.cpp)
target_link_libraries(chess PRIVATE chess_core)

add_executable(chess_engine_api src/engine_api.cpp)
target_link_libraries(chess_engine_api PRIVATE chess_core)

add_executable(dataset_export tools/dataset_export.cpp)
target_link_libraries(dataset_export PRIVATE chess_core)

add_executable(match_engines tools/match_engines.cpp)
target_link_libraries(match_engines PRIVATE chess_core)

add_executable(match_nn_models tools/match_nn_models.cpp)
target_link_libraries(match_nn_models PRIVATE chess_core)

add_executable(material_stress_export tools/material_stress_export.cpp)
target_link_libraries(material_stress_export PRIVATE chess_core)

add_executable(material_imbalance_export tools/material_imbalance_export.cpp)
target_link_libraries(material_imbalance_export PRIVATE chess_core)

add_executable(legal_move_dataset_export tools/legal_move_dataset_export.cpp)
target_link_libraries(legal_move_dataset_export PRIVATE chess_core)

add_executable(legal_move_dataset_view tools/legal_move_dataset_view.cpp)
target_link_libraries(legal_move_dataset_view PRIVATE chess_core)

add_executable(robotmoon_binpack_to_cbin tools/robotmoon_binpack_to_cbin.cpp)
add_executable(validate_robotmoon_cbin tools/validate_robotmoon_cbin.cpp)
add_executable(robotmoon_binpack_fixture tests/robotmoon_binpack_fixture.cpp)

add_executable(evaluate_heuristic_on_cbin tools/evaluate_heuristic_on_cbin.cpp)
target_link_libraries(evaluate_heuristic_on_cbin PRIVATE chess_core)

add_executable(evaluate_nnue_v38_selective tools/evaluate_nnue_v38_selective.cpp)
target_link_libraries(evaluate_nnue_v38_selective PRIVATE chess_core)

add_executable(evaluate_nnue_v39_selective tools/evaluate_nnue_v38_selective.cpp)
target_compile_definitions(
    evaluate_nnue_v39_selective PRIVATE CHESS_EVALUATE_NNUE_V39)
target_link_libraries(evaluate_nnue_v39_selective PRIVATE chess_core)

add_executable(trace_replay_ply tools/trace_replay_ply.cpp)
target_link_libraries(trace_replay_ply PRIVATE chess_core)

add_executable(match_searchers tools/match_searchers.cpp)
target_link_libraries(match_searchers PRIVATE chess_core)

add_executable(nn_engine tools/nn_engine.cpp)
target_link_libraries(nn_engine PRIVATE chess_core)

add_executable(benchmark_searchers_on_dataset tools/benchmark_searchers_on_dataset.cpp)
target_link_libraries(benchmark_searchers_on_dataset PRIVATE chess_core)

add_executable(benchmark_v7_v8_on_random_match tools/benchmark_v7_v8_on_random_match.cpp)
target_link_libraries(benchmark_v7_v8_on_random_match PRIVATE chess_core)

add_executable(benchmark_v10_order_profiles tools/benchmark_v10_order_profiles.cpp)
target_link_libraries(benchmark_v10_order_profiles PRIVATE chess_core)

add_executable(benchmark_v9_v10_v11_on_dataset tools/benchmark_v9_v10_v11_on_dataset.cpp)
target_link_libraries(benchmark_v9_v10_v11_on_dataset PRIVATE chess_core)

add_executable(tune_v12_move_order_weights tools/tune_v12_move_order_weights.cpp)
target_link_libraries(tune_v12_move_order_weights PRIVATE chess_core)

add_executable(measure_v12_ordering_stats tools/measure_v12_ordering_stats.cpp)
target_link_libraries(measure_v12_ordering_stats PRIVATE chess_core)

add_executable(compare_v12_v13_logic tools/compare_v12_v13_logic.cpp)
target_link_libraries(compare_v12_v13_logic PRIVATE chess_core)

add_executable(benchmark_v14_pvs tools/benchmark_v14_pvs.cpp)
target_link_libraries(benchmark_v14_pvs PRIVATE chess_core)

add_executable(benchmark_v15_pvs_compare tools/benchmark_v15_pvs_compare.cpp)
target_link_libraries(benchmark_v15_pvs_compare PRIVATE chess_core)

add_executable(benchmark_fast_v15_v35 tools/profile/benchmark_fast_v15_v35.cpp)
target_link_libraries(benchmark_fast_v15_v35 PRIVATE chess_core)

add_executable(benchmark_fast_v10_v35 tools/profile/benchmark_fast_v10_v35.cpp)
target_link_libraries(benchmark_fast_v10_v35 PRIVATE chess_core)

add_executable(benchmark_nnue_v36_vs_heuristic_v35 tools/profile/benchmark_nnue_v36_vs_heuristic_v35.cpp)
target_link_libraries(benchmark_nnue_v36_vs_heuristic_v35 PRIVATE chess_core)

add_executable(benchmark_nnue_v36_vs_v37 tools/profile/benchmark_nnue_v36_vs_v37.cpp)
target_link_libraries(benchmark_nnue_v36_vs_v37 PRIVATE chess_core)

add_executable(
    benchmark_nnue_v36_ordering_weights
    tools/profile/benchmark_nnue_v36_ordering_weights.cpp
)
target_link_libraries(benchmark_nnue_v36_ordering_weights PRIVATE chess_core)

add_executable(benchmark_nnue_rebuild_perspective tools/profile/benchmark_nnue_rebuild_perspective.cpp)
target_link_libraries(benchmark_nnue_rebuild_perspective PRIVATE chess_core)

add_executable(phase_nnue_strict_paired_match tools/phase_nnue_strict_paired_match.cpp)
target_link_libraries(phase_nnue_strict_paired_match PRIVATE chess_core)

add_executable(measure_nnue_v36_cutoff_stats tools/measure_nnue_v36_cutoff_stats.cpp)
target_link_libraries(measure_nnue_v36_cutoff_stats PRIVATE chess_core)

add_executable(benchmark_strict_v32_fast_v32 tools/profile/benchmark_strict_v32_fast_v32.cpp)
target_link_libraries(benchmark_strict_v32_fast_v32 PRIVATE chess_core)

add_executable(tune_fast_v32_lmr_match tools/profile/tune_fast_v32_lmr_match.cpp)
target_link_libraries(tune_fast_v32_lmr_match PRIVATE chess_core)

add_executable(render_fast_v32_lmr_win tools/profile/render_fast_v32_lmr_win.cpp)
target_link_libraries(render_fast_v32_lmr_win PRIVATE chess_core)

add_executable(benchmark_v15_aspiration tools/benchmark_v15_aspiration.cpp)
target_link_libraries(benchmark_v15_aspiration PRIVATE chess_core)

add_executable(benchmark_v15_v16_history tools/benchmark_v15_v16_history.cpp)
target_link_libraries(benchmark_v15_v16_history PRIVATE chess_core)

add_executable(benchmark_v10_v16_depth_sweep tools/benchmark_v10_v16_depth_sweep.cpp)
target_link_libraries(benchmark_v10_v16_depth_sweep PRIVATE chess_core)

add_executable(benchmark_v10_v24_depth_sweep tools/benchmark_v10_v24_depth_sweep.cpp)
target_link_libraries(benchmark_v10_v24_depth_sweep PRIVATE chess_core)

add_executable(benchmark_v16_v17_counterhistory tools/benchmark_v16_v17_counterhistory.cpp)
target_link_libraries(benchmark_v16_v17_counterhistory PRIVATE chess_core)

add_executable(benchmark_v17_v18 tools/benchmark_v17_v18.cpp)
target_link_libraries(benchmark_v17_v18 PRIVATE chess_core)

add_executable(analyze_v17_v18_move_ordering tools/analyze_v17_v18_move_ordering.cpp)
target_link_libraries(analyze_v17_v18_move_ordering PRIVATE chess_core)

add_executable(benchmark_v18_counterhistory_toggle tools/benchmark_v18_counterhistory_toggle.cpp)
target_link_libraries(benchmark_v18_counterhistory_toggle PRIVATE chess_core)

add_executable(benchmark_v18_v19_bucket_tt tools/benchmark_v18_v19_bucket_tt.cpp)
target_link_libraries(benchmark_v18_v19_bucket_tt PRIVATE chess_core)

add_executable(benchmark_v19_v20 tools/benchmark_v19_v20.cpp)
target_link_libraries(benchmark_v19_v20 PRIVATE chess_core)

add_executable(benchmark_v19_v21 tools/benchmark_v19_v21.cpp)
target_link_libraries(benchmark_v19_v21 PRIVATE chess_core)

add_executable(benchmark_v22_v23 tools/benchmark_v22_v23.cpp)
target_link_libraries(benchmark_v22_v23 PRIVATE chess_core)

add_executable(benchmark_v23_v24 tools/benchmark_v23_v24.cpp)
target_link_libraries(benchmark_v23_v24 PRIVATE chess_core)

add_executable(benchmark_v24_v25_depth_sweep tools/benchmark_v24_v25_depth_sweep.cpp)
target_link_libraries(benchmark_v24_v25_depth_sweep PRIVATE chess_core)

add_executable(profile_v25_line tools/profile_v25_line.cpp)
target_link_libraries(profile_v25_line PRIVATE chess_core)

add_executable(benchmark_make_unmake tools/benchmark_make_unmake.cpp)
target_link_libraries(benchmark_make_unmake PRIVATE chess_core)

add_executable(benchmark_v25_v26_v27 tools/benchmark_v25_v26_v27.cpp)
target_link_libraries(benchmark_v25_v26_v27 PRIVATE chess_core)

add_executable(benchmark_v27_v28 tools/benchmark_v27_v28.cpp)
target_link_libraries(benchmark_v27_v28 PRIVATE chess_core)

add_executable(benchmark_v29_v30 tools/benchmark_v29_v30.cpp)
target_link_libraries(benchmark_v29_v30 PRIVATE chess_core)

add_executable(benchmark_v30_v31 tools/benchmark_v30_v31.cpp)
target_link_libraries(benchmark_v30_v31 PRIVATE chess_core)

add_executable(benchmark_v31_v32 tools/benchmark_v31_v32.cpp)
target_link_libraries(benchmark_v31_v32 PRIVATE chess_core)

add_executable(benchmark_v32_v33 tools/benchmark_v32_v33.cpp)
target_link_libraries(benchmark_v32_v33 PRIVATE chess_core)

add_executable(benchmark_v32_v34 tools/benchmark_v32_v34.cpp)
target_link_libraries(benchmark_v32_v34 PRIVATE chess_core)

add_executable(benchmark_v32_v35 tools/benchmark_v32_v35.cpp)
target_link_libraries(benchmark_v32_v35 PRIVATE chess_core)

add_executable(check_v31_ordering_score tools/check_v31_ordering_score.cpp)
target_link_libraries(check_v31_ordering_score PRIVATE chess_core)

add_executable(profile_v31_iterative_once tools/profile/profile_v31_iterative_once.cpp)
target_link_libraries(profile_v31_iterative_once PRIVATE chess_core)

add_executable(profile_v32_iterative_once tools/profile/profile_v32_iterative_once.cpp)
target_link_libraries(profile_v32_iterative_once PRIVATE chess_core)

add_executable(profile_v34_iterative_once tools/profile/profile_v34_iterative_once.cpp)
target_link_libraries(profile_v34_iterative_once PRIVATE chess_core)

add_executable(profile_v32_tt_stats tools/profile/profile_v32_tt_stats.cpp)
target_link_libraries(profile_v32_tt_stats PRIVATE chess_core)

add_executable(profile_v32_v34_tt_stats tools/profile/profile_v32_v34_tt_stats.cpp)
target_link_libraries(profile_v32_v34_tt_stats PRIVATE chess_core)

add_executable(profile_v32_v34_tt_timing tools/profile/profile_v32_v34_tt_timing.cpp)
target_compile_definitions(profile_v32_v34_tt_timing PRIVATE CHESS_PROFILE_TT_TIMING)
target_link_libraries(profile_v32_v34_tt_timing PRIVATE chess_core)

add_executable(profile_v32_v34_tt_probe_paths tools/profile/profile_v32_v34_tt_probe_paths.cpp)
target_compile_definitions(profile_v32_v34_tt_probe_paths PRIVATE CHESS_PROFILE_TT_PATH_TIMING)
target_link_libraries(profile_v32_v34_tt_probe_paths PRIVATE chess_core)

add_executable(profile_v34_tt_cache_lines tools/profile/profile_v34_tt_cache_lines.cpp)
target_compile_definitions(profile_v34_tt_cache_lines PRIVATE CHESS_PROFILE_TT_CACHE_LINES)
target_link_libraries(profile_v34_tt_cache_lines PRIVATE chess_core)

add_executable(benchmark_tt_probe_cases tools/profile/benchmark_tt_probe_cases.cpp)
target_link_libraries(benchmark_tt_probe_cases PRIVATE chess_core)

add_executable(benchmark_v32_v34_tt_functions tools/profile/benchmark_v32_v34_tt_functions.cpp)
target_link_libraries(benchmark_v32_v34_tt_functions PRIVATE chess_core)

add_executable(benchmark_v32_v33_single_fen tools/profile/benchmark_v32_v33_single_fen.cpp)
target_link_libraries(benchmark_v32_v33_single_fen PRIVATE chess_core)

add_executable(benchmark_v32_v34_single_fen tools/profile/benchmark_v32_v34_single_fen.cpp)
target_link_libraries(benchmark_v32_v34_single_fen PRIVATE chess_core)

add_executable(benchmark_v34_bucket_sizes tools/profile/benchmark_v34_bucket_sizes.cpp)
target_link_libraries(benchmark_v34_bucket_sizes PRIVATE chess_core)

add_executable(benchmark_v35_tt_mb tools/profile/benchmark_v35_tt_mb.cpp)
target_link_libraries(benchmark_v35_tt_mb PRIVATE chess_core)

add_executable(benchmark_v32_v34_tt_mb tools/profile/benchmark_v32_v34_tt_mb.cpp)
target_link_libraries(benchmark_v32_v34_tt_mb PRIVATE chess_core)

add_executable(benchmark_incremental_nnue tools/profile/benchmark_custom_nnue.cpp)
target_include_directories(benchmark_incremental_nnue PRIVATE include)

add_executable(analyze_nn_buckets tools/analyze_nn_buckets.cpp)
target_link_libraries(analyze_nn_buckets PRIVATE chess_core)

add_executable(tactical_disagreement_export tools/tactical_disagreement_export.cpp)
target_link_libraries(tactical_disagreement_export PRIVATE chess_core)

add_executable(measure_tt_stats tools/measure_tt_stats.cpp)
target_link_libraries(measure_tt_stats PRIVATE chess_core)

add_executable(generate_magic_bitboards tools/generate_magic_bitboards.cpp)
target_link_libraries(generate_magic_bitboards PRIVATE chess_core)

add_executable(opening_book_match tools/opening_book_match.cpp)
target_link_libraries(opening_book_match PRIVATE chess_core)

add_executable(nnue_hard_negative_export tools/nnue_hard_negative_export.cpp)
target_link_libraries(nnue_hard_negative_export PRIVATE chess_core)

add_executable(nnue_spectator_gap_export tools/nnue_spectator_gap_export.cpp)
target_link_libraries(nnue_spectator_gap_export PRIVATE chess_core)

add_executable(uci_v7 tools/uci_v7.cpp)
target_link_libraries(uci_v7 PRIVATE chess_core)

add_executable(uci_nnue_v38 tools/uci_nnue_v38.cpp)
target_link_libraries(uci_nnue_v38 PRIVATE chess_core)

add_executable(nnue_v38_time_gauntlet tools/nnue_v38_time_gauntlet.cpp)
target_link_libraries(nnue_v38_time_gauntlet PRIVATE chess_core)

add_executable(nnue_v39_time_gauntlet tools/nnue_v38_time_gauntlet.cpp)
target_compile_definitions(
    nnue_v39_time_gauntlet PRIVATE CHESS_NNUE_V39_TIME_GAUNTLET)
target_link_libraries(nnue_v39_time_gauntlet PRIVATE chess_core)

add_executable(collect_nnue_wdl_calibration tools/collect_nnue_wdl_calibration.cpp)
target_link_libraries(collect_nnue_wdl_calibration PRIVATE chess_core)

enable_testing()

function(add_chess_test name)
    add_executable(${name} tests/${name}.cpp)
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
add_chess_test(rook_attack_tests)
add_chess_test(search_tests)
add_chess_test(search_engine_tests)
add_chess_test(see_tests)
add_chess_test(slider_movegen_tests)
add_chess_test(startpos_tests)

add_executable(nn_search_tests tests/nn_search_tests.cpp)
target_link_libraries(nn_search_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nn_search_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nn_search_tests PRIVATE -UNDEBUG)
endif()
add_test(NAME nn_search_tests COMMAND nn_search_tests ${CMAKE_SOURCE_DIR}/models/value_net_stream_100k.bin)

add_executable(nn_value_tests tests/nn_value_tests.cpp)
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

add_executable(nnue_value_tests tests/nnue_value_tests.cpp)
target_link_libraries(nnue_value_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_value_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_value_tests PRIVATE -UNDEBUG)
endif()

add_executable(incremental_nnue_tests tests/incremental_nnue_tests.cpp)
target_include_directories(incremental_nnue_tests PRIVATE include)
if(MSVC)
    target_compile_options(incremental_nnue_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(incremental_nnue_tests PRIVATE -UNDEBUG)
endif()
add_test(NAME incremental_nnue_tests COMMAND incremental_nnue_tests)

add_executable(nnue_searcher_v10_tests tests/nnue_searcher_v10_tests.cpp)
target_link_libraries(nnue_searcher_v10_tests PRIVATE chess_core)
if(MSVC)
    target_compile_options(nnue_searcher_v10_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(nnue_searcher_v10_tests PRIVATE -UNDEBUG)
endif()

add_executable(nnue_searcher_v36_strict_tests tests/nnue_searcher_v36_strict_tests.cpp)
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

add_executable(nnue_searcher_v37_strict_tests tests/nnue_searcher_v37_strict_tests.cpp)
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

add_executable(nnue_searcher_v39_selective_tests tests/nnue_searcher_v39_selective_tests.cpp)
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

add_executable(phase_quantized_nnue_tests tests/phase_quantized_nnue_tests.cpp)
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
