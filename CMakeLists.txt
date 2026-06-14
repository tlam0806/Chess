cmake_minimum_required(VERSION 3.20)

project(ChessEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chess_core
    src/attacks.cpp
    src/bitboard.cpp
    src/board_encoder.cpp
    src/counter_history_table.cpp
    src/evaluate.cpp
    src/game_state.cpp
    src/history_table.cpp
    src/history_table_v16.cpp
    src/killer_move_table.cpp
    src/move.cpp
    src/nn_search.cpp
    src/nn_value.cpp
    src/nnue_value.cpp
    src/perft.cpp
    src/position.cpp
    src/range_transposition_table.cpp
    src/range_bucket_transposition_table.cpp
    src/heuristic_searcher.cpp
    src/heuristic_searcher_v2.cpp
    src/heuristic_searcher_v3.cpp
    src/heuristic_searcher_v4.cpp
    src/heuristic_searcher_v5.cpp
    src/heuristic_searcher_v6.cpp
    src/heuristic_searcher_v7.cpp
    src/heuristic_searcher_v8.cpp
    src/heuristic_searcher_v9.cpp
    src/heuristic_searcher_v10.cpp
    src/heuristic_searcher_v11.cpp
    src/heuristic_searcher_v12.cpp
    src/heuristic_searcher_v13.cpp
    src/heuristic_searcher_v14.cpp
    src/heuristic_searcher_v14_experimental.cpp
    src/searchers/strict/heuristic_searcher_v15.cpp
    src/searchers/strict/heuristic_searcher_v16.cpp
    src/searchers/strict/heuristic_searcher_v17.cpp
    src/searchers/strict/heuristic_searcher_v18.cpp
    src/searchers/strict/heuristic_searcher_v19.cpp
    src/searchers/fast/heuristic_searcher_v15_pvs.cpp
    src/nn_searcher.cpp
    src/nn_searcher_v6.cpp
    src/nn_searcher_v7.cpp
    src/nn_searcher_v10.cpp
    src/nnue_searcher_v10.cpp
    src/searcher.cpp
    src/transposition_table.cpp
    src/zobrist.cpp
)

target_include_directories(chess_core PUBLIC include)

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

add_executable(benchmark_v15_aspiration tools/benchmark_v15_aspiration.cpp)
target_link_libraries(benchmark_v15_aspiration PRIVATE chess_core)

add_executable(benchmark_v15_v16_history tools/benchmark_v15_v16_history.cpp)
target_link_libraries(benchmark_v15_v16_history PRIVATE chess_core)

add_executable(benchmark_v10_v16_depth_sweep tools/benchmark_v10_v16_depth_sweep.cpp)
target_link_libraries(benchmark_v10_v16_depth_sweep PRIVATE chess_core)

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

enable_testing()

function(add_chess_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE chess_core)
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
add_chess_test(king_attack_tests)
add_chess_test(king_movegen_tests)
add_chess_test(knight_attack_tests)
add_chess_test(knight_movegen_tests)
add_chess_test(legal_movegen_tests)
add_chess_test(make_move_tests)
add_chess_test(magic_bitboard_tests)
add_chess_test(move_tests)
add_chess_test(movegen_regression_tests)
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
add_test(NAME nn_search_tests COMMAND nn_search_tests ${CMAKE_SOURCE_DIR}/models/value_net_stream_100k.bin)

add_executable(nn_value_tests tests/nn_value_tests.cpp)
target_link_libraries(nn_value_tests PRIVATE chess_core)
add_test(NAME nn_value_tests
    COMMAND nn_value_tests
        ${CMAKE_SOURCE_DIR}/models/value_net_stream_100k.bin
        ${CMAKE_SOURCE_DIR}/data/nn_value_compare_10000.jsonl
)

add_executable(nnue_value_tests tests/nnue_value_tests.cpp)
target_link_libraries(nnue_value_tests PRIVATE chess_core)

add_executable(nnue_searcher_v10_tests tests/nnue_searcher_v10_tests.cpp)
target_link_libraries(nnue_searcher_v10_tests PRIVATE chess_core)
