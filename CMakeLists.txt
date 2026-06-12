cmake_minimum_required(VERSION 3.20)

project(ChessEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chess_core
    src/attacks.cpp
    src/bitboard.cpp
    src/board_encoder.cpp
    src/evaluate.cpp
    src/game_state.cpp
    src/move.cpp
    src/nn_search.cpp
    src/nn_value.cpp
    src/perft.cpp
    src/position.cpp
    src/heuristic_searcher.cpp
    src/heuristic_searcher_v2.cpp
    src/heuristic_searcher_v3.cpp
    src/heuristic_searcher_v4.cpp
    src/heuristic_searcher_v5.cpp
    src/heuristic_searcher_v6.cpp
    src/heuristic_searcher_v7.cpp
    src/nn_searcher.cpp
    src/nn_searcher_v6.cpp
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
add_chess_test(heuristic_searcher_v2_tests)
add_chess_test(heuristic_searcher_v3_tests)
add_chess_test(heuristic_searcher_v4_tests)
add_chess_test(heuristic_searcher_v5_tests)
add_chess_test(heuristic_searcher_v6_tests)
add_chess_test(heuristic_searcher_v7_tests)
add_chess_test(king_attack_tests)
add_chess_test(king_movegen_tests)
add_chess_test(knight_attack_tests)
add_chess_test(knight_movegen_tests)
add_chess_test(legal_movegen_tests)
add_chess_test(make_move_tests)
add_chess_test(move_tests)
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
