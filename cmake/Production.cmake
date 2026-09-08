# User-facing production engine and opt-in historical UCI entry points.

add_executable(uci_nnue_v43 tools/ops/uci_nnue_v38.cpp)
target_compile_definitions(uci_nnue_v43 PRIVATE CHESS_UCI_NNUE_V43)
target_link_libraries(uci_nnue_v43 PRIVATE chess_core Threads::Threads)

add_custom_target(production DEPENDS uci_nnue_v43)

if(CHESS_BUILD_EXPERIMENTS)
    add_executable(chess src/main.cpp)
    target_link_libraries(chess PRIVATE chess_core)

    add_executable(chess_engine_api src/engine_api.cpp)
    target_link_libraries(chess_engine_api PRIVATE chess_core)

    add_executable(uci_v7 tools/ops/uci_v7.cpp)
    target_link_libraries(uci_v7 PRIVATE chess_core)

    add_executable(uci_nnue_v38 tools/ops/uci_nnue_v38.cpp)
    target_link_libraries(uci_nnue_v38 PRIVATE chess_core Threads::Threads)

    add_executable(uci_nnue_v40 tools/ops/uci_nnue_v38.cpp)
    target_compile_definitions(uci_nnue_v40 PRIVATE CHESS_UCI_NNUE_V40)
    target_link_libraries(uci_nnue_v40 PRIVATE chess_core Threads::Threads)

    add_executable(uci_nnue_v41 tools/ops/uci_nnue_v38.cpp)
    target_compile_definitions(uci_nnue_v41 PRIVATE CHESS_UCI_NNUE_V41)
    target_link_libraries(uci_nnue_v41 PRIVATE chess_core Threads::Threads)

    add_executable(uci_nnue_v42 tools/ops/uci_nnue_v38.cpp)
    target_compile_definitions(uci_nnue_v42 PRIVATE CHESS_UCI_NNUE_V42)
    target_link_libraries(uci_nnue_v42 PRIVATE chess_core Threads::Threads)

    add_executable(uci_nnue_v44 tools/ops/uci_nnue_v38.cpp)
    target_compile_definitions(uci_nnue_v44 PRIVATE CHESS_UCI_NNUE_V44)
    target_link_libraries(uci_nnue_v44 PRIVATE chess_core Threads::Threads)
endif()
