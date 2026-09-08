cmake_minimum_required(VERSION 3.20)

project(ChessEngine LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHESS_ENABLE_TT_STATS "Enable transposition-table counters in hot paths" OFF)
option(CHESS_ENABLE_LTO "Enable link-time optimization for release builds" OFF)
option(
    CHESS_ENABLE_NNUE_STAGE_BENCHMARK
    "Compile the non-production phase NNUE stage benchmark hooks"
    OFF
)

if(CHESS_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT CHESS_IPO_SUPPORTED OUTPUT CHESS_IPO_ERROR)
    if(NOT CHESS_IPO_SUPPORTED)
        message(FATAL_ERROR "CHESS_ENABLE_LTO requested but unsupported: ${CHESS_IPO_ERROR}")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

include(cmake/Core.cmake)
include(cmake/Production.cmake)
include(cmake/Experiments.cmake)
include(cmake/Tools.cmake)
include(cmake/Tests.cmake)
