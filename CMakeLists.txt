cmake_minimum_required(VERSION 3.20)

project(ChessEngine LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHESS_ENABLE_TT_STATS "Enable transposition-table counters in hot paths" OFF)
option(CHESS_ENABLE_LTO "Enable link-time optimization for release builds" OFF)
option(
    CHESS_BUILD_EXPERIMENTS
    "Build historical searchers, research tools, and experiment tests"
    OFF
)
set(
    CHESS_TT_PROFILE_MODE "none" CACHE STRING
    "Instrument historical transposition tables: none, timing, probe-paths, or cache-lines"
)
set_property(
    CACHE CHESS_TT_PROFILE_MODE PROPERTY STRINGS
    none timing probe-paths cache-lines
)
option(
    CHESS_ENABLE_NNUE_STAGE_BENCHMARK
    "Compile the non-production phase NNUE stage benchmark hooks"
    OFF
)

if(CHESS_ENABLE_NNUE_STAGE_BENCHMARK AND NOT CHESS_BUILD_EXPERIMENTS)
    message(FATAL_ERROR
        "CHESS_ENABLE_NNUE_STAGE_BENCHMARK requires CHESS_BUILD_EXPERIMENTS=ON"
    )
endif()

set(CHESS_TT_PROFILE_MODES none timing probe-paths cache-lines)
if(NOT CHESS_TT_PROFILE_MODE IN_LIST CHESS_TT_PROFILE_MODES)
    message(FATAL_ERROR
        "Invalid CHESS_TT_PROFILE_MODE='${CHESS_TT_PROFILE_MODE}'; "
        "choose one of: ${CHESS_TT_PROFILE_MODES}"
    )
endif()
if(
    NOT CHESS_TT_PROFILE_MODE STREQUAL "none"
    AND NOT CHESS_BUILD_EXPERIMENTS
)
    message(FATAL_ERROR
        "CHESS_TT_PROFILE_MODE requires CHESS_BUILD_EXPERIMENTS=ON"
    )
endif()

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
if(CHESS_BUILD_EXPERIMENTS)
    include(cmake/Experiments.cmake)
    include(cmake/Tools.cmake)
endif()
include(cmake/Tests.cmake)
