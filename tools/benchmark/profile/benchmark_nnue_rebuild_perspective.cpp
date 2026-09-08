#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::array<std::string_view, 10> Fens{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
    "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
    "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
    "8/8/2p5/3p4/3P4/2P5/8/4K1k1 w - - 0 1",
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
};

int parse_iterations(const char* text) {
    const int value = std::stoi(text);
    if (value <= 0) {
        throw std::runtime_error("iterations must be positive");
    }
    return value;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: benchmark_nnue_rebuild_perspective MODEL.bin [ITERATIONS]\n";
        return 2;
    }

    try {
        const int iterations = argc == 3 ? parse_iterations(argv[2]) : 20'000;
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(argv[1])) {
            throw std::runtime_error("failed to load model");
        }

        std::array<chess::Position, Fens.size()> positions;
        for (std::size_t index = 0; index < Fens.size(); ++index) {
            if (!positions[index].set_fen(Fens[index])) {
                throw std::runtime_error("invalid benchmark FEN");
            }
        }

        chess::PhaseQuantizedNnueAccumulator accumulator;
        std::uint64_t checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (const chess::Position& pos : positions) {
                accumulator.reset(model, pos);
                checksum += static_cast<std::uint32_t>(
                    accumulator.positional_accumulators()[0][
                        static_cast<std::size_t>(iteration) & 127]);
            }
        }
        const auto end = std::chrono::steady_clock::now();

        const std::uint64_t resets =
            static_cast<std::uint64_t>(iterations) * positions.size();
        const std::uint64_t perspectives = resets * 2;
        const std::uint64_t elapsed_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count());
        std::cout
            << "iterations=" << iterations
            << " positions=" << positions.size()
            << " resets=" << resets
            << " perspectives=" << perspectives
            << " elapsed_ns=" << elapsed_ns
            << " ns_per_perspective="
            << static_cast<double>(elapsed_ns)
                / static_cast<double>(perspectives)
            << " checksum=" << checksum
            << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
