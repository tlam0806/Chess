#include "nnue_searcher_v40.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::array<std::string_view, 10> Fens{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
    "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
    "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
    "8/8/2p5/3p4/3P4/2P5/8/4K1k1 w - - 0 1",
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
    "2r3k1/5ppp/1p2p3/p2pP3/P2P1P2/1P1B2P1/5K1P/2R5 b - - 0 28",
};

struct Totals {
    std::uint64_t nodes = 0;
    std::uint64_t microseconds = 0;
};

int parse_positive(const char* text, std::string_view name) {
    const int value = std::stoi(text);
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return value;
}

chess::SearchResult run_one(
    chess::NnueSearcherV40& searcher,
    const chess::Position& position,
    int depth,
    Totals& totals
) {
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = std::chrono::steady_clock::now();
    const chess::SearchResult result = searcher.search_best_move(position, limits);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);
    totals.nodes += result.nodes;
    totals.microseconds += static_cast<std::uint64_t>(elapsed.count());
    return result;
}

double nps(const Totals& totals) {
    return totals.microseconds == 0
        ? 0.0
        : static_cast<double>(totals.nodes) * 1'000'000.0
            / static_cast<double>(totals.microseconds);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr
            << "usage: benchmark_nnue_horizontal_mirror "
            << "SYMMETRIC_F2.bin F2M.bin [DEPTH] [REPEATS]\n";
        return 2;
    }
    try {
        const int depth = argc >= 4 ? parse_positive(argv[3], "depth") : 6;
        const int repeats = argc >= 5 ? parse_positive(argv[4], "repeats") : 3;
        chess::PhaseQuantizedNnueModel reference_model;
        chess::PhaseQuantizedNnueModel mirror_model;
        if (!reference_model.load(argv[1]) || !mirror_model.load(argv[2])) {
            throw std::runtime_error("failed to load one or both models");
        }
        if (reference_model.uses_horizontal_mirror()
            || !mirror_model.uses_horizontal_mirror()) {
            throw std::runtime_error("model encodings do not match benchmark roles");
        }
        std::array<chess::Position, Fens.size()> positions;
        for (std::size_t index = 0; index < Fens.size(); ++index) {
            if (!positions[index].set_fen(Fens[index])) {
                throw std::runtime_error("invalid benchmark FEN");
            }
            if (reference_model.evaluate_cp_rounded(positions[index])
                != mirror_model.evaluate_cp_rounded(positions[index])) {
                throw std::runtime_error("model evaluation mismatch before search");
            }
        }

        chess::NnueSearcherV40 reference(reference_model, 64, 4);
        chess::NnueSearcherV40 mirror(mirror_model, 64, 4);
        Totals reference_totals;
        Totals mirror_totals;
        int result_mismatches = 0;
        int node_mismatches = 0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (std::size_t index = 0; index < positions.size(); ++index) {
                chess::SearchResult first;
                chess::SearchResult second;
                if (((static_cast<std::size_t>(repeat) + index) & 1U) == 0) {
                    first = run_one(reference, positions[index], depth, reference_totals);
                    second = run_one(mirror, positions[index], depth, mirror_totals);
                } else {
                    second = run_one(mirror, positions[index], depth, mirror_totals);
                    first = run_one(reference, positions[index], depth, reference_totals);
                }
                result_mismatches += first.best_move != second.best_move
                    || first.score != second.score;
                node_mismatches += first.nodes != second.nodes;
            }
        }

        const double reference_nps = nps(reference_totals);
        const double mirror_nps = nps(mirror_totals);
        std::cout
            << "depth=" << depth
            << " repeats=" << repeats
            << " positions=" << positions.size()
            << " reference_nodes=" << reference_totals.nodes
            << " mirror_nodes=" << mirror_totals.nodes
            << " reference_us=" << reference_totals.microseconds
            << " mirror_us=" << mirror_totals.microseconds
            << " reference_nps=" << reference_nps
            << " mirror_nps=" << mirror_nps
            << " mirror_vs_reference_nps="
            << (reference_nps == 0.0 ? 0.0 : mirror_nps / reference_nps)
            << " result_mismatches=" << result_mismatches
            << " node_mismatches=" << node_mismatches
            << '\n';
        return result_mismatches == 0 && node_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
