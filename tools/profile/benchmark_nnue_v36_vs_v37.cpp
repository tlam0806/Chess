#include "move.hpp"
#include "nnue_searcher_v36.hpp"
#include "nnue_searcher_v37.hpp"
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
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
    "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
    "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
    "8/8/2p5/3p4/3P4/2P5/8/4K1k1 w - - 0 1",
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
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

template<typename Searcher>
chess::SearchResult run_one(
    Searcher& searcher,
    const chess::Position& position,
    int depth,
    Totals* totals
) {
    searcher.clear_tt();
    chess::SearchLimits limits;
    limits.max_depth = depth;
    const auto start = std::chrono::steady_clock::now();
    const chess::SearchResult result =
        searcher.search_best_move(position, limits);
    const auto end = std::chrono::steady_clock::now();
    if (totals != nullptr) {
        totals->nodes += result.nodes;
        totals->microseconds += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count());
    }
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
    try {
        const int depth = argc >= 2 ? parse_positive(argv[1], "depth") : 7;
        const int repeats = argc >= 3 ? parse_positive(argv[2], "repeats") : 1;
        const std::string model_path = argc >= 4
            ? argv[3]
            : std::string(chess::DefaultPhaseQuantizedNnueModelPath);

        chess::PhaseQuantizedNnueModel model;
        if (!model.load(model_path)) {
            throw std::runtime_error("failed to load model: " + model_path);
        }
        model.set_neon_dotprod_enabled(true);

        std::array<chess::Position, Fens.size()> positions;
        for (std::size_t index = 0; index < Fens.size(); ++index) {
            if (!positions[index].set_fen(Fens[index])) {
                throw std::runtime_error("invalid benchmark FEN");
            }
        }

        constexpr std::size_t V36TtMb = 64;
        constexpr std::size_t V37TtMb = 1024;
        constexpr std::size_t BucketSize = 4;
        chess::NnueSearcherV36 v36(model, V36TtMb, BucketSize);
        chess::NnueSearcherV37 v37(model, V37TtMb, BucketSize);

        run_one(v36, positions[0], 4, nullptr);
        run_one(v37, positions[0], 4, nullptr);

        Totals v36_totals;
        Totals v37_totals;
        int score_mismatches = 0;
        int move_mismatches = 0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (std::size_t index = 0; index < positions.size(); ++index) {
                chess::SearchResult result36;
                chess::SearchResult result37;
                if (((static_cast<std::size_t>(repeat) + index) & 1U) == 0) {
                    result36 = run_one(v36, positions[index], depth, &v36_totals);
                    result37 = run_one(v37, positions[index], depth, &v37_totals);
                } else {
                    result37 = run_one(v37, positions[index], depth, &v37_totals);
                    result36 = run_one(v36, positions[index], depth, &v36_totals);
                }
                score_mismatches += result36.score != result37.score;
                move_mismatches += result36.best_move != result37.best_move;
            }
        }

        const double v36_nps = nps(v36_totals);
        const double v37_nps = nps(v37_totals);
        std::cout
            << "depth=" << depth
            << " positions=" << positions.size()
            << " repeats=" << repeats
            << " v36_tt_mb=" << V36TtMb
            << " v36_tt_entries=" << v36.tt_entry_count()
            << " v36_nodes=" << v36_totals.nodes
            << " v36_us=" << v36_totals.microseconds
            << " v36_nps=" << v36_nps
            << " v37_tt_mb=" << V37TtMb
            << " v37_tt_entries=" << v37.tt_entry_count()
            << " v37_nodes=" << v37_totals.nodes
            << " v37_us=" << v37_totals.microseconds
            << " v37_nps=" << v37_nps
            << " v37_vs_v36_time="
            << static_cast<double>(v37_totals.microseconds)
                / static_cast<double>(v36_totals.microseconds)
            << " v37_vs_v36_nodes="
            << static_cast<double>(v37_totals.nodes)
                / static_cast<double>(v36_totals.nodes)
            << " v37_vs_v36_nps="
            << (v36_nps == 0.0 ? 0.0 : v37_nps / v36_nps)
            << " score_mismatches=" << score_mismatches
            << " move_mismatches=" << move_mismatches
            << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
