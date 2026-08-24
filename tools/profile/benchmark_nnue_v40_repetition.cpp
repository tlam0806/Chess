#include "nnue_searcher_v40.hpp"
#include "nnue_searcher_v41.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model = std::string(chess::DefaultPhaseQuantizedNnueModelPath);
    int depth = 7;
    int repeats = 5;
};

int parse_int(std::string_view value, std::string_view name) {
    try {
        return std::stoi(std::string(value));
    } catch (...) {
        throw std::runtime_error("invalid integer for " + std::string(name));
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + std::string(arg));
            }
            return argv[i];
        };
        if (arg == "--model") {
            options.model = std::string(next());
        } else if (arg == "--depth") {
            options.depth = parse_int(next(), arg);
        } else if (arg == "--repeats") {
            options.repeats = parse_int(next(), arg);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.depth < 1 || options.repeats < 1) {
        throw std::runtime_error("depth and repeats must be positive");
    }
    return options;
}

std::vector<chess::Position> positions() {
    constexpr std::string_view Fens[] = {
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
        "2r3k1/1p1bqppp/p3pn2/3p4/3P4/P1NBPN2/1PQ2PPP/2R2RK1 b - - 0 16",
        "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
    };

    std::vector<chess::Position> result;
    result.reserve(std::size(Fens));
    for (const std::string_view fen : Fens) {
        chess::Position pos;
        if (!pos.set_fen(fen)) {
            throw std::runtime_error("invalid benchmark FEN");
        }
        result.push_back(pos);
    }
    return result;
}

struct Measurement {
    std::uint64_t nodes = 0;
    std::uint64_t us = 0;
};

template <typename Searcher>
Measurement measure_position(
    Searcher& searcher,
    const chess::Position& pos,
    int depth
) {
    Measurement result;
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    const auto start = std::chrono::steady_clock::now();
    result.nodes = searcher.search_best_move(pos, depth).nodes;
    result.us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count());
    return result;
}

void print_round(std::string_view version, int repeat, const Measurement& value) {
    std::cout << "round=" << repeat
              << " version=" << version
              << " nodes=" << value.nodes
              << " us=" << value.us
              << " nps=" << (value.nodes * 1'000'000.0 / value.us)
              << '\n';
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model: " + options.model);
        }
        model.set_neon_dotprod_enabled(true);
        const std::vector<chess::Position> suite = positions();

        Measurement v40_total;
        Measurement v41_total;
        std::vector<double> paired_nps_ratios;
        std::vector<double> paired_time_ratios;
        paired_nps_ratios.reserve(static_cast<std::size_t>(options.repeats));
        paired_time_ratios.reserve(static_cast<std::size_t>(options.repeats));
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            chess::NnueSearcherV40 v40(model, 64, 4);
            chess::NnueSearcherV41 v41(model, 64, 4);
            Measurement v40_round;
            Measurement v41_round;
            for (std::size_t index = 0; index < suite.size(); ++index) {
                Measurement v40_position;
                Measurement v41_position;
                if (((repeat + static_cast<int>(index)) & 1) == 0) {
                    v40_position = measure_position(
                        v40, suite[index], options.depth);
                    v41_position = measure_position(
                        v41, suite[index], options.depth);
                } else {
                    v41_position = measure_position(
                        v41, suite[index], options.depth);
                    v40_position = measure_position(
                        v40, suite[index], options.depth);
                }
                v40_round.nodes += v40_position.nodes;
                v40_round.us += v40_position.us;
                v41_round.nodes += v41_position.nodes;
                v41_round.us += v41_position.us;
            }
            v40_total.nodes += v40_round.nodes;
            v40_total.us += v40_round.us;
            v41_total.nodes += v41_round.nodes;
            v41_total.us += v41_round.us;
            const double v40_round_nps =
                v40_round.nodes * 1'000'000.0 / v40_round.us;
            const double v41_round_nps =
                v41_round.nodes * 1'000'000.0 / v41_round.us;
            paired_nps_ratios.push_back(v41_round_nps / v40_round_nps);
            paired_time_ratios.push_back(
                static_cast<double>(v41_round.us) / v40_round.us);
            print_round("v40", repeat, v40_round);
            print_round("v41", repeat, v41_round);
        }
        const double v40_nps = v40_total.nodes * 1'000'000.0 / v40_total.us;
        const double v41_nps = v41_total.nodes * 1'000'000.0 / v41_total.us;
        std::cout << "summary version=v40"
                  << " depth=" << options.depth
                  << " samples=" << suite.size()
                  << " repeats=" << options.repeats
                  << " nodes=" << v40_total.nodes
                  << " us=" << v40_total.us
                  << " nps=" << v40_nps
                  << '\n';
        std::cout << "summary version=v41"
                  << " depth=" << options.depth
                  << " samples=" << suite.size()
                  << " repeats=" << options.repeats
                  << " nodes=" << v41_total.nodes
                  << " us=" << v41_total.us
                  << " nps=" << v41_nps
                  << " nps_ratio=" << (v41_nps / v40_nps)
                  << " node_ratio="
                  << (static_cast<double>(v41_total.nodes) / v40_total.nodes)
                  << " median_paired_nps_ratio="
                  << median(paired_nps_ratios)
                  << " median_paired_time_ratio="
                  << median(paired_time_ratios)
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
