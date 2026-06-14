#include "heuristic_searcher_v15.hpp"
#include "heuristic_searcher_v15_pvs.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    int depth = 7;
    int fixed_count = 3;
    int random_positions = 0;
    int random_plies = 40;
    std::uint32_t seed = 20260618;
    bool iterative = true;
};

struct Totals {
    std::uint64_t nodes = 0;
    double ms = 0.0;
};

int parse_int(std::string_view value, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const int result = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + std::string(name) + ": " + std::string(value));
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };

        if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--fixed-count") {
            options.fixed_count = parse_int(require_value(arg), arg);
        } else if (arg == "--random-positions") {
            options.random_positions = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--fixed-depth") {
            options.iterative = false;
        } else if (arg == "--iterative") {
            options.iterative = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_v15_pvs_compare [--depth N] [--iterative|--fixed-depth]\n"
                << "                                 [--fixed-count N] [--random-positions N]\n"
                << "                                 [--random-plies N] [--seed N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    return options;
}

std::vector<chess::Position> fixed_positions(int fixed_count) {
    const std::vector<std::string_view> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
        "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
        "4r3/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23",
        "2b1qb1r/rpppkp1p/2n1p2n/p7/P3N3/1P3P2/2PNP1PP/R1BQKB1R w KQ - 3 14",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
    };

    std::vector<chess::Position> positions;
    const int count = std::min<int>(fixed_count, static_cast<int>(fens.size()));
    for (int i = 0; i < count; ++i) {
        chess::Position pos;
        if (!pos.set_fen(fens[static_cast<std::size_t>(i)])) {
            throw std::runtime_error("invalid hardcoded FEN");
        }
        positions.push_back(pos);
    }
    return positions;
}

std::vector<chess::Position> random_positions(const Options& options) {
    std::mt19937 rng(options.seed);
    std::vector<chess::Position> positions;
    for (int sample = 0; sample < options.random_positions; ++sample) {
        chess::Position pos;
        pos.set_startpos();
        std::uniform_int_distribution<int> plies_dist(0, options.random_plies);
        const int plies = plies_dist(rng);
        for (int ply = 0; ply < plies; ++ply) {
            const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
            if (moves.empty()) {
                break;
            }
            std::uniform_int_distribution<std::size_t> move_dist(0, moves.size() - 1);
            pos.make_move(moves[move_dist(rng)]);
        }
        positions.push_back(pos);
    }
    return positions;
}

char piece_char(chess::Color color, chess::PieceType piece) {
    char result = '?';
    switch (piece) {
        case chess::PieceType::Pawn: result = 'p'; break;
        case chess::PieceType::Knight: result = 'n'; break;
        case chess::PieceType::Bishop: result = 'b'; break;
        case chess::PieceType::Rook: result = 'r'; break;
        case chess::PieceType::Queen: result = 'q'; break;
        case chess::PieceType::King: result = 'k'; break;
        case chess::PieceType::None: result = '?'; break;
    }
    if (color == chess::Color::White && result >= 'a' && result <= 'z') {
        result = static_cast<char>(result - 'a' + 'A');
    }
    return result;
}

std::string to_fen(const chess::Position& pos) {
    std::ostringstream out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const chess::Square square = chess::make_square(file, rank);
            if (pos.is_empty(square)) {
                ++empty;
                continue;
            }
            if (empty != 0) {
                out << empty;
                empty = 0;
            }
            out << piece_char(pos.color_on_occupied(square), pos.piece_type_on_occupied(square));
        }
        if (empty != 0) {
            out << empty;
        }
        if (rank != 0) {
            out << '/';
        }
    }
    out << (pos.side_to_move == chess::Color::White ? " w " : " b ");
    std::string castling;
    if (pos.white_can_castle_kingside) castling += 'K';
    if (pos.white_can_castle_queenside) castling += 'Q';
    if (pos.black_can_castle_kingside) castling += 'k';
    if (pos.black_can_castle_queenside) castling += 'q';
    out << (castling.empty() ? "-" : castling);
    out << ' ';
    if (pos.en_passant_square == chess::NoSquare) {
        out << '-';
    } else {
        out << static_cast<char>('a' + chess::file_of(pos.en_passant_square))
            << static_cast<char>('1' + chess::rank_of(pos.en_passant_square));
    }
    out << ' ' << pos.halfmove_clock << ' ' << pos.fullmove_number;
    return out.str();
}

template <typename Searcher>
chess::SearchResult search(Searcher& searcher, const chess::Position& pos, const Options& options) {
    searcher.clear_tt();
    if (!options.iterative) {
        return searcher.search_best_move(pos, options.depth);
    }

    chess::SearchResult best;
    std::uint64_t total_nodes = 0;
    for (int depth = 1; depth <= options.depth; ++depth) {
        best = searcher.search_best_move(pos, depth);
        total_nodes += best.nodes;
    }
    best.nodes = total_nodes;
    return best;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        std::vector<chess::Position> positions = fixed_positions(options.fixed_count);
        std::vector<chess::Position> random = random_positions(options);
        positions.insert(positions.end(), random.begin(), random.end());

        Totals base_totals;
        Totals pvs_totals;
        int score_mismatches = 0;
        int move_mismatches = 0;

        for (std::size_t i = 0; i < positions.size(); ++i) {
            chess::HeuristicSearcherV15 base(64);
            chess::HeuristicSearcherV15Pvs pvs(64);

            const auto base_start = std::chrono::steady_clock::now();
            const chess::SearchResult base_result = search(base, positions[i], options);
            const auto base_stop = std::chrono::steady_clock::now();

            const auto pvs_start = std::chrono::steady_clock::now();
            const chess::SearchResult pvs_result = search(pvs, positions[i], options);
            const auto pvs_stop = std::chrono::steady_clock::now();

            const double base_ms = std::chrono::duration<double, std::milli>(base_stop - base_start).count();
            const double pvs_ms = std::chrono::duration<double, std::milli>(pvs_stop - pvs_start).count();
            base_totals.nodes += base_result.nodes;
            base_totals.ms += base_ms;
            pvs_totals.nodes += pvs_result.nodes;
            pvs_totals.ms += pvs_ms;

            const bool score_mismatch = base_result.score != pvs_result.score;
            const bool move_mismatch = base_result.best_move != pvs_result.best_move;
            if (score_mismatch) {
                ++score_mismatches;
            }
            if (move_mismatch) {
                ++move_mismatches;
            }

            std::cout
                << "sample=" << i
                << " depth=" << options.depth
                << " mode=" << (options.iterative ? "iterative" : "fixed")
                << " base_score=" << base_result.score
                << " pvs_score=" << pvs_result.score
                << " base_move=" << chess::move_to_string(base_result.best_move)
                << " pvs_move=" << chess::move_to_string(pvs_result.best_move)
                << " base_nodes=" << base_result.nodes
                << " pvs_nodes=" << pvs_result.nodes
                << " node_ratio=" << (base_result.nodes == 0 ? 0.0 : static_cast<double>(pvs_result.nodes) / static_cast<double>(base_result.nodes))
                << " base_ms=" << base_ms
                << " pvs_ms=" << pvs_ms
                << " time_ratio=" << (base_ms == 0.0 ? 0.0 : pvs_ms / base_ms)
                << " score_mismatch=" << (score_mismatch ? 1 : 0)
                << " move_mismatch=" << (move_mismatch ? 1 : 0)
                << " fen=\"" << to_fen(positions[i]) << "\""
                << '\n' << std::flush;
        }

        std::cout
            << "SUMMARY"
            << " samples=" << positions.size()
            << " depth=" << options.depth
            << " mode=" << (options.iterative ? "iterative" : "fixed")
            << " base_nodes=" << base_totals.nodes
            << " pvs_nodes=" << pvs_totals.nodes
            << " node_ratio=" << (base_totals.nodes == 0 ? 0.0 : static_cast<double>(pvs_totals.nodes) / static_cast<double>(base_totals.nodes))
            << " base_ms=" << base_totals.ms
            << " pvs_ms=" << pvs_totals.ms
            << " time_ratio=" << (base_totals.ms == 0.0 ? 0.0 : pvs_totals.ms / base_totals.ms)
            << " score_mismatches=" << score_mismatches
            << " move_mismatches=" << move_mismatches
            << '\n';
    } catch (const std::exception& error) {
        std::cerr << "benchmark_v15_pvs_compare: " << error.what() << '\n';
        return 1;
    }
}
