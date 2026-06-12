#include "attacks.hpp"
#include "board_encoder.hpp"
#include "heuristic_searcher_v6.hpp"
#include "move.hpp"
#include "position.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    int games = 100;
    int depth = 6;
    int random_plies = 12;
    int max_plies = 40;
    int limit = 20000;
    int progress_interval = 10;
    std::uint32_t seed = 20260621;
    std::string output = "data/material_imbalance_depth6_v6.jsonl";
};

struct PieceOnBoard {
    chess::Square square = chess::NoSquare;
    chess::Color color = chess::Color::White;
    chess::PieceType piece = chess::PieceType::None;
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

        if (arg == "--games") {
            options.games = parse_int(require_value(arg), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--limit") {
            options.limit = parse_int(require_value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--output") {
            options.output = std::string(require_value(arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: material_imbalance_export [--games N] [--depth D]\n"
                << "                                 [--random-plies N] [--max-plies N]\n"
                << "                                 [--limit N] [--seed N] [--output path]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.games <= 0) {
        throw std::runtime_error("--games must be positive");
    }
    if (options.depth < 0) {
        throw std::runtime_error("--depth must be non-negative");
    }
    if (options.random_plies < 0) {
        throw std::runtime_error("--random-plies must be non-negative");
    }
    if (options.max_plies <= 0) {
        throw std::runtime_error("--max-plies must be positive");
    }
    if (options.limit <= 0) {
        throw std::runtime_error("--limit must be positive");
    }
    if (options.progress_interval <= 0) {
        throw std::runtime_error("--progress-interval must be positive");
    }
    return options;
}

void write_sample(std::ostream& out, const chess::EncodedPosition& encoded, int target) {
    out << "{\"features\":[";
    for (std::size_t i = 0; i < encoded.features.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << encoded.features[i];
    }

    out << "],\"aux\":[";
    for (std::size_t i = 0; i < encoded.aux.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << static_cast<int>(encoded.aux[i]);
    }

    out << "],\"target\":" << target << "}\n";
}

chess::Move choose_random_move(const std::vector<chess::Move>& moves, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

std::vector<PieceOnBoard> pieces_of(
    const chess::Position& pos,
    chess::Color color,
    chess::PieceType piece
) {
    std::vector<PieceOnBoard> pieces;
    chess::Bitboard bb = pos.pieces[static_cast<int>(color)][static_cast<int>(piece)];
    while (bb) {
        pieces.push_back(PieceOnBoard{chess::pop_lsb(bb), color, piece});
    }
    return pieces;
}

bool valid_after_removal(const chess::Position& pos) {
    return !chess::in_check(pos, chess::opposite(pos.side_to_move));
}

int write_labeled_position(
    std::ostream& out,
    const chess::Position& pos,
    int depth,
    chess::HeuristicSearcherV6& teacher
) {
    const int target = teacher.search_best_move(pos, depth).score;
    write_sample(out, chess::encode_position(pos), target);
    return 1;
}

int write_removed_piece_variants(
    std::ostream& out,
    const chess::Position& pos,
    int depth,
    chess::HeuristicSearcherV6& teacher,
    int remaining
) {
    int written = 0;
    for (chess::Color color : {chess::Color::White, chess::Color::Black}) {
        for (chess::PieceType piece : {
                 chess::PieceType::Pawn,
                 chess::PieceType::Knight,
                 chess::PieceType::Bishop,
                 chess::PieceType::Rook,
                 chess::PieceType::Queen
             }) {
            for (const PieceOnBoard& candidate : pieces_of(pos, color, piece)) {
                if (written >= remaining) {
                    return written;
                }

                chess::Position variant = pos;
                variant.clear_square(candidate.square);
                if (!valid_after_removal(variant)) {
                    continue;
                }
                written += write_labeled_position(out, variant, depth, teacher);
            }
        }
    }
    return written;
}

int write_exchange_variants(
    std::ostream& out,
    const chess::Position& pos,
    int depth,
    chess::HeuristicSearcherV6& teacher,
    int remaining
) {
    int written = 0;
    for (chess::Color rook_color : {chess::Color::White, chess::Color::Black}) {
        const chess::Color minor_color = chess::opposite(rook_color);
        for (const PieceOnBoard& rook : pieces_of(pos, rook_color, chess::PieceType::Rook)) {
            for (chess::PieceType minor_piece : {chess::PieceType::Knight, chess::PieceType::Bishop}) {
                for (const PieceOnBoard& minor : pieces_of(pos, minor_color, minor_piece)) {
                    if (written >= remaining) {
                        return written;
                    }

                    chess::Position variant = pos;
                    variant.clear_square(rook.square);
                    variant.clear_square(minor.square);
                    if (!valid_after_removal(variant)) {
                        continue;
                    }
                    written += write_labeled_position(out, variant, depth, teacher);
                }
            }
        }
    }
    return written;
}

int write_two_pawn_variants(
    std::ostream& out,
    const chess::Position& pos,
    int depth,
    chess::HeuristicSearcherV6& teacher,
    int remaining
) {
    int written = 0;
    for (chess::Color color : {chess::Color::White, chess::Color::Black}) {
        const std::vector<PieceOnBoard> pawns = pieces_of(pos, color, chess::PieceType::Pawn);
        for (std::size_t first = 0; first < pawns.size(); ++first) {
            for (std::size_t second = first + 1; second < pawns.size(); ++second) {
                if (written >= remaining) {
                    return written;
                }

                chess::Position variant = pos;
                variant.clear_square(pawns[first].square);
                variant.clear_square(pawns[second].square);
                if (!valid_after_removal(variant)) {
                    continue;
                }
                written += write_labeled_position(out, variant, depth, teacher);
            }
        }
    }
    return written;
}

int write_material_imbalance(
    std::ostream& out,
    const chess::Position& pos,
    int depth,
    chess::HeuristicSearcherV6& teacher,
    int remaining
) {
    int written = 0;
    written += write_removed_piece_variants(out, pos, depth, teacher, remaining - written);
    if (written >= remaining) {
        return written;
    }
    written += write_two_pawn_variants(out, pos, depth, teacher, remaining - written);
    if (written >= remaining) {
        return written;
    }
    written += write_exchange_variants(out, pos, depth, teacher, remaining - written);
    return written;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        const std::filesystem::path output_path(options.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("failed to open output: " + options.output);
        }

        std::mt19937 rng(options.seed);
        chess::HeuristicSearcherV6 teacher;
        int written = 0;

        for (int game = 0; game < options.games && written < options.limit; ++game) {
            chess::Position pos;
            pos.set_startpos();

            for (int ply = 0; ply < options.max_plies && written < options.limit; ++ply) {
                const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
                if (moves.empty()) {
                    break;
                }

                written += write_material_imbalance(
                    out,
                    pos,
                    options.depth,
                    teacher,
                    options.limit - written);

                chess::Move move = ply < options.random_plies
                    ? choose_random_move(moves, rng)
                    : teacher.search_best_move(pos, options.depth).best_move;
                pos.make_move(move);
            }

            if ((game + 1) % options.progress_interval == 0 || game + 1 == options.games) {
                std::cerr << "game " << (game + 1) << " / " << options.games
                          << ", material imbalance samples " << written << '\n';
            }
        }

        std::cerr << "wrote " << written << " material imbalance samples to "
                  << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "material_imbalance_export: " << error.what() << '\n';
        return 1;
    }
}
