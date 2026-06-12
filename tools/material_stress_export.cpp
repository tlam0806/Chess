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
    int depth = 4;
    int random_plies = 12;
    int max_plies = 120;
    int limit = 50000;
    int progress_interval = 10;
    std::uint32_t seed = 20260612;
    std::string output = "data/material_stress.jsonl";
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
                << "Usage: material_stress_export [--games N] [--depth D]\n"
                << "                              [--random-plies N] [--max-plies N]\n"
                << "                              [--limit N] [--seed N] [--output path]\n";
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

std::vector<chess::Move> capture_moves(const chess::Position& pos) {
    std::vector<chess::Move> captures;
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::is_capture(move)) {
            captures.push_back(move);
        }
    }
    return captures;
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

int write_capture_stress(
    std::ostream& out,
    const chess::Position& pos,
    int depth,
    chess::HeuristicSearcherV6& teacher,
    int remaining
) {
    int written = 0;
    for (chess::Move capture : capture_moves(pos)) {
        if (written >= remaining) {
            return written;
        }

        chess::Position after_capture = pos;
        after_capture.make_move(capture);
        written += write_labeled_position(out, after_capture, depth, teacher);

        for (chess::Move recapture : capture_moves(after_capture)) {
            if (written >= remaining) {
                return written;
            }
            chess::Position after_recapture = after_capture;
            after_recapture.make_move(recapture);
            written += write_labeled_position(out, after_recapture, depth, teacher);
        }
    }
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

                written += write_capture_stress(
                    out,
                    pos,
                    options.depth,
                    teacher,
                    options.limit - written);

                chess::Move move;
                if (ply < options.random_plies) {
                    move = choose_random_move(moves, rng);
                } else {
                    move = teacher.search_best_move(pos, options.depth).best_move;
                }
                pos.make_move(move);
            }

            if ((game + 1) % options.progress_interval == 0 || game + 1 == options.games) {
                std::cerr << "game " << (game + 1) << " / " << options.games
                          << ", stress samples " << written << '\n';
            }
        }

        std::cerr << "wrote " << written << " material stress samples to "
                  << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "material_stress_export: " << error.what() << '\n';
        return 1;
    }
}
