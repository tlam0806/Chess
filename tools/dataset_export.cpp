#include "board_encoder.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search.hpp"

#include <cstdint>
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
    int positions = 1000;
    int games = 0;
    int depth = 2;
    int random_plies = 8;
    int max_plies_per_game = 160;
    int progress_interval = 100;
    std::uint32_t seed = 1;
    std::string output = "data/value_train.jsonl";
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

        if (arg == "--positions") {
            options.positions = parse_int(require_value(arg), arg);
        } else if (arg == "--games") {
            options.games = parse_int(require_value(arg), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies_per_game = parse_int(require_value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--output") {
            options.output = std::string(require_value(arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: dataset_export [--positions N] [--games N] [--depth D]\n"
                << "                      [--random-plies N] [--max-plies N]\n"
                << "                      [--progress-interval N]\n"
                << "                      [--seed N] [--output path]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.positions <= 0) {
        throw std::runtime_error("--positions must be positive");
    }
    if (options.games < 0) {
        throw std::runtime_error("--games must be non-negative");
    }
    if (options.depth < 0) {
        throw std::runtime_error("--depth must be non-negative");
    }
    if (options.random_plies < 0) {
        throw std::runtime_error("--random-plies must be non-negative");
    }
    if (options.max_plies_per_game <= 0) {
        throw std::runtime_error("--max-plies must be positive");
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

void reset_game(chess::Position& pos, int& ply) {
    pos.set_startpos();
    ply = 0;
}

chess::Move choose_random_move(const std::vector<chess::Move>& moves, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

int export_position_sample(std::ostream& out, const chess::Position& pos, int depth) {
    const int target = chess::search_best_move(pos, depth).score;
    write_sample(out, chess::encode_position(pos), target);
    return target;
}

chess::SearchResult export_search_sample(std::ostream& out, const chess::Position& pos, int depth) {
    const chess::SearchResult result = chess::search_best_move(pos, depth);
    write_sample(out, chess::encode_position(pos), result.score);
    return result;
}

int export_games(std::ostream& out, const Options& options, std::mt19937& rng) {
    int written = 0;

    for (int game = 0; game < options.games; ++game) {
        chess::Position pos;
        pos.set_startpos();

        for (int ply = 0; ply < options.max_plies_per_game; ++ply) {
            const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
            if (moves.empty()) {
                break;
            }

            const chess::SearchResult search_result = export_search_sample(out, pos, options.depth);
            ++written;

            chess::Move move;
            if (ply < options.random_plies) {
                move = choose_random_move(moves, rng);
            } else {
                move = search_result.best_move;
            }
            pos.make_move(move);
        }

        if ((game + 1) % options.progress_interval == 0 || game + 1 == options.games) {
            std::cerr << "game " << (game + 1) << " / " << options.games
                      << ", samples " << written << '\n';
        }
    }

    return written;
}

int export_positions(std::ostream& out, const Options& options, std::mt19937& rng) {
    chess::Position pos;
    int ply = 0;
    reset_game(pos, ply);

    for (int written = 0; written < options.positions;) {
        std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (moves.empty() || ply >= options.max_plies_per_game) {
            reset_game(pos, ply);
            continue;
        }

        export_position_sample(out, pos, options.depth);
        ++written;

        pos.make_move(choose_random_move(moves, rng));
        ++ply;

        if (written % 1000 == 0) {
            std::cerr << "written " << written << " / " << options.positions << '\n';
        }
    }

    return options.positions;
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
        const int written = options.games > 0
            ? export_games(out, options, rng)
            : export_positions(out, options, rng);

        std::cerr << "wrote " << written << " samples to " << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "dataset_export: " << error.what() << '\n';
        return 1;
    }
}
