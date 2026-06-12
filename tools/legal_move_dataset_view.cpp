#include "attacks.hpp"
#include "heuristic_searcher_v6.hpp"
#include "move.hpp"
#include "position.hpp"

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
    std::string input = "data/legal_moves_depth6_v6_r12_5k.jsonl";
    std::string output = "data/legal_moves_depth6_v6_r12_5k_view.json";
    int limit = 160;
    int depth = 6;
    int random_plies = 12;
    int max_plies = 160;
    std::uint32_t seed = 20260613;
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
        throw std::runtime_error("invalid integer for " + std::string(name));
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

        if (arg == "--input") {
            options.input = std::string(require_value(arg));
        } else if (arg == "--output") {
            options.output = std::string(require_value(arg));
        } else if (arg == "--limit") {
            options.limit = parse_int(require_value(arg), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: legal_move_dataset_view [--input path] [--output path]\n"
                << "                               [--limit N] [--depth N]\n"
                << "                               [--random-plies N] [--max-plies N]\n"
                << "                               [--seed N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.limit <= 0) {
        throw std::runtime_error("--limit must be positive");
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
    return options;
}

int extract_target(const std::string& line) {
    constexpr std::string_view key = "\"target\":";
    const std::size_t start = line.find(key);
    if (start == std::string::npos) {
        throw std::runtime_error("sample is missing target");
    }
    const std::size_t value_start = start + key.size();
    std::size_t value_end = value_start;
    while (value_end < line.size() && line[value_end] != ',' && line[value_end] != '}') {
        ++value_end;
    }
    return parse_int(std::string_view(line).substr(value_start, value_end - value_start), "target");
}

chess::Move choose_random_move(const std::vector<chess::Move>& moves, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

std::string square_name(chess::Square square) {
    std::string name;
    name.push_back(static_cast<char>('a' + chess::file_of(square)));
    name.push_back(static_cast<char>('1' + chess::rank_of(square)));
    return name;
}

char piece_char(chess::PieceType piece, chess::Color color) {
    char value = '.';
    switch (piece) {
        case chess::PieceType::Pawn:
            value = 'P';
            break;
        case chess::PieceType::Knight:
            value = 'N';
            break;
        case chess::PieceType::Bishop:
            value = 'B';
            break;
        case chess::PieceType::Rook:
            value = 'R';
            break;
        case chess::PieceType::Queen:
            value = 'Q';
            break;
        case chess::PieceType::King:
            value = 'K';
            break;
        case chess::PieceType::None:
            value = '.';
            break;
    }
    if (color == chess::Color::Black && value != '.') {
        value = static_cast<char>(value - 'A' + 'a');
    }
    return value;
}

void write_board(std::ostream& out, const chess::Position& pos) {
    out << "\"board\":{";
    bool first = true;
    for (int square = 0; square < 64; ++square) {
        if (pos.is_empty(square)) {
            continue;
        }
        if (!first) {
            out << ',';
        }
        first = false;
        const chess::Color color = pos.color_on_occupied(square);
        const chess::PieceType piece = pos.piece_type_on_occupied(square);
        out << '"' << square_name(square) << "\":\"" << piece_char(piece, color) << '"';
    }
    out << '}';
}

std::string color_name(chess::Color color) {
    return color == chess::Color::White ? "white" : "black";
}

int white_target(int target, chess::Color side_to_move) {
    return side_to_move == chess::Color::White ? target : -target;
}

void write_sample(
    std::ostream& out,
    int index,
    int parent_ply,
    chess::Move move,
    const chess::Position& child,
    int target
) {
    if (index != 0) {
        out << ",\n";
    }
    out << "    {";
    out << "\"index\":" << index;
    out << ",\"parent_ply\":" << parent_ply;
    out << ",\"move\":\"" << chess::move_to_string(move) << '"';
    out << ",\"side_to_move\":\"" << color_name(child.side_to_move) << '"';
    out << ",\"target\":" << target;
    out << ",\"white_target\":" << white_target(target, child.side_to_move);
    out << ',';
    write_board(out, child);
    out << '}';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        std::ifstream input(options.input);
        if (!input) {
            throw std::runtime_error("failed to open input: " + options.input);
        }

        const std::filesystem::path output_path(options.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        std::ofstream output(output_path);
        if (!output) {
            throw std::runtime_error("failed to open output: " + options.output);
        }

        chess::Position pos;
        pos.set_startpos();
        std::mt19937 rng(options.seed);
        chess::HeuristicSearcherV6 teacher;

        output << "{\n";
        output << "  \"ok\":true,\n";
        output << "  \"dataset\":\"" << options.input << "\",\n";
        output << "  \"samples\":[\n";

        int written = 0;
        std::string line;
        for (int ply = 0; ply < options.max_plies && written < options.limit; ++ply) {
            const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
            if (moves.empty()) {
                break;
            }

            for (chess::Move move : moves) {
                if (written >= options.limit) {
                    break;
                }
                if (!std::getline(input, line)) {
                    break;
                }
                chess::Position child = pos;
                child.make_move(move);
                write_sample(output, written, ply + 1, move, child, extract_target(line));
                ++written;
            }

            if (written >= options.limit) {
                break;
            }

            chess::Move played = ply < options.random_plies
                ? choose_random_move(moves, rng)
                : teacher.search_best_move(pos, options.depth).best_move;
            pos.make_move(played);
        }

        output << "\n  ]\n";
        output << "}\n";
        std::cerr << "wrote " << written << " view samples to " << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "legal_move_dataset_view: " << error.what() << '\n';
        return 1;
    }
}
