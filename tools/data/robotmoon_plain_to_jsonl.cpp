#include "board_encoder.hpp"
#include "move.hpp"
#include "position.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string input;
    std::string output;
    std::int64_t limit = 5'000'000;
    int max_abs_score = 10'000;
    std::int64_t progress_interval = 100'000;
};

struct PlainRecord {
    std::string fen;
    std::string move;
    int score = 0;
    int ply = 0;
    int result = 0;
};

struct Stats {
    std::int64_t records = 0;
    std::int64_t written = 0;
    std::int64_t invalid_fen = 0;
    std::int64_t illegal_move = 0;
    std::int64_t score_filtered = 0;
    std::int64_t malformed = 0;
};

std::int64_t parse_i64(std::string_view value, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const std::int64_t result = std::stoll(std::string(value), &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + std::string(name) + ": " + std::string(value));
    }
}

int parse_int(std::string_view value, std::string_view name) {
    const std::int64_t parsed = parse_i64(value, name);
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error("integer out of range for " + std::string(name));
    }
    return static_cast<int>(parsed);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };

        if (arg == "--input") {
            options.input = std::string(value(arg));
        } else if (arg == "--output") {
            options.output = std::string(value(arg));
        } else if (arg == "--limit") {
            options.limit = parse_i64(value(arg), arg);
        } else if (arg == "--max-abs-score") {
            options.max_abs_score = parse_int(value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_i64(value(arg), arg);
        } else if (arg == "--help") {
            std::cout
                << "Usage: robotmoon_plain_to_jsonl --input file.plain --output file.jsonl\n"
                << "                                  [--limit N] [--max-abs-score CP]\n"
                << "                                  [--progress-interval N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    if (options.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (options.limit <= 0) {
        throw std::runtime_error("--limit must be positive");
    }
    if (options.max_abs_score <= 0) {
        throw std::runtime_error("--max-abs-score must be positive");
    }
    if (options.progress_interval <= 0) {
        throw std::runtime_error("--progress-interval must be positive");
    }
    return options;
}

bool read_record(std::istream& in, PlainRecord& record) {
    record = PlainRecord{};

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            break;
        }
    }
    if (!in && line.empty()) {
        return false;
    }
    if (!line.starts_with("fen ")) {
        throw std::runtime_error("expected fen line, got: " + line);
    }
    record.fen = line.substr(4);

    if (!std::getline(in, line) || !line.starts_with("move ")) {
        throw std::runtime_error("expected move line");
    }
    record.move = line.substr(5);

    if (!std::getline(in, line) || !line.starts_with("score ")) {
        throw std::runtime_error("expected score line");
    }
    record.score = parse_int(line.substr(6), "score");

    if (!std::getline(in, line) || !line.starts_with("ply ")) {
        throw std::runtime_error("expected ply line");
    }
    record.ply = parse_int(line.substr(4), "ply");

    if (!std::getline(in, line) || !line.starts_with("result ")) {
        throw std::runtime_error("expected result line");
    }
    record.result = parse_int(line.substr(7), "result");

    if (!std::getline(in, line) || line != "e") {
        throw std::runtime_error("expected record terminator");
    }

    return true;
}

bool has_both_kings(const chess::Position& pos) {
    return pos.king_squares[static_cast<int>(chess::Color::White)] != chess::NoSquare
        && pos.king_squares[static_cast<int>(chess::Color::Black)] != chess::NoSquare;
}

bool is_legal_move_text(const chess::Position& pos, const std::string& move_text) {
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    return std::any_of(moves.begin(), moves.end(), [&](chess::Move move) {
        return chess::move_to_string(move) == move_text;
    });
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
        std::ofstream output(options.output);
        if (!output) {
            throw std::runtime_error("failed to open output: " + options.output);
        }

        Stats stats;
        PlainRecord record;
        while (stats.written < options.limit && read_record(input, record)) {
            ++stats.records;
            if (std::abs(record.score) > options.max_abs_score) {
                ++stats.score_filtered;
                continue;
            }

            chess::Position pos;
            if (!pos.set_fen(record.fen) || !has_both_kings(pos)) {
                ++stats.invalid_fen;
                continue;
            }
            if (!is_legal_move_text(pos, record.move)) {
                ++stats.illegal_move;
                continue;
            }

            write_sample(output, chess::encode_position(pos), record.score);
            ++stats.written;

            if (stats.written % options.progress_interval == 0) {
                std::cerr
                    << "written=" << stats.written
                    << " records=" << stats.records
                    << " score_filtered=" << stats.score_filtered
                    << " invalid_fen=" << stats.invalid_fen
                    << " illegal_move=" << stats.illegal_move
                    << '\n';
            }
        }

        std::cerr
            << "done"
            << " written=" << stats.written
            << " records=" << stats.records
            << " score_filtered=" << stats.score_filtered
            << " invalid_fen=" << stats.invalid_fen
            << " illegal_move=" << stats.illegal_move
            << '\n';
        return stats.written == options.limit ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "robotmoon_plain_to_jsonl: " << error.what() << '\n';
        return 1;
    }
}
