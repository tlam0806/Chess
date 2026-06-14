#include "board_encoder.hpp"
#include "game_state.hpp"
#include "heuristic_searcher_v10.hpp"
#include "move.hpp"
#include "nnue_value.hpp"
#include "position.hpp"
#include "search_types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BookLine {
    int line_number = 0;
    std::string text;
    std::vector<std::string> moves;
};

struct Options {
    std::string book_path = "data/opening_book_6plies.txt";
    std::string model_path = "models/nnue_value_mix_666k_hardneg_lr1e4.bin";
    std::string output = "data/nnue_spectator_gap_depth7.jsonl";
    int games = 1000000;
    int depth = 7;
    int max_plies = 90;
    int max_samples = 100000;
    int min_gap = 180;
    int time_limit_seconds = 10800;
    int progress_interval = 100;
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

        if (arg == "--book") {
            options.book_path = std::string(require_value(arg));
        } else if (arg == "--model") {
            options.model_path = std::string(require_value(arg));
        } else if (arg == "--output") {
            options.output = std::string(require_value(arg));
        } else if (arg == "--games") {
            options.games = parse_int(require_value(arg), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--max-samples") {
            options.max_samples = parse_int(require_value(arg), arg);
        } else if (arg == "--min-gap") {
            options.min_gap = parse_int(require_value(arg), arg);
        } else if (arg == "--time-limit-seconds") {
            options.time_limit_seconds = parse_int(require_value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: nnue_spectator_gap_export [--book path] [--model path]\n"
                << "                                  [--output path] [--games N]\n"
                << "                                  [--depth N] [--max-plies N]\n"
                << "                                  [--max-samples N] [--min-gap CP]\n"
                << "                                  [--time-limit-seconds N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.games <= 0 || options.depth <= 0 || options.max_plies <= 0 || options.max_samples <= 0) {
        throw std::runtime_error("games/depth/max-plies/max-samples must be positive");
    }
    if (options.min_gap < 0 || options.time_limit_seconds <= 0 || options.progress_interval <= 0) {
        throw std::runtime_error("min-gap must be non-negative and time/progress limits must be positive");
    }
    return options;
}

std::string strip_comment(std::string line) {
    const std::size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
        line.resize(comment_pos);
    }
    return line;
}

std::vector<std::string> split_words(const std::string& line) {
    std::istringstream input(line);
    std::vector<std::string> words;
    std::string word;
    while (input >> word) {
        words.push_back(word);
    }
    return words;
}

chess::Move find_legal_uci_move(const chess::Position& pos, const std::string& uci) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    throw std::runtime_error("illegal book move: " + uci);
}

std::vector<BookLine> load_book(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open book: " + path);
    }

    std::vector<BookLine> book;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string clean = strip_comment(line);
        std::vector<std::string> moves = split_words(clean);
        if (moves.empty()) {
            continue;
        }
        if (moves.size() != 6) {
            throw std::runtime_error("book line " + std::to_string(line_number) + " must have 6 plies");
        }
        book.push_back(BookLine{line_number, clean, std::move(moves)});
    }
    if (book.empty()) {
        throw std::runtime_error("book is empty");
    }
    return book;
}

chess::Position position_after_book(const BookLine& line) {
    chess::Position pos;
    pos.set_startpos();
    for (const std::string& uci : line.moves) {
        const chess::Move move = find_legal_uci_move(pos, uci);
        pos.make_move(move);
    }
    return pos;
}

void write_sample(
    std::ostream& out,
    const chess::EncodedPosition& encoded,
    int target,
    int nnue_score,
    int gap,
    int depth,
    std::string_view kind
) {
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

    out << "],\"target\":" << target
        << ",\"kind\":\"" << kind << '"'
        << ",\"nnue\":" << nnue_score
        << ",\"gap\":" << gap
        << ",\"depth\":" << depth
        << "}\n";
}

chess::SearchResult search_iterative(chess::HeuristicSearcherV10& searcher, const chess::Position& pos, int depth) {
    return searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = depth,
        .move_time = std::chrono::milliseconds{0},
    });
}

bool time_expired(Clock::time_point start, int time_limit_seconds) {
    return Clock::now() - start >= std::chrono::seconds{time_limit_seconds};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<BookLine> book = load_book(options.book_path);

        chess::NnueValueModel model;
        if (!model.load(options.model_path)) {
            throw std::runtime_error("failed to load model: " + options.model_path);
        }

        const std::filesystem::path output_path(options.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("failed to open output: " + options.output);
        }

        std::vector<int> indices(book.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(options.seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        const Clock::time_point start = Clock::now();
        int samples = 0;
        int searched = 0;
        int games = 0;
        std::uint64_t total_nodes = 0;

        for (int game_index = 0;
             game_index < options.games
                 && samples < options.max_samples
                 && !time_expired(start, options.time_limit_seconds);
             ++game_index) {
            const BookLine& line = book[indices[static_cast<std::size_t>(game_index) % indices.size()]];
            chess::Position pos = position_after_book(line);
            std::vector<chess::HashKey> position_hashes{pos.zobrist_key};
            chess::HeuristicSearcherV10 white;
            chess::HeuristicSearcherV10 black;

            for (int ply = 0;
                 ply < options.max_plies
                     && samples < options.max_samples
                     && !time_expired(start, options.time_limit_seconds);
                 ++ply) {
                if (chess::is_threefold_repetition(pos.zobrist_key, position_hashes)) {
                    break;
                }
                if (chess::generate_legal_moves(pos).empty()) {
                    break;
                }

                chess::HeuristicSearcherV10& player =
                    pos.side_to_move == chess::Color::White ? white : black;
                const chess::SearchResult result = search_iterative(player, pos, options.depth);
                ++searched;
                total_nodes += result.nodes;

                const int nnue_score = model.evaluate_cp_rounded(pos);
                const int gap = std::abs(result.score - nnue_score);
                if (gap >= options.min_gap) {
                    write_sample(
                        out,
                        chess::encode_position(pos),
                        result.score,
                        nnue_score,
                        gap,
                        options.depth,
                        "nnue_spectator_gap"
                    );
                    ++samples;
                }

                if (options.progress_interval > 0 && searched % options.progress_interval == 0) {
                    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count();
                    std::cerr << "searched=" << searched
                              << " samples=" << samples
                              << " games=" << games
                              << " elapsed_s=" << elapsed
                              << " avg_nodes=" << (searched == 0 ? 0 : total_nodes / searched)
                              << '\n';
                }

                pos.make_move(result.best_move);
                position_hashes.push_back(pos.zobrist_key);
            }

            ++games;
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count();
        std::cerr << "wrote " << samples << " spectator-gap samples from "
                  << games << " games and " << searched << " searched positions to "
                  << options.output << " elapsed_s=" << elapsed << '\n';
    } catch (const std::exception& error) {
        std::cerr << "nnue_spectator_gap_export: " << error.what() << '\n';
        return 1;
    }
}
