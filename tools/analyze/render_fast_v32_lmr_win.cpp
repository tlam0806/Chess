#include "attacks.hpp"
#include "game_state.hpp"
#include "heuristic_searcher_fast_v32.hpp"
#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string book_path = "data/opening_book_6plies.txt";
    std::string output_dir = "data/matches_lmr_variant";
    std::string output_name = "lmr_v32_b05_d26_md4_mi4_win.txt";
    int max_games = 200;
    int movetime_ms = 100;
    int max_depth = 64;
    int max_plies_after_book = 120;
    int tt_mb = 24;
    std::size_t book_start = 120;
};

struct BookLine {
    int line_number = 0;
    std::string text;
    std::vector<std::string> moves;
};

struct PlyRecord {
    int ply = 0;
    chess::Color side = chess::Color::White;
    std::string engine;
    std::string move;
    int score = 0;
    std::uint64_t nodes = 0;
    int depth = 0;
    bool stopped = false;
};

struct GameRecord {
    std::string result;
    std::string reason;
    std::string white;
    std::string black;
    int depth = 0;
    int book_line_number = 0;
    std::string book_line;
    std::vector<PlyRecord> plies;
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
        } else if (arg == "--output-dir") {
            options.output_dir = std::string(require_value(arg));
        } else if (arg == "--output-name") {
            options.output_name = std::string(require_value(arg));
        } else if (arg == "--max-games") {
            options.max_games = parse_int(require_value(arg), arg);
        } else if (arg == "--movetime-ms") {
            options.movetime_ms = parse_int(require_value(arg), arg);
        } else if (arg == "--max-depth") {
            options.max_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies-after-book") {
            options.max_plies_after_book = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-mb") {
            options.tt_mb = parse_int(require_value(arg), arg);
        } else if (arg == "--book-start") {
            options.book_start = static_cast<std::size_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: render_fast_v32_lmr_win [--book path] [--output-dir path]\n"
                << "                               [--output-name file.txt] [--max-games N]\n"
                << "                               [--movetime-ms N] [--max-depth N]\n"
                << "                               [--max-plies-after-book N] [--tt-mb N]\n"
                << "                               [--book-start N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.max_games <= 0 || options.movetime_ms < 0 || options.max_depth <= 0
        || options.max_plies_after_book <= 0 || options.tt_mb <= 0) {
        throw std::runtime_error("invalid non-positive option");
    }
    return options;
}

std::string strip_comment(std::string_view line) {
    const std::size_t hash = line.find('#');
    std::string clean(line.substr(0, hash == std::string_view::npos ? line.size() : hash));
    while (!clean.empty() && (clean.back() == ' ' || clean.back() == '\t' || clean.back() == '\r')) {
        clean.pop_back();
    }
    return clean;
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
    chess::MoveList moves;
    chess::generate_legal_moves(pos, moves);
    for (chess::Move move : moves) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    throw std::runtime_error("illegal move: " + uci);
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

std::string color_name(chess::Color color) {
    return color == chess::Color::White ? "white" : "black";
}

bool contains_move(const chess::MoveList& moves, chess::Move target) {
    for (chess::Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

std::string terminal_result(const chess::Position& pos, std::string& reason) {
    if (chess::in_check(pos, pos.side_to_move)) {
        reason = "checkmate";
        return pos.side_to_move == chess::Color::White ? "0-1" : "1-0";
    }
    reason = "stalemate";
    return "1/2-1/2";
}

chess::SearchResult choose_move(
    chess::HeuristicSearcherFastV32& searcher,
    const chess::Position& pos,
    const Options& options
) {
    return searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = options.max_depth,
        .move_time = std::chrono::milliseconds{options.movetime_ms},
    });
}

bool candidate_won(const GameRecord& game, bool candidate_is_white) {
    return (candidate_is_white && game.result == "1-0")
        || (!candidate_is_white && game.result == "0-1");
}

GameRecord play_game(
    const BookLine& line,
    bool candidate_is_white,
    const Options& options
) {
    const std::string candidate_name = "lmr_v32_b05_d26_md4_mi4";
    const std::string baseline_name = "baseline_v32";

    GameRecord game;
    game.depth = options.max_depth;
    game.white = candidate_is_white ? candidate_name : baseline_name;
    game.black = candidate_is_white ? baseline_name : candidate_name;
    game.book_line_number = line.line_number;
    game.book_line = line.text;

    chess::Position pos;
    pos.set_startpos();
    std::vector<chess::HashKey> position_hashes{pos.zobrist_key};

    int replay_ply = 1;
    for (const std::string& uci : line.moves) {
        const chess::Move move = find_legal_uci_move(pos, uci);
        game.plies.push_back(PlyRecord{
            replay_ply++,
            pos.side_to_move,
            "book",
            chess::move_to_string(move),
            0,
            0,
            options.max_depth,
            false
        });
        pos.make_move(move);
        position_hashes.push_back(pos.zobrist_key);
    }

    chess::HeuristicSearcherFastV32 candidate(static_cast<std::size_t>(options.tt_mb));
    chess::HeuristicSearcherFastV32 baseline(static_cast<std::size_t>(options.tt_mb));
    candidate.set_lmr_config(chess::HeuristicSearcherFastV32::LmrConfig{
        .enabled = true,
        .base = 0.5,
        .divisor = 2.6,
        .min_depth = 4,
        .min_move_index = 4,
    });

    for (int ply = 0; ply < options.max_plies_after_book; ++ply) {
        if (chess::is_threefold_repetition(pos.zobrist_key, position_hashes)) {
            game.result = "1/2-1/2";
            game.reason = "threefold repetition";
            return game;
        }

        chess::MoveList legal_moves;
        chess::generate_legal_moves(pos, legal_moves);
        if (legal_moves.empty()) {
            game.result = terminal_result(pos, game.reason);
            return game;
        }

        const bool candidate_to_move =
            (pos.side_to_move == chess::Color::White) == candidate_is_white;
        chess::HeuristicSearcherFastV32& searcher = candidate_to_move ? candidate : baseline;
        const chess::SearchResult result = choose_move(searcher, pos, options);
        if (!contains_move(legal_moves, result.best_move)) {
            throw std::runtime_error(std::string(candidate_to_move ? "candidate" : "baseline")
                + " returned illegal move " + chess::move_to_string(result.best_move));
        }

        game.plies.push_back(PlyRecord{
            replay_ply++,
            pos.side_to_move,
            candidate_to_move ? candidate_name : baseline_name,
            chess::move_to_string(result.best_move),
            result.score,
            result.nodes,
            result.depth,
            result.stopped
        });
        pos.make_move(result.best_move);
        position_hashes.push_back(pos.zobrist_key);
    }

    game.result = "1/2-1/2";
    game.reason = "max plies";
    return game;
}

void write_replay(const GameRecord& game, const Options& options) {
    std::filesystem::create_directories(options.output_dir);
    const std::filesystem::path path = std::filesystem::path(options.output_dir) / options.output_name;
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write " + path.string());
    }

    out << "depth " << game.depth << '\n';
    out << "white " << game.white << '\n';
    out << "black " << game.black << '\n';
    out << "result " << game.result << '\n';
    out << "reason " << game.reason << '\n';
    out << "book_line " << game.book_line_number << ": " << game.book_line << "\n\n";

    for (const PlyRecord& ply : game.plies) {
        out << ply.ply << ' '
            << color_name(ply.side) << ' '
            << ply.engine << ' '
            << ply.move << " score " << ply.score
            << " nodes " << ply.nodes
            << " depth " << ply.depth
            << " stopped " << (ply.stopped ? 1 : 0)
            << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<BookLine> book = load_book(options.book_path);
        const Clock::time_point start = Clock::now();

        for (int game_index = 0; game_index < options.max_games; ++game_index) {
            const BookLine& line =
                book[(options.book_start + static_cast<std::size_t>(game_index)) % book.size()];
            const bool candidate_is_white = game_index % 2 == 0;
            const GameRecord game = play_game(line, candidate_is_white, options);
            std::cout << "game=" << game_index
                      << " book_line=" << line.line_number
                      << " candidate_color=" << (candidate_is_white ? "white" : "black")
                      << " result=" << game.result
                      << " reason=" << game.reason
                      << '\n'
                      << std::flush;
            if (candidate_won(game, candidate_is_white)) {
                write_replay(game, options);
                const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    Clock::now() - start).count();
                std::cout << "wrote " << (std::filesystem::path(options.output_dir) / options.output_name)
                          << " elapsed=" << elapsed << "s\n";
                return 0;
            }
        }

        std::cerr << "no candidate win found in " << options.max_games << " games\n";
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
