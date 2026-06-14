#include "attacks.hpp"
#include "game_state.hpp"
#include "heuristic_searcher_v10.hpp"
#include "move.hpp"
#include "nnue_searcher_v10.hpp"
#include "nnue_value.hpp"
#include "position.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
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

enum class EngineKind {
    Heuristic,
    Nnue
};

struct BookLine {
    int line_number = 0;
    std::string text;
    std::vector<std::string> moves;
};

struct Options {
    std::string book_path = "data/opening_book_6plies.txt";
    std::string model_path = "models/nnue_value_smoke.bin";
    int games = 10;
    int depth = 7;
    int max_plies = 120;
    int progress_interval = 1;
    std::uint32_t seed = 20260613;
};

struct Score {
    int nnue_wins = 0;
    int heuristic_wins = 0;
    int draws = 0;
};

struct GameSummary {
    int game_index = 0;
    int book_line = 0;
    std::string book;
    EngineKind white = EngineKind::Nnue;
    EngineKind black = EngineKind::Heuristic;
    std::string result;
    std::string reason;
    int played_plies = 0;
    std::uint64_t nnue_nodes = 0;
    std::uint64_t heuristic_nodes = 0;
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

        if (arg == "--book") {
            options.book_path = std::string(require_value(arg));
        } else if (arg == "--model") {
            options.model_path = std::string(require_value(arg));
        } else if (arg == "--games") {
            options.games = parse_int(require_value(arg), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: opening_book_match [--book path] [--model path] [--games N]\n"
                << "                          [--depth N] [--max-plies N] [--seed N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.games <= 0) {
        throw std::runtime_error("--games must be positive");
    }
    if (options.depth <= 0) {
        throw std::runtime_error("--depth must be positive");
    }
    if (options.max_plies <= 0) {
        throw std::runtime_error("--max-plies must be positive");
    }
    if (options.progress_interval <= 0) {
        throw std::runtime_error("--progress-interval must be positive");
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

bool contains_move(const std::vector<chess::Move>& moves, chess::Move target) {
    for (chess::Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

chess::Move find_legal_uci_move(const chess::Position& pos, const std::string& uci) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    throw std::runtime_error("illegal book move: " + uci);
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

        BookLine book_line{line_number, clean, std::move(moves)};
        static_cast<void>(position_after_book(book_line));
        book.push_back(std::move(book_line));
    }

    if (book.size() < 100) {
        throw std::runtime_error("book must contain at least 100 valid lines");
    }
    return book;
}

std::string engine_name(EngineKind engine) {
    return engine == EngineKind::Nnue ? "nnue" : "heuristic";
}

std::string terminal_result(const chess::Position& pos, std::string& reason) {
    if (chess::in_check(pos, pos.side_to_move)) {
        reason = "checkmate";
        return pos.side_to_move == chess::Color::White ? "0-1" : "1-0";
    }
    reason = "stalemate";
    return "1/2-1/2";
}

chess::SearchResult search_iterative(
    chess::Searcher& searcher,
    const chess::Position& pos,
    int depth
) {
    return searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = depth,
        .move_time = std::chrono::milliseconds{0},
    });
}

GameSummary play_game(
    int game_index,
    const BookLine& book_line,
    const Options& options,
    chess::HeuristicSearcherV10& heuristic,
    chess::NnueSearcherV10& nnue
) {
    GameSummary summary;
    summary.game_index = game_index;
    summary.book_line = book_line.line_number;
    summary.book = book_line.text;
    summary.white = (game_index % 2 == 0) ? EngineKind::Nnue : EngineKind::Heuristic;
    summary.black = (game_index % 2 == 0) ? EngineKind::Heuristic : EngineKind::Nnue;

    chess::Position pos = position_after_book(book_line);
    std::vector<chess::HashKey> position_hashes{pos.zobrist_key};

    for (int ply = 0; ply < options.max_plies; ++ply) {
        if (chess::is_threefold_repetition(pos.zobrist_key, position_hashes)) {
            summary.result = "1/2-1/2";
            summary.reason = "threefold repetition";
            summary.played_plies = ply;
            return summary;
        }

        const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (moves.empty()) {
            summary.result = terminal_result(pos, summary.reason);
            summary.played_plies = ply;
            return summary;
        }

        const EngineKind engine = pos.side_to_move == chess::Color::White ? summary.white : summary.black;
        chess::Searcher& searcher = engine == EngineKind::Nnue
            ? static_cast<chess::Searcher&>(nnue)
            : static_cast<chess::Searcher&>(heuristic);
        const chess::SearchResult result = search_iterative(searcher, pos, options.depth);
        if (!contains_move(moves, result.best_move)) {
            throw std::runtime_error(std::string(searcher.name()) + " returned illegal move");
        }
        std::cerr << "game " << summary.game_index
                  << " ply " << (ply + 1)
                  << " engine=" << engine_name(engine)
                  << " move=" << chess::move_to_string(result.best_move)
                  << " score=" << result.score
                  << " nodes=" << result.nodes << '\n';

        if (engine == EngineKind::Nnue) {
            summary.nnue_nodes += result.nodes;
        } else {
            summary.heuristic_nodes += result.nodes;
        }

        pos.make_move(result.best_move);
        position_hashes.push_back(pos.zobrist_key);
    }

    summary.result = "1/2-1/2";
    summary.reason = "max plies";
    summary.played_plies = options.max_plies;
    return summary;
}

void update_score(Score& score, const GameSummary& game) {
    if (game.result == "1/2-1/2") {
        ++score.draws;
        return;
    }
    const bool white_won = game.result == "1-0";
    const bool nnue_is_white = game.white == EngineKind::Nnue;
    if (white_won == nnue_is_white) {
        ++score.nnue_wins;
    } else {
        ++score.heuristic_wins;
    }
}

void print_game_summary(const GameSummary& game) {
    std::cout << "game " << game.game_index
              << " book_line=" << game.book_line
              << " white=" << engine_name(game.white)
              << " black=" << engine_name(game.black)
              << " result=" << game.result
              << " reason=" << game.reason
              << " plies=" << game.played_plies
              << " nnue_nodes=" << game.nnue_nodes
              << " heuristic_nodes=" << game.heuristic_nodes
              << " book=\"" << game.book << "\"\n";
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

        std::vector<int> indices(book.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(options.seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        const int games = std::min<int>(options.games, static_cast<int>(book.size()));
        Score score;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < games; ++i) {
            chess::HeuristicSearcherV10 heuristic;
            chess::NnueSearcherV10 nnue(model);
            const GameSummary game = play_game(i + 1, book[indices[i]], options, heuristic, nnue);
            update_score(score, game);
            print_game_summary(game);

            if ((i + 1) % options.progress_interval == 0) {
                std::cerr << "completed " << (i + 1) << "/" << games << '\n';
            }
        }

        const auto end = std::chrono::steady_clock::now();
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::cout << "summary: nnue_wins=" << score.nnue_wins
                  << " heuristic_wins=" << score.heuristic_wins
                  << " draws=" << score.draws
                  << " depth=" << options.depth
                  << " iterative=1"
                  << " elapsed=" << seconds << "s\n";
    } catch (const std::exception& error) {
        std::cerr << "opening_book_match: " << error.what() << '\n';
        return 1;
    }
}
