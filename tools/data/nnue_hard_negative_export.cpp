#include "attacks.hpp"
#include "board_encoder.hpp"
#include "game_state.hpp"
#include "heuristic_searcher_v10.hpp"
#include "move.hpp"
#include "nnue_searcher_v10.hpp"
#include "nnue_value.hpp"
#include "position.hpp"

#include <algorithm>
#include <chrono>
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

struct BookLine {
    int line_number = 0;
    std::string text;
    std::vector<std::string> moves;
};

struct Options {
    std::string book_path = "data/opening_book_6plies.txt";
    std::string model_path = "models/nnue_value_mix_666k_more_lr3e4.bin";
    std::string output = "data/nnue_hard_negative_depth8.jsonl";
    int games = 6;
    int depth = 8;
    int max_plies = 80;
    int max_samples = 2000;
    int min_score_gap = 150;
    int max_noisy_children = 4;
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
        } else if (arg == "--min-score-gap") {
            options.min_score_gap = parse_int(require_value(arg), arg);
        } else if (arg == "--max-noisy-children") {
            options.max_noisy_children = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: nnue_hard_negative_export [--book path] [--model path]\n"
                << "                                 [--output path] [--games N]\n"
                << "                                 [--depth N] [--max-plies N]\n"
                << "                                 [--max-samples N] [--min-score-gap CP]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.games <= 0 || options.depth <= 0 || options.max_plies <= 0 || options.max_samples <= 0) {
        throw std::runtime_error("games/depth/max-plies/max-samples must be positive");
    }
    if (options.min_score_gap < 0 || options.max_noisy_children < 0) {
        throw std::runtime_error("min-score-gap and max-noisy-children must be non-negative");
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

void write_sample(std::ostream& out, const chess::EncodedPosition& encoded, int target, std::string_view kind) {
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
        << ",\"kind\":\"" << kind << "\"}\n";
}

chess::SearchResult search_iterative(chess::Searcher& searcher, const chess::Position& pos, int depth) {
    return searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = depth,
        .move_time = std::chrono::milliseconds{0},
    });
}

bool is_noisy_or_check(const chess::Position& pos, chess::Move move) {
    if (chess::is_capture(move) || chess::promotion_piece(move) != chess::PieceType::None) {
        return true;
    }
    chess::Position next = pos;
    next.make_move(move);
    return chess::in_check(next, next.side_to_move);
}

int label_and_write(
    std::ostream& out,
    chess::HeuristicSearcherV10& teacher,
    const chess::Position& pos,
    int depth,
    std::string_view kind,
    int& samples,
    int max_samples
) {
    if (samples >= max_samples) {
        return 0;
    }
    const chess::SearchResult label = search_iterative(teacher, pos, depth);
    write_sample(out, chess::encode_position(pos), label.score, kind);
    ++samples;
    return label.score;
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

        int samples = 0;
        int games = 0;
        for (int game_index = 0; game_index < options.games && samples < options.max_samples; ++game_index) {
            chess::Position pos = position_after_book(book[indices[static_cast<std::size_t>(game_index) % indices.size()]]);
            std::vector<chess::HashKey> position_hashes{pos.zobrist_key};
            chess::HeuristicSearcherV10 heuristic;
            chess::HeuristicSearcherV10 teacher;
            chess::NnueSearcherV10 nnue(model);
            const bool nnue_is_white = (game_index % 2 == 0);

            for (int ply = 0; ply < options.max_plies && samples < options.max_samples; ++ply) {
                if (chess::is_threefold_repetition(pos.zobrist_key, position_hashes)) {
                    break;
                }
                const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
                if (moves.empty()) {
                    break;
                }

                const bool nnue_to_move = (pos.side_to_move == chess::Color::White) == nnue_is_white;
                chess::SearchResult played{};
                if (nnue_to_move) {
                    played = search_iterative(nnue, pos, options.depth);
                    const chess::SearchResult teacher_result = search_iterative(teacher, pos, options.depth);
                    const bool move_differs = !(played.best_move == teacher_result.best_move);
                    const int score_gap = std::abs(teacher_result.score - played.score);
                    if (move_differs || score_gap >= options.min_score_gap) {
                        write_sample(out, chess::encode_position(pos), teacher_result.score, "nnue_disagreement_parent");
                        ++samples;

                        chess::Position nnue_child = pos;
                        nnue_child.make_move(played.best_move);
                        label_and_write(out, teacher, nnue_child, options.depth, "nnue_chosen_child", samples, options.max_samples);

                        chess::Position teacher_child = pos;
                        teacher_child.make_move(teacher_result.best_move);
                        label_and_write(out, teacher, teacher_child, options.depth, "teacher_chosen_child", samples, options.max_samples);

                        int noisy_written = 0;
                        for (chess::Move move : moves) {
                            if (samples >= options.max_samples || noisy_written >= options.max_noisy_children) {
                                break;
                            }
                            if (move == played.best_move || move == teacher_result.best_move || !is_noisy_or_check(pos, move)) {
                                continue;
                            }
                            chess::Position child = pos;
                            child.make_move(move);
                            label_and_write(out, teacher, child, options.depth, "noisy_candidate_child", samples, options.max_samples);
                            ++noisy_written;
                        }
                    }
                } else {
                    played = search_iterative(heuristic, pos, options.depth);
                }

                std::cerr << "game " << (game_index + 1)
                          << " ply " << (ply + 1)
                          << " side=" << (nnue_to_move ? "nnue" : "heuristic")
                          << " move=" << chess::move_to_string(played.best_move)
                          << " score=" << played.score
                          << " samples=" << samples << '\n';

                pos.make_move(played.best_move);
                position_hashes.push_back(pos.zobrist_key);
            }
            ++games;
        }

        std::cerr << "wrote " << samples << " hard-negative samples from "
                  << games << " games to " << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "nnue_hard_negative_export: " << error.what() << '\n';
        return 1;
    }
}
