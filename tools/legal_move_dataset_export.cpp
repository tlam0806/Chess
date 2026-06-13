#include "board_encoder.hpp"
#include "attacks.hpp"
#include "game_state.hpp"
#include "heuristic_searcher_v7.hpp"
#include "move.hpp"
#include "position.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

enum class SelectionKind {
    All,
    Important
};

struct Options {
    int games = 10;
    int depth = 6;
    int label_depth = 6;
    int candidate_depth = 6;
    int top_moves = 3;
    int random_options = 2;
    int random_plies = 12;
    int max_plies = 160;
    int limit = 50000;
    int progress_interval = 1;
    int threads = 1;
    std::uint32_t seed = 20260612;
    SelectionKind selection = SelectionKind::All;
    std::string output = "data/legal_moves_depth6_v7.jsonl";
};

struct WorkerResult {
    int written = 0;
};

struct MoveLabel {
    chess::Move move{};
    chess::Position child{};
    int target = 0;
    int parent_score = 0;
    bool gives_check = false;
    bool capture = false;
    bool selected = false;
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

SelectionKind parse_selection_kind(std::string_view value) {
    if (value == "all") {
        return SelectionKind::All;
    }
    if (value == "important") {
        return SelectionKind::Important;
    }
    throw std::runtime_error("invalid selection: " + std::string(value));
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
        } else if (arg == "--label-depth") {
            options.label_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--candidate-depth") {
            options.candidate_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--top-moves") {
            options.top_moves = parse_int(require_value(arg), arg);
        } else if (arg == "--random-options") {
            options.random_options = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--limit") {
            options.limit = parse_int(require_value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_int(require_value(arg), arg);
        } else if (arg == "--threads") {
            options.threads = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--selection") {
            options.selection = parse_selection_kind(require_value(arg));
        } else if (arg == "--output") {
            options.output = std::string(require_value(arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: legal_move_dataset_export [--games N] [--depth D] [--label-depth D]\n"
                << "                                 [--selection all|important]\n"
                << "                                 [--candidate-depth D]\n"
                << "                                 [--top-moves N] [--random-options N]\n"
                << "                                 [--random-plies N] [--max-plies N]\n"
                << "                                 [--limit N] [--progress-interval N]\n"
                << "                                 [--threads N] [--seed N] [--output path]\n";
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
    if (options.label_depth < 0) {
        throw std::runtime_error("--label-depth must be non-negative");
    }
    if (options.candidate_depth < 0) {
        throw std::runtime_error("--candidate-depth must be non-negative");
    }
    if (options.top_moves < 0) {
        throw std::runtime_error("--top-moves must be non-negative");
    }
    if (options.random_options < 0) {
        throw std::runtime_error("--random-options must be non-negative");
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
    if (options.threads <= 0) {
        throw std::runtime_error("--threads must be positive");
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

int write_legal_move_children(
    std::ostream& out,
    const chess::Position& pos,
    const std::vector<chess::Move>& moves,
    int label_depth,
    chess::HeuristicSearcherV7& teacher,
    int remaining
) {
    int written = 0;
    for (chess::Move move : moves) {
        if (written >= remaining) {
            break;
        }

        chess::Position child = pos;
        child.make_move(move);
        const int target = teacher.search_best_move(child, label_depth).score;
        write_sample(out, chess::encode_position(child), target);
        ++written;
    }
    return written;
}

std::vector<MoveLabel> score_legal_move_children(
    const chess::Position& pos,
    const std::vector<chess::Move>& moves,
    int candidate_depth,
    chess::HeuristicSearcherV7& teacher
) {
    std::vector<MoveLabel> labels;
    labels.reserve(moves.size());

    for (chess::Move move : moves) {
        chess::Position child = pos;
        child.make_move(move);
        const int target = teacher.search_best_move(child, candidate_depth).score;
        labels.push_back(MoveLabel{
            move,
            child,
            target,
            -target,
            chess::in_check(child, child.side_to_move),
            chess::is_capture(move),
            false
        });
    }

    return labels;
}

void mark_selected(MoveLabel& label) {
    label.selected = true;
}

int write_important_move_children(
    std::ostream& out,
    const chess::Position& pos,
    const std::vector<chess::Move>& moves,
    const Options& options,
    chess::HeuristicSearcherV7& teacher,
    std::mt19937& rng,
    int remaining
) {
    std::vector<MoveLabel> labels = score_legal_move_children(pos, moves, options.candidate_depth, teacher);

    for (MoveLabel& label : labels) {
        if (label.gives_check || label.capture) {
            mark_selected(label);
        }
    }

    std::vector<std::size_t> ranked(labels.size());
    for (std::size_t index = 0; index < labels.size(); ++index) {
        ranked[index] = index;
    }
    std::sort(ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
        return labels[lhs].parent_score > labels[rhs].parent_score;
    });

    const int top_count = std::min(options.top_moves, static_cast<int>(ranked.size()));
    for (int index = 0; index < top_count; ++index) {
        mark_selected(labels[ranked[static_cast<std::size_t>(index)]]);
    }

    std::vector<std::size_t> random_candidates;
    for (std::size_t index = 0; index < labels.size(); ++index) {
        if (!labels[index].selected) {
            random_candidates.push_back(index);
        }
    }
    std::shuffle(random_candidates.begin(), random_candidates.end(), rng);

    const int random_count = std::min(options.random_options, static_cast<int>(random_candidates.size()));
    for (int index = 0; index < random_count; ++index) {
        mark_selected(labels[random_candidates[static_cast<std::size_t>(index)]]);
    }

    int written = 0;
    for (const MoveLabel& label : labels) {
        if (!label.selected) {
            continue;
        }
        if (written >= remaining) {
            break;
        }
        const int target = teacher.search_best_move(label.child, options.label_depth).score;
        write_sample(out, chess::encode_position(label.child), target);
        ++written;
    }

    return written;
}

int write_selected_move_children(
    std::ostream& out,
    const chess::Position& pos,
    const std::vector<chess::Move>& moves,
    const Options& options,
    chess::HeuristicSearcherV7& teacher,
    std::mt19937& rng,
    int remaining
) {
    if (options.selection == SelectionKind::Important) {
        return write_important_move_children(out, pos, moves, options, teacher, rng, remaining);
    }
    return write_legal_move_children(out, pos, moves, options.label_depth, teacher, remaining);
}

int export_games(std::ostream& out, const Options& options, std::mt19937& rng, int worker_index = 0) {
    chess::HeuristicSearcherV7 teacher;
    int written = 0;

    for (int game = 0; game < options.games && written < options.limit; ++game) {
        chess::Position pos;
        pos.set_startpos();
        std::vector<chess::HashKey> position_hashes{pos.zobrist_key};

        for (int ply = 0; ply < options.max_plies && written < options.limit; ++ply) {
            if (chess::is_threefold_repetition(pos.zobrist_key, position_hashes)) {
                break;
            }

            const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
            if (moves.empty()) {
                break;
            }

            written += write_selected_move_children(
                out,
                pos,
                moves,
                options,
                teacher,
                rng,
                options.limit - written);

            chess::Move move;
            if (ply < options.random_plies) {
                move = choose_random_move(moves, rng);
            } else {
                move = teacher.search_best_move(pos, options.depth).best_move;
            }

            pos.make_move(move);
            position_hashes.push_back(pos.zobrist_key);
        }

        if ((game + 1) % options.progress_interval == 0 || game + 1 == options.games) {
            std::cerr << "worker " << worker_index
                      << " game " << (game + 1) << " / " << options.games
                      << ", legal-move samples " << written << '\n';
        }
    }

    return written;
}

std::filesystem::path make_part_path(const std::filesystem::path& output_path, int worker_index) {
    std::filesystem::path part_path = output_path;
    part_path += ".part" + std::to_string(worker_index);
    return part_path;
}

int split_count(int total, int worker_index, int worker_count) {
    const int base = total / worker_count;
    const int extra = total % worker_count;
    return base + (worker_index < extra ? 1 : 0);
}

int export_single_thread(const Options& options, const std::filesystem::path& output_path) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("failed to open output: " + output_path.string());
    }

    std::mt19937 rng(options.seed);
    return export_games(out, options, rng);
}

int export_parallel(const Options& options, const std::filesystem::path& output_path) {
    const int worker_count = std::min(options.threads, options.games);

    std::vector<std::thread> workers;
    std::vector<WorkerResult> results(static_cast<std::size_t>(worker_count));
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(worker_count));
    std::mutex progress_mutex;

    for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
        workers.emplace_back([&, worker_index] {
            try {
                Options worker_options = options;
                worker_options.games = split_count(options.games, worker_index, worker_count);
                worker_options.limit = split_count(options.limit, worker_index, worker_count);
                worker_options.seed = options.seed + static_cast<std::uint32_t>(worker_index * 0x9E37U + 1U);

                const std::filesystem::path part_path = make_part_path(output_path, worker_index);
                std::ofstream part_out(part_path);
                if (!part_out) {
                    throw std::runtime_error("failed to open part output: " + part_path.string());
                }

                std::mt19937 rng(worker_options.seed);
                results[static_cast<std::size_t>(worker_index)].written =
                    export_games(part_out, worker_options, rng, worker_index);

                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    std::cerr << "thread " << worker_index << " wrote "
                              << results[static_cast<std::size_t>(worker_index)].written
                              << " samples\n";
                }
            } catch (...) {
                errors[static_cast<std::size_t>(worker_index)] = std::current_exception();
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    for (const std::exception_ptr& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("failed to open output: " + output_path.string());
    }

    int total_written = 0;
    for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
        const std::filesystem::path part_path = make_part_path(output_path, worker_index);
        std::ifstream part_in(part_path);
        if (!part_in) {
            throw std::runtime_error("failed to open part input: " + part_path.string());
        }
        out << part_in.rdbuf();
        total_written += results[static_cast<std::size_t>(worker_index)].written;
        part_in.close();
        std::filesystem::remove(part_path);
    }

    return total_written;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        const std::filesystem::path output_path(options.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        const int written = options.threads == 1
            ? export_single_thread(options, output_path)
            : export_parallel(options, output_path);

        std::cerr << "wrote " << written << " legal-move samples to "
                  << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "legal_move_dataset_export: " << error.what() << '\n';
        return 1;
    }
}
