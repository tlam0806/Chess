#include "board_encoder.hpp"
#include "heuristic_searcher.hpp"
#include "heuristic_searcher_v2.hpp"
#include "heuristic_searcher_v3.hpp"
#include "heuristic_searcher_v4.hpp"
#include "heuristic_searcher_v5.hpp"
#include "heuristic_searcher_v6.hpp"
#include "move.hpp"
#include "position.hpp"

#include <cstdint>
#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

enum class TeacherKind {
    V1,
    V2,
    V3,
    V4,
    V5,
    V6
};

struct Options {
    int positions = 1000;
    int games = 0;
    int depth = 2;
    int random_plies = 8;
    int max_plies_per_game = 160;
    int progress_interval = 100;
    int threads = 1;
    std::uint32_t seed = 1;
    TeacherKind teacher = TeacherKind::V1;
    std::string output = "data/value_train.jsonl";
};

struct WorkerResult {
    int written = 0;
};

TeacherKind parse_teacher_kind(std::string_view value) {
    if (value == "v1") {
        return TeacherKind::V1;
    }
    if (value == "v2") {
        return TeacherKind::V2;
    }
    if (value == "v3") {
        return TeacherKind::V3;
    }
    if (value == "v4") {
        return TeacherKind::V4;
    }
    if (value == "v5") {
        return TeacherKind::V5;
    }
    if (value == "v6") {
        return TeacherKind::V6;
    }
    throw std::runtime_error("invalid teacher: " + std::string(value));
}

std::unique_ptr<chess::Searcher> make_teacher(TeacherKind teacher) {
    switch (teacher) {
        case TeacherKind::V1:
            return std::make_unique<chess::HeuristicSearcher>();
        case TeacherKind::V2:
            return std::make_unique<chess::HeuristicSearcherV2>();
        case TeacherKind::V3:
            return std::make_unique<chess::HeuristicSearcherV3>();
        case TeacherKind::V4:
            return std::make_unique<chess::HeuristicSearcherV4>();
        case TeacherKind::V5:
            return std::make_unique<chess::HeuristicSearcherV5>();
        case TeacherKind::V6:
            return std::make_unique<chess::HeuristicSearcherV6>();
    }
    return std::make_unique<chess::HeuristicSearcher>();
}

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
        } else if (arg == "--threads") {
            options.threads = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--teacher") {
            options.teacher = parse_teacher_kind(require_value(arg));
        } else if (arg == "--output") {
            options.output = std::string(require_value(arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: dataset_export [--positions N] [--games N] [--depth D]\n"
                << "                      [--random-plies N] [--max-plies N]\n"
                << "                      [--progress-interval N] [--threads N]\n"
                << "                      [--seed N] [--teacher v1|v2|v3|v4|v5|v6]\n"
                << "                      [--output path]\n";
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

void reset_game(chess::Position& pos, int& ply) {
    pos.set_startpos();
    ply = 0;
}

chess::Move choose_random_move(const std::vector<chess::Move>& moves, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

int export_position_sample(
    std::ostream& out,
    const chess::Position& pos,
    int depth,
    chess::Searcher& searcher
) {
    const int target = searcher.search_best_move(pos, depth).score;
    write_sample(out, chess::encode_position(pos), target);
    return target;
}

chess::SearchResult export_search_sample(
    std::ostream& out,
    const chess::Position& pos,
    int depth,
    chess::Searcher& searcher
) {
    const chess::SearchResult result = searcher.search_best_move(pos, depth);
    write_sample(out, chess::encode_position(pos), result.score);
    return result;
}

int export_games(std::ostream& out, const Options& options, std::mt19937& rng) {
    int written = 0;
    std::unique_ptr<chess::Searcher> searcher = make_teacher(options.teacher);

    for (int game = 0; game < options.games; ++game) {
        chess::Position pos;
        pos.set_startpos();

        for (int ply = 0; ply < options.max_plies_per_game; ++ply) {
            const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
            if (moves.empty()) {
                break;
            }

            const chess::SearchResult search_result = export_search_sample(out, pos, options.depth, *searcher);
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
    std::unique_ptr<chess::Searcher> searcher = make_teacher(options.teacher);
    int ply = 0;
    reset_game(pos, ply);

    for (int written = 0; written < options.positions;) {
        std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (moves.empty() || ply >= options.max_plies_per_game) {
            reset_game(pos, ply);
            continue;
        }

        export_position_sample(out, pos, options.depth, *searcher);
        ++written;

        pos.make_move(choose_random_move(moves, rng));
        ++ply;

        if (written % 1000 == 0) {
            std::cerr << "written " << written << " / " << options.positions << '\n';
        }
    }

    return options.positions;
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

int export_single_thread(
    const Options& options,
    const std::filesystem::path& output_path
) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("failed to open output: " + output_path.string());
    }

    std::mt19937 rng(options.seed);
    return options.games > 0
        ? export_games(out, options, rng)
        : export_positions(out, options, rng);
}

int export_parallel(
    const Options& options,
    const std::filesystem::path& output_path
) {
    const int total_units = options.games > 0 ? options.games : options.positions;
    const int worker_count = std::min(options.threads, total_units);

    std::vector<std::thread> workers;
    std::vector<WorkerResult> results(static_cast<std::size_t>(worker_count));
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(worker_count));
    std::mutex progress_mutex;

    for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
        workers.emplace_back([&, worker_index] {
            try {
                Options worker_options = options;
                worker_options.seed = options.seed + static_cast<std::uint32_t>(worker_index * 0x9E37U + 1U);
                if (options.games > 0) {
                    worker_options.games = split_count(options.games, worker_index, worker_count);
                } else {
                    worker_options.positions = split_count(options.positions, worker_index, worker_count);
                }

                const std::filesystem::path part_path = make_part_path(output_path, worker_index);
                std::ofstream part_out(part_path);
                if (!part_out) {
                    throw std::runtime_error("failed to open part output: " + part_path.string());
                }

                std::mt19937 rng(worker_options.seed);
                results[static_cast<std::size_t>(worker_index)].written = options.games > 0
                    ? export_games(part_out, worker_options, rng)
                    : export_positions(part_out, worker_options, rng);

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

        std::cerr << "wrote " << written << " samples to " << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "dataset_export: " << error.what() << '\n';
        return 1;
    }
}
