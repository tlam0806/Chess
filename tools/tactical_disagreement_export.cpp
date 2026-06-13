#include "attacks.hpp"
#include "board_encoder.hpp"
#include "game_state.hpp"
#include "heuristic_searcher_v7.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search_types.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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

using Clock = std::chrono::steady_clock;

struct Options {
    int shallow_depth = 3;
    int deep_depth = 7;
    int min_diff = 300;
    int random_plies = 10;
    int max_plies = 120;
    int limit = 1000000;
    int time_limit_seconds = 28800;
    int progress_interval = 5000;
    int threads = 1;
    int random_play_percent = 20;
    std::uint32_t seed = 20260612;
    std::string output = "data/tactical_disagreement_depth7_overnight.jsonl";
};

struct WorkerResult {
    int written = 0;
    int games = 0;
};

struct SearchScores {
    int shallow = 0;
    int deep = 0;
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

        if (arg == "--shallow-depth") {
            options.shallow_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--deep-depth") {
            options.deep_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--min-diff") {
            options.min_diff = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--limit") {
            options.limit = parse_int(require_value(arg), arg);
        } else if (arg == "--time-limit-seconds") {
            options.time_limit_seconds = parse_int(require_value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_int(require_value(arg), arg);
        } else if (arg == "--threads") {
            options.threads = parse_int(require_value(arg), arg);
        } else if (arg == "--random-play-percent") {
            options.random_play_percent = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--output") {
            options.output = std::string(require_value(arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: tactical_disagreement_export [--shallow-depth D] [--deep-depth D]\n"
                << "                                      [--min-diff CP]\n"
                << "                                      [--random-plies N] [--max-plies N]\n"
                << "                                      [--limit N] [--time-limit-seconds N]\n"
                << "                                      [--threads N] [--random-play-percent N]\n"
                << "                                      [--seed N] [--output path]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.shallow_depth < 0 || options.deep_depth < 0) {
        throw std::runtime_error("depths must be non-negative");
    }
    if (options.deep_depth < options.shallow_depth) {
        throw std::runtime_error("--deep-depth must be >= --shallow-depth");
    }
    if (options.min_diff < 0) {
        throw std::runtime_error("--min-diff must be non-negative");
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
    if (options.time_limit_seconds <= 0) {
        throw std::runtime_error("--time-limit-seconds must be positive");
    }
    if (options.progress_interval <= 0) {
        throw std::runtime_error("--progress-interval must be positive");
    }
    if (options.threads <= 0) {
        throw std::runtime_error("--threads must be positive");
    }
    if (options.random_play_percent < 0 || options.random_play_percent > 100) {
        throw std::runtime_error("--random-play-percent must be in [0,100]");
    }
    return options;
}

void write_sample(
    std::ostream& out,
    const chess::EncodedPosition& encoded,
    int target,
    std::string_view kind,
    int shallow,
    int deep,
    int diff
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
        << ",\"shallow\":" << shallow
        << ",\"deep\":" << deep
        << ",\"diff\":" << diff
        << "}\n";
}

bool gives_check(const chess::Position& pos, chess::Move move) {
    chess::Position next = pos;
    next.make_move(move);
    return chess::in_check(next, next.side_to_move);
}

bool is_tactical_move(const chess::Position& pos, chess::Move move) {
    return chess::is_capture(move)
        || chess::promotion_piece(move) != chess::PieceType::None
        || gives_check(pos, move);
}

chess::Move choose_random_move(const std::vector<chess::Move>& moves, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

chess::SearchResult iterative_search(chess::HeuristicSearcherV7& searcher, const chess::Position& pos, int depth) {
    searcher.clear_tt();
    return searcher.search_best_move(pos, chess::SearchLimits{
        .max_depth = depth,
        .move_time = std::chrono::milliseconds{0}
    });
}

SearchScores score_position(
    chess::HeuristicSearcherV7& shallow_teacher,
    chess::HeuristicSearcherV7& deep_teacher,
    const chess::Position& pos,
    const Options& options
) {
    const int shallow = iterative_search(shallow_teacher, pos, options.shallow_depth).score;
    const int deep = iterative_search(deep_teacher, pos, options.deep_depth).score;
    return SearchScores{shallow, deep};
}

int write_tactical_samples(
    std::ostream& out,
    chess::HeuristicSearcherV7& shallow_teacher,
    chess::HeuristicSearcherV7& deep_teacher,
    const chess::Position& pos,
    const std::vector<chess::Move>& moves,
    const Options& options,
    int remaining,
    const Clock::time_point deadline
) {
    int written = 0;

    const SearchScores parent = score_position(shallow_teacher, deep_teacher, pos, options);
    const int parent_diff = std::abs(parent.deep - parent.shallow);
    if (parent_diff >= options.min_diff && written < remaining) {
        write_sample(
            out,
            chess::encode_position(pos),
            parent.deep,
            "parent_disagreement",
            parent.shallow,
            parent.deep,
            parent_diff);
        ++written;
    }

    for (chess::Move move : moves) {
        if (written >= remaining || Clock::now() >= deadline) {
            break;
        }
        if (!is_tactical_move(pos, move)) {
            continue;
        }

        chess::Position child = pos;
        child.make_move(move);
        const SearchScores child_scores = score_position(shallow_teacher, deep_teacher, child, options);
        const int diff = std::abs(child_scores.deep - child_scores.shallow);
        const bool is_promotion = chess::promotion_piece(move) != chess::PieceType::None;
        const std::string_view kind = is_promotion
            ? "promotion_child"
            : chess::is_capture(move)
                ? "capture_child"
                : "check_child";

        if (diff >= options.min_diff || chess::is_capture(move) || is_promotion || gives_check(pos, move)) {
            write_sample(
                out,
                chess::encode_position(child),
                child_scores.deep,
                kind,
                child_scores.shallow,
                child_scores.deep,
                diff);
            ++written;
        }
    }

    return written;
}

chess::Move choose_play_move(
    chess::HeuristicSearcherV7& teacher,
    const chess::Position& pos,
    const std::vector<chess::Move>& moves,
    const Options& options,
    int ply,
    std::mt19937& rng
) {
    if (ply < options.random_plies) {
        return choose_random_move(moves, rng);
    }

    std::uniform_int_distribution<int> percent(1, 100);
    if (percent(rng) <= options.random_play_percent) {
        return choose_random_move(moves, rng);
    }

    return iterative_search(teacher, pos, options.deep_depth).best_move;
}

std::filesystem::path make_part_path(const std::filesystem::path& output_path, int worker_index) {
    std::filesystem::path part_path = output_path;
    part_path += ".part" + std::to_string(worker_index);
    return part_path;
}

WorkerResult export_worker(
    const Options& options,
    int worker_index,
    std::ofstream& out,
    const Clock::time_point deadline,
    std::atomic<int>& total_written,
    std::mutex& progress_mutex
) {
    WorkerResult result;
    std::mt19937 rng(options.seed + static_cast<std::uint32_t>(worker_index * 0x9E37U + 1U));
    chess::HeuristicSearcherV7 shallow_teacher;
    chess::HeuristicSearcherV7 deep_teacher;
    chess::HeuristicSearcherV7 play_teacher;

    while (Clock::now() < deadline && total_written.load() < options.limit) {
        chess::Position pos;
        pos.set_startpos();
        std::vector<chess::HashKey> position_hashes{pos.zobrist_key};

        for (int ply = 0; ply < options.max_plies
             && Clock::now() < deadline
             && total_written.load() < options.limit;
             ++ply) {
            if (chess::is_threefold_repetition(pos.zobrist_key, position_hashes)) {
                break;
            }

            const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
            if (moves.empty()) {
                break;
            }

            const int remaining = options.limit - total_written.load();
            const int written = write_tactical_samples(
                out,
                shallow_teacher,
                deep_teacher,
                pos,
                moves,
                options,
                remaining,
                deadline);
            if (written > 0) {
                result.written += written;
                const int total = total_written.fetch_add(written) + written;
                if (total % options.progress_interval < written) {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    std::cerr << "samples " << total
                              << " worker " << worker_index
                              << " games " << result.games
                              << '\n';
                }
            }

            const chess::Move played = choose_play_move(play_teacher, pos, moves, options, ply, rng);
            pos.make_move(played);
            position_hashes.push_back(pos.zobrist_key);
        }

        ++result.games;
    }

    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::filesystem::path output_path(options.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        const Clock::time_point deadline = Clock::now() + std::chrono::seconds(options.time_limit_seconds);
        const int worker_count = options.threads;
        std::vector<std::thread> workers;
        std::vector<WorkerResult> results(static_cast<std::size_t>(worker_count));
        std::vector<std::exception_ptr> errors(static_cast<std::size_t>(worker_count));
        std::atomic<int> total_written{0};
        std::mutex progress_mutex;

        for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
            workers.emplace_back([&, worker_index] {
                try {
                    const std::filesystem::path part_path = make_part_path(output_path, worker_index);
                    std::ofstream part_out(part_path);
                    if (!part_out) {
                        throw std::runtime_error("failed to open part output: " + part_path.string());
                    }
                    results[static_cast<std::size_t>(worker_index)] = export_worker(
                        options,
                        worker_index,
                        part_out,
                        deadline,
                        total_written,
                        progress_mutex);
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

        int combined = 0;
        int games = 0;
        for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
            const std::filesystem::path part_path = make_part_path(output_path, worker_index);
            std::ifstream part_in(part_path);
            if (!part_in) {
                continue;
            }
            out << part_in.rdbuf();
            combined += results[static_cast<std::size_t>(worker_index)].written;
            games += results[static_cast<std::size_t>(worker_index)].games;
            part_in.close();
            std::filesystem::remove(part_path);
        }

        std::cerr << "wrote " << combined
                  << " tactical disagreement samples across " << games
                  << " games to " << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "tactical_disagreement_export: " << error.what() << '\n';
        return 1;
    }
}
