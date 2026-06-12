#include "attacks.hpp"
#include "heuristic_searcher.hpp"
#include "heuristic_searcher_v2.hpp"
#include "heuristic_searcher_v3.hpp"
#include "heuristic_searcher_v4.hpp"
#include "heuristic_searcher_v5.hpp"
#include "heuristic_searcher_v6.hpp"
#include "heuristic_searcher_v7.hpp"
#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

enum class SideResult {
    EngineAWin,
    EngineBWin,
    Draw
};

enum class EngineKind {
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7
};

struct Options {
    int games = 500;
    int max_depth = 4;
    int movetime_ms = 1000;
    int random_plies = 8;
    int max_plies = 160;
    int progress_interval = 10;
    int threads = 0;
    std::uint32_t seed = 1;
    EngineKind engine_a = EngineKind::V2;
    EngineKind engine_b = EngineKind::V1;
};

struct Score {
    int wins = 0;
    int losses = 0;
    int draws = 0;
};

struct SearchTiming {
    std::uint64_t moves = 0;
    std::uint64_t time_us = 0;
};

struct MatchTotals {
    Score engine_a_score;
    std::uint64_t total_plies = 0;
    SearchTiming engine_a_timing;
    SearchTiming engine_b_timing;
};

std::string_view engine_name(EngineKind engine) {
    switch (engine) {
        case EngineKind::V1:
            return "v1";
        case EngineKind::V2:
            return "v2";
        case EngineKind::V3:
            return "v3";
        case EngineKind::V4:
            return "v4";
        case EngineKind::V5:
            return "v5";
        case EngineKind::V6:
            return "v6";
        case EngineKind::V7:
            return "v7";
    }
    return "unknown";
}

EngineKind parse_engine_kind(std::string_view value) {
    if (value == "v1") {
        return EngineKind::V1;
    }
    if (value == "v2") {
        return EngineKind::V2;
    }
    if (value == "v3") {
        return EngineKind::V3;
    }
    if (value == "v4") {
        return EngineKind::V4;
    }
    if (value == "v5") {
        return EngineKind::V5;
    }
    if (value == "v6") {
        return EngineKind::V6;
    }
    if (value == "v7") {
        return EngineKind::V7;
    }
    throw std::runtime_error("invalid engine kind: " + std::string(value));
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

        if (arg == "--games") {
            options.games = parse_int(require_value(arg), arg);
        } else if (arg == "--max-depth") {
            options.max_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--movetime-ms") {
            options.movetime_ms = parse_int(require_value(arg), arg);
        } else if (arg == "--random-plies") {
            options.random_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--progress-interval") {
            options.progress_interval = parse_int(require_value(arg), arg);
        } else if (arg == "--threads") {
            options.threads = parse_int(require_value(arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(require_value(arg), arg));
        } else if (arg == "--engine-a") {
            options.engine_a = parse_engine_kind(require_value(arg));
        } else if (arg == "--engine-b") {
            options.engine_b = parse_engine_kind(require_value(arg));
        } else if (arg == "--help") {
            std::cout
                << "Usage: match_searchers [--games N] [--max-depth D] [--movetime-ms MS]\n"
                << "                       [--random-plies N] [--max-plies N]\n"
                << "                       [--progress-interval N] [--threads N] [--seed N]\n"
                << "                       [--engine-a v1|v2|v3|v4|v5|v6|v7] [--engine-b v1|v2|v3|v4|v5|v6|v7]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.games <= 0) {
        throw std::runtime_error("--games must be positive");
    }
    if (options.max_depth < 0) {
        throw std::runtime_error("--max-depth must be non-negative");
    }
    if (options.movetime_ms < 0) {
        throw std::runtime_error("--movetime-ms must be non-negative");
    }
    if (options.random_plies < 0) {
        throw std::runtime_error("--random-plies must be non-negative");
    }
    if (options.max_plies <= 0) {
        throw std::runtime_error("--max-plies must be positive");
    }
    if (options.progress_interval <= 0) {
        throw std::runtime_error("--progress-interval must be positive");
    }
    if (options.threads < 0) {
        throw std::runtime_error("--threads must be non-negative");
    }

    return options;
}

chess::Searcher& select_searcher(
    EngineKind engine,
    chess::HeuristicSearcher& v1,
    chess::HeuristicSearcherV2& v2,
    chess::HeuristicSearcherV3& v3,
    chess::HeuristicSearcherV4& v4,
    chess::HeuristicSearcherV5& v5,
    chess::HeuristicSearcherV6& v6,
    chess::HeuristicSearcherV7& v7
) {
    switch (engine) {
        case EngineKind::V1:
            return v1;
        case EngineKind::V2:
            return v2;
        case EngineKind::V3:
            return v3;
        case EngineKind::V4:
            return v4;
        case EngineKind::V5:
            return v5;
        case EngineKind::V6:
            return v6;
        case EngineKind::V7:
            return v7;
    }
    return v1;
}

bool contains_move(const std::vector<chess::Move>& moves, chess::Move target) {
    for (chess::Move move : moves) {
        if (move == target) {
            return true;
        }
    }
    return false;
}

chess::Move random_move(const std::vector<chess::Move>& moves, std::mt19937& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

chess::Move choose_move(
    chess::Searcher& searcher,
    const chess::Position& pos,
    const chess::SearchLimits& limits
) {
    const chess::SearchResult result = searcher.search_best_move(pos, limits);
    const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
    if (!contains_move(moves, result.best_move)) {
        throw std::runtime_error(std::string(searcher.name()) + " returned illegal move");
    }
    return result.best_move;
}

SideResult play_game(
    int game_index,
    const Options& options,
    std::uint64_t& total_plies,
    SearchTiming& engine_a_timing,
    SearchTiming& engine_b_timing
) {
    std::mt19937 rng(options.seed + static_cast<std::uint32_t>(game_index) * 9973U);

    chess::Position pos;
    pos.set_startpos();

    chess::HeuristicSearcher v1;
    chess::HeuristicSearcherV2 v2;
    chess::HeuristicSearcherV3 v3;
    chess::HeuristicSearcherV4 v4;
    chess::HeuristicSearcherV5 v5;
    chess::HeuristicSearcherV6 v6;
    chess::HeuristicSearcherV7 v7;

    const bool engine_a_is_white = (game_index % 2 == 0);
    const chess::SearchLimits limits{
        .max_depth = options.max_depth,
        .move_time = std::chrono::milliseconds{options.movetime_ms}
    };

    for (int ply = 0; ply < options.max_plies; ++ply) {
        std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (moves.empty()) {
            if (!chess::in_check(pos, pos.side_to_move)) {
                return SideResult::Draw;
            }
            const bool white_won = pos.side_to_move == chess::Color::Black;
            const bool engine_a_won = white_won == engine_a_is_white;
            return engine_a_won ? SideResult::EngineAWin : SideResult::EngineBWin;
        }

        chess::Move move{};
        if (ply < options.random_plies) {
            move = random_move(moves, rng);
        } else {
            const bool engine_a_to_move = (pos.side_to_move == chess::Color::White) == engine_a_is_white;
            chess::Searcher& searcher = engine_a_to_move
                ? select_searcher(options.engine_a, v1, v2, v3, v4, v5, v6, v7)
                : select_searcher(options.engine_b, v1, v2, v3, v4, v5, v6, v7);
            const auto search_start = std::chrono::steady_clock::now();
            move = choose_move(searcher, pos, limits);
            const auto search_end = std::chrono::steady_clock::now();
            const auto search_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count());

            SearchTiming& timing = engine_a_to_move ? engine_a_timing : engine_b_timing;
            ++timing.moves;
            timing.time_us += search_us;
        }

        pos.make_move(move);
        ++total_plies;
    }

    return SideResult::Draw;
}

void update_score(Score& engine_a_score, SideResult result) {
    if (result == SideResult::EngineAWin) {
        ++engine_a_score.wins;
    } else if (result == SideResult::EngineBWin) {
        ++engine_a_score.losses;
    } else {
        ++engine_a_score.draws;
    }
}

double score_points(const Score& score) {
    return score.wins + 0.5 * score.draws;
}

int default_thread_count() {
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) {
        return 1;
    }
    return static_cast<int>(hardware_threads);
}

void merge_score(Score& dst, const Score& src) {
    dst.wins += src.wins;
    dst.losses += src.losses;
    dst.draws += src.draws;
}

void merge_timing(SearchTiming& dst, const SearchTiming& src) {
    dst.moves += src.moves;
    dst.time_us += src.time_us;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        if (options.threads == 0) {
            options.threads = default_thread_count();
        }
        options.threads = std::min(options.threads, options.games);

        MatchTotals totals;
        int completed_games = 0;
        int next_game = 0;
        std::mutex mutex;
        std::exception_ptr worker_error;

        const auto start = std::chrono::steady_clock::now();

        auto worker = [&]() {
            while (true) {
                int game = 0;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (worker_error != nullptr || next_game >= options.games) {
                        return;
                    }
                    game = next_game++;
                }

                try {
                    std::uint64_t game_plies = 0;
                    SearchTiming engine_a_timing;
                    SearchTiming engine_b_timing;
                    const SideResult result = play_game(
                        game,
                        options,
                        game_plies,
                        engine_a_timing,
                        engine_b_timing);

                    std::lock_guard<std::mutex> lock(mutex);
                    update_score(totals.engine_a_score, result);
                    totals.total_plies += game_plies;
                    merge_timing(totals.engine_a_timing, engine_a_timing);
                    merge_timing(totals.engine_b_timing, engine_b_timing);
                    ++completed_games;

                    if (completed_games % options.progress_interval == 0
                        || completed_games == options.games) {
                        const auto now = std::chrono::steady_clock::now();
                        const auto elapsed_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                        std::cerr << "games " << completed_games << " / " << options.games
                                  << " | " << engine_name(options.engine_a) << " W-L-D "
                                  << totals.engine_a_score.wins << '-'
                                  << totals.engine_a_score.losses << '-' << totals.engine_a_score.draws
                                  << " | plies " << totals.total_plies
                                  << " | threads " << options.threads
                                  << " | elapsed_ms " << elapsed_ms << '\n';
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (worker_error == nullptr) {
                        worker_error = std::current_exception();
                    }
                    return;
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(options.threads);
        for (int i = 0; i < options.threads; ++i) {
            workers.emplace_back(worker);
        }
        for (std::thread& thread : workers) {
            thread.join();
        }

        if (worker_error != nullptr) {
            std::rethrow_exception(worker_error);
        }

        const double points = score_points(totals.engine_a_score);
        const auto avg_a_us = totals.engine_a_timing.moves == 0
            ? 0
            : totals.engine_a_timing.time_us / totals.engine_a_timing.moves;
        const auto avg_b_us = totals.engine_b_timing.moves == 0
            ? 0
            : totals.engine_b_timing.time_us / totals.engine_b_timing.moves;
        std::cout << engine_name(options.engine_a) << "_vs_" << engine_name(options.engine_b)
                  << " games=" << options.games
                  << " W-L-D=" << totals.engine_a_score.wins << '-'
                  << totals.engine_a_score.losses << '-'
                  << totals.engine_a_score.draws
                  << " score=" << points << '/' << options.games
                  << " score_rate=" << (points / options.games)
                  << " total_plies=" << totals.total_plies
                  << " " << engine_name(options.engine_a) << "_search_moves=" << totals.engine_a_timing.moves
                  << " " << engine_name(options.engine_a) << "_search_ms=" << (totals.engine_a_timing.time_us / 1000)
                  << " " << engine_name(options.engine_a) << "_avg_us=" << avg_a_us
                  << " " << engine_name(options.engine_b) << "_search_moves=" << totals.engine_b_timing.moves
                  << " " << engine_name(options.engine_b) << "_search_ms=" << (totals.engine_b_timing.time_us / 1000)
                  << " " << engine_name(options.engine_b) << "_avg_us=" << avg_b_us
                  << " threads=" << options.threads << '\n';
    } catch (const std::exception& error) {
        std::cerr << "match_searchers: " << error.what() << '\n';
        return 1;
    }
}
