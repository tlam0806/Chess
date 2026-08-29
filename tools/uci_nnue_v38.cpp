#include "attacks.hpp"
#include "game_state.hpp"
#include "move.hpp"
#if defined(CHESS_UCI_NNUE_V41)
#include "nnue_searcher_v41.hpp"
#elif defined(CHESS_UCI_NNUE_V40)
#include "nnue_searcher_v40.hpp"
#else
#include "nnue_searcher_v38.hpp"
#endif
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int DefaultDepth = 8;

#if defined(CHESS_UCI_NNUE_V41)
using UciSearcher = chess::NnueSearcherV41;
constexpr const char* EngineName = "ChessNNUEV41";
#elif defined(CHESS_UCI_NNUE_V40)
using UciSearcher = chess::NnueSearcherV40;
constexpr const char* EngineName = "ChessNNUEV40";
#else
using UciSearcher = chess::NnueSearcherV38;
constexpr const char* EngineName = "ChessNNUEV38";
#endif

struct AdapterOptions {
    bool avoid_draw = true;
    bool ponder = true;
    int avoid_draw_min_cp = 120;
    int avoid_draw_max_loss_cp = 80;
    int move_overhead_ms = 200;
};

struct ParsedGo {
    chess::SearchLimits limits{};
    std::chrono::milliseconds budget{0};
    bool ponder = false;
    bool infinite = false;
};

std::uint8_t castling_rights(const chess::Position& pos) {
    return static_cast<std::uint8_t>(
        (pos.white_can_castle_kingside ? 1 : 0)
        | (pos.white_can_castle_queenside ? 2 : 0)
        | (pos.black_can_castle_kingside ? 4 : 0)
        | (pos.black_can_castle_queenside ? 8 : 0));
}

bool apply_uci_move(
    chess::Position& pos,
    const std::string& uci,
    bool& irreversible
) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == uci) {
            const std::uint8_t rights_before = castling_rights(pos);
            pos.make_move(move);
            irreversible = pos.halfmove_clock == 0
                || castling_rights(pos) != rights_before;
            return true;
        }
    }
    return false;
}

void set_position(
    chess::Position& pos,
    std::vector<chess::HashKey>& history,
    UciSearcher& searcher,
    std::istringstream& input
) {
    std::string token;
    input >> token;
    if (token == "startpos") {
        pos.set_startpos();
        history = {pos.zobrist_key};
        if (input >> token && token != "moves") {
            return;
        }
    } else if (token == "fen") {
        std::vector<std::string> parts;
        while (input >> token && token != "moves") {
            parts.push_back(token);
        }
        std::string fen;
        for (const std::string& part : parts) {
            if (!fen.empty()) {
                fen += ' ';
            }
            fen += part;
        }
        if (!pos.set_fen(fen)) {
            pos.set_startpos();
        }
        history = {pos.zobrist_key};
    } else {
        return;
    }

#if !defined(CHESS_UCI_NNUE_V41)
    searcher.clear_tt();
#endif
    if (token == "moves") {
        while (input >> token) {
            bool irreversible = false;
            if (!apply_uci_move(pos, token, irreversible)) {
                break;
            }
            if (irreversible) {
                history.clear();
            }
            history.push_back(pos.zobrist_key);
        }
    }
}

ParsedGo parse_go_limits(
    std::istringstream& input,
    chess::Color side,
    const AdapterOptions& options
) {
    ParsedGo parsed;
    parsed.limits.max_depth = DefaultDepth;
    int wtime = -1;
    int btime = -1;
    int winc = 0;
    int binc = 0;
    int moves_to_go = 0;
    bool explicit_limit = false;
    std::string token;
    while (input >> token) {
        int value = 0;
        if (token == "ponder") {
            parsed.ponder = true;
        } else if (token == "infinite") {
            parsed.infinite = true;
            parsed.limits.max_depth = 64;
            explicit_limit = true;
        } else if (token == "depth" && input >> value) {
            parsed.limits.max_depth = std::max(1, value);
            explicit_limit = true;
        } else if (token == "movetime" && input >> value) {
            parsed.limits.max_depth = 64;
            parsed.budget = std::chrono::milliseconds{std::max(1, value)};
            explicit_limit = true;
        } else if (token == "wtime" && input >> value) {
            wtime = value;
        } else if (token == "btime" && input >> value) {
            btime = value;
        } else if (token == "winc" && input >> value) {
            winc = value;
        } else if (token == "binc" && input >> value) {
            binc = value;
        } else if (token == "movestogo" && input >> value) {
            moves_to_go = std::max(1, value);
        }
    }
    if (!explicit_limit) {
        const int remaining = side == chess::Color::White ? wtime : btime;
        const int increment = side == chess::Color::White ? winc : binc;
        if (remaining > 0) {
            const int reserve = std::min(
                remaining / 2,
                std::max(options.move_overhead_ms, remaining / 50));
            const int usable = std::max(1, remaining - reserve);
            const int horizon = moves_to_go > 0 ? moves_to_go : 20;
            int budget = usable / horizon + increment / 2;
            budget = std::max(20, budget);
            budget = std::min(budget, std::max(1, usable / 5));
            parsed.limits.max_depth = 64;
            parsed.budget = std::chrono::milliseconds{budget};
        }
    }
    return parsed;
}

bool move_causes_draw(
    const chess::Position& child,
    const std::vector<chess::HashKey>& history
) {
    if (chess::repetition_count(child.zobrist_key, history) >= 2
        || child.halfmove_clock >= 100) {
        return true;
    }
    const std::vector<chess::Move> replies = chess::generate_legal_moves(child);
    return replies.empty() && !chess::in_check(child, child.side_to_move);
}

chess::Move choose_non_drawing_alternative(
    const chess::Position& pos,
    const std::vector<chess::HashKey>& history,
    const chess::PhaseQuantizedNnueModel& model,
    const chess::SearchResult& result,
    const AdapterOptions& options
) {
    if (!options.avoid_draw || result.score < options.avoid_draw_min_cp) {
        return result.best_move;
    }

    chess::Position selected = pos;
    selected.make_move(result.best_move);
    if (!move_causes_draw(selected, history)) {
        return result.best_move;
    }

    const int selected_static_cp = -model.evaluate_cp_rounded(selected);
    chess::Move alternative{};
    int alternative_cp = -chess::Infinity;
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (move == result.best_move) {
            continue;
        }
        chess::Position child = pos;
        child.make_move(move);
        if (move_causes_draw(child, history)) {
            continue;
        }
        const int cp = -model.evaluate_cp_rounded(child);
        if (cp >= selected_static_cp - options.avoid_draw_max_loss_cp
            && cp > alternative_cp) {
            alternative = move;
            alternative_cp = cp;
        }
    }
    return alternative.value != 0 ? alternative : result.best_move;
}

void set_option(
    std::istringstream& input,
    AdapterOptions& adapter,
    UciSearcher& searcher
) {
    std::string token;
    input >> token;
    if (token != "name") {
        return;
    }
    std::string name;
    while (input >> token && token != "value") {
        if (!name.empty()) name += ' ';
        name += token;
    }
    std::string value;
    std::getline(input >> std::ws, value);
    auto config = searcher.selective_config();
    try {
        if (name == "AvoidDraw") adapter.avoid_draw = value == "true";
        else if (name == "Ponder") adapter.ponder = value == "true";
        else if (name == "TwofoldSearchDraw")
            searcher.set_twofold_search_draw_enabled(value == "true");
        else if (name == "AvoidDrawMinCp") adapter.avoid_draw_min_cp = std::stoi(value);
        else if (name == "AvoidDrawMaxLossCp") adapter.avoid_draw_max_loss_cp = std::stoi(value);
        else if (name == "MoveOverhead") adapter.move_overhead_ms = std::stoi(value);
        else if (name == "LmrBase") config.lmr_base = std::stod(value);
        else if (name == "LmrDivisor") config.lmr_divisor = std::stod(value);
        else if (name == "LmrMinDepth") config.lmr_min_depth = std::stoi(value);
        else if (name == "LmrMinMoveIndex") config.lmr_min_move_index = std::stoul(value);
        else if (name == "NullMoveMinDepth") config.null_move_min_depth = std::stoi(value);
        else if (name == "NullMoveReduction") config.null_move_reduction = std::stoi(value);
        else if (name == "RfpEnabled") config.enable_reverse_futility = value == "true";
        else if (name == "RfpMaxDepth") config.reverse_futility_max_depth = std::stoi(value);
        else if (name == "RfpBaseMargin") config.reverse_futility_base_margin = std::stoi(value);
        else if (name == "RfpMarginPerDepth") config.reverse_futility_margin_per_depth = std::stoi(value);
        else if (name == "LmpEnabled") config.enable_late_move_pruning = value == "true";
        else if (name == "LmpMaxDepth") config.late_move_pruning_max_depth = std::stoi(value);
        else if (name == "LmpBase") config.late_move_pruning_base = std::stoul(value);
        else if (name == "LmpDepthMultiplier") config.late_move_pruning_depth_multiplier = std::stoul(value);
        else if (name == "QseeEnabled") config.enable_qsearch_see_pruning = value == "true";
        else if (name == "QseeThreshold") config.qsearch_see_threshold = std::stoi(value);
        else if (name == "MainSeeEnabled")
            config.enable_main_search_see_pruning = value == "true";
        else if (name == "MainSeeMaxDepth")
            config.main_search_see_max_depth = std::stoi(value);
        else if (name == "MainSeeMarginPerDepth")
            config.main_search_see_margin_per_depth = std::stoi(value);
        searcher.set_selective_config(config);
        searcher.clear_tt();
    } catch (...) {
        std::cerr << "info string ignored invalid option " << name << '\n';
    }
}

std::mutex UciOutputMutex;

void write_uci(const std::string& text) {
    const std::lock_guard<std::mutex> lock(UciOutputMutex);
    std::cout << text << std::flush;
}

std::string format_search_result(const chess::SearchResult& result) {
    const std::string best_move = result.best_move.value != 0
        ? chess::move_to_string(result.best_move)
        : "0000";
    std::ostringstream output;
    output << "info depth " << result.depth << " score cp " << result.score
           << " nodes " << result.nodes;
    if (result.best_move.value != 0) {
        output << " pv " << best_move;
        if (result.ponder_move.value != 0) {
            output << ' ' << chess::move_to_string(result.ponder_move);
        }
    }
    output << '\n' << "bestmove " << best_move;
    if (result.ponder_move.value != 0) {
        output << " ponder " << chess::move_to_string(result.ponder_move);
    }
    output << '\n';
    return output.str();
}

struct SearchJob {
    std::atomic<bool> stop_requested{false};
    std::mutex mutex;
    std::condition_variable changed;
    std::chrono::milliseconds budget{0};
    bool ponder = false;
    bool hold_output = false;
    bool ponder_hit = false;
    bool stop_received = false;
    bool cancel_timer = false;
};

class UciSearchController {
public:
    UciSearchController(
        UciSearcher& searcher,
        const chess::PhaseQuantizedNnueModel& model
    ) : searcher_(searcher), model_(model) {
    }

    ~UciSearchController() {
        finish_active(true);
    }

    void start(
        const chess::Position& pos,
        const std::vector<chess::HashKey>& history,
        ParsedGo parsed,
        const AdapterOptions& adapter
    ) {
        finish_active(true);

        auto job = std::make_shared<SearchJob>();
        job->budget = parsed.budget;
        job->ponder = parsed.ponder && adapter.ponder;
        job->hold_output = job->ponder || parsed.infinite;
        parsed.limits.move_time = std::chrono::milliseconds{0};
        parsed.limits.stop_requested = &job->stop_requested;
        active_ = job;

        if (!job->ponder && !parsed.infinite && job->budget.count() > 0) {
            start_timer(job);
        }

        search_thread_ = std::thread([
            this,
            job,
            search_pos = pos,
            search_history = history,
            limits = parsed.limits,
            adapter
        ]() mutable {
            chess::SearchResult result =
#if defined(CHESS_UCI_NNUE_V41)
                searcher_.search_best_move(search_pos, limits, search_history);
#else
                searcher_.search_best_move(search_pos, limits);
#endif
#if !defined(CHESS_UCI_NNUE_V41)
            const chess::Move selected = choose_non_drawing_alternative(
                search_pos, search_history, model_, result, adapter);
            if (selected != result.best_move) {
                result.best_move = selected;
                result.ponder_move = {};
            }
#endif

            {
                std::unique_lock<std::mutex> lock(job->mutex);
                job->changed.wait(lock, [&] {
                    return !job->hold_output || job->stop_received;
                });
                job->cancel_timer = true;
                job->changed.notify_all();
            }
            write_uci(format_search_result(result));
        });
    }

    void ponder_hit() {
        const std::shared_ptr<SearchJob> job = active_;
        if (!job) {
            return;
        }

        bool arm_timer = false;
        {
            const std::lock_guard<std::mutex> lock(job->mutex);
            if (!job->ponder || job->ponder_hit || job->stop_received) {
                return;
            }
            job->ponder_hit = true;
            job->hold_output = false;
            arm_timer = job->budget.count() > 0;
            job->changed.notify_all();
        }
        if (arm_timer) {
            start_timer(job);
        }
    }

    void stop() {
        signal_stop(active_);
    }

    void finish_active(bool request_stop) {
        const std::shared_ptr<SearchJob> job = active_;
        if (!job) {
            return;
        }
        if (request_stop) {
            signal_stop(job);
        }
        if (search_thread_.joinable()) {
            search_thread_.join();
        }
        {
            const std::lock_guard<std::mutex> lock(job->mutex);
            job->cancel_timer = true;
            job->changed.notify_all();
        }
        if (timer_thread_.joinable()) {
            timer_thread_.join();
        }
        active_.reset();
    }

private:
    void start_timer(const std::shared_ptr<SearchJob>& job) {
        if (timer_thread_.joinable()) {
            timer_thread_.join();
        }
        timer_thread_ = std::thread([job] {
            std::unique_lock<std::mutex> lock(job->mutex);
            const bool cancelled = job->changed.wait_for(
                lock,
                job->budget,
                [&] { return job->cancel_timer || job->stop_received; });
            if (!cancelled) {
                job->stop_requested.store(true, std::memory_order_relaxed);
            }
        });
    }

    static void signal_stop(const std::shared_ptr<SearchJob>& job) {
        if (!job) {
            return;
        }
        job->stop_requested.store(true, std::memory_order_relaxed);
        const std::lock_guard<std::mutex> lock(job->mutex);
        job->stop_received = true;
        job->hold_output = false;
        job->cancel_timer = true;
        job->changed.notify_all();
    }

    UciSearcher& searcher_;
    const chess::PhaseQuantizedNnueModel& model_;
    std::shared_ptr<SearchJob> active_;
    std::thread search_thread_;
    std::thread timer_thread_;
};

} // namespace

int main(int argc, char** argv) {
    const char* env_model = std::getenv("CHESS_NNUE_MODEL");
    const std::string model_path = argc > 1
        ? argv[1]
        : env_model != nullptr
            ? env_model
            : std::string(chess::DefaultPhaseQuantizedNnueModelPath);

    chess::PhaseQuantizedNnueModel model;
    if (!model.load(model_path)) {
        std::cerr << "Failed to load NNUE model: " << model_path << '\n';
        return 2;
    }
    UciSearcher searcher(model);
    auto selective_config = searcher.selective_config();
#if !defined(CHESS_UCI_NNUE_V40) && !defined(CHESS_UCI_NNUE_V41)
    selective_config.lmr_base = 0.5;
    selective_config.lmr_divisor = 2.45;
    selective_config.lmr_min_depth = 4;
    selective_config.lmr_min_move_index = 5;
    selective_config.null_move_min_depth = 2;
    selective_config.null_move_reduction = 3;
#endif
    searcher.set_selective_config(selective_config);
    chess::Position pos;
    pos.set_startpos();
    std::vector<chess::HashKey> history{pos.zobrist_key};
    AdapterOptions adapter;
    UciSearchController controller(searcher, model);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command == "uci") {
            const auto& config = searcher.selective_config();
            std::ostringstream output;
            output << "id name " << EngineName << '\n'
                      << "id author TungLamNguyen\n"
                      << "info string nnue_kernel="
                      << model.forward_kernel_name()
                      << '\n'
                      << "info string nnue_accumulator_kernel="
                      << model.accumulator_kernel_name()
                      << '\n';
#if !defined(CHESS_UCI_NNUE_V41)
            output
                      << "option name AvoidDraw type check default true\n"
                      << "option name AvoidDrawMinCp type spin default 120 min 0 max 2000\n"
                      << "option name AvoidDrawMaxLossCp type spin default 80 min 0 max 1000\n";
#endif
#if defined(CHESS_UCI_NNUE_V41)
            output
                      << "option name TwofoldSearchDraw type check default "
                      << (searcher.twofold_search_draw_enabled()
                            ? "true" : "false")
                      << '\n';
#endif
            output
                      << "option name Ponder type check default true\n"
                      << "option name MoveOverhead type spin default 200 min 0 max 5000\n"
                      << "option name LmrBase type string default " << config.lmr_base << '\n'
                      << "option name LmrDivisor type string default " << config.lmr_divisor << '\n'
                      << "option name LmrMinDepth type spin default " << config.lmr_min_depth << " min 1 max 16\n"
                      << "option name LmrMinMoveIndex type spin default " << config.lmr_min_move_index << " min 1 max 64\n"
                      << "option name NullMoveMinDepth type spin default " << config.null_move_min_depth << " min 1 max 16\n"
                      << "option name NullMoveReduction type spin default " << config.null_move_reduction << " min 1 max 8\n"
                      << "option name RfpEnabled type check default " << (config.enable_reverse_futility ? "true" : "false") << '\n'
                      << "option name RfpMaxDepth type spin default " << config.reverse_futility_max_depth << " min 1 max 16\n"
                      << "option name RfpBaseMargin type spin default " << config.reverse_futility_base_margin << " min 0 max 5000\n"
                      << "option name RfpMarginPerDepth type spin default " << config.reverse_futility_margin_per_depth << " min 0 max 5000\n"
                      << "option name LmpEnabled type check default " << (config.enable_late_move_pruning ? "true" : "false") << '\n'
                      << "option name LmpMaxDepth type spin default " << config.late_move_pruning_max_depth << " min 1 max 16\n"
                      << "option name LmpBase type spin default " << config.late_move_pruning_base << " min 1 max 128\n"
                      << "option name LmpDepthMultiplier type spin default " << config.late_move_pruning_depth_multiplier << " min 0 max 32\n"
                      << "option name QseeEnabled type check default " << (config.enable_qsearch_see_pruning ? "true" : "false") << '\n'
                      << "option name QseeThreshold type spin default " << config.qsearch_see_threshold << " min -2000 max 2000\n"
                      << "option name MainSeeEnabled type check default " << (config.enable_main_search_see_pruning ? "true" : "false") << '\n'
                      << "option name MainSeeMaxDepth type spin default " << config.main_search_see_max_depth << " min 1 max 16\n"
                      << "option name MainSeeMarginPerDepth type spin default " << config.main_search_see_margin_per_depth << " min 0 max 5000\n"
                      << "uciok\n";
            write_uci(output.str());
        } else if (command == "isready") {
            write_uci("readyok\n");
        } else if (command == "setoption") {
            controller.finish_active(true);
            set_option(input, adapter, searcher);
        } else if (command == "ucinewgame") {
            controller.finish_active(true);
            pos.set_startpos();
            history = {pos.zobrist_key};
            searcher.clear_tt();
            searcher.clear_search_heuristics();
        } else if (command == "position") {
            controller.finish_active(true);
            set_position(pos, history, searcher, input);
        } else if (command == "go") {
            const ParsedGo parsed =
                parse_go_limits(input, pos.side_to_move, adapter);
            controller.start(pos, history, parsed, adapter);
        } else if (command == "ponderhit") {
            controller.ponder_hit();
        } else if (command == "stop") {
            controller.stop();
        } else if (command == "quit") {
            controller.finish_active(true);
            break;
        }
    }
}
