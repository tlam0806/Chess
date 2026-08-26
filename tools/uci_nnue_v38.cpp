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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
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
    int avoid_draw_min_cp = 120;
    int avoid_draw_max_loss_cp = 80;
    int move_overhead_ms = 200;
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

chess::SearchLimits parse_go_limits(
    std::istringstream& input,
    chess::Color side,
    const AdapterOptions& options
) {
    chess::SearchLimits limits;
    limits.max_depth = DefaultDepth;
    int wtime = -1;
    int btime = -1;
    int winc = 0;
    int binc = 0;
    int moves_to_go = 0;
    bool explicit_limit = false;
    std::string token;
    while (input >> token) {
        int value = 0;
        if (token == "depth" && input >> value) {
            limits.max_depth = std::max(1, value);
            explicit_limit = true;
        } else if (token == "movetime" && input >> value) {
            limits.max_depth = 64;
            limits.move_time = std::chrono::milliseconds{std::max(1, value)};
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
            limits.max_depth = 64;
            limits.move_time = std::chrono::milliseconds{budget};
        }
    }
    return limits;
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
        searcher.set_selective_config(config);
        searcher.clear_tt();
    } catch (...) {
        std::cerr << "info string ignored invalid option " << name << '\n';
    }
}

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

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command == "uci") {
            const auto& config = searcher.selective_config();
            std::cout << "id name " << EngineName << '\n'
                      << "id author TungLamNguyen\n"
                      << "info string nnue_kernel="
                      << model.forward_kernel_name()
                      << '\n';
#if !defined(CHESS_UCI_NNUE_V41)
            std::cout
                      << "option name AvoidDraw type check default true\n"
                      << "option name AvoidDrawMinCp type spin default 120 min 0 max 2000\n"
                      << "option name AvoidDrawMaxLossCp type spin default 80 min 0 max 1000\n";
#endif
            std::cout
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
                      << "uciok" << std::endl;
        } else if (command == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (command == "setoption") {
            set_option(input, adapter, searcher);
        } else if (command == "ucinewgame") {
            pos.set_startpos();
            history = {pos.zobrist_key};
            searcher.clear_tt();
            searcher.clear_search_heuristics();
        } else if (command == "position") {
            set_position(pos, history, searcher, input);
        } else if (command == "go") {
            const chess::SearchLimits limits =
                parse_go_limits(input, pos.side_to_move, adapter);
            chess::SearchResult result =
#if defined(CHESS_UCI_NNUE_V41)
                searcher.search_best_move(pos, limits, history);
#else
                searcher.search_best_move(pos, limits);
#endif
#if !defined(CHESS_UCI_NNUE_V41)
            result.best_move = choose_non_drawing_alternative(
                pos, history, model, result, adapter);
#endif
            std::cout << "info depth " << result.depth << " score cp " << result.score
                      << " nodes " << result.nodes << '\n'
                      << "bestmove " << chess::move_to_string(result.best_move)
                      << std::endl;
        } else if (command == "quit") {
            break;
        }
    }
}
