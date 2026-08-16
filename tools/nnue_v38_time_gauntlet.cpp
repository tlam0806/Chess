#include "attacks.hpp"
#include "game_state.hpp"
#include "move.hpp"
#include "nnue_searcher_v36.hpp"
#ifdef CHESS_NNUE_V39_TIME_GAUNTLET
#include "nnue_searcher_v39.hpp"
#else
#include "nnue_searcher_v38.hpp"
#endif
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

#ifdef CHESS_NNUE_V39_TIME_GAUNTLET
using MatchSearcher = chess::NnueSearcherV39;
#else
using MatchSearcher = chess::NnueSearcherV38;
#endif

struct Profile {
    std::string name;
    MatchSearcher::SelectiveConfig config;
};

struct Opening {
    int source_line = 0;
    std::string text;
    std::vector<std::string> moves;
};

struct Options {
    std::string book;
    std::string model;
    std::string output;
    int openings = 500;
    int base_ms = 10'000;
    int increment_ms = 100;
    int overhead_ms = 20;
    int max_plies = 200;
    std::size_t tt_mb = 64;
    std::uint32_t seed = 20260727;
    bool round_robin = false;
    bool fast_balanced_ci = false;
    bool stop_on_ci = false;
    bool balanced_rematch = false;
    std::string balanced_new_config;
    std::vector<std::string> profile_specs;
    int ci_min_pairs = 40;
    std::string trace_key;
    std::string stop_after_key;
};

struct GameResult {
    std::string result;
    std::string reason;
    int plies = 0;
    std::uint64_t control_nodes = 0;
    std::uint64_t candidate_nodes = 0;
    std::int64_t control_time_ms = 0;
    std::int64_t candidate_time_ms = 0;
};

int parse_int(std::string_view value, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const int result = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) throw std::invalid_argument("trailing");
        return result;
    } catch (...) {
        throw std::runtime_error("invalid integer for " + std::string(name));
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + std::string(arg));
            }
            return argv[i];
        };
        if (arg == "--book") options.book = next();
        else if (arg == "--model") options.model = next();
        else if (arg == "--output") options.output = next();
        else if (arg == "--openings") options.openings = parse_int(next(), arg);
        else if (arg == "--base-ms") options.base_ms = parse_int(next(), arg);
        else if (arg == "--increment-ms")
            options.increment_ms = parse_int(next(), arg);
        else if (arg == "--overhead-ms")
            options.overhead_ms = parse_int(next(), arg);
        else if (arg == "--max-plies") options.max_plies = parse_int(next(), arg);
        else if (arg == "--tt-mb") {
            options.tt_mb = static_cast<std::size_t>(parse_int(next(), arg));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(parse_int(next(), arg));
        } else if (arg == "--round-robin") {
            options.round_robin = true;
        } else if (arg == "--fast-balanced-ci") {
            options.fast_balanced_ci = true;
            options.round_robin = true;
        } else if (arg == "--stop-on-ci") {
            options.stop_on_ci = true;
            options.round_robin = true;
        } else if (arg == "--balanced-rematch") {
            options.balanced_rematch = true;
            options.round_robin = true;
        } else if (arg == "--balanced-new-config") {
            options.balanced_new_config = next();
        } else if (arg == "--profile") {
            options.profile_specs.push_back(next());
            options.round_robin = true;
        } else if (arg == "--ci-min-pairs") {
            options.ci_min_pairs = parse_int(next(), arg);
        } else if (arg == "--trace-key") {
            options.trace_key = next();
        } else if (arg == "--stop-after-key") {
            options.stop_after_key = next();
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.book.empty() || options.model.empty() || options.output.empty()) {
        throw std::runtime_error("--book, --model and --output are required");
    }
    if (options.openings <= 0 || options.base_ms <= 0
        || options.increment_ms < 0 || options.overhead_ms < 0
        || options.max_plies <= 0 || options.tt_mb == 0
        || options.ci_min_pairs < 2) {
        throw std::runtime_error("invalid non-positive option");
    }
    if (options.fast_balanced_ci && options.balanced_rematch) {
        throw std::runtime_error(
            "--fast-balanced-ci and --balanced-rematch are mutually exclusive");
    }
    if (!options.balanced_new_config.empty() && !options.balanced_rematch) {
        throw std::runtime_error(
            "--balanced-new-config requires --balanced-rematch");
    }
    if (!options.profile_specs.empty() && options.profile_specs.size() < 2) {
        throw std::runtime_error("at least two --profile values are required");
    }
    if (!options.profile_specs.empty()
        && (options.fast_balanced_ci || options.balanced_rematch
            || !options.balanced_new_config.empty())) {
        throw std::runtime_error(
            "--profile cannot be combined with legacy profile modes");
    }
    return options;
}

std::vector<std::string> words(std::string_view text) {
    std::istringstream input{std::string(text)};
    std::vector<std::string> result;
    std::string word;
    while (input >> word) result.push_back(word);
    return result;
}

chess::Move legal_uci(const chess::Position& position, std::string_view uci) {
    for (const chess::Move move : chess::generate_legal_moves(position)) {
        if (chess::move_to_string(move) == uci) return move;
    }
    throw std::runtime_error("illegal opening move: " + std::string(uci));
}

chess::Position opening_position(const Opening& opening) {
    chess::Position position;
    position.set_startpos();
    for (const std::string& uci : opening.moves) {
        position.make_move(legal_uci(position, uci));
    }
    return position;
}

std::vector<Opening> load_openings(const Options& options) {
    std::ifstream input(options.book);
    if (!input) throw std::runtime_error("failed to open book");
    std::vector<Opening> openings;
    std::string line;
    for (int line_number = 1; std::getline(input, line); ++line_number) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        const std::vector<std::string> moves = words(line);
        if (moves.empty()) continue;
        Opening opening{line_number, line, moves};
        static_cast<void>(opening_position(opening));
        openings.push_back(std::move(opening));
    }
    if (static_cast<int>(openings.size()) < options.openings) {
        throw std::runtime_error("book has fewer openings than requested");
    }
    std::mt19937 rng(options.seed);
    std::shuffle(openings.begin(), openings.end(), rng);
    openings.resize(options.openings);
    return openings;
}

int move_budget_ms(int remaining_ms, const Options& options) {
    const int reserve = std::min(
        remaining_ms / 2,
        std::max(options.overhead_ms, remaining_ms / 50));
    const int usable = std::max(1, remaining_ms - reserve);
    int budget = usable / 20 + options.increment_ms / 2;
    budget = std::max(5, budget);
    return std::min(budget, std::max(1, usable / 5));
}

char piece_char(chess::Color color, chess::PieceType piece) {
    char result = '?';
    switch (piece) {
        case chess::PieceType::Pawn: result = 'p'; break;
        case chess::PieceType::Knight: result = 'n'; break;
        case chess::PieceType::Bishop: result = 'b'; break;
        case chess::PieceType::Rook: result = 'r'; break;
        case chess::PieceType::Queen: result = 'q'; break;
        case chess::PieceType::King: result = 'k'; break;
        case chess::PieceType::None: result = '?'; break;
    }
    if (color == chess::Color::White && result >= 'a' && result <= 'z') {
        result = static_cast<char>(result - 'a' + 'A');
    }
    return result;
}

std::string to_fen(const chess::Position& position) {
    std::ostringstream output;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const chess::Square square = chess::make_square(file, rank);
            if (position.is_empty(square)) {
                ++empty;
                continue;
            }
            if (empty > 0) {
                output << empty;
                empty = 0;
            }
            output << piece_char(
                position.color_on_occupied(square),
                position.piece_type_on_occupied(square));
        }
        if (empty > 0) output << empty;
        if (rank > 0) output << '/';
    }
    output << (position.side_to_move == chess::Color::White ? " w " : " b ");
    std::string castling;
    if (position.white_can_castle_kingside) castling += 'K';
    if (position.white_can_castle_queenside) castling += 'Q';
    if (position.black_can_castle_kingside) castling += 'k';
    if (position.black_can_castle_queenside) castling += 'q';
    output << (castling.empty() ? "-" : castling) << ' ';
    if (position.en_passant_square == chess::NoSquare) {
        output << '-';
    } else {
        output << static_cast<char>(
            'a' + chess::file_of(position.en_passant_square));
        output << static_cast<char>(
            '1' + chess::rank_of(position.en_passant_square));
    }
    output << ' ' << position.halfmove_clock
           << ' ' << position.fullmove_number;
    return output.str();
}

std::string terminal_result(const chess::Position& position) {
    if (!chess::in_check(position, position.side_to_move)) return "1/2-1/2";
    return position.side_to_move == chess::Color::White ? "0-1" : "1-0";
}

GameResult play_game(
    const Opening& opening,
    bool candidate_white,
    const Options& options,
    chess::Searcher& control,
    chess::Searcher& candidate
) {
    chess::Position position = opening_position(opening);
    std::vector<chess::HashKey> history{position.zobrist_key};
    std::array<int, 2> remaining{options.base_ms, options.base_ms};
    GameResult game;

    for (int ply = 0; ply < options.max_plies; ++ply) {
        if (chess::is_threefold_repetition(position.zobrist_key, history)
            || position.halfmove_clock >= 100) {
            game.result = "1/2-1/2";
            game.reason = "rule_draw";
            game.plies = ply;
            return game;
        }
        const std::vector<chess::Move> legal =
            chess::generate_legal_moves(position);
        if (legal.empty()) {
            game.result = terminal_result(position);
            game.reason = chess::in_check(position, position.side_to_move)
                ? "checkmate" : "stalemate";
            game.plies = ply;
            return game;
        }

        const bool white = position.side_to_move == chess::Color::White;
        const bool use_candidate = white == candidate_white;
        const int clock_index = white ? 0 : 1;
        const int budget = move_budget_ms(remaining[clock_index], options);
        const chess::SearchLimits limits{
            .max_depth = 64,
            .move_time = std::chrono::milliseconds{budget},
        };
        const bool trace_search =
            std::getenv("CHESS_TRACE_V38_ROOT") != nullptr;
        if (trace_search) {
            std::cerr << "search_start"
                      << " opening_line=" << opening.source_line
                      << " candidate_white=" << candidate_white
                      << " ply=" << ply
                      << " engine=" << (use_candidate ? "candidate" : "opponent")
                      << " budget_ms=" << budget
                      << " remaining_ms=" << remaining[clock_index]
                      << " fen=\"" << to_fen(position) << "\"\n";
            std::cerr.flush();
        }
        const auto start = std::chrono::steady_clock::now();
        const chess::SearchResult result = use_candidate
            ? candidate.search_best_move(position, limits)
            : control.search_best_move(position, limits);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (trace_search) {
            std::cerr << "search_done"
                      << " opening_line=" << opening.source_line
                      << " candidate_white=" << candidate_white
                      << " ply=" << ply
                      << " engine=" << (use_candidate ? "candidate" : "opponent")
                      << " elapsed_ms=" << elapsed
                      << " depth=" << result.depth
                      << " nodes=" << result.nodes
                      << " stopped=" << result.stopped
                      << " score=" << result.score
                      << " move=" << chess::move_to_string(result.best_move)
                      << '\n';
            std::cerr.flush();
        }
        remaining[clock_index] -= static_cast<int>(elapsed);
        if (use_candidate) {
            game.candidate_nodes += result.nodes;
            game.candidate_time_ms += elapsed;
        } else {
            game.control_nodes += result.nodes;
            game.control_time_ms += elapsed;
        }
        if (remaining[clock_index] < 0) {
            game.result = white ? "0-1" : "1-0";
            game.reason = use_candidate ? "candidate_time" : "control_time";
            game.plies = ply;
            return game;
        }
        remaining[clock_index] += options.increment_ms;
        if (std::find(legal.begin(), legal.end(), result.best_move) == legal.end()) {
            game.result = white ? "0-1" : "1-0";
            game.reason = use_candidate ? "candidate_illegal" : "control_illegal";
            game.plies = ply;
            return game;
        }
        position.make_move(result.best_move);
        history.push_back(position.zobrist_key);
    }
    game.result = "1/2-1/2";
    game.reason = "max_plies";
    game.plies = options.max_plies;
    return game;
}

std::set<std::string> completed_keys(const std::string& path) {
    std::ifstream input(path);
    std::set<std::string> keys;
    std::string line;
    while (std::getline(input, line)) {
        const std::string marker = "\"key\":\"";
        const std::size_t start = line.find(marker);
        if (start == std::string::npos) continue;
        const std::size_t value_start = start + marker.size();
        const std::size_t end = line.find('"', value_start);
        if (end != std::string::npos) {
            keys.insert(line.substr(value_start, end - value_start));
        }
    }
    return keys;
}

std::string candidate_outcome(const GameResult& game, bool candidate_white) {
    if (game.result == "1/2-1/2") return "draw";
    const bool white_won = game.result == "1-0";
    return white_won == candidate_white ? "win" : "loss";
}

double candidate_points(const GameResult& game, bool candidate_white) {
    if (game.result == "1/2-1/2") return 0.5;
    const bool white_won = game.result == "1-0";
    return white_won == candidate_white ? 1.0 : 0.0;
}

struct ConfidenceInterval {
    double mean = 0.5;
    double lower = 0.0;
    double upper = 1.0;
};

ConfidenceInterval paired_score_ci95(const std::vector<double>& pair_scores) {
    ConfidenceInterval result;
    if (pair_scores.empty()) return result;
    const double n = static_cast<double>(pair_scores.size());
    double sum = 0.0;
    for (double score : pair_scores) sum += score;
    result.mean = sum / n;
    if (pair_scores.size() < 2) return result;

    double squared_error = 0.0;
    for (double score : pair_scores) {
        const double error = score - result.mean;
        squared_error += error * error;
    }
    const double sample_variance = squared_error / (n - 1.0);
    const double half_width = 1.96 * std::sqrt(sample_variance / n);
    result.lower = std::max(0.0, result.mean - half_width);
    result.upper = std::min(1.0, result.mean + half_width);
    return result;
}

void append_game(
    std::ofstream& output,
    const std::string& key,
    const Profile& profile,
    std::string_view opponent,
    const Opening& opening,
    bool candidate_white,
    const GameResult& game
) {
    output << "{\"kind\":\"game\",\"key\":\"" << key
           << "\",\"profile\":\"" << profile.name
           << "\",\"opponent\":\"" << opponent
           << "\",\"opening_line\":" << opening.source_line
           << ",\"candidate_color\":\""
           << (candidate_white ? "white" : "black")
           << "\",\"result\":\"" << game.result
           << "\",\"candidate_outcome\":\""
           << candidate_outcome(game, candidate_white)
           << "\",\"reason\":\"" << game.reason
           << "\",\"plies\":" << game.plies
           << ",\"candidate_nodes\":" << game.candidate_nodes
           << ",\"control_nodes\":" << game.control_nodes
           << ",\"candidate_time_ms\":" << game.candidate_time_ms
           << ",\"control_time_ms\":" << game.control_time_ms
           << "}\n";
    output.flush();
}

MatchSearcher::SelectiveConfig config(
    double base, double divisor, int min_depth, std::size_t move_index,
    int null_depth, int null_reduction
) {
    MatchSearcher::SelectiveConfig result;
    result.lmr_base = base;
    result.lmr_divisor = divisor;
    result.lmr_min_depth = min_depth;
    result.lmr_min_move_index = move_index;
    result.null_move_min_depth = null_depth;
    result.null_move_reduction = null_reduction;
    return result;
}

MatchSearcher::SelectiveConfig parse_config(std::string text) {
    std::replace(text.begin(), text.end(), ',', ' ');
    const std::vector<std::string> fields = words(text);
    if (fields.size() != 6) {
        throw std::runtime_error(
            "--balanced-new-config needs base,divisor,min_depth,"
            "move_index,null_depth,null_reduction");
    }
    try {
        return config(
            std::stod(fields[0]),
            std::stod(fields[1]),
            parse_int(fields[2], "lmr_min_depth"),
            static_cast<std::size_t>(
                parse_int(fields[3], "lmr_min_move_index")),
            parse_int(fields[4], "null_min_depth"),
            parse_int(fields[5], "null_reduction"));
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("invalid --balanced-new-config");
    } catch (const std::out_of_range&) {
        throw std::runtime_error("out-of-range --balanced-new-config");
    }
}

bool parse_bool01(std::string_view value, std::string_view name) {
    const int parsed = parse_int(value, name);
    if (parsed != 0 && parsed != 1) {
        throw std::runtime_error(
            std::string(name) + " must be either 0 or 1");
    }
    return parsed != 0;
}

Profile parse_profile(std::string text) {
    std::replace(text.begin(), text.end(), ',', ' ');
    const std::vector<std::string> fields = words(text);
    if (fields.size() != 15) {
        throw std::runtime_error(
            "--profile needs name,lmr_base,lmr_divisor,lmr_min_depth,"
            "lmr_min_move,null_min_depth,null_reduction,rfp_enabled,"
            "rfp_max_depth,rfp_base_margin,rfp_margin_per_depth,"
            "lmp_enabled,lmp_max_depth,lmp_base,lmp_depth_multiplier");
    }
    if (fields[0].empty()
        || fields[0].find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
            != std::string::npos) {
        throw std::runtime_error(
            "profile name may contain only letters, digits, '_' and '-'");
    }
    try {
        MatchSearcher::SelectiveConfig result = config(
            std::stod(fields[1]),
            std::stod(fields[2]),
            parse_int(fields[3], "lmr_min_depth"),
            static_cast<std::size_t>(parse_int(fields[4], "lmr_min_move")),
            parse_int(fields[5], "null_min_depth"),
            parse_int(fields[6], "null_reduction"));
        result.enable_reverse_futility =
            parse_bool01(fields[7], "rfp_enabled");
        result.reverse_futility_max_depth =
            parse_int(fields[8], "rfp_max_depth");
        result.reverse_futility_base_margin =
            parse_int(fields[9], "rfp_base_margin");
        result.reverse_futility_margin_per_depth =
            parse_int(fields[10], "rfp_margin_per_depth");
        result.enable_late_move_pruning =
            parse_bool01(fields[11], "lmp_enabled");
        result.late_move_pruning_max_depth =
            parse_int(fields[12], "lmp_max_depth");
        result.late_move_pruning_base = static_cast<std::size_t>(
            parse_int(fields[13], "lmp_base"));
        result.late_move_pruning_depth_multiplier = static_cast<std::size_t>(
            parse_int(fields[14], "lmp_depth_multiplier"));
        return {fields[0], result};
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("invalid --profile");
    } catch (const std::out_of_range&) {
        throw std::runtime_error("out-of-range --profile");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Opening> openings = load_openings(options);
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model");
        }
        model.set_neon_dotprod_enabled(true);
        const MatchSearcher::SelectiveConfig balanced_new =
            options.balanced_new_config.empty()
            ? config(0.55, 2.8, 5, 8, 6, 2)
            : parse_config(options.balanced_new_config);
        std::vector<Profile> profiles = options.balanced_rematch
            ? std::vector<Profile>{
                {"balanced_new", balanced_new},
                {"balanced_old", config(0.45, 2.45, 5, 5, 3, 2)},
            }
            : std::vector<Profile>{
                {"fast", config(0.25, 1.95, 3, 8, 3, 2)},
                {"balanced", config(0.45, 2.45, 5, 5, 3, 2)},
                {"safe", config(0.65, 2.9, 7, 3, 3, 2)},
            };
        if (!options.profile_specs.empty()) {
            profiles.clear();
            std::set<std::string> names;
            for (const std::string& spec : options.profile_specs) {
                Profile profile = parse_profile(spec);
                if (!names.insert(profile.name).second) {
                    throw std::runtime_error(
                        "duplicate profile name: " + profile.name);
                }
                profiles.push_back(std::move(profile));
            }
        }
        const std::set<std::string> done = completed_keys(options.output);
        if ((options.fast_balanced_ci || options.stop_on_ci) && !done.empty()) {
            throw std::runtime_error(
                "CI stopping requires a fresh output file");
        }
        std::ofstream output(options.output, std::ios::app);
        if (!output) throw std::runtime_error("failed to open output");
        int newly_completed = 0;
        if (options.round_robin) {
            for (std::size_t first = 0; first < profiles.size(); ++first) {
                for (std::size_t second = first + 1;
                     second < profiles.size(); ++second) {
                    if (options.fast_balanced_ci
                        && !(first == 0 && second == 1)) {
                        continue;
                    }
                    const Profile& profile = profiles[first];
                    const Profile& opponent = profiles[second];
                    std::vector<double> pair_scores;
                    double current_pair_points = 0.0;
                    bool matchup_ci_stopped = false;
                    MatchSearcher first_engine(
                        model, options.tt_mb, 4, 10, 14, 14'000,
                        MatchSearcher::MoveOrderingWeights{},
                        profile.config);
                    MatchSearcher second_engine(
                        model, options.tt_mb, 4, 10, 14, 14'000,
                        MatchSearcher::MoveOrderingWeights{},
                        opponent.config);
                    for (std::size_t index = 0; index < openings.size(); ++index) {
                        for (int color = 0; color < 2; ++color) {
                            const bool candidate_white = color == 0;
                            const std::string key = profile.name + "_vs_"
                                + opponent.name + ":" + std::to_string(index)
                                + ":" + (candidate_white ? "w" : "b");
                            if (done.contains(key)) continue;
                            if (!options.trace_key.empty()
                                && (options.trace_key == "all"
                                    || key == options.trace_key)) {
                                setenv("CHESS_TRACE_V38_ROOT", "1", 1);
                            } else {
                                unsetenv("CHESS_TRACE_V38_ROOT");
                            }
                            first_engine.clear_tt();
                            second_engine.clear_tt();
                            const GameResult game = play_game(
                                openings[index], candidate_white, options,
                                second_engine, first_engine);
                            current_pair_points += candidate_points(
                                game, candidate_white);
                            append_game(
                                output, key, profile, opponent.name,
                                openings[index], candidate_white, game);
                            ++newly_completed;
                            if ((options.fast_balanced_ci || options.stop_on_ci)
                                && color == 1) {
                                pair_scores.push_back(current_pair_points / 2.0);
                                current_pair_points = 0.0;
                                const ConfidenceInterval ci =
                                    paired_score_ci95(pair_scores);
                                const std::size_t pairs = pair_scores.size();
                                if (pairs % 10 == 0
                                    || static_cast<int>(pairs)
                                        == options.ci_min_pairs) {
                                    std::cerr << "ci_progress"
                                              << " matchup=" << profile.name
                                              << "_vs_" << opponent.name
                                              << " pairs=" << pairs
                                              << " games=" << pairs * 2
                                              << " first_score=" << ci.mean
                                              << " ci95_lower=" << ci.lower
                                              << " ci95_upper=" << ci.upper
                                              << '\n';
                                }
                                if (static_cast<int>(pairs)
                                        >= options.ci_min_pairs
                                    && (ci.lower > 0.5
                                        || ci.upper < 0.5)) {
                                    std::cerr << "ci_stop"
                                              << " matchup=" << profile.name
                                              << "_vs_" << opponent.name
                                              << " pairs=" << pairs
                                              << " games=" << pairs * 2
                                              << " first_score=" << ci.mean
                                              << " ci95_lower=" << ci.lower
                                              << " ci95_upper=" << ci.upper
                                              << '\n';
                                    if (options.fast_balanced_ci) return 0;
                                    matchup_ci_stopped = true;
                                }
                            }
                            if (!options.stop_after_key.empty()
                                && key == options.stop_after_key) {
                                std::cerr << "complete stop_after_key=" << key
                                          << " new_games=" << newly_completed
                                          << '\n';
                                return 0;
                            }
                            if (newly_completed % 10 == 0) {
                                std::cerr << "progress new_games="
                                          << newly_completed
                                          << " last=" << key << '\n';
                            }
                        }
                        if (matchup_ci_stopped) break;
                    }
                    if ((options.fast_balanced_ci || options.stop_on_ci)
                        && !matchup_ci_stopped) {
                        const ConfidenceInterval ci =
                            paired_score_ci95(pair_scores);
                        std::cerr << "ci_inconclusive"
                                  << " matchup=" << profile.name
                                  << "_vs_" << opponent.name
                                  << " pairs=" << pair_scores.size()
                                  << " games=" << pair_scores.size() * 2
                                  << " first_score=" << ci.mean
                                  << " ci95_lower=" << ci.lower
                                  << " ci95_upper=" << ci.upper
                                  << '\n';
                    }
                }
            }
            std::cerr << "complete new_games=" << newly_completed << '\n';
            return 0;
        }
        for (const Profile& profile : profiles) {
            chess::NnueSearcherV36 control(model, options.tt_mb);
            MatchSearcher candidate(
                model, options.tt_mb, 4, 10, 14, 14'000,
                MatchSearcher::MoveOrderingWeights{},
                profile.config);
            for (std::size_t index = 0; index < openings.size(); ++index) {
                for (int color = 0; color < 2; ++color) {
                    const bool candidate_white = color == 0;
                    const std::string key = profile.name + ":"
                        + std::to_string(index) + ":"
                        + (candidate_white ? "w" : "b");
                    if (done.contains(key)) continue;
                    control.clear_tt();
                    candidate.clear_tt();
                    const GameResult game = play_game(
                        openings[index], candidate_white, options,
                        control, candidate);
                    append_game(
                        output, key, profile, "v36", openings[index],
                        candidate_white, game);
                    ++newly_completed;
                    if (newly_completed % 10 == 0) {
                        std::cerr << "progress new_games=" << newly_completed
                                  << " last=" << key << '\n';
                    }
                }
            }
        }
        std::cerr << "complete new_games=" << newly_completed << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
