#include "attacks.hpp"
#include "game_state.hpp"
#include "move.hpp"
#include "nnue_searcher_v43.hpp"
#include "nnue_searcher_v44.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ControlSearcher = chess::NnueSearcherV43;
using CandidateSearcher = chess::NnueSearcherV44;

constexpr std::string_view CandidateName = "v44_clear_each_search";
constexpr std::string_view ControlName = "v43_production";
constexpr std::string_view CombinedConfigHash =
    "98b7732c9587da35554cc274a072a0a5b5f55902aaa77605e78c1ae13e88b4f2";

volatile std::sig_atomic_t stop_requested = 0;

void handle_stop_signal(int) {
    stop_requested = 1;
}

struct Options {
    std::string book;
    std::string model;
    std::string output;
    int openings = 300;
    int base_ms = 10'000;
    int increment_ms = 100;
    int overhead_ms = 20;
    int max_plies = 240;
    std::size_t tt_mb = 64;
    std::uint32_t seed = 2'026'090'155;
    int stop_after_games = 0;
};

struct Opening {
    int source_line = 0;
    std::string text;
    std::vector<std::string> moves;
};

struct PlySearch {
    int game_ply = 0;
    int absolute_ply = 0;
    bool candidate = false;
    bool white = false;
    std::string move;
    int depth = 0;
    int score = 0;
    std::uint64_t nodes = 0;
    std::int64_t time_ms = 0;
    int budget_ms = 0;
    int remaining_before_ms = 0;
    int remaining_after_ms = 0;
    bool stopped = false;
    bool applied = false;
};

struct GameResult {
    std::string result;
    std::string reason;
    bool aborted = false;
    int played_plies = 0;
    std::uint64_t control_nodes = 0;
    std::uint64_t candidate_nodes = 0;
    std::int64_t control_time_ms = 0;
    std::int64_t candidate_time_ms = 0;
    std::uint64_t control_search_cycle_draws = 0;
    std::uint64_t candidate_search_cycle_draws = 0;
    std::vector<std::string> played_moves;
    std::vector<PlySearch> searches;
};

int parse_int(std::string_view value, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const long long result = std::stoll(std::string(value), &parsed);
        if (parsed != value.size()
            || result < std::numeric_limits<int>::min()
            || result > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("out of range");
        }
        return static_cast<int>(result);
    } catch (...) {
        throw std::runtime_error("invalid integer for " + std::string(name));
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(
                    "missing value for " + std::string(argument));
            }
            return argv[index];
        };
        if (argument == "--book") options.book = next();
        else if (argument == "--model") options.model = next();
        else if (argument == "--output") options.output = next();
        else if (argument == "--openings") {
            options.openings = parse_int(next(), argument);
        } else if (argument == "--base-ms") {
            options.base_ms = parse_int(next(), argument);
        } else if (argument == "--increment-ms") {
            options.increment_ms = parse_int(next(), argument);
        } else if (argument == "--overhead-ms") {
            options.overhead_ms = parse_int(next(), argument);
        } else if (argument == "--max-plies") {
            options.max_plies = parse_int(next(), argument);
        } else if (argument == "--tt-mb") {
            const int value = parse_int(next(), argument);
            if (value <= 0) throw std::runtime_error("--tt-mb must be positive");
            options.tt_mb = static_cast<std::size_t>(value);
        } else if (argument == "--seed") {
            const int value = parse_int(next(), argument);
            if (value < 0) throw std::runtime_error("--seed must be non-negative");
            options.seed = static_cast<std::uint32_t>(value);
        } else if (argument == "--stop-after-games") {
            options.stop_after_games = parse_int(next(), argument);
        } else {
            throw std::runtime_error(
                "unknown argument: " + std::string(argument));
        }
    }
    if (options.book.empty() || options.model.empty() || options.output.empty()) {
        throw std::runtime_error("--book, --model and --output are required");
    }
    if (options.openings <= 0 || options.base_ms <= 0
        || options.increment_ms < 0 || options.overhead_ms < 0
        || options.max_plies <= 0 || options.stop_after_games < 0) {
        throw std::runtime_error("invalid non-positive match option");
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

void append_json_string(std::ostream& output, std::string_view value) {
    static constexpr char hex[] = "0123456789abcdef";
    output << '"';
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (byte < 0x20) {
                output << "\\u00" << hex[byte >> 4] << hex[byte & 0xf];
            } else {
                output << static_cast<char>(byte);
            }
        }
    }
    output << '"';
}

chess::Move legal_uci(const chess::Position& position, std::string_view uci) {
    for (const chess::Move move : chess::generate_legal_moves(position)) {
        if (chess::move_to_string(move) == uci) return move;
    }
    throw std::runtime_error("illegal opening move: " + std::string(uci));
}

std::uint8_t castling_rights(const chess::Position& position) {
    return static_cast<std::uint8_t>(
        (position.white_can_castle_kingside ? 1 : 0)
        | (position.white_can_castle_queenside ? 2 : 0)
        | (position.black_can_castle_kingside ? 4 : 0)
        | (position.black_can_castle_queenside ? 8 : 0));
}

bool move_was_irreversible(
    std::uint8_t rights_before,
    const chess::Position& after
) {
    return after.halfmove_clock == 0
        || castling_rights(after) != rights_before;
}

chess::Position opening_position(
    const Opening& opening,
    std::vector<chess::HashKey>* history = nullptr
) {
    chess::Position position;
    position.set_startpos();
    if (history != nullptr) {
        history->clear();
        history->push_back(position.zobrist_key);
    }
    for (const std::string& uci : opening.moves) {
        const chess::Move move = legal_uci(position, uci);
        const std::uint8_t rights_before = castling_rights(position);
        position.make_move(move);
        if (history != nullptr) {
            if (move_was_irreversible(rights_before, position)) {
                history->clear();
            }
            history->push_back(position.zobrist_key);
        }
    }
    return position;
}

std::vector<Opening> load_openings(const Options& options) {
    std::ifstream input(options.book);
    if (!input) throw std::runtime_error("failed to open book");
    std::vector<Opening> openings;
    std::set<std::string> unique;
    std::string line;
    for (int line_number = 1; std::getline(input, line); ++line_number) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        const std::vector<std::string> moves = words(line);
        if (moves.empty()) continue;
        std::ostringstream canonical;
        for (std::size_t index = 0; index < moves.size(); ++index) {
            if (index != 0) canonical << ' ';
            canonical << moves[index];
        }
        if (!unique.insert(canonical.str()).second) {
            throw std::runtime_error(
                "duplicate opening at source line "
                + std::to_string(line_number));
        }
        Opening opening{line_number, canonical.str(), moves};
        static_cast<void>(opening_position(opening));
        openings.push_back(std::move(opening));
    }
    if (static_cast<int>(openings.size()) < options.openings) {
        throw std::runtime_error("book has fewer openings than requested");
    }
    std::mt19937 rng(options.seed);
    std::shuffle(openings.begin(), openings.end(), rng);
    openings.resize(static_cast<std::size_t>(options.openings));
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

std::string terminal_result(const chess::Position& position) {
    if (!chess::in_check(position, position.side_to_move)) return "1/2-1/2";
    return position.side_to_move == chess::Color::White ? "0-1" : "1-0";
}

template <typename Engine>
chess::SearchResult search_with_history(
    Engine& engine,
    const chess::Position& position,
    const chess::SearchLimits& limits,
    const std::vector<chess::HashKey>& history
) {
    return engine.search_best_move(
        position, limits, std::span<const chess::HashKey>{history});
}

template <typename Engine>
std::uint64_t search_cycle_draws(const Engine& engine) {
    return engine.repetition_stats().search_cycle_draws;
}

template <typename Config>
Config production_selective_config() {
    Config result{};
    result.enable_lmr = true;
    result.lmr_base = 0.45;
    result.lmr_divisor = 2.9;
    result.lmr_min_depth = 3;
    result.lmr_min_move_index = 6;
    result.enable_null_move = true;
    result.null_move_min_depth = 2;
    result.null_move_reduction = 4;
    result.enable_reverse_futility = true;
    result.reverse_futility_max_depth = 4;
    result.reverse_futility_base_margin = 50;
    result.reverse_futility_margin_per_depth = 100;
    result.enable_late_move_pruning = true;
    result.late_move_pruning_max_depth = 3;
    result.late_move_pruning_base = 4;
    result.late_move_pruning_depth_multiplier = 1;
    result.enable_qsearch_see_pruning = true;
    result.qsearch_see_threshold = -25;
    result.enable_main_search_see_pruning = false;
    result.main_search_see_max_depth = 5;
    result.main_search_see_margin_per_depth = 100;
    return result;
}

template <typename Config>
Config production_aspiration_config() {
    Config result{};
    result.enabled = true;
    result.min_depth = 2;
    result.delta_base_cp = 68;
    result.delta_divisor = 33'700;
    result.expansion_factor_per_mille = 2'290;
    result.max_fail_high_reductions = 1;
    result.mean_score_new_weight_per_mille = 370;
    result.max_researches = 6;
    result.mean_score_clamp_cp = 1'500;
    return result;
}

template <typename Engine>
Engine make_engine(const chess::PhaseQuantizedNnueModel& model, std::size_t tt_mb) {
    Engine result(
        model, tt_mb, 4, 10, 14, 14'000,
        typename Engine::MoveOrderingWeights{},
        production_selective_config<typename Engine::SelectiveConfig>());
    result.set_aspiration_config(
        production_aspiration_config<typename Engine::AspirationConfig>());
    result.set_twofold_search_draw_enabled(true);
    result.set_reuse_deeper_tt_scores(false);
    if constexpr (requires { result.set_reuse_stale_tt_scores(false); }) {
        result.set_reuse_stale_tt_scores(false);
    }
    return result;
}

GameResult play_game(
    const Opening& opening,
    bool candidate_white,
    const Options& options,
    ControlSearcher& control,
    CandidateSearcher& candidate
) {
    // Each game is isolated. V44 performs one additional clear inside every
    // top-level search; that clear is intentionally inside the elapsed timer.
    control.clear_tt();
    candidate.clear_tt();
    control.clear_search_heuristics();
    candidate.clear_search_heuristics();

    std::vector<chess::HashKey> history;
    chess::Position position = opening_position(opening, &history);
    std::array<int, 2> remaining{options.base_ms, options.base_ms};
    GameResult game;

    for (int ply = 0; ply < options.max_plies; ++ply) {
        // A detached runner asks the process to stop with SIGTERM. Finish the
        // current root search, then discard this incomplete game so resume can
        // replay its immutable key from a clean game boundary.
        if (stop_requested != 0) {
            game.aborted = true;
            game.played_plies = ply;
            return game;
        }
        if (chess::is_threefold_repetition(position.zobrist_key, history)
            || position.halfmove_clock >= 100) {
            game.result = "1/2-1/2";
            game.reason = "rule_draw";
            game.played_plies = ply;
            return game;
        }
        const std::vector<chess::Move> legal =
            chess::generate_legal_moves(position);
        if (legal.empty()) {
            game.result = terminal_result(position);
            game.reason = chess::in_check(position, position.side_to_move)
                ? "checkmate" : "stalemate";
            game.played_plies = ply;
            return game;
        }

        const bool white = position.side_to_move == chess::Color::White;
        const bool use_candidate = white == candidate_white;
        const int clock_index = white ? 0 : 1;
        const int remaining_before = remaining[clock_index];
        const int budget = move_budget_ms(remaining_before, options);
        const chess::SearchLimits limits{
            .max_depth = 64,
            .move_time = std::chrono::milliseconds{budget},
        };
        const auto start = std::chrono::steady_clock::now();
        const chess::SearchResult result = use_candidate
            ? search_with_history(candidate, position, limits, history)
            : search_with_history(control, position, limits, history);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        remaining[clock_index] -= static_cast<int>(elapsed);

        const std::string selected_move = chess::move_to_string(result.best_move);
        PlySearch search{
            .game_ply = ply,
            .absolute_ply = static_cast<int>(opening.moves.size()) + ply,
            .candidate = use_candidate,
            .white = white,
            .move = selected_move,
            .depth = result.depth,
            .score = result.score,
            .nodes = result.nodes,
            .time_ms = elapsed,
            .budget_ms = budget,
            .remaining_before_ms = remaining_before,
            .remaining_after_ms = remaining[clock_index],
            .stopped = result.stopped,
            .applied = false,
        };
        if (use_candidate) {
            game.candidate_nodes += result.nodes;
            game.candidate_time_ms += elapsed;
            game.candidate_search_cycle_draws += search_cycle_draws(candidate);
        } else {
            game.control_nodes += result.nodes;
            game.control_time_ms += elapsed;
            game.control_search_cycle_draws += search_cycle_draws(control);
        }

        if (remaining[clock_index] < 0) {
            game.searches.push_back(std::move(search));
            game.result = white ? "0-1" : "1-0";
            game.reason = use_candidate ? "candidate_time" : "control_time";
            game.played_plies = ply;
            return game;
        }
        remaining[clock_index] += options.increment_ms;
        search.remaining_after_ms = remaining[clock_index];
        if (std::find(legal.begin(), legal.end(), result.best_move) == legal.end()) {
            game.searches.push_back(std::move(search));
            game.result = white ? "0-1" : "1-0";
            game.reason = use_candidate ? "candidate_illegal" : "control_illegal";
            game.played_plies = ply;
            return game;
        }

        search.applied = true;
        game.played_moves.push_back(selected_move);
        game.searches.push_back(std::move(search));
        const std::uint8_t rights_before = castling_rights(position);
        position.make_move(result.best_move);
        if (move_was_irreversible(rights_before, position)) history.clear();
        history.push_back(position.zobrist_key);
    }

    game.result = "1/2-1/2";
    game.reason = "max_plies";
    game.played_plies = options.max_plies;
    return game;
}

std::string candidate_outcome(const GameResult& game, bool candidate_white) {
    if (game.result == "1/2-1/2") return "draw";
    const bool white_won = game.result == "1-0";
    return white_won == candidate_white ? "win" : "loss";
}

std::uint64_t fnv1a_update(
    std::uint64_t hash,
    const char* data,
    std::size_t size
) {
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= static_cast<unsigned char>(data[index]);
        hash *= 1'099'511'628'211ULL;
    }
    return hash;
}

std::string fnv1a_hex(std::string_view value) {
    const std::uint64_t hash = fnv1a_update(
        14'695'981'039'346'656'037ULL, value.data(), value.size());
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0') << hash;
    return output.str();
}

struct FileFingerprint {
    std::string path;
    std::uintmax_t size = 0;
    std::string fnv1a64;
};

std::filesystem::path resolve_executable(std::string_view argv0) {
    std::filesystem::path candidate{std::string(argv0)};
    std::error_code error;
    if (candidate.has_parent_path()) {
        const auto result = std::filesystem::canonical(candidate, error);
        if (!error) return result;
    } else if (const char* path = std::getenv("PATH")) {
        std::istringstream entries(path);
        std::string directory;
        while (std::getline(entries, directory, ':')) {
            if (directory.empty()) directory = ".";
            const auto resolved = std::filesystem::canonical(
                std::filesystem::path(directory) / candidate, error);
            if (!error) return resolved;
            error.clear();
        }
    }
    throw std::runtime_error(
        "cannot resolve match executable: " + std::string(argv0));
}

FileFingerprint fingerprint_file(const std::filesystem::path& input_path) {
    std::error_code error;
    const std::filesystem::path path =
        std::filesystem::canonical(input_path, error);
    if (error) {
        throw std::runtime_error(
            "cannot resolve manifest input: " + input_path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "cannot fingerprint manifest input: " + path.string());
    }
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    std::uintmax_t size = 0;
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count <= 0) break;
        hash = fnv1a_update(hash, buffer.data(), static_cast<std::size_t>(count));
        size += static_cast<std::uintmax_t>(count);
    }
    if (!input.eof()) throw std::runtime_error("failed to fingerprint input");
    std::ostringstream encoded;
    encoded << std::hex << std::setw(16) << std::setfill('0') << hash;
    return {path.string(), size, encoded.str()};
}

void append_file_fingerprint_json(
    std::ostream& output,
    const FileFingerprint& fingerprint
) {
    output << "{\"path\":";
    append_json_string(output, fingerprint.path);
    output << ",\"size\":" << fingerprint.size
           << ",\"fnv1a64\":\"" << fingerprint.fnv1a64 << "\"}";
}

template <typename Config>
void append_selective_config_json(std::ostream& output, const Config& config) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\"enable_lmr\":" << (config.enable_lmr ? "true" : "false")
           << ",\"lmr_base\":" << config.lmr_base
           << ",\"lmr_divisor\":" << config.lmr_divisor
           << ",\"lmr_min_depth\":" << config.lmr_min_depth
           << ",\"lmr_min_move_index\":" << config.lmr_min_move_index
           << ",\"enable_null_move\":"
           << (config.enable_null_move ? "true" : "false")
           << ",\"null_move_min_depth\":" << config.null_move_min_depth
           << ",\"null_move_reduction\":" << config.null_move_reduction
           << ",\"enable_reverse_futility\":"
           << (config.enable_reverse_futility ? "true" : "false")
           << ",\"reverse_futility_max_depth\":"
           << config.reverse_futility_max_depth
           << ",\"reverse_futility_base_margin\":"
           << config.reverse_futility_base_margin
           << ",\"reverse_futility_margin_per_depth\":"
           << config.reverse_futility_margin_per_depth
           << ",\"enable_late_move_pruning\":"
           << (config.enable_late_move_pruning ? "true" : "false")
           << ",\"late_move_pruning_max_depth\":"
           << config.late_move_pruning_max_depth
           << ",\"late_move_pruning_base\":" << config.late_move_pruning_base
           << ",\"late_move_pruning_depth_multiplier\":"
           << config.late_move_pruning_depth_multiplier
           << ",\"enable_qsearch_see_pruning\":"
           << (config.enable_qsearch_see_pruning ? "true" : "false")
           << ",\"qsearch_see_threshold\":" << config.qsearch_see_threshold
           << ",\"enable_main_search_see_pruning\":"
           << (config.enable_main_search_see_pruning ? "true" : "false")
           << ",\"main_search_see_max_depth\":"
           << config.main_search_see_max_depth
           << ",\"main_search_see_margin_per_depth\":"
           << config.main_search_see_margin_per_depth << '}';
}

template <typename Config>
void append_aspiration_config_json(std::ostream& output, const Config& config) {
    output << "{\"enabled\":" << (config.enabled ? "true" : "false")
           << ",\"min_depth\":" << config.min_depth
           << ",\"delta_base_cp\":" << config.delta_base_cp
           << ",\"delta_divisor\":" << config.delta_divisor
           << ",\"expansion_factor_per_mille\":"
           << config.expansion_factor_per_mille
           << ",\"max_fail_high_reductions\":"
           << config.max_fail_high_reductions
           << ",\"mean_score_new_weight_per_mille\":"
           << config.mean_score_new_weight_per_mille
           << ",\"max_researches\":" << config.max_researches
           << ",\"mean_score_clamp_cp\":" << config.mean_score_clamp_cp
           << '}';
}

struct MatchManifest {
    std::string json;
    std::string run_fingerprint;
};

MatchManifest make_manifest(const Options& options, std::string_view argv0) {
    const auto selective =
        production_selective_config<CandidateSearcher::SelectiveConfig>();
    const auto aspiration =
        production_aspiration_config<CandidateSearcher::AspirationConfig>();
    std::ostringstream identity;
    identity << "{\"binary\":";
    append_file_fingerprint_json(
        identity, fingerprint_file(resolve_executable(argv0)));
    identity << ",\"book\":";
    append_file_fingerprint_json(identity, fingerprint_file(options.book));
    identity << ",\"model\":";
    append_file_fingerprint_json(identity, fingerprint_file(options.model));
    identity << ",\"candidate\":{\"name\":\"" << CandidateName
             << "\",\"engine\":\"nnue_clear_each_search_v44\","
             << "\"clear_tt_each_top_level_search\":true,"
             << "\"generation_fields\":false},"
             << "\"control\":{\"name\":\"" << ControlName
             << "\",\"engine\":\"nnue_single_bound_v43\","
             << "\"clear_tt_each_top_level_search\":false,"
             << "\"generation_policy\":\"current_only_scores_stale_move_hint\"},"
             << "\"combined_config_hash\":\"" << CombinedConfigHash << "\","
             << "\"selective_config\":";
    append_selective_config_json(identity, selective);
    identity << ",\"aspiration_config\":";
    append_aspiration_config_json(identity, aspiration);
    identity << ",\"common_policy\":{"
             << "\"twofold_search_draw\":true,"
             << "\"reuse_deeper_tt_scores\":false,"
             << "\"tt_mb\":" << options.tt_mb << "},"
             << "\"match_options\":{"
             << "\"openings\":" << options.openings
             << ",\"games\":" << options.openings * 2
             << ",\"base_ms\":" << options.base_ms
             << ",\"increment_ms\":" << options.increment_ms
             << ",\"overhead_ms\":" << options.overhead_ms
             << ",\"max_plies\":" << options.max_plies
             << ",\"seed\":" << options.seed
             << ",\"paired_color_reversed\":true,"
             << "\"alternating_first_color\":true,"
             << "\"per_ply_telemetry\":true}}";
    const std::string identity_json = identity.str();
    const std::string fingerprint = fnv1a_hex(identity_json);
    std::ostringstream manifest;
    manifest << "{\"kind\":\"nnue_v43_v44_match_manifest\","
             << "\"schema_version\":1,\"run_fingerprint\":\""
             << fingerprint << "\",\"identity\":" << identity_json << '}';
    return {manifest.str(), fingerprint};
}

void ensure_manifest(const std::string& output_path, const MatchManifest& expected) {
    const std::filesystem::path games_path(output_path);
    const std::filesystem::path manifest_path(output_path + ".manifest.json");
    std::error_code error;
    const bool games_nonempty = std::filesystem::exists(games_path, error)
        && std::filesystem::file_size(games_path, error) != 0;
    if (error) throw std::runtime_error("cannot inspect match output");
    std::ifstream existing(manifest_path, std::ios::binary);
    if (existing) {
        const std::string actual{
            std::istreambuf_iterator<char>(existing),
            std::istreambuf_iterator<char>()};
        if (actual != expected.json && actual != expected.json + "\n") {
            throw std::runtime_error(
                "match manifest mismatch; use a new --output path");
        }
        return;
    }
    if (games_nonempty) {
        throw std::runtime_error(
            "refusing to resume nonempty output without a manifest");
    }
    std::ofstream created(manifest_path, std::ios::binary | std::ios::trunc);
    if (!created) throw std::runtime_error("failed to create match manifest");
    created << expected.json << '\n';
    created.flush();
    if (!created) throw std::runtime_error("failed to write match manifest");
}

std::set<std::string> completed_games(
    const std::string& path,
    std::string_view run_fingerprint
) {
    std::ifstream input(path);
    std::set<std::string> keys;
    std::string line;
    const std::string fingerprint_marker =
        "\"run_fingerprint\":\"" + std::string(run_fingerprint) + "\"";
    for (int line_number = 1; std::getline(input, line); ++line_number) {
        if (line.empty()) continue;
        if (line.find(fingerprint_marker) == std::string::npos) {
            throw std::runtime_error(
                "output fingerprint mismatch at line "
                + std::to_string(line_number));
        }
        if (line.find("\"kind\":\"game\"") == std::string::npos
            || line.find("\"candidate\":\"v44_clear_each_search\"")
                == std::string::npos
            || line.find("\"control\":\"v43_production\"")
                == std::string::npos
            || line.find("\"searches\":[") == std::string::npos) {
            throw std::runtime_error(
                "output policy/telemetry mismatch at line "
                + std::to_string(line_number));
        }
        constexpr std::string_view marker = "\"key\":\"";
        const std::size_t start = line.find(marker);
        const std::size_t value_start = start == std::string::npos
            ? start : start + marker.size();
        const std::size_t end = start == std::string::npos
            ? start : line.find('"', value_start);
        if (start == std::string::npos || end == std::string::npos) {
            throw std::runtime_error(
                "malformed output record at line "
                + std::to_string(line_number));
        }
        const std::string key = line.substr(value_start, end - value_start);
        if (!keys.insert(key).second) {
            throw std::runtime_error(
                "duplicate output key at line " + std::to_string(line_number));
        }
    }
    return keys;
}

void append_string_array(
    std::ostream& output,
    std::span<const std::string> values
) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        append_json_string(output, values[index]);
    }
    output << ']';
}

void append_searches(std::ostream& output, std::span<const PlySearch> searches) {
    output << '[';
    for (std::size_t index = 0; index < searches.size(); ++index) {
        if (index != 0) output << ',';
        const PlySearch& search = searches[index];
        output << "{\"game_ply\":" << search.game_ply
               << ",\"absolute_ply\":" << search.absolute_ply
               << ",\"engine\":\""
               << (search.candidate ? CandidateName : ControlName)
               << "\",\"color\":\"" << (search.white ? "white" : "black")
               << "\",\"move\":";
        append_json_string(output, search.move);
        output << ",\"depth\":" << search.depth
               << ",\"score\":" << search.score
               << ",\"nodes\":" << search.nodes
               << ",\"time_ms\":" << search.time_ms
               << ",\"budget_ms\":" << search.budget_ms
               << ",\"remaining_before_ms\":" << search.remaining_before_ms
               << ",\"remaining_after_ms\":" << search.remaining_after_ms
               << ",\"stopped\":" << (search.stopped ? "true" : "false")
               << ",\"applied\":" << (search.applied ? "true" : "false")
               << '}';
    }
    output << ']';
}

void append_game(
    std::ofstream& output,
    std::string_view run_fingerprint,
    std::string_view key,
    std::string_view logical_key,
    const Opening& opening,
    bool candidate_white,
    const GameResult& game
) {
    output << "{\"kind\":\"game\",\"schema_version\":1,"
           << "\"run_fingerprint\":\"" << run_fingerprint << "\",\"key\":";
    append_json_string(output, key);
    output << ",\"logical_key\":";
    append_json_string(output, logical_key);
    output << ",\"candidate\":\"" << CandidateName
           << "\",\"control\":\"" << ControlName
           << "\",\"combined_config_hash\":\"" << CombinedConfigHash
           << "\",\"opening_source_line\":" << opening.source_line
           << ",\"opening\":";
    append_json_string(output, opening.text);
    output << ",\"opening_moves\":";
    append_string_array(output, opening.moves);
    output << ",\"candidate_color\":\""
           << (candidate_white ? "white" : "black")
           << "\",\"result\":";
    append_json_string(output, game.result);
    output << ",\"candidate_outcome\":";
    append_json_string(output, candidate_outcome(game, candidate_white));
    output << ",\"reason\":";
    append_json_string(output, game.reason);
    output << ",\"played_plies\":" << game.played_plies
           << ",\"candidate_nodes\":" << game.candidate_nodes
           << ",\"control_nodes\":" << game.control_nodes
           << ",\"candidate_time_ms\":" << game.candidate_time_ms
           << ",\"control_time_ms\":" << game.control_time_ms
           << ",\"candidate_search_cycle_draws\":"
           << game.candidate_search_cycle_draws
           << ",\"control_search_cycle_draws\":"
           << game.control_search_cycle_draws
           << ",\"played_moves\":";
    append_string_array(output, game.played_moves);
    output << ",\"searches\":";
    append_searches(output, game.searches);
    output << "}\n";
    output.flush();
    if (!output) throw std::runtime_error("failed to append game record");
}

void print_summary(
    const std::set<std::string>& completed,
    int expected,
    int newly_completed,
    bool interrupted,
    std::string_view stop_reason
) {
    std::cout << "{\"kind\":\"nnue_v43_v44_match_process_summary\","
              << "\"completed_games\":" << completed.size()
              << ",\"expected_games\":" << expected
              << ",\"new_games\":" << newly_completed
              << ",\"interrupted\":" << (interrupted ? "true" : "false")
              << ",\"stopped_early\":"
              << (completed.size() < static_cast<std::size_t>(expected)
                      ? "true" : "false")
              << ",\"stop_reason\":";
    if (stop_reason.empty()) {
        std::cout << "null";
    } else {
        append_json_string(std::cout, stop_reason);
    }
    std::cout << "}\n";
    std::cout.flush();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Opening> openings = load_openings(options);
        const MatchManifest manifest = make_manifest(options, argv[0]);
        ensure_manifest(options.output, manifest);
        std::set<std::string> completed =
            completed_games(options.output, manifest.run_fingerprint);
        std::set<std::string> expected_keys;
        for (std::size_t opening_index = 0;
             opening_index < openings.size(); ++opening_index) {
            for (const char color : {'w', 'b'}) {
                expected_keys.insert(
                    manifest.run_fingerprint + ":opening:"
                    + std::to_string(opening_index) + ":" + color);
            }
        }
        if (!std::includes(
                expected_keys.begin(), expected_keys.end(),
                completed.begin(), completed.end())) {
            throw std::runtime_error(
                "saved output contains a game outside the immutable protocol");
        }

        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model");
        }
        model.set_neon_dotprod_enabled(true);
        ControlSearcher control = make_engine<ControlSearcher>(model, options.tt_mb);
        CandidateSearcher candidate =
            make_engine<CandidateSearcher>(model, options.tt_mb);
        if (control.name() != "nnue_single_bound_v43"
            || candidate.name() != "nnue_clear_each_search_v44") {
            throw std::runtime_error("unexpected engine identity");
        }

        std::signal(SIGINT, handle_stop_signal);
        std::signal(SIGTERM, handle_stop_signal);
        std::ofstream output(options.output, std::ios::app);
        if (!output) throw std::runtime_error("failed to open match output");

        int newly_completed = 0;
        for (std::size_t opening_index = 0;
             opening_index < openings.size(); ++opening_index) {
            const std::array<bool, 2> colors = opening_index % 2 == 0
                ? std::array<bool, 2>{true, false}
                : std::array<bool, 2>{false, true};
            for (const bool candidate_white : colors) {
                if (stop_requested != 0) {
                    print_summary(
                        completed, options.openings * 2,
                        newly_completed, true, "signal");
                    return 0;
                }
                const std::string logical_key =
                    "opening:" + std::to_string(opening_index) + ":"
                    + (candidate_white ? "w" : "b");
                const std::string key =
                    manifest.run_fingerprint + ":" + logical_key;
                if (completed.contains(key)) continue;

                const GameResult game = play_game(
                    openings[opening_index], candidate_white,
                    options, control, candidate);
                if (game.aborted) {
                    print_summary(
                        completed, options.openings * 2,
                        newly_completed, true, "signal");
                    return 0;
                }
                append_game(
                    output, manifest.run_fingerprint, key, logical_key,
                    openings[opening_index], candidate_white, game);
                completed.insert(key);
                ++newly_completed;
                if (newly_completed % 10 == 0) {
                    std::cerr << "progress completed=" << completed.size()
                              << '/' << options.openings * 2
                              << " new_games=" << newly_completed << '\n';
                    std::cerr.flush();
                }
                if (options.stop_after_games > 0
                    && newly_completed >= options.stop_after_games) {
                    print_summary(
                        completed, options.openings * 2,
                        newly_completed, false, "stop_after_games");
                    return 0;
                }
            }
        }
        print_summary(
            completed, options.openings * 2, newly_completed, false, {});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
