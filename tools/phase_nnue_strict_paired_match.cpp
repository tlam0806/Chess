#include "attacks.hpp"
#include "game_state.hpp"
#include "heuristic_searcher_v35.hpp"
#include "move.hpp"
#include "nnue_searcher_v36.hpp"
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct ModelSpec {
    std::string label;
    std::string path;
};

struct BookLine {
    int line_number = 0;
    std::string text;
    std::vector<std::string> moves;
};

struct Options {
    std::string book_path = "data/opening_book_6plies.txt";
    std::vector<ModelSpec> models;
    std::optional<ModelSpec> opponent_model;
    std::vector<std::string> resume_logs;
    int games = 10;
    int candidate_depth = 4;
    int heuristic_depth = 4;
    int max_plies = 200;
    std::size_t tt_mb = 64;
    std::uint32_t seed = 20260721;
    bool stop_on_ci = false;
    int min_pairs = 30;
    int check_every_pairs = 5;
    int bootstrap_samples = 100000;
    bool generated_openings = false;
    int generated_extra_plies = 2;
};

struct Score {
    int wins = 0;
    int losses = 0;
    int draws = 0;
};

struct GameResult {
    std::string result;
    std::string reason;
    int played_plies = 0;
    std::uint64_t candidate_nodes = 0;
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
        throw std::runtime_error(
            "invalid integer for " + std::string(name) + ": "
            + std::string(value));
    }
}

ModelSpec parse_model(std::string_view value) {
    const std::size_t separator = value.find('=');
    if (separator == std::string_view::npos
        || separator == 0
        || separator + 1 == value.size()) {
        throw std::runtime_error("--model must be label=path");
    }
    return ModelSpec{
        std::string(value.substr(0, separator)),
        std::string(value.substr(separator + 1)),
    };
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        auto require_value = [&](std::string_view name) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::runtime_error(
                    "missing value for " + std::string(name));
            }
            return argv[++index];
        };
        if (arg == "--book") {
            options.book_path = std::string(require_value(arg));
        } else if (arg == "--model") {
            options.models.push_back(parse_model(require_value(arg)));
        } else if (arg == "--opponent-model") {
            options.opponent_model = parse_model(require_value(arg));
        } else if (arg == "--resume-log") {
            options.resume_logs.emplace_back(require_value(arg));
        } else if (arg == "--games") {
            options.games = parse_int(require_value(arg), arg);
        } else if (arg == "--depth") {
            const int depth = parse_int(require_value(arg), arg);
            options.candidate_depth = depth;
            options.heuristic_depth = depth;
        } else if (arg == "--candidate-depth") {
            options.candidate_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--heuristic-depth") {
            options.heuristic_depth = parse_int(require_value(arg), arg);
        } else if (arg == "--max-plies") {
            options.max_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--tt-mb") {
            const int value = parse_int(require_value(arg), arg);
            if (value <= 0) {
                throw std::runtime_error("--tt-mb must be positive");
            }
            options.tt_mb = static_cast<std::size_t>(value);
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(
                parse_int(require_value(arg), arg));
        } else if (arg == "--stop-on-ci") {
            options.stop_on_ci = true;
        } else if (arg == "--min-pairs") {
            options.min_pairs = parse_int(require_value(arg), arg);
        } else if (arg == "--check-every-pairs") {
            options.check_every_pairs = parse_int(require_value(arg), arg);
        } else if (arg == "--bootstrap-samples") {
            options.bootstrap_samples = parse_int(require_value(arg), arg);
        } else if (arg == "--generated-openings") {
            options.generated_openings = true;
        } else if (arg == "--generated-extra-plies") {
            options.generated_extra_plies = parse_int(require_value(arg), arg);
        } else if (arg == "--help") {
            std::cout
                << "Usage: phase_nnue_strict_paired_match\n"
                << "  --model label=path [--model label=path ...]\n"
                << "  [--book path] [--games 10] [--depth 4]\n"
                << "  [--candidate-depth 4] [--heuristic-depth 6]\n"
                << "  [--opponent-model label=path]\n"
                << "  [--stop-on-ci] [--min-pairs 30]\n"
                << "  [--check-every-pairs 5] [--bootstrap-samples 100000]\n"
                << "  [--generated-openings] [--generated-extra-plies 2]\n"
                << "  [--resume-log path]\n"
                << "  [--max-plies 200] [--tt-mb 64] [--seed N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.models.empty()) {
        throw std::runtime_error("at least one --model is required");
    }
    if (options.games <= 0 || options.games % 2 != 0) {
        throw std::runtime_error("--games must be a positive even number");
    }
    if (options.candidate_depth <= 0 || options.heuristic_depth <= 0) {
        throw std::runtime_error("search depths must be positive");
    }
    if (options.max_plies <= 0) {
        throw std::runtime_error("--max-plies must be positive");
    }
    if (options.min_pairs <= 0 || options.check_every_pairs <= 0
        || options.bootstrap_samples <= 0
        || options.generated_extra_plies <= 0) {
        throw std::runtime_error("CI parameters must be positive");
    }
    return options;
}

std::string strip_comment(std::string line) {
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) {
        line.resize(comment);
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

chess::Move find_legal_uci_move(
    const chess::Position& pos,
    std::string_view uci
) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == uci) {
            return move;
        }
    }
    throw std::runtime_error("illegal book move: " + std::string(uci));
}

std::vector<BookLine> load_book(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open book: " + path);
    }
    std::vector<BookLine> lines;
    std::set<chess::HashKey> unique_positions;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string clean = strip_comment(line);
        std::vector<std::string> moves = split_words(clean);
        if (moves.empty()) {
            continue;
        }
        chess::Position pos;
        pos.set_startpos();
        for (const std::string& move_text : moves) {
            pos.make_move(find_legal_uci_move(pos, move_text));
        }
        if (!unique_positions.insert(pos.zobrist_key).second) {
            continue;
        }
        lines.push_back(BookLine{line_number, clean, std::move(moves)});
    }
    return lines;
}

chess::Position play_book(
    const BookLine& line,
    std::vector<chess::HashKey>& hashes
) {
    chess::Position pos;
    pos.set_startpos();
    hashes.clear();
    hashes.push_back(pos.zobrist_key);
    for (const std::string& move_text : line.moves) {
        pos.make_move(find_legal_uci_move(pos, move_text));
        hashes.push_back(pos.zobrist_key);
    }
    return pos;
}

std::vector<BookLine> generate_opening_extensions(
    const std::vector<BookLine>& base_book,
    int count,
    std::uint32_t seed,
    int extra_plies
) {
    std::set<chess::HashKey> unique_positions;
    std::vector<chess::HashKey> hashes;
    for (const BookLine& line : base_book) {
        unique_positions.insert(play_book(line, hashes).zobrist_key);
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> base_pick(
        0, base_book.size() - 1);
    std::vector<BookLine> generated;
    generated.reserve(static_cast<std::size_t>(count));
    int attempts = 0;
    const int max_attempts = std::max(10000, count * 100);
    while (static_cast<int>(generated.size()) < count
           && attempts++ < max_attempts) {
        const BookLine& base = base_book[base_pick(rng)];
        chess::Position pos = play_book(base, hashes);
        std::vector<std::string> moves = base.moves;
        bool valid = true;
        for (int extra_ply = 0; extra_ply < extra_plies; ++extra_ply) {
            const std::vector<chess::Move> legal =
                chess::generate_legal_moves(pos);
            if (legal.empty()) {
                valid = false;
                break;
            }
            std::uniform_int_distribution<std::size_t> move_pick(
                0, legal.size() - 1);
            const chess::Move move = legal[move_pick(rng)];
            moves.push_back(chess::move_to_string(move));
            pos.make_move(move);
        }
        if (!valid || !unique_positions.insert(pos.zobrist_key).second) {
            continue;
        }
        std::ostringstream text;
        for (std::size_t index = 0; index < moves.size(); ++index) {
            if (index != 0) {
                text << ' ';
            }
            text << moves[index];
        }
        generated.push_back(BookLine{
            100000 + static_cast<int>(generated.size()),
            text.str(),
            std::move(moves),
        });
    }
    if (static_cast<int>(generated.size()) != count) {
        throw std::runtime_error("failed to generate enough unique openings");
    }
    return generated;
}

bool contains_move(const std::vector<chess::Move>& moves, chess::Move target) {
    return std::find(moves.begin(), moves.end(), target) != moves.end();
}

bool insufficient_material(const chess::Position& pos) {
    const auto count = [&](chess::Color color, chess::PieceType piece) {
        return chess::popcount(
            pos.pieces[static_cast<int>(color)][static_cast<int>(piece)]);
    };
    for (chess::Color color : {chess::Color::White, chess::Color::Black}) {
        if (count(color, chess::PieceType::Pawn) != 0
            || count(color, chess::PieceType::Rook) != 0
            || count(color, chess::PieceType::Queen) != 0) {
            return false;
        }
    }
    int minor_count = 0;
    for (chess::Color color : {chess::Color::White, chess::Color::Black}) {
        minor_count += count(color, chess::PieceType::Knight);
        minor_count += count(color, chess::PieceType::Bishop);
    }
    return minor_count <= 1;
}

std::string terminal_result(const chess::Position& pos, std::string& reason) {
    if (chess::in_check(pos, pos.side_to_move)) {
        reason = "checkmate";
        return pos.side_to_move == chess::Color::White ? "0-1" : "1-0";
    }
    reason = "stalemate";
    return "1/2-1/2";
}

GameResult play_game(
    const BookLine& opening,
    bool candidate_is_white,
    int candidate_depth,
    int heuristic_depth,
    int max_plies,
    std::size_t tt_mb,
    const chess::PhaseQuantizedNnueModel& model,
    const chess::PhaseQuantizedNnueModel* opponent_model
) {
    chess::NnueSearcherV36 candidate(model, tt_mb, 4);
    std::optional<chess::NnueSearcherV36> opponent;
    std::optional<chess::HeuristicSearcherV35> heuristic;
    if (opponent_model != nullptr) {
        opponent.emplace(*opponent_model, tt_mb, 4);
    } else {
        heuristic.emplace(tt_mb, 4);
    }
    std::vector<chess::HashKey> hashes;
    chess::Position pos = play_book(opening, hashes);
    GameResult game;

    for (int ply = 0; ply < max_plies; ++ply) {
        if (chess::is_threefold_repetition(pos.zobrist_key, hashes)) {
            game.result = "1/2-1/2";
            game.reason = "threefold";
            game.played_plies = ply;
            return game;
        }
        if (pos.halfmove_clock >= 100) {
            game.result = "1/2-1/2";
            game.reason = "fifty-move";
            game.played_plies = ply;
            return game;
        }
        if (insufficient_material(pos)) {
            game.result = "1/2-1/2";
            game.reason = "insufficient-material";
            game.played_plies = ply;
            return game;
        }

        const std::vector<chess::Move> legal_moves =
            chess::generate_legal_moves(pos);
        if (legal_moves.empty()) {
            game.result = terminal_result(pos, game.reason);
            game.played_plies = ply;
            return game;
        }

        const bool candidate_to_move =
            (pos.side_to_move == chess::Color::White) == candidate_is_white;
        chess::SearchResult search;
        if (candidate_to_move) {
            search = candidate.search_best_move(pos, candidate_depth);
        } else if (opponent.has_value()) {
            search = opponent->search_best_move(pos, heuristic_depth);
        } else {
            search = heuristic->search_best_move(pos, heuristic_depth);
        }
        if (!contains_move(legal_moves, search.best_move)) {
            throw std::runtime_error(
                std::string(candidate_to_move ? "candidate " : "opponent ")
                + std::string(candidate_to_move
                    ? candidate.name()
                    : opponent.has_value() ? opponent->name() : heuristic->name())
                + " returned illegal move "
                + chess::move_to_string(search.best_move)
                + " at played_ply=" + std::to_string(ply));
        }
        if (candidate_to_move) {
            game.candidate_nodes += search.nodes;
        } else {
            game.heuristic_nodes += search.nodes;
        }
        pos.make_move(search.best_move);
        hashes.push_back(pos.zobrist_key);
    }

    game.result = "1/2-1/2";
    game.reason = "max-plies";
    game.played_plies = max_plies;
    return game;
}

struct ConfidenceInterval {
    double lower = 0.0;
    double upper = 0.0;
};

ConfidenceInterval paired_bootstrap_ci(
    const std::vector<double>& pair_points,
    int samples,
    std::uint32_t seed
) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(
        0, pair_points.size() - 1);
    std::vector<double> rates;
    rates.reserve(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample) {
        double points = 0.0;
        for (std::size_t index = 0; index < pair_points.size(); ++index) {
            points += pair_points[pick(rng)];
        }
        rates.push_back(points / (2.0 * pair_points.size()));
    }
    std::sort(rates.begin(), rates.end());
    const std::size_t lower_index = static_cast<std::size_t>(
        0.025 * static_cast<double>(rates.size()));
    const std::size_t upper_index = std::min(
        rates.size() - 1,
        static_cast<std::size_t>(0.975 * static_cast<double>(rates.size())));
    return ConfidenceInterval{rates[lower_index], rates[upper_index]};
}

void update_score(Score& score, const GameResult& game, bool candidate_is_white) {
    if (game.result == "1/2-1/2") {
        ++score.draws;
        return;
    }
    const bool white_won = game.result == "1-0";
    if (white_won == candidate_is_white) {
        ++score.wins;
    } else {
        ++score.losses;
    }
}

struct ResumeStats {
    Score score;
    std::vector<double> pair_points;
};

std::string log_field(const std::string& line, std::string_view key) {
    const std::size_t found = line.find(key);
    if (found == std::string::npos) {
        return {};
    }
    const std::size_t begin = found + key.size();
    const std::size_t end = line.find(' ', begin);
    return line.substr(begin, end == std::string::npos ? end : end - begin);
}

ResumeStats load_resume_stats(
    const std::vector<std::string>& paths,
    const std::string& model_label
) {
    ResumeStats combined;
    for (const std::string& path : paths) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to open resume log: " + path);
        }
        struct Record {
            bool candidate_is_white = false;
            std::string result;
        };
        std::map<int, std::vector<Record>> pairs;
        std::string line;
        const std::string prefix = "game model=" + model_label + " ";
        while (std::getline(input, line)) {
            if (line.rfind(prefix, 0) != 0) {
                continue;
            }
            const std::string pair_text = log_field(line, " pair=");
            const std::string color = log_field(line, " candidate_color=");
            const std::string result = log_field(line, " result=");
            if (pair_text.empty() || color.empty() || result.empty()) {
                throw std::runtime_error("malformed game in resume log: " + path);
            }
            pairs[parse_int(pair_text, "resume pair")].push_back(
                Record{color == "white", result});
        }
        for (const auto& [pair, records] : pairs) {
            (void) pair;
            if (records.size() != 2) {
                continue;
            }
            const double before =
                combined.score.wins + 0.5 * combined.score.draws;
            for (const Record& record : records) {
                GameResult game;
                game.result = record.result;
                update_score(
                    combined.score, game, record.candidate_is_white);
            }
            const double after =
                combined.score.wins + 0.5 * combined.score.draws;
            combined.pair_points.push_back(after - before);
        }
    }
    return combined;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<BookLine> base_book = load_book(options.book_path);
        const int pair_count = options.games / 2;
        const std::vector<BookLine> book = options.generated_openings
            ? generate_opening_extensions(
                base_book,
                pair_count,
                options.seed,
                options.generated_extra_plies)
            : base_book;
        if (static_cast<int>(book.size()) < pair_count) {
            throw std::runtime_error("opening book has too few lines");
        }

        std::vector<std::size_t> opening_indices(book.size());
        std::iota(opening_indices.begin(), opening_indices.end(), 0);
        std::mt19937 rng(options.seed);
        std::shuffle(opening_indices.begin(), opening_indices.end(), rng);
        opening_indices.resize(static_cast<std::size_t>(pair_count));

        std::optional<chess::PhaseQuantizedNnueModel> opponent_model;
        if (options.opponent_model.has_value()) {
            opponent_model.emplace();
            if (!opponent_model->load(options.opponent_model->path)) {
                throw std::runtime_error(
                    "failed to load opponent model: "
                    + options.opponent_model->path);
            }
        }

        std::cout << "match_settings candidate_depth=" << options.candidate_depth
                  << " heuristic_depth=" << options.heuristic_depth
                  << " fixed_depth=1 strict=1 games_per_model=" << options.games
                  << " opening_plies=" << book.front().moves.size()
                  << " opening_pairs=" << pair_count
                  << " seed=" << options.seed
                  << " max_plies=" << options.max_plies
                  << " opponent="
                  << (options.opponent_model.has_value()
                        ? options.opponent_model->label : "heuristic_v35")
                  << " stop_on_ci=" << options.stop_on_ci << '\n';
        for (int pair = 0; pair < pair_count; ++pair) {
            const BookLine& line = book[opening_indices[static_cast<std::size_t>(pair)]];
            std::cout << "opening pair=" << pair
                      << " line=" << line.line_number
                      << " moves=\"" << line.text << "\"\n";
        }

        for (const ModelSpec& spec : options.models) {
            chess::PhaseQuantizedNnueModel model;
            if (!model.load(spec.path)) {
                throw std::runtime_error("failed to load model: " + spec.path);
            }
            Score score;
            std::uint64_t total_candidate_nodes = 0;
            std::uint64_t total_heuristic_nodes = 0;
            const ResumeStats resume = load_resume_stats(
                options.resume_logs, spec.label);
            score = resume.score;
            std::vector<double> pair_points = resume.pair_points;
            if (!pair_points.empty()) {
                std::cout << "resume model=" << spec.label
                          << " pairs=" << pair_points.size()
                          << " games=" << 2 * pair_points.size()
                          << " W=" << score.wins
                          << " D=" << score.draws
                          << " L=" << score.losses << '\n';
            }
            bool ci_decision = false;
            for (int pair = 0; pair < pair_count; ++pair) {
                const BookLine& line =
                    book[opening_indices[static_cast<std::size_t>(pair)]];
                const double points_before = score.wins + 0.5 * score.draws;
                for (int side = 0; side < 2; ++side) {
                    const bool candidate_is_white = side == 0;
                    const GameResult game = play_game(
                        line,
                        candidate_is_white,
                        options.candidate_depth,
                        options.heuristic_depth,
                        options.max_plies,
                        options.tt_mb,
                        model,
                        opponent_model.has_value() ? &*opponent_model : nullptr);
                    update_score(score, game, candidate_is_white);
                    total_candidate_nodes += game.candidate_nodes;
                    total_heuristic_nodes += game.heuristic_nodes;
                    std::cout << "game model=" << spec.label
                              << " pair=" << pair
                              << " opening_line=" << line.line_number
                              << " candidate_color="
                              << (candidate_is_white ? "white" : "black")
                              << " result=" << game.result
                              << " reason=" << game.reason
                              << " played_plies=" << game.played_plies
                              << " candidate_nodes=" << game.candidate_nodes
                              << " heuristic_nodes=" << game.heuristic_nodes
                              << '\n';
                }
                const double points_after = score.wins + 0.5 * score.draws;
                pair_points.push_back(points_after - points_before);
                const int completed_pairs =
                    static_cast<int>(pair_points.size());
                if (options.stop_on_ci
                    && completed_pairs >= options.min_pairs
                    && completed_pairs % options.check_every_pairs == 0) {
                    const ConfidenceInterval ci = paired_bootstrap_ci(
                        pair_points,
                        options.bootstrap_samples,
                        options.seed ^ static_cast<std::uint32_t>(completed_pairs));
                    const double rate = points_after / (2.0 * completed_pairs);
                    std::cout << "ci model=" << spec.label
                              << " pairs=" << completed_pairs
                              << " games=" << 2 * completed_pairs
                              << " score_rate=" << rate
                              << " bootstrap95_lower=" << ci.lower
                              << " bootstrap95_upper=" << ci.upper << '\n';
                    if (ci.lower > 0.5 || ci.upper < 0.5) {
                        ci_decision = true;
                        const std::string winner = ci.lower > 0.5
                            ? spec.label
                            : options.opponent_model.has_value()
                                ? options.opponent_model->label
                                : "heuristic_v35";
                        std::cout << "decision model=" << spec.label
                                  << " winner="
                                  << winner
                                  << " pairs=" << completed_pairs
                                  << " games=" << 2 * completed_pairs << '\n';
                        break;
                    }
                }
            }
            const double points = score.wins + 0.5 * score.draws;
            const int completed_games = 2 * static_cast<int>(pair_points.size());
            std::cout << "summary model=" << spec.label
                      << " W=" << score.wins
                      << " D=" << score.draws
                      << " L=" << score.losses
                      << " points=" << points
                      << " score_rate=" << points / completed_games
                      << " games=" << completed_games
                      << " ci_decision=" << ci_decision
                      << " candidate_nodes=" << total_candidate_nodes
                      << " heuristic_nodes=" << total_heuristic_nodes
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "phase_nnue_strict_paired_match: " << error.what() << '\n';
        return 1;
    }
}
