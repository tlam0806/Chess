#include "board_encoder.hpp"
#include "heuristic_searcher_v25.hpp"
#include "move.hpp"
#include "position.hpp"
#include "search_types.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string input = "data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl";
    std::uint64_t line = 1;
    int depth = 7;
    bool instrument_ordering = false;
    bool instrument_time = false;
    bool describe_only = false;
    bool fixed_depth = false;
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

std::uint64_t parse_u64(std::string_view value, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const auto result = static_cast<std::uint64_t>(std::stoull(std::string(value), &parsed));
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

        if (arg == "--input") {
            options.input = std::string(require_value(arg));
        } else if (arg == "--line") {
            options.line = parse_u64(require_value(arg), arg);
        } else if (arg == "--depth") {
            options.depth = parse_int(require_value(arg), arg);
        } else if (arg == "--instrument-ordering") {
            options.instrument_ordering = true;
        } else if (arg == "--instrument-time") {
            options.instrument_time = true;
        } else if (arg == "--describe-only") {
            options.describe_only = true;
        } else if (arg == "--fixed-depth") {
            options.fixed_depth = true;
        } else if (arg == "--help") {
            std::cout << "Usage: profile_v25_line [--input path] --line N [--depth D] [--instrument-ordering] [--instrument-time] [--describe-only] [--fixed-depth]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.line == 0 || options.depth < 0) {
        throw std::runtime_error("line/depth arguments are invalid");
    }
    return options;
}

std::vector<int> extract_int_array(const std::string& line, std::string_view key) {
    const std::size_t key_pos = line.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("sample is missing " + std::string(key));
    }

    const std::size_t open = line.find('[', key_pos);
    const std::size_t close = line.find(']', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("invalid array for " + std::string(key));
    }

    std::vector<int> values;
    std::size_t pos = open + 1;
    while (pos < close) {
        while (pos < close && (line[pos] == ' ' || line[pos] == ',')) {
            ++pos;
        }
        if (pos >= close) {
            break;
        }
        std::size_t end = pos;
        while (end < close && line[end] != ',') {
            ++end;
        }
        values.push_back(parse_int(std::string_view(line).substr(pos, end - pos), std::string(key)));
        pos = end + 1;
    }
    return values;
}

bool decode_sample(const std::string& line, chess::Position& pos) {
    const std::vector<int> aux_values = extract_int_array(line, "\"aux\"");
    if (aux_values.size() != chess::AuxFeatureCount) {
        throw std::runtime_error("unexpected aux feature count");
    }
    if (aux_values[chess::HasEnPassant] != 0) {
        return false;
    }

    pos.clear();
    pos.side_to_move = chess::Color::White;
    pos.white_can_castle_kingside = aux_values[chess::FriendlyCanCastleKingside] != 0;
    pos.white_can_castle_queenside = aux_values[chess::FriendlyCanCastleQueenside] != 0;
    pos.black_can_castle_kingside = aux_values[chess::EnemyCanCastleKingside] != 0;
    pos.black_can_castle_queenside = aux_values[chess::EnemyCanCastleQueenside] != 0;

    std::array<bool, chess::EncodedFeatureCount> seen{};
    const std::vector<int> features = extract_int_array(line, "\"features\"");
    for (int raw_feature : features) {
        if (raw_feature < 0 || raw_feature >= chess::EncodedFeatureCount) {
            throw std::runtime_error("feature index out of range");
        }

        chess::FeatureIndex index = static_cast<chess::FeatureIndex>(raw_feature);
        const chess::Square piece_square = static_cast<chess::Square>(index % chess::EncoderSquares);
        index /= chess::EncoderSquares;
        index /= chess::EncoderSquares;
        const auto king_context = static_cast<chess::EncodedKingContext>(index % chess::EncoderKingContexts);
        index /= chess::EncoderKingContexts;
        const auto piece_side = static_cast<chess::EncodedPieceSide>(index % chess::EncoderPieceSides);
        index /= chess::EncoderPieceSides;
        const auto piece = static_cast<chess::PieceType>(index);

        if (king_context != chess::EncodedKingContext::FriendlyKing || seen[raw_feature]) {
            continue;
        }
        seen[raw_feature] = true;

        const chess::Color color = piece_side == chess::EncodedPieceSide::Friendly
            ? chess::Color::White
            : chess::Color::Black;
        if (!pos.is_empty(piece_square)) {
            throw std::runtime_error("decoded duplicate piece square");
        }
        pos.set_piece(color, piece, piece_square);
    }

    return chess::popcount(pos.occupancy(chess::Color::White, chess::PieceType::King)) == 1
        && chess::popcount(pos.occupancy(chess::Color::Black, chess::PieceType::King)) == 1;
}

chess::Position load_line(const Options& options) {
    std::ifstream input(options.input);
    if (!input) {
        throw std::runtime_error("failed to open input: " + options.input);
    }

    std::string line;
    for (std::uint64_t line_index = 1; std::getline(input, line); ++line_index) {
        if (line_index != options.line) {
            continue;
        }
        chess::Position pos;
        if (!decode_sample(line, pos)) {
            throw std::runtime_error("line is not a decodable sample");
        }
        return pos;
    }
    throw std::runtime_error("line not found");
}

char piece_char(const chess::Position& pos, chess::Square square) {
    if (pos.is_empty(square)) {
        return '1';
    }
    const chess::Color color = pos.color_on_occupied(square);
    const chess::PieceType piece = pos.piece_type_on_occupied(square);
    char ch = '?';
    switch (piece) {
        case chess::PieceType::Pawn: ch = 'p'; break;
        case chess::PieceType::Knight: ch = 'n'; break;
        case chess::PieceType::Bishop: ch = 'b'; break;
        case chess::PieceType::Rook: ch = 'r'; break;
        case chess::PieceType::Queen: ch = 'q'; break;
        case chess::PieceType::King: ch = 'k'; break;
        case chess::PieceType::None: ch = '?'; break;
    }
    return color == chess::Color::White ? static_cast<char>(std::toupper(ch)) : ch;
}

std::string to_fen(const chess::Position& pos) {
    std::ostringstream out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const chess::Square square = chess::make_square(file, rank);
            if (pos.is_empty(square)) {
                ++empty;
                continue;
            }
            if (empty != 0) {
                out << empty;
                empty = 0;
            }
            out << piece_char(pos, square);
        }
        if (empty != 0) {
            out << empty;
        }
        if (rank != 0) {
            out << '/';
        }
    }
    out << (pos.side_to_move == chess::Color::White ? " w " : " b ");
    std::string castling;
    if (pos.white_can_castle_kingside) castling.push_back('K');
    if (pos.white_can_castle_queenside) castling.push_back('Q');
    if (pos.black_can_castle_kingside) castling.push_back('k');
    if (pos.black_can_castle_queenside) castling.push_back('q');
    out << (castling.empty() ? "-" : castling);
    out << " - " << pos.halfmove_clock << ' ' << pos.fullmove_number;
    return out.str();
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const chess::Position pos = load_line(options);

        if (options.describe_only) {
            chess::MoveList pseudo_moves;
            chess::MoveList legal_moves;
            chess::MoveList legal_noisy_moves;
            chess::generate_pseudo_legal_moves(pos, pseudo_moves);
            chess::generate_legal_moves(pos, legal_moves);
            const chess::KingSafetyContext king_safety = chess::make_king_safety_context(pos);
            chess::generate_legal_noisy_moves(pos, king_safety, legal_noisy_moves);
            std::cout << "line=" << options.line
                      << " fen=\"" << to_fen(pos) << "\""
                      << " pseudo_moves=" << pseudo_moves.size()
                      << " legal_moves=" << legal_moves.size()
                      << " legal_noisy_moves=" << legal_noisy_moves.size()
                      << " in_check=" << (king_safety.checkers != chess::EmptyBB ? 1 : 0)
                      << " double_check=" << (chess::popcount(king_safety.checkers) > 1 ? 1 : 0)
                      << '\n';
            return 0;
        }

        chess::HeuristicSearcherV25 searcher;
        searcher.clear_tt_stats();
        if (options.instrument_ordering) {
            searcher.clear_move_ordering_stats();
            searcher.set_move_ordering_stats_enabled(true);
        }
        if (options.instrument_time) {
            searcher.clear_timing_stats();
            searcher.set_timing_stats_enabled(true);
        }

        const auto start = std::chrono::steady_clock::now();
        const chess::SearchResult result = options.fixed_depth
            ? searcher.search_best_move(pos, options.depth)
            : searcher.search_best_move(pos, chess::SearchLimits{
                .max_depth = options.depth,
                .move_time = std::chrono::milliseconds{0},
            });
        const auto end = std::chrono::steady_clock::now();
        const auto us = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
        );

        const auto tt = searcher.tt_stats();
        std::cout << "line=" << options.line
                  << " depth=" << options.depth
                  << " fixed=" << (options.fixed_depth ? 1 : 0)
                  << " score=" << result.score
                  << " best=" << chess::move_to_string(result.best_move)
                  << " nodes=" << result.nodes
                  << " us=" << us
                  << " nps=" << (us == 0 ? 0.0 : static_cast<double>(result.nodes) * 1'000'000.0 / static_cast<double>(us))
                  << " tt_probes=" << tt.probes
                  << " tt_key_hit_rate=" << ratio(tt.key_hits, tt.probes)
                  << " tt_score_returns=" << tt.score_returns
                  << " tt_probe_collision_rate=" << ratio(tt.index_collisions, tt.probes);

        if (options.instrument_ordering) {
            const auto stats = searcher.move_ordering_stats();
            const std::uint64_t main_cutoffs =
                stats.tt_lower_stage0_cutoffs + stats.priority_stage_cutoffs + stats.quiet_stage_cutoffs;
            std::cout << " stage0_attempts=" << stats.tt_lower_stage0_attempts
                      << " stage0_cutoffs=" << stats.tt_lower_stage0_cutoffs
                      << " stage0_cutoff_rate=" << ratio(stats.tt_lower_stage0_cutoffs, stats.tt_lower_stage0_attempts)
                      << " priority_cutoffs=" << stats.priority_stage_cutoffs
                      << " priority_cutoff_share=" << ratio(stats.priority_stage_cutoffs, main_cutoffs)
                      << " quiet_cutoffs=" << stats.quiet_stage_cutoffs
                      << " quiet_cutoff_share=" << ratio(stats.quiet_stage_cutoffs, main_cutoffs)
                      << " qsearch_cutoffs=" << stats.qsearch.beta_cutoffs
                      << " priority_contexts=" << stats.priority_king_safety_contexts
                      << " quiet_contexts=" << stats.quiet_king_safety_contexts;
        }
        if (options.instrument_time) {
            const auto timing = searcher.timing_stats();
            auto pct = [&](std::uint64_t measured_ns) {
                return us == 0
                    ? 0.0
                    : 100.0 * static_cast<double>(measured_ns) / (static_cast<double>(us) * 1000.0);
            };
            std::cout << " timing_tt_probe_ns=" << timing.tt_probe_ns
                      << " timing_tt_probe_pct=" << pct(timing.tt_probe_ns)
                      << " timing_tt_probe_calls=" << timing.tt_probe_calls
                      << " timing_king_safety_ns=" << timing.king_safety_ns
                      << " timing_king_safety_pct=" << pct(timing.king_safety_ns)
                      << " timing_king_safety_calls=" << timing.king_safety_calls
                      << " timing_move_generation_ns=" << timing.move_generation_ns
                      << " timing_move_generation_pct=" << pct(timing.move_generation_ns)
                      << " timing_move_generation_calls=" << timing.move_generation_calls
                      << " timing_score_move_ns=" << timing.score_move_ns
                      << " timing_score_move_pct=" << pct(timing.score_move_ns)
                      << " timing_score_move_calls=" << timing.score_move_calls
                      << " timing_main_score_move_ns=" << timing.main_score_move_ns
                      << " timing_main_score_move_pct=" << pct(timing.main_score_move_ns)
                      << " timing_main_score_move_calls=" << timing.main_score_move_calls
                      << " timing_qsearch_score_move_ns=" << timing.qsearch_score_move_ns
                      << " timing_qsearch_score_move_pct=" << pct(timing.qsearch_score_move_ns)
                      << " timing_qsearch_score_move_calls=" << timing.qsearch_score_move_calls
                      << " timing_order_score_ns=" << timing.order_score_ns
                      << " timing_order_score_pct=" << pct(timing.order_score_ns)
                      << " timing_order_score_calls=" << timing.order_score_calls
                      << " timing_history_score_ns=" << timing.history_score_ns
                      << " timing_history_score_pct=" << pct(timing.history_score_ns)
                      << " timing_history_score_calls=" << timing.history_score_calls
                      << " timing_counter_history_score_ns=" << timing.counter_history_score_ns
                      << " timing_counter_history_score_pct=" << pct(timing.counter_history_score_ns)
                      << " timing_counter_history_score_calls=" << timing.counter_history_score_calls
                      << " timing_killer_score_ns=" << timing.killer_score_ns
                      << " timing_killer_score_pct=" << pct(timing.killer_score_ns)
                      << " timing_killer_score_calls=" << timing.killer_score_calls
                      << " timing_see_ns=" << timing.see_ns
                      << " timing_see_pct=" << pct(timing.see_ns)
                      << " timing_see_calls=" << timing.see_calls
                      << " timing_main_see_ns=" << timing.main_see_ns
                      << " timing_main_see_pct=" << pct(timing.main_see_ns)
                      << " timing_main_see_calls=" << timing.main_see_calls
                      << " timing_qsearch_see_ns=" << timing.qsearch_see_ns
                      << " timing_qsearch_see_pct=" << pct(timing.qsearch_see_ns)
                      << " timing_qsearch_see_calls=" << timing.qsearch_see_calls
                      << " timing_gives_check_ns=" << timing.gives_check_ns
                      << " timing_gives_check_pct=" << pct(timing.gives_check_ns)
                      << " timing_gives_check_calls=" << timing.gives_check_calls
                      << " timing_sort_ns=" << timing.sort_ns
                      << " timing_sort_pct=" << pct(timing.sort_ns)
                      << " timing_sort_calls=" << timing.sort_calls
                      << " timing_copy_make_move_ns=" << timing.make_move_ns
                      << " timing_copy_make_move_pct=" << pct(timing.make_move_ns)
                      << " timing_copy_make_move_calls=" << timing.make_move_calls
                      << " timing_evaluate_ns=" << timing.evaluate_ns
                      << " timing_evaluate_pct=" << pct(timing.evaluate_ns)
                      << " timing_evaluate_calls=" << timing.evaluate_calls
                      << " timing_qsearch_inclusive_ns=" << timing.qsearch_ns
                      << " timing_qsearch_inclusive_pct=" << pct(timing.qsearch_ns)
                      << " timing_qsearch_calls=" << timing.qsearch_calls;
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "profile_v25_line: " << error.what() << '\n';
        return 1;
    }
}
