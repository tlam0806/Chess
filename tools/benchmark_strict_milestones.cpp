#include "heuristic_searcher_v15.hpp"
#include "heuristic_searcher_v19.hpp"
#include "heuristic_searcher_v24.hpp"
#include "heuristic_searcher_v27.hpp"
#include "heuristic_searcher_v29.hpp"
#include "heuristic_searcher_v32.hpp"
#include "heuristic_searcher_v33.hpp"
#include "heuristic_searcher_v35.hpp"
#include "move.hpp"
#include "position.hpp"
#include "searcher.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t TtMegabytes = 64;
constexpr int DefaultDepth = 7;
constexpr int DefaultRounds = 8;

enum class Version : std::size_t {
    V15,
    V19,
    V24,
    V27,
    V29,
    V32,
    V33,
    V35,
};

constexpr std::array<Version, 8> Versions = {
    Version::V15,
    Version::V19,
    Version::V24,
    Version::V27,
    Version::V29,
    Version::V32,
    Version::V33,
    Version::V35,
};

constexpr std::array<std::string_view, Versions.size()> VersionNames = {
    "V15", "V19", "V24", "V27", "V29", "V32", "V33", "V35",
};

// Even-order Williams design.  Rotating these offsets through all eight
// labels balances both execution slots and ordered predecessor/carryover
// pairs.  The second leg reverses the resulting order for drift symmetry.
constexpr std::array<std::size_t, Versions.size()> WilliamsOffsets = {
    0, 1, 7, 2, 6, 3, 5, 4,
};

struct CorpusPosition {
    std::string_view id;
    std::string_view fen;
};

// Versioned in the benchmark source itself.  The exact corpus is also
// identified in every run by a deterministic FNV-1a hash and can be emitted
// with --emit-corpus.
constexpr std::array<CorpusPosition, 25> Corpus = {{
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
    {"opening_knights", "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4"},
    {"tactical_castling", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"},
    {"closed_center", "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9"},
    {"middlegame_rooks", "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12"},
    {"advanced_pawns", "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23"},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"},
    {"promotion_race", "8/P6k/8/8/8/8/6Kp/8 w - - 0 1"},
    {"pawn_structure", "8/8/2p5/3p4/3P4/2P5/8/4K1k1 w - - 0 1"},
    {"development", "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10"},
    {"queenside_pressure", "2r3k1/1p1bqppp/p3pn2/3p4/3P4/P1NBPN2/1PQ2PPP/2R2RK1 b - - 0 16"},
    {"simple_king_pawns", "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"},
    {"rook_endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"},
    {"promotion_and_castling", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"},
    {"opposite_attacks", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"},
    {"undeveloped_tension", "2b1qb1r/rpppkp1p/2n1p2n/p7/P3N3/1P3P2/2PNP1PP/R1BQKB1R w KQ - 3 14"},
    {"minimal_castling", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"},
    {"en_passant", "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"},
    {"in_check_evasion", "4r1k1/8/8/8/8/8/8/2B1K3 w - - 0 1"},
    {"queen_vs_pawn", "5k2/2p5/8/8/1Q6/8/8/6K1 b - - 0 1"},
    {"bishop_vs_pawn", "k7/8/1b6/8/8/8/3P4/6K1 w - - 0 1"},
    {"single_pawn_endgame", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"},
    {"promotion_technique", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"},
    {"rook_technique", "8/8/2k5/8/8/8/4K3/3R4 w - - 0 1"},
    {"black_to_move_center", "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 b - - 0 9"},
}};

struct Options {
    std::vector<int> depths{DefaultDepth};
    int rounds = DefaultRounds;
    bool emit_corpus = false;
};

struct PreparedPosition {
    chess::Position position;
    chess::MoveList legal_moves;
};

struct TimedRow {
    std::uint64_t order = 0;
    int requested_depth = 0;
    int round = 0;
    bool reverse = false;
    std::size_t position_index = 0;
    std::size_t slot = 0;
    Version version = Version::V15;
    chess::SearchResult result{};
    std::uint64_t elapsed_ns = 0;
    bool legal = false;
    bool score_match = false;
    bool depth_match = false;
};

struct VersionSummary {
    std::uint64_t elapsed_ns = 0;
    std::uint64_t nodes = 0;
    std::uint64_t samples = 0;
    std::uint64_t legal_failures = 0;
    std::uint64_t stopped_failures = 0;
    std::uint64_t score_failures = 0;
    std::uint64_t depth_failures = 0;
    std::vector<std::uint64_t> row_elapsed_ns;
};

std::size_t version_index(Version version) {
    return static_cast<std::size_t>(version);
}

std::string_view version_name(Version version) {
    return VersionNames[version_index(version)];
}

int parse_int(std::string_view text, std::string_view option) {
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(std::string(text), &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + std::string(option) + ": " + std::string(text));
    }
    if (consumed != text.size()) {
        throw std::runtime_error("invalid integer for " + std::string(option) + ": " + std::string(text));
    }
    return value;
}

std::vector<int> parse_depths(std::string_view text) {
    if (text.empty()) {
        throw std::runtime_error("--depths must not be empty");
    }

    std::vector<int> depths;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        if (end == begin) {
            throw std::runtime_error("--depths contains an empty item: " + std::string(text));
        }
        const int depth = parse_int(text.substr(begin, end - begin), "--depths");
        if (depth < 1) {
            throw std::runtime_error("every --depths value must be at least 1");
        }
        if (std::find(depths.begin(), depths.end(), depth) != depths.end()) {
            throw std::runtime_error("--depths contains a duplicate value: " + std::to_string(depth));
        }
        depths.push_back(depth);
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1;
    }
    return depths;
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool depth_option_seen = false;
    bool depths_option_seen = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };

        if (arg == "--depth") {
            if (depths_option_seen) {
                throw std::runtime_error("--depth and --depths are mutually exclusive");
            }
            const int depth = parse_int(require_value(arg), arg);
            options.depths = {depth};
            depth_option_seen = true;
        } else if (arg == "--depths") {
            if (depth_option_seen) {
                throw std::runtime_error("--depth and --depths are mutually exclusive");
            }
            options.depths = parse_depths(require_value(arg));
            depths_option_seen = true;
        } else if (arg == "--rounds") {
            options.rounds = parse_int(require_value(arg), arg);
        } else if (arg == "--emit-corpus") {
            options.emit_corpus = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: benchmark_strict_milestones [--depth N | --depths N,N,...]\n"
                << "                                    [--rounds N] [--emit-corpus]\n\n"
                << "Direct fixed-depth benchmark of Strict V15,V19,V24,V27,V29,V32,V33,V35.\n"
                << "The approved multi-depth protocol is --depths 6,7,8 --rounds 8.\n"
                << "Each round contains a Williams-balanced forward leg and its mirrored reverse leg.\n"
                << "Each depth gets an independent unmeasured correctness preflight/warmup.\n"
                << "Output is newline-delimited JSON; elapsed_ns is the primary metric.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (options.depths.empty()) {
        throw std::runtime_error("at least one search depth is required");
    }
    if (std::any_of(options.depths.begin(), options.depths.end(), [](int depth) { return depth < 1; })) {
        throw std::runtime_error("every search depth must be at least 1");
    }
    if (options.rounds < 1) {
        throw std::runtime_error("--rounds must be at least 1");
    }
    return options;
}

std::string json_string(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    out << '"';
    return out.str();
}

std::string corpus_hash() {
    constexpr std::uint64_t offset_basis = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset_basis;
    for (const CorpusPosition& item : Corpus) {
        for (const unsigned char ch : item.fen) {
            hash ^= ch;
            hash *= prime;
        }
        hash ^= static_cast<unsigned char>('\n');
        hash *= prime;
    }
    std::ostringstream out;
    out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

std::unique_ptr<chess::Searcher> make_searcher(Version version) {
    switch (version) {
        case Version::V15: return std::make_unique<chess::HeuristicSearcherV15>(TtMegabytes);
        case Version::V19: return std::make_unique<chess::HeuristicSearcherV19>(TtMegabytes);
        case Version::V24: return std::make_unique<chess::HeuristicSearcherV24>(TtMegabytes);
        case Version::V27: return std::make_unique<chess::HeuristicSearcherV27>(TtMegabytes);
        case Version::V29: return std::make_unique<chess::HeuristicSearcherV29>(TtMegabytes);
        case Version::V32: return std::make_unique<chess::HeuristicSearcherV32>(TtMegabytes);
        case Version::V33: return std::make_unique<chess::HeuristicSearcherV33>(TtMegabytes);
        case Version::V35: return std::make_unique<chess::HeuristicSearcherV35>(TtMegabytes);
    }
    throw std::logic_error("unknown Strict milestone version");
}

std::vector<PreparedPosition> prepare_corpus() {
    std::vector<PreparedPosition> prepared;
    prepared.reserve(Corpus.size());
    for (const CorpusPosition& item : Corpus) {
        PreparedPosition value;
        if (!value.position.set_fen(item.fen)) {
            throw std::runtime_error("invalid embedded FEN: " + std::string(item.id));
        }
        chess::generate_legal_moves(value.position, value.legal_moves);
        if (value.legal_moves.empty()) {
            throw std::runtime_error("embedded FEN has no legal root move: " + std::string(item.id));
        }
        prepared.push_back(value);
    }
    return prepared;
}

bool contains_move(const chess::MoveList& legal_moves, chess::Move move) {
    return std::find(legal_moves.begin(), legal_moves.end(), move) != legal_moves.end();
}

std::array<Version, Versions.size()> scheduled_versions(
    int round,
    std::size_t position_index,
    bool reverse
) {
    std::array<Version, Versions.size()> order{};
    const std::size_t rotation =
        (static_cast<std::size_t>(round) + position_index) % Versions.size();
    for (std::size_t i = 0; i < Versions.size(); ++i) {
        order[i] = Versions[(rotation + WilliamsOffsets[i]) % Versions.size()];
    }
    if (reverse) {
        std::reverse(order.begin(), order.end());
    }
    return order;
}

void emit_meta(const Options& options, const std::string& hash) {
    std::cout
        << "{\"type\":\"meta\",\"schema_version\":2"
        << ",\"benchmark\":\"strict_milestones_v2\""
        << ",\"versions\":[\"V15\",\"V19\",\"V24\",\"V27\",\"V29\",\"V32\",\"V33\",\"V35\"]"
        << ",\"depths\":[";
    for (std::size_t i = 0; i < options.depths.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << options.depths[i];
    }
    std::cout
        << "]"
        << ",\"rounds_per_depth\":" << options.rounds
        << ",\"legs_per_round\":2"
        << ",\"positions\":" << Corpus.size()
        << ",\"tt_mb\":" << TtMegabytes
        << ",\"fixed_depth\":true"
        << ",\"fresh_searcher_per_tuple\":true"
        << ",\"tt_or_history_shared_across_tuples\":false"
        << ",\"searcher_construction_timed\":false"
        << ",\"schedule\":\"paired_williams_forward_mirrored_reverse_v2\""
        << ",\"schedule_base_offsets\":[0,1,7,2,6,3,5,4]"
        << ",\"schedule_rotation\":\"(round+position_index)%8\""
        << ",\"schedule_balance_cycle_rounds\":8"
        << ",\"schedule_balance_cycle_complete\":"
        << (options.rounds % static_cast<int>(Versions.size()) == 0 ? "true" : "false")
        << ",\"warmup\":\"unmeasured_full_corpus_correctness_preflight\""
        << ",\"all_depth_preflights_complete_before_timing\":true"
        << ",\"row_order_scope\":\"requested_depth\""
        << ",\"score_gate_reference\":\"V15_per_position_at_requested_depth\""
        << ",\"move_gate\":\"returned_move_must_be_legal; equivalent_best_moves_allowed\""
        << ",\"primary_metric\":\"elapsed_ns\""
        << ",\"corpus_embedded\":true"
        << ",\"corpus_hash\":" << json_string(hash)
        << "}\n";
}

void emit_corpus(const std::string& hash) {
    for (std::size_t i = 0; i < Corpus.size(); ++i) {
        std::cout
            << "{\"type\":\"corpus_position\""
            << ",\"corpus_hash\":" << json_string(hash)
            << ",\"position\":" << i
            << ",\"position_id\":" << json_string(Corpus[i].id)
            << ",\"fen\":" << json_string(Corpus[i].fen)
            << "}\n";
    }
}

bool run_preflight(
    int requested_depth,
    const std::vector<PreparedPosition>& prepared,
    std::array<int, Corpus.size()>& expected_scores
) {
    std::uint64_t failures = 0;
    std::uint64_t tuples = 0;
    for (std::size_t position_index = 0; position_index < Corpus.size(); ++position_index) {
        std::optional<int> expected_score;
        for (Version version : Versions) {
            std::unique_ptr<chess::Searcher> searcher = make_searcher(version);
            const chess::SearchResult result =
                searcher->search_best_move(prepared[position_index].position, requested_depth);
            ++tuples;

            if (!expected_score.has_value()) {
                expected_score = result.score;
                expected_scores[position_index] = result.score;
            }
            const bool legal = contains_move(prepared[position_index].legal_moves, result.best_move);
            const bool score_match = result.score == *expected_score;
            const bool depth_match = result.depth == requested_depth;
            const bool pass = legal && !result.stopped && score_match && depth_match;
            if (!pass) {
                ++failures;
                std::cout
                    << "{\"type\":\"gate_error\",\"phase\":\"preflight_warmup\""
                    << ",\"requested_depth\":" << requested_depth
                    << ",\"position\":" << position_index
                    << ",\"position_id\":" << json_string(Corpus[position_index].id)
                    << ",\"version\":" << json_string(version_name(version))
                    << ",\"reported_depth\":" << result.depth
                    << ",\"score\":" << result.score
                    << ",\"expected_score\":" << *expected_score
                    << ",\"move\":" << json_string(chess::move_to_string(result.best_move))
                    << ",\"nodes\":" << result.nodes
                    << ",\"legal\":" << (legal ? "true" : "false")
                    << ",\"stopped\":" << (result.stopped ? "true" : "false")
                    << ",\"score_match\":" << (score_match ? "true" : "false")
                    << ",\"depth_match\":" << (depth_match ? "true" : "false")
                    << "}\n";
            }
        }
    }

    std::cout
        << "{\"type\":\"gate_summary\",\"phase\":\"preflight_warmup\""
        << ",\"requested_depth\":" << requested_depth
        << ",\"measured\":false"
        << ",\"tuples\":" << tuples
        << ",\"failures\":" << failures
        << ",\"status\":\"" << (failures == 0 ? "pass" : "fail") << "\"}\n";
    return failures == 0;
}

std::uint64_t median(std::vector<std::uint64_t> values) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

void emit_row(const TimedRow& row, int expected_score) {
    const bool pass = row.legal && !row.result.stopped && row.score_match && row.depth_match;
    std::cout
        << "{\"type\":\"row\""
        << ",\"order\":" << row.order
        << ",\"requested_depth\":" << row.requested_depth
        << ",\"round\":" << row.round
        << ",\"leg\":\"" << (row.reverse ? "reverse" : "forward") << "\""
        << ",\"slot\":" << row.slot
        << ",\"position\":" << row.position_index
        << ",\"position_id\":" << json_string(Corpus[row.position_index].id)
        << ",\"version\":" << json_string(version_name(row.version))
        << ",\"depth\":" << row.result.depth
        << ",\"elapsed_ns\":" << row.elapsed_ns
        << ",\"nodes\":" << row.result.nodes
        << ",\"score\":" << row.result.score
        << ",\"expected_score\":" << expected_score
        << ",\"move\":" << json_string(chess::move_to_string(row.result.best_move))
        << ",\"legal\":" << (row.legal ? "true" : "false")
        << ",\"stopped\":" << (row.result.stopped ? "true" : "false")
        << ",\"score_match\":" << (row.score_match ? "true" : "false")
        << ",\"depth_match\":" << (row.depth_match ? "true" : "false")
        << ",\"gate_pass\":" << (pass ? "true" : "false")
        << "}\n";
}

struct DepthRunOutcome {
    bool pass = false;
    std::uint64_t timed_rows = 0;
    std::uint64_t expected_timed_rows = 0;
};

DepthRunOutcome run_timed_depth(
    const Options& options,
    int requested_depth,
    const std::vector<PreparedPosition>& prepared,
    const std::array<int, Corpus.size()>& expected_scores,
    const std::string& hash
) {
    const std::size_t row_count = static_cast<std::size_t>(options.rounds)
        * 2 * Corpus.size() * Versions.size();
    std::vector<TimedRow> rows;
    rows.reserve(row_count);
    std::array<VersionSummary, Versions.size()> summaries{};
    std::uint64_t next_order = 0;
    bool all_gates_pass = true;

    for (int round = 0; round < options.rounds; ++round) {
        for (const bool reverse : {false, true}) {
            for (std::size_t position_slot = 0; position_slot < Corpus.size(); ++position_slot) {
                const std::size_t position_index = reverse
                    ? (static_cast<std::size_t>(round) + Corpus.size() - 1 - position_slot) % Corpus.size()
                    : (static_cast<std::size_t>(round) + position_slot) % Corpus.size();
                const auto version_order = scheduled_versions(round, position_index, reverse);

                for (std::size_t slot = 0; slot < version_order.size(); ++slot) {
                    const Version version = version_order[slot];
                    // Construction/destruction are intentionally outside the timer.  A new
                    // instance still guarantees a logically empty 64 MiB TT and fresh history.
                    std::unique_ptr<chess::Searcher> searcher = make_searcher(version);
                    const auto start = Clock::now();
                    const chess::SearchResult result = searcher->search_best_move(
                        prepared[position_index].position,
                        requested_depth);
                    const auto stop = Clock::now();
                    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
                    if (elapsed.count() < 0) {
                        throw std::runtime_error("steady clock returned a negative elapsed duration");
                    }

                    TimedRow row;
                    row.order = next_order++;
                    row.requested_depth = requested_depth;
                    row.round = round;
                    row.reverse = reverse;
                    row.position_index = position_index;
                    row.slot = slot;
                    row.version = version;
                    row.result = result;
                    row.elapsed_ns = static_cast<std::uint64_t>(elapsed.count());
                    row.legal = contains_move(prepared[position_index].legal_moves, result.best_move);
                    row.score_match = result.score == expected_scores[position_index];
                    row.depth_match = result.depth == requested_depth;
                    const bool pass = row.legal && !result.stopped && row.score_match && row.depth_match;
                    all_gates_pass = all_gates_pass && pass;

                    VersionSummary& summary = summaries[version_index(version)];
                    summary.elapsed_ns += row.elapsed_ns;
                    summary.nodes += result.nodes;
                    ++summary.samples;
                    summary.legal_failures += row.legal ? 0 : 1;
                    summary.stopped_failures += result.stopped ? 1 : 0;
                    summary.score_failures += row.score_match ? 0 : 1;
                    summary.depth_failures += row.depth_match ? 0 : 1;
                    summary.row_elapsed_ns.push_back(row.elapsed_ns);
                    rows.push_back(row);
                }
            }
        }
    }

    for (const TimedRow& row : rows) {
        emit_row(row, expected_scores[row.position_index]);
    }

    const std::uint64_t baseline_elapsed = summaries[version_index(Version::V15)].elapsed_ns;
    for (std::size_t i = 0; i < Versions.size(); ++i) {
        const VersionSummary& summary = summaries[i];
        const std::uint64_t min_elapsed = summary.row_elapsed_ns.empty()
            ? 0
            : *std::min_element(summary.row_elapsed_ns.begin(), summary.row_elapsed_ns.end());
        const std::uint64_t max_elapsed = summary.row_elapsed_ns.empty()
            ? 0
            : *std::max_element(summary.row_elapsed_ns.begin(), summary.row_elapsed_ns.end());
        const double nps = summary.elapsed_ns == 0
            ? 0.0
            : static_cast<double>(summary.nodes) * 1'000'000'000.0
                / static_cast<double>(summary.elapsed_ns);
        const double ratio_to_v15 = baseline_elapsed == 0
            ? 0.0
            : static_cast<double>(summary.elapsed_ns) / static_cast<double>(baseline_elapsed);
        const bool pass = summary.legal_failures == 0
            && summary.stopped_failures == 0
            && summary.score_failures == 0
            && summary.depth_failures == 0;

        std::cout
            << "{\"type\":\"depth_version_summary\""
            << ",\"requested_depth\":" << requested_depth
            << ",\"order\":" << i
            << ",\"version\":" << json_string(VersionNames[i])
            << ",\"samples\":" << summary.samples
            << ",\"elapsed_ns\":" << summary.elapsed_ns
            << ",\"nodes\":" << summary.nodes
            << ",\"nps\":" << std::setprecision(12) << nps
            << ",\"elapsed_ratio_to_v15\":" << std::setprecision(12) << ratio_to_v15
            << ",\"median_row_elapsed_ns\":" << median(summary.row_elapsed_ns)
            << ",\"min_row_elapsed_ns\":" << min_elapsed
            << ",\"max_row_elapsed_ns\":" << max_elapsed
            << ",\"legal_failures\":" << summary.legal_failures
            << ",\"stopped_failures\":" << summary.stopped_failures
            << ",\"score_failures\":" << summary.score_failures
            << ",\"depth_failures\":" << summary.depth_failures
            << ",\"status\":\"" << (pass ? "pass" : "fail") << "\"}\n";
    }

    std::cout
        << "{\"type\":\"depth_run_summary\""
        << ",\"requested_depth\":" << requested_depth
        << ",\"status\":\"" << (all_gates_pass ? "pass" : "fail") << "\""
        << ",\"timed_rows\":" << rows.size()
        << ",\"expected_timed_rows\":" << row_count
        << ",\"primary_metric\":\"elapsed_ns\""
        << ",\"corpus_hash\":" << json_string(hash)
        << "}\n";
    return DepthRunOutcome{
        .pass = all_gates_pass,
        .timed_rows = static_cast<std::uint64_t>(rows.size()),
        .expected_timed_rows = static_cast<std::uint64_t>(row_count),
    };
}

int run_benchmark(const Options& options) {
    const std::string hash = corpus_hash();
    const std::vector<PreparedPosition> prepared = prepare_corpus();
    emit_meta(options, hash);
    if (options.emit_corpus) {
        emit_corpus(hash);
    }

    std::vector<std::array<int, Corpus.size()>> expected_scores_by_depth(options.depths.size());
    bool all_preflights_pass = true;
    for (std::size_t i = 0; i < options.depths.size(); ++i) {
        const bool pass = run_preflight(
            options.depths[i],
            prepared,
            expected_scores_by_depth[i]);
        all_preflights_pass = all_preflights_pass && pass;
    }

    const std::uint64_t expected_rows_per_depth = static_cast<std::uint64_t>(options.rounds)
        * 2 * Corpus.size() * Versions.size();
    const std::uint64_t expected_rows =
        expected_rows_per_depth * static_cast<std::uint64_t>(options.depths.size());
    if (!all_preflights_pass) {
        std::cout
            << "{\"type\":\"run_summary\",\"status\":\"fail\""
            << ",\"phase\":\"preflight_warmup\""
            << ",\"depths_requested\":" << options.depths.size()
            << ",\"timed_rows\":0"
            << ",\"expected_timed_rows\":" << expected_rows
            << ",\"corpus_hash\":" << json_string(hash)
            << "}\n";
        return 2;
    }

    bool all_depths_pass = true;
    std::uint64_t timed_rows = 0;
    std::uint64_t expected_timed_rows = 0;
    for (std::size_t i = 0; i < options.depths.size(); ++i) {
        const DepthRunOutcome outcome = run_timed_depth(
            options,
            options.depths[i],
            prepared,
            expected_scores_by_depth[i],
            hash);
        all_depths_pass = all_depths_pass && outcome.pass;
        timed_rows += outcome.timed_rows;
        expected_timed_rows += outcome.expected_timed_rows;
    }

    std::cout
        << "{\"type\":\"run_summary\""
        << ",\"status\":\"" << (all_depths_pass ? "pass" : "fail") << "\""
        << ",\"depths_completed\":" << options.depths.size()
        << ",\"timed_rows\":" << timed_rows
        << ",\"expected_timed_rows\":" << expected_timed_rows
        << ",\"primary_metric\":\"elapsed_ns\""
        << ",\"corpus_hash\":" << json_string(hash)
        << "}\n";
    return all_depths_pass ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run_benchmark(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "benchmark_strict_milestones: " << error.what() << '\n';
        std::cout
            << "{\"type\":\"error\",\"message\":" << json_string(error.what()) << "}\n";
        return 1;
    }
}
