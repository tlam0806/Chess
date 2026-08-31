#include "attacks.hpp"
#include "move.hpp"
#include "nnue_wdl_calibration.hpp"
#include "nnue_searcher_v36.hpp"
#if defined(CHESS_EVALUATE_NNUE_V43)
#include "nnue_searcher_v43.hpp"
#elif defined(CHESS_EVALUATE_NNUE_V42)
#include "nnue_searcher_v42.hpp"
#elif defined(CHESS_EVALUATE_NNUE_V41)
#include "nnue_searcher_v41.hpp"
#elif defined(CHESS_EVALUATE_NNUE_V40)
#include "nnue_searcher_v40.hpp"
#elif defined(CHESS_EVALUATE_NNUE_V39)
#include "nnue_searcher_v39.hpp"
#else
#include "nnue_searcher_v38.hpp"
#endif
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

#if defined(CHESS_EVALUATE_NNUE_V43)
using CandidateSearcher = chess::NnueSearcherV43;
static_assert(
    sizeof(chess::V43SingleBoundTranspositionTable::Entry) == 16);
static_assert(
    sizeof(chess::V43SingleBoundTranspositionTable::Bucket) == 64);
#elif defined(CHESS_EVALUATE_NNUE_V42)
using CandidateSearcher = chess::NnueSearcherV42;
#elif defined(CHESS_EVALUATE_NNUE_V41)
using CandidateSearcher = chess::NnueSearcherV41;
#elif defined(CHESS_EVALUATE_NNUE_V40)
using CandidateSearcher = chess::NnueSearcherV40;
#elif defined(CHESS_EVALUATE_NNUE_V39)
using CandidateSearcher = chess::NnueSearcherV39;
#else
using CandidateSearcher = chess::NnueSearcherV38;
#endif

#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
CandidateSearcher::AspirationConfig default_aspiration_config() {
    CandidateSearcher::AspirationConfig config;
    config.enabled = true;
    return config;
}
#if defined(CHESS_EVALUATE_NNUE_V43)
constexpr std::string_view AdaptiveEvaluatorVersion = "V43";
#else
constexpr std::string_view AdaptiveEvaluatorVersion = "V42";
#endif
#endif

struct Options {
    std::string dataset;
    std::string model = std::string(chess::DefaultPhaseQuantizedNnueModelPath);
    int depth = 6;
    int offset = 0;
    int count = 0;
    int detail_threshold = 0;
    int ranking_target_abs_cp = 1500;
    int candidate_time_ms = 0;
    int candidate_max_depth = 64;
    bool include_all_in_objective = false;
    bool twofold_search_draw_enabled = false;
    std::string objective = "cp";
    CandidateSearcher::SelectiveConfig config{};
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
    CandidateSearcher::AspirationConfig aspiration_config =
        default_aspiration_config();
    std::string write_control_cache;
    std::string control_cache;
    std::string control_cache_identity_sha256;
    std::string control_cache_evaluator_sha256;
    std::string control_cache_model_sha256;
    std::string control_cache_dataset_sha256;
    std::string write_detail_jsonl;
#endif
#if defined(CHESS_EVALUATE_NNUE_V43)
    std::size_t tt_mb = 64;
    std::size_t tt_bucket_size = 4;
    bool reuse_stale_tt_scores = false;
    bool reuse_deeper_tt_scores = false;
#endif
};

struct Sample {
    std::string category;
    std::string hash;
    std::string payload;
    chess::Position position;
    int calibration_ply = 0;
    bool has_explicit_ply = false;
};

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
    throw std::runtime_error("illegal book move: " + std::string(uci));
}

chess::Position position_from_payload(std::string_view payload) {
    chess::Position position;
    if (payload.starts_with("book:")) {
        position.set_startpos();
        for (const std::string& uci : words(payload.substr(5))) {
            position.make_move(legal_uci(position, uci));
        }
    } else if (!position.set_fen(payload)) {
        throw std::runtime_error("invalid FEN: " + std::string(payload));
    }
    return position;
}

std::vector<Sample> load_samples(const Options& options) {
    std::ifstream input(options.dataset);
    if (!input) throw std::runtime_error("failed to open dataset: " + options.dataset);
    std::vector<Sample> all;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t first = line.find('\t');
        const std::size_t second =
            first == std::string::npos ? first : line.find('\t', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            throw std::runtime_error("bad dataset row");
        }
        const std::size_t third = line.find('\t', second + 1);
        const bool has_explicit_ply = third != std::string::npos;
        int calibration_ply = 0;
        std::string payload;
        if (has_explicit_ply) {
            const std::string ply_text =
                line.substr(second + 1, third - second - 1);
            std::size_t consumed = 0;
            try {
                calibration_ply = std::stoi(ply_text, &consumed);
            } catch (...) {
                throw std::runtime_error("bad dataset ply");
            }
            if (consumed != ply_text.size()
                || calibration_ply < 0
                || calibration_ply > 0x3FFF) {
                throw std::runtime_error("bad dataset ply");
            }
            payload = line.substr(third + 1);
        } else {
            payload = line.substr(second + 1);
        }
        chess::Position position = position_from_payload(payload);
        if (!has_explicit_ply) {
            calibration_ply = 2 * (position.fullmove_number - 1)
                + (position.side_to_move == chess::Color::Black ? 1 : 0);
        }
        all.push_back(Sample{
            line.substr(0, first),
            line.substr(first + 1, second - first - 1),
            payload,
            std::move(position),
            calibration_ply,
            has_explicit_ply,
        });
    }
    if (options.offset < 0 || options.offset >= static_cast<int>(all.size())) {
        throw std::runtime_error("offset outside dataset");
    }
    const int count = options.count == 0
        ? static_cast<int>(all.size()) - options.offset
        : options.count;
    if (count <= 0 || options.offset + count > static_cast<int>(all.size())) {
        throw std::runtime_error("count outside dataset");
    }
    return std::vector<Sample>(
        all.begin() + options.offset, all.begin() + options.offset + count);
}

int integer(const char* value, std::string_view name) {
    const std::string text(value);
    std::size_t consumed = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(text, &consumed);
    } catch (...) {
        throw std::runtime_error("bad integer for " + std::string(name));
    }
    if (consumed != text.size()
        || parsed < std::numeric_limits<int>::min()
        || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error("bad integer for " + std::string(name));
    }
    return static_cast<int>(parsed);
}

double real(const char* value, std::string_view name) {
    const std::string text(value);
    std::size_t consumed = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(text, &consumed);
    } catch (...) {
        throw std::runtime_error("bad number for " + std::string(name));
    }
    if (consumed != text.size() || !std::isfinite(parsed)) {
        throw std::runtime_error("bad finite number for " + std::string(name));
    }
    return parsed;
}

std::size_t bounded_size(
    const char* value,
    std::string_view name,
    std::size_t maximum
) {
    // Parse as signed first. Casting a negative CLI value directly to size_t
    // used to turn -1 into a huge, superficially valid pruning threshold.
    const int parsed = integer(value, name);
    if (parsed < 0 || static_cast<std::size_t>(parsed) > maximum) {
        throw std::runtime_error(
            std::string(name) + " must be in [0, "
            + std::to_string(maximum) + "]");
    }
    return static_cast<std::size_t>(parsed);
}

template<typename Value>
void require_closed_range(
    Value value,
    Value minimum,
    Value maximum,
    std::string_view name
) {
    if (value < minimum || value > maximum) {
        throw std::runtime_error(
            std::string(name) + " is outside the evaluator-safe range");
    }
}

void validate_selective_config(
    const CandidateSearcher::SelectiveConfig& config
) {
    constexpr int MaximumSearchDepth = 64;
    constexpr std::size_t MaximumLegalMoveIndex = 256;
    constexpr int MaximumMarginCp = 1'000'000;
    constexpr std::size_t MaximumLmpCoefficient = 1'024;

    if (!std::isfinite(config.lmr_base)) {
        throw std::runtime_error("lmr base must be finite");
    }
    if (!std::isfinite(config.lmr_divisor)
        || config.lmr_divisor <= 0.0) {
        throw std::runtime_error("lmr divisor must be finite and > 0");
    }
    require_closed_range(config.lmr_base, 0.0, 16.0, "lmr base");
    require_closed_range(config.lmr_divisor, 0.01, 100.0, "lmr divisor");
    require_closed_range(
        config.lmr_min_depth, 1, MaximumSearchDepth, "lmr min depth");
    require_closed_range(
        config.lmr_min_move_index,
        std::size_t{0}, MaximumLegalMoveIndex, "lmr min move index");

    require_closed_range(
        config.null_move_min_depth,
        1, MaximumSearchDepth, "null-move min depth");
    require_closed_range(
        config.null_move_reduction,
        1, MaximumSearchDepth - 1, "null-move reduction");

    require_closed_range(
        config.reverse_futility_max_depth,
        1, MaximumSearchDepth, "reverse-futility max depth");
    require_closed_range(
        config.reverse_futility_base_margin,
        0, MaximumMarginCp, "reverse-futility base margin");
    require_closed_range(
        config.reverse_futility_margin_per_depth,
        0, MaximumMarginCp, "reverse-futility margin per depth");

    require_closed_range(
        config.late_move_pruning_max_depth,
        1, MaximumSearchDepth, "late-move-pruning max depth");
    require_closed_range(
        config.late_move_pruning_base,
        std::size_t{0}, MaximumLmpCoefficient,
        "late-move-pruning base");
    require_closed_range(
        config.late_move_pruning_depth_multiplier,
        std::size_t{0}, MaximumLmpCoefficient,
        "late-move-pruning depth multiplier");

    require_closed_range(
        config.qsearch_see_threshold,
        -MaximumMarginCp, MaximumMarginCp, "qsearch SEE threshold");
    require_closed_range(
        config.main_search_see_max_depth,
        1, MaximumSearchDepth, "main-search SEE max depth");
    require_closed_range(
        config.main_search_see_margin_per_depth,
        0, MaximumMarginCp, "main-search SEE margin per depth");
}

void write_selective_config_json(
    std::ostream& output,
    const CandidateSearcher::SelectiveConfig& config
) {
    output
        << '{'
        << "\"enable_lmr\":" << (config.enable_lmr ? "true" : "false")
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
        << ",\"late_move_pruning_base\":"
        << config.late_move_pruning_base
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
        << config.main_search_see_margin_per_depth
        << '}';
}

#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
void write_aspiration_config_json(
    std::ostream& output,
    const CandidateSearcher::AspirationConfig& config
) {
    output
        << '{'
        << "\"enabled\":" << (config.enabled ? "true" : "false")
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
#endif

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> const char* {
            if (++i >= argc) throw std::runtime_error("missing value for " + std::string(arg));
            return argv[i];
        };
        if (arg == "--dataset") options.dataset = next();
        else if (arg == "--model") options.model = next();
        else if (arg == "--depth") options.depth = integer(next(), arg);
        else if (arg == "--offset") options.offset = integer(next(), arg);
        else if (arg == "--count") options.count = integer(next(), arg);
        else if (arg == "--detail-threshold")
            options.detail_threshold = integer(next(), arg);
        else if (arg == "--ranking-target-abs-cp")
            options.ranking_target_abs_cp = integer(next(), arg);
        else if (arg == "--candidate-time-ms")
            options.candidate_time_ms = integer(next(), arg);
        else if (arg == "--candidate-max-depth")
            options.candidate_max_depth = integer(next(), arg);
        else if (arg == "--include-all-in-objective")
            options.include_all_in_objective = true;
        else if (arg == "--objective") options.objective = next();
        else if (arg == "--enable-lmr") options.config.enable_lmr = true;
        else if (arg == "--disable-lmr") options.config.enable_lmr = false;
        else if (arg == "--enable-null-move")
            options.config.enable_null_move = true;
        else if (arg == "--disable-null-move")
            options.config.enable_null_move = false;
        else if (arg == "--lmr-base") options.config.lmr_base = real(next(), arg);
        else if (arg == "--lmr-divisor") options.config.lmr_divisor = real(next(), arg);
        else if (arg == "--lmr-min-depth") options.config.lmr_min_depth = integer(next(), arg);
        else if (arg == "--lmr-min-move-index")
            options.config.lmr_min_move_index = bounded_size(next(), arg, 256);
        else if (arg == "--null-min-depth")
            options.config.null_move_min_depth = integer(next(), arg);
        else if (arg == "--null-reduction")
            options.config.null_move_reduction = integer(next(), arg);
        else if (arg == "--disable-reverse-futility")
            options.config.enable_reverse_futility = false;
        else if (arg == "--enable-reverse-futility")
            options.config.enable_reverse_futility = true;
        else if (arg == "--reverse-futility-max-depth")
            options.config.reverse_futility_max_depth = integer(next(), arg);
        else if (arg == "--reverse-futility-base-margin")
            options.config.reverse_futility_base_margin = integer(next(), arg);
        else if (arg == "--reverse-futility-margin-per-depth")
            options.config.reverse_futility_margin_per_depth =
                integer(next(), arg);
        else if (arg == "--disable-late-move-pruning")
            options.config.enable_late_move_pruning = false;
        else if (arg == "--enable-late-move-pruning")
            options.config.enable_late_move_pruning = true;
        else if (arg == "--late-move-pruning-max-depth")
            options.config.late_move_pruning_max_depth = integer(next(), arg);
        else if (arg == "--late-move-pruning-base")
            options.config.late_move_pruning_base =
                bounded_size(next(), arg, 1'024);
        else if (arg == "--late-move-pruning-depth-multiplier")
            options.config.late_move_pruning_depth_multiplier =
                bounded_size(next(), arg, 1'024);
        else if (arg == "--disable-qsearch-see-pruning")
            options.config.enable_qsearch_see_pruning = false;
        else if (arg == "--enable-qsearch-see-pruning")
            options.config.enable_qsearch_see_pruning = true;
        else if (arg == "--qsearch-see-threshold")
            options.config.qsearch_see_threshold = integer(next(), arg);
        else if (arg == "--disable-main-search-see-pruning")
            options.config.enable_main_search_see_pruning = false;
        else if (arg == "--enable-main-search-see-pruning")
            options.config.enable_main_search_see_pruning = true;
        else if (arg == "--main-search-see-max-depth")
            options.config.main_search_see_max_depth = integer(next(), arg);
        else if (arg == "--main-search-see-margin-per-depth")
            options.config.main_search_see_margin_per_depth = integer(next(), arg);
        else if (arg == "--enable-twofold-search-draw")
            options.twofold_search_draw_enabled = true;
        else if (arg == "--disable-twofold-search-draw")
            options.twofold_search_draw_enabled = false;
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
        else if (arg == "--enable-adaptive-aspiration")
            options.aspiration_config.enabled = true;
        else if (arg == "--disable-adaptive-aspiration")
            options.aspiration_config.enabled = false;
        else if (arg == "--aspiration-min-depth")
            options.aspiration_config.min_depth = integer(next(), arg);
        else if (arg == "--aspiration-delta-base-cp")
            options.aspiration_config.delta_base_cp = integer(next(), arg);
        else if (arg == "--aspiration-delta-divisor")
            options.aspiration_config.delta_divisor = integer(next(), arg);
        else if (arg == "--aspiration-expansion-factor-per-mille")
            options.aspiration_config.expansion_factor_per_mille =
                integer(next(), arg);
        else if (arg == "--aspiration-max-fail-high-reductions")
            options.aspiration_config.max_fail_high_reductions =
                integer(next(), arg);
        else if (arg == "--aspiration-mean-score-new-weight-per-mille")
            options.aspiration_config.mean_score_new_weight_per_mille =
                integer(next(), arg);
        else if (arg == "--aspiration-max-researches")
            options.aspiration_config.max_researches = integer(next(), arg);
        else if (arg == "--aspiration-mean-score-clamp-cp")
            options.aspiration_config.mean_score_clamp_cp = integer(next(), arg);
        else if (arg == "--write-control-cache")
            options.write_control_cache = next();
        else if (arg == "--control-cache")
            options.control_cache = next();
        else if (arg == "--control-cache-identity-sha256")
            options.control_cache_identity_sha256 = next();
        else if (arg == "--control-cache-evaluator-sha256")
            options.control_cache_evaluator_sha256 = next();
        else if (arg == "--control-cache-model-sha256")
            options.control_cache_model_sha256 = next();
        else if (arg == "--control-cache-dataset-sha256")
            options.control_cache_dataset_sha256 = next();
        else if (arg == "--write-detail-jsonl")
            options.write_detail_jsonl = next();
        // Kept as a compatibility alias for a short-lived pre-merge runner.
        else if (arg == "--position-details")
            options.write_detail_jsonl = next();
#endif
#if defined(CHESS_EVALUATE_NNUE_V43)
        else if (arg == "--tt-mb")
            options.tt_mb = bounded_size(next(), arg, 4'096);
        else if (arg == "--tt-bucket-size")
            options.tt_bucket_size =
                bounded_size(next(), arg, 64);
        else if (arg == "--enable-reuse-stale-tt-scores")
            options.reuse_stale_tt_scores = true;
        else if (arg == "--disable-reuse-stale-tt-scores")
            options.reuse_stale_tt_scores = false;
        else if (arg == "--enable-reuse-deeper-tt-scores")
            options.reuse_deeper_tt_scores = true;
        else if (arg == "--disable-reuse-deeper-tt-scores")
            options.reuse_deeper_tt_scores = false;
#endif
        else throw std::runtime_error("unknown argument: " + std::string(arg));
    }
    if (options.dataset.empty()) throw std::runtime_error("--dataset is required");
    if (options.depth < 2) throw std::runtime_error("depth must be >= 2");
    if (options.candidate_time_ms < 0) {
        throw std::runtime_error("candidate time must be >= 0");
    }
    if (options.candidate_max_depth < 2) {
        throw std::runtime_error("candidate max depth must be >= 2");
    }
    if (options.depth > 64 || options.candidate_max_depth > 64) {
        throw std::runtime_error("evaluator search depth must be <= 64");
    }
    if (options.detail_threshold < 0) {
        throw std::runtime_error("detail threshold must be >= 0");
    }
    if (options.ranking_target_abs_cp < 1
        || options.ranking_target_abs_cp > 1'000'000) {
        throw std::runtime_error(
            "ranking target abs cp must be in [1, 1000000]");
    }
    validate_selective_config(options.config);
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
    require_closed_range(
        options.aspiration_config.min_depth, 2, 64, "aspiration min depth");
    require_closed_range(
        options.aspiration_config.delta_base_cp,
        1, 1'000'000, "aspiration base delta");
    require_closed_range(
        options.aspiration_config.delta_divisor,
        1, 1'000'000'000, "aspiration delta divisor");
    if (options.aspiration_config.expansion_factor_per_mille < 1'000
        || options.aspiration_config.expansion_factor_per_mille > 1'000'000) {
        throw std::runtime_error(
            "aspiration expansion factor must be in [1000, 1000000] per mille");
    }
    require_closed_range(
        options.aspiration_config.max_fail_high_reductions,
        0, 64, "aspiration max fail-high reductions");
    if (options.aspiration_config.mean_score_new_weight_per_mille < 0
        || options.aspiration_config.mean_score_new_weight_per_mille > 1'000) {
        throw std::runtime_error(
            "aspiration mean-score weight must be in [0, 1000]");
    }
    require_closed_range(
        options.aspiration_config.max_researches,
        1, 64, "aspiration max researches");
    require_closed_range(
        options.aspiration_config.mean_score_clamp_cp,
        1, 1'000'000, "aspiration mean-score clamp");
    const bool writes_control_cache = !options.write_control_cache.empty();
    const bool reads_control_cache = !options.control_cache.empty();
    if (writes_control_cache && reads_control_cache) {
        throw std::runtime_error(
            "--write-control-cache and --control-cache are mutually exclusive");
    }
    const auto valid_sha256 = [](std::string_view value) {
        return value.size() == 64
            && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
                return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
            });
    };
    const bool has_cache_identity =
        !options.control_cache_identity_sha256.empty()
        || !options.control_cache_evaluator_sha256.empty()
        || !options.control_cache_model_sha256.empty()
        || !options.control_cache_dataset_sha256.empty();
    if (writes_control_cache || reads_control_cache) {
        if (!valid_sha256(options.control_cache_identity_sha256)
            || !valid_sha256(options.control_cache_evaluator_sha256)
            || !valid_sha256(options.control_cache_model_sha256)
            || !valid_sha256(options.control_cache_dataset_sha256)) {
            throw std::runtime_error(
                "control-cache identity requires four lowercase SHA-256 values");
        }
    } else if (has_cache_identity) {
        throw std::runtime_error(
            "control-cache identity supplied without a cache mode");
    }
    if (writes_control_cache && !options.write_detail_jsonl.empty()) {
        throw std::runtime_error(
            "--write-detail-jsonl cannot be used while building a control cache");
    }
#endif
#if defined(CHESS_EVALUATE_NNUE_V43)
    // Fail closed: the first V43 aspiration experiment is intentionally a
    // single-variable test.  Do not allow a CLI typo or a future default to
    // silently mix TT capacity/replacement semantics into the tune.
    if (options.tt_mb != 64) {
        throw std::runtime_error("V43 aspiration tune requires --tt-mb 64");
    }
    if (options.tt_bucket_size != 4) {
        throw std::runtime_error(
            "V43 aspiration tune requires --tt-bucket-size 4");
    }
    if (options.reuse_stale_tt_scores) {
        throw std::runtime_error(
            "V43 aspiration tune requires stale TT score reuse disabled");
    }
    if (options.reuse_deeper_tt_scores) {
        throw std::runtime_error(
            "V43 aspiration tune requires deeper TT score reuse disabled");
    }
    if (!options.twofold_search_draw_enabled) {
        throw std::runtime_error(
            "V43 aspiration tune requires twofold search draw enabled");
    }
#endif
    if (options.objective != "cp" && options.objective != "wdl") {
        throw std::runtime_error("--objective must be cp or wdl");
    }
    return options;
}

struct ControlExactnessTotals {
    std::uint64_t searches = 0;
    std::uint64_t live_root_searches = 0;
    std::uint64_t live_child_searches = 0;
    std::uint64_t cached_root_results = 0;
    std::uint64_t full_window_fallbacks = 0;
    std::uint64_t range_conflicts = 0;
    std::uint64_t unresolved_ranges = 0;
};

enum class ControlSearchRole {
    Root,
    CandidateChild,
};

struct CandidateRepetitionTotals {
    std::uint64_t history_aware_searches = 0;
    std::uint64_t threefold_draws = 0;
    std::uint64_t search_cycle_draws = 0;
    std::uint64_t fifty_move_draws = 0;
    std::uint64_t tt_score_suppressions = 0;
};

constexpr int EvaluatorMateScoreThreshold = chess::CheckmateScore - 1024;

bool evaluator_mate_score(int score) {
    return std::abs(score) >= EvaluatorMateScoreThreshold;
}

int normalize_child_score_to_parent(int score) {
    if (score >= EvaluatorMateScoreThreshold) {
        return score - 1;
    }
    if (score <= -EvaluatorMateScoreThreshold) {
        return score + 1;
    }
    return score;
}

chess::SearchResult run_control(
    chess::NnueSearcherV36& searcher,
    const chess::Position& position,
    int depth,
    ControlExactnessTotals& totals,
    ControlSearchRole role
) {
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    const chess::SearchResult result = searcher.search_best_move(position, depth);
    const auto& exactness = searcher.exactness_stats();
    ++totals.searches;
    if (role == ControlSearchRole::Root) {
        ++totals.live_root_searches;
    } else {
        ++totals.live_child_searches;
    }
    totals.full_window_fallbacks += exactness.full_window_fallbacks;
    totals.range_conflicts += exactness.range_conflicts;
    totals.unresolved_ranges += exactness.unresolved_ranges;
    if (!searcher.last_search_exact()
        || exactness.unresolved_ranges != 0) {
        throw std::runtime_error(
            std::string("V36 strict teacher returned a non-exact ")
            + (role == ControlSearchRole::Root ? "root" : "candidate-child")
            + " result");
    }
    return result;
}

#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
#if defined(CHESS_EVALUATE_NNUE_V43)
constexpr std::string_view ControlCacheMagic =
    "nnue_v43_v36_root_cache_v1";
#else
constexpr std::string_view ControlCacheMagic =
    "nnue_v42_v36_root_cache_v1";
#endif
constexpr std::string_view ControlCacheColumns =
    "index\tsample_hash\tcalibration_ply\tzobrist_key\tmove_value\tscore"
    "\tnodes\tresult_depth\tstopped\tfull_window_fallbacks\trange_conflicts"
    "\tunresolved_ranges";

struct CachedControlRoot {
    chess::SearchResult result{};
    chess::NnueSearcherV36::ExactnessStats exactness{};
};

std::vector<std::string> tab_fields(std::string_view line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t tab = line.find('\t', start);
        fields.emplace_back(line.substr(
            start, tab == std::string_view::npos ? tab : tab - start));
        if (tab == std::string_view::npos) return fields;
        start = tab + 1;
    }
}

void write_json_string(std::ostream& output, std::string_view value) {
    constexpr char Hex[] = "0123456789abcdef";
    output.put('"');
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u00" << Hex[ch >> 4] << Hex[ch & 0x0F];
                } else {
                    output.put(static_cast<char>(ch));
                }
                break;
        }
    }
    output.put('"');
}

std::uint64_t unsigned_decimal(
    std::string_view text,
    std::string_view field
) {
    if (text.empty()
        || !std::all_of(text.begin(), text.end(), [](unsigned char ch) {
            return std::isdigit(ch);
        })) {
        throw std::runtime_error(
            "bad unsigned control-cache field: " + std::string(field));
    }
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(std::string(text), &consumed);
    } catch (...) {
        throw std::runtime_error(
            "bad unsigned control-cache field: " + std::string(field));
    }
    if (consumed != text.size()) {
        throw std::runtime_error(
            "bad unsigned control-cache field: " + std::string(field));
    }
    return static_cast<std::uint64_t>(value);
}

std::int64_t signed_decimal(
    std::string_view text,
    std::string_view field
) {
    std::size_t consumed = 0;
    long long value = 0;
    try {
        value = std::stoll(std::string(text), &consumed);
    } catch (...) {
        throw std::runtime_error(
            "bad signed control-cache field: " + std::string(field));
    }
    if (consumed != text.size()) {
        throw std::runtime_error(
            "bad signed control-cache field: " + std::string(field));
    }
    return static_cast<std::int64_t>(value);
}

void require_cache_header(
    std::ifstream& input,
    std::string_view expected_name,
    std::string_view expected_value
) {
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("truncated control cache header");
    }
    const auto fields = tab_fields(line);
    if (fields.size() != 2
        || fields[0] != expected_name
        || fields[1] != expected_value) {
        throw std::runtime_error(
            "control cache header mismatch: " + std::string(expected_name));
    }
}

bool is_legal_cached_move(
    const chess::Position& position,
    chess::Move move
) {
    const auto moves = chess::generate_legal_moves(position);
    if (moves.empty()) return move.value == 0;
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

void sync_cache_file(const std::filesystem::path& path) {
#if defined(__unix__) || defined(__APPLE__)
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::runtime_error("failed to open control cache for fsync");
    }
    const int sync_result = ::fsync(descriptor);
    const int close_result = ::close(descriptor);
    if (sync_result != 0 || close_result != 0) {
        throw std::runtime_error("failed to fsync control cache");
    }
#else
    (void)path;
#endif
}

constexpr std::string_view PositionDetailsSchema =
    "nnue_selective_position_details_jsonl_v1";

class AtomicPositionDetails {
public:
    explicit AtomicPositionDetails(std::string_view requested_path) {
        if (requested_path.empty()) return;
        final_path_ = std::filesystem::path(requested_path);
        if (std::filesystem::exists(final_path_)) {
            throw std::runtime_error(
                "refusing to overwrite existing position details: "
                + final_path_.string());
        }
        const std::string nonce = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count())
            + "-" + std::to_string(std::random_device{}());
        temporary_path_ = final_path_.string() + ".tmp-" + nonce;
        output_.open(temporary_path_, std::ios::out | std::ios::trunc);
        if (!output_) {
            throw std::runtime_error(
                "failed to create temporary position details: "
                + temporary_path_.string());
        }
        output_ << std::setprecision(
            std::numeric_limits<double>::max_digits10);
        enabled_ = true;
    }

    AtomicPositionDetails(const AtomicPositionDetails&) = delete;
    AtomicPositionDetails& operator=(const AtomicPositionDetails&) = delete;

    ~AtomicPositionDetails() {
        if (!temporary_path_.empty()) {
            output_.close();
            std::error_code ignored;
            std::filesystem::remove(temporary_path_, ignored);
        }
    }

    bool enabled() const { return enabled_; }

    std::ostream& stream() {
        if (!enabled_) {
            throw std::logic_error("position-details stream is disabled");
        }
        return output_;
    }

    void commit() {
        if (!enabled_) return;
        output_.flush();
        if (!output_) {
            throw std::runtime_error("failed to flush position details");
        }
        output_.close();
        if (!output_) {
            throw std::runtime_error("failed to close position details");
        }
        sync_cache_file(temporary_path_);

        // The temporary lives beside the final path. Hard-link publication is
        // atomic and, unlike rename-overwrite, fails closed if a competing
        // evaluator already published this candidate's sidecar.
        std::error_code publish_error;
        std::filesystem::create_hard_link(
            temporary_path_, final_path_, publish_error);
        if (publish_error) {
            if (std::filesystem::exists(final_path_)) {
                throw std::runtime_error(
                    "refusing to overwrite existing position details: "
                    + final_path_.string());
            }
            throw std::runtime_error(
                "failed to atomically publish position details: "
                + publish_error.message());
        }
        std::error_code remove_error;
        std::filesystem::remove(temporary_path_, remove_error);
        if (remove_error) {
            throw std::runtime_error(
                "published position details but failed to remove temporary "
                "link: " + remove_error.message());
        }
        temporary_path_.clear();
        enabled_ = false;
        committed_ = true;
    }

    bool committed() const { return committed_; }

private:
    std::filesystem::path final_path_;
    std::filesystem::path temporary_path_;
    std::ofstream output_;
    bool enabled_ = false;
    bool committed_ = false;
};

void write_control_cache(
    const Options& options,
    const std::vector<Sample>& samples,
    chess::NnueSearcherV36& control
) {
    const std::filesystem::path path = options.write_control_cache;
    if (std::filesystem::exists(path)) {
        throw std::runtime_error(
            "refusing to overwrite existing control cache: " + path.string());
    }
    const std::string nonce = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count())
        + "-" + std::to_string(std::random_device{}());
    const std::filesystem::path temporary =
        path.string() + ".tmp-" + nonce;
    struct TemporaryGuard {
        std::filesystem::path path;
        ~TemporaryGuard() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } temporary_guard{temporary};
    std::ofstream output(temporary, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "failed to create temporary control cache: "
            + temporary.string());
    }
    output << ControlCacheMagic << '\n'
           << "identity_sha256\t"
           << options.control_cache_identity_sha256 << '\n'
           << "evaluator_sha256\t"
           << options.control_cache_evaluator_sha256 << '\n'
           << "model_sha256\t"
           << options.control_cache_model_sha256 << '\n'
           << "dataset_sha256\t"
           << options.control_cache_dataset_sha256 << '\n'
           << "depth\t" << options.depth << '\n'
           << "offset\t" << options.offset << '\n'
           << "count\t" << samples.size() << '\n'
           << ControlCacheColumns << '\n';

    ControlExactnessTotals totals;
    std::uint64_t nodes = 0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const chess::SearchResult result = run_control(
            control,
            samples[index].position,
            options.depth,
            totals,
            ControlSearchRole::Root);
        const auto exactness = control.exactness_stats();
        if (result.stopped || result.depth != options.depth
            || result.nodes == 0
            || !is_legal_cached_move(
                samples[index].position, result.best_move)
            || exactness.unresolved_ranges != 0) {
            throw std::runtime_error(
                "invalid exact V36 root result while building cache at row "
                + std::to_string(index));
        }
        nodes += result.nodes;
        output
            << index << '\t'
            << samples[index].hash << '\t'
            << samples[index].calibration_ply << '\t'
            << samples[index].position.zobrist_key << '\t'
            << result.best_move.value << '\t'
            << result.score << '\t'
            << result.nodes << '\t'
            << result.depth << '\t'
            << (result.stopped ? 1 : 0) << '\t'
            << exactness.full_window_fallbacks << '\t'
            << exactness.range_conflicts << '\t'
            << exactness.unresolved_ranges << '\n';
        if (!output) {
            throw std::runtime_error(
                "failed while writing control cache row "
                + std::to_string(index));
        }
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to flush control cache");
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to close control cache");
    }
    sync_cache_file(temporary);
    // Publish without a replacement window. Hard-link creation is atomic and
    // fails if another builder already published the immutable final path.
    // A crash before this point leaves only an ignored unique temporary file;
    // a crash after it leaves a complete final cache.
    std::error_code publish_error;
    std::filesystem::create_hard_link(temporary, path, publish_error);
    if (publish_error) {
        if (std::filesystem::exists(path)) {
            throw std::runtime_error(
                "refusing to overwrite existing control cache: "
                + path.string());
        }
        throw std::runtime_error(
            "failed to atomically publish control cache: "
            + publish_error.message());
    }
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    if (remove_error) {
        throw std::runtime_error(
            "published control cache but failed to remove temporary link: "
            + remove_error.message());
    }
    std::cout
        << "{\"control_cache_written\":true"
        << ",\"control_cache_schema\":\"" << ControlCacheMagic << '"'
        << ",\"control_cache_identity_sha256\":\""
        << options.control_cache_identity_sha256 << '"'
        << ",\"count\":" << samples.size()
        << ",\"depth\":" << options.depth
        << ",\"control_nodes\":" << nodes
        << ",\"control_exact_searches\":" << totals.searches
        << ",\"control_exact_full_window_fallbacks\":"
        << totals.full_window_fallbacks
        << ",\"control_exact_range_conflicts\":"
        << totals.range_conflicts
        << ",\"control_exact_unresolved_ranges\":"
        << totals.unresolved_ranges
        << "}\n";
}

std::vector<CachedControlRoot> load_control_cache(
    const Options& options,
    const std::vector<Sample>& samples
) {
    std::ifstream input(options.control_cache);
    if (!input) {
        throw std::runtime_error(
            "failed to open control cache: " + options.control_cache);
    }
    std::string line;
    if (!std::getline(input, line) || line != ControlCacheMagic) {
        throw std::runtime_error("control cache magic mismatch");
    }
    require_cache_header(
        input, "identity_sha256", options.control_cache_identity_sha256);
    require_cache_header(
        input, "evaluator_sha256", options.control_cache_evaluator_sha256);
    require_cache_header(
        input, "model_sha256", options.control_cache_model_sha256);
    require_cache_header(
        input, "dataset_sha256", options.control_cache_dataset_sha256);
    require_cache_header(input, "depth", std::to_string(options.depth));
    require_cache_header(input, "offset", std::to_string(options.offset));
    require_cache_header(input, "count", std::to_string(samples.size()));
    if (!std::getline(input, line) || line != ControlCacheColumns) {
        throw std::runtime_error("control cache columns mismatch");
    }

    std::vector<CachedControlRoot> roots;
    roots.reserve(samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (!std::getline(input, line)) {
            throw std::runtime_error(
                "truncated control cache at row " + std::to_string(index));
        }
        const auto fields = tab_fields(line);
        if (fields.size() != 12) {
            throw std::runtime_error(
                "control cache column count mismatch at row "
                + std::to_string(index));
        }
        if (unsigned_decimal(fields[0], "index") != index
            || fields[1] != samples[index].hash
            || unsigned_decimal(fields[2], "calibration_ply")
                != static_cast<std::uint64_t>(
                    samples[index].calibration_ply)
            || unsigned_decimal(fields[3], "zobrist_key")
                != samples[index].position.zobrist_key) {
            throw std::runtime_error(
                "control cache row identity/order mismatch at row "
                + std::to_string(index));
        }
        const std::uint64_t move_value =
            unsigned_decimal(fields[4], "move_value");
        const std::int64_t score = signed_decimal(fields[5], "score");
        const std::uint64_t nodes = unsigned_decimal(fields[6], "nodes");
        const std::int64_t result_depth =
            signed_decimal(fields[7], "result_depth");
        const std::uint64_t stopped = unsigned_decimal(fields[8], "stopped");
        if (move_value > std::numeric_limits<std::uint16_t>::max()
            || score < -chess::Infinity || score > chess::Infinity
            || nodes == 0
            || result_depth != options.depth
            || stopped != 0) {
            throw std::runtime_error(
                "invalid control cache root result at row "
                + std::to_string(index));
        }
        CachedControlRoot root;
        root.result.best_move = chess::Move{
            static_cast<std::uint16_t>(move_value)};
        root.result.score = static_cast<int>(score);
        root.result.nodes = nodes;
        root.result.depth = static_cast<int>(result_depth);
        root.result.stopped = false;
        root.exactness.full_window_fallbacks =
            unsigned_decimal(fields[9], "full_window_fallbacks");
        root.exactness.range_conflicts =
            unsigned_decimal(fields[10], "range_conflicts");
        root.exactness.unresolved_ranges =
            unsigned_decimal(fields[11], "unresolved_ranges");
        if (root.exactness.unresolved_ranges != 0
            || !is_legal_cached_move(
                samples[index].position, root.result.best_move)) {
            throw std::runtime_error(
                "non-exact or illegal control cache result at row "
                + std::to_string(index));
        }
        roots.push_back(root);
    }
    if (std::getline(input, line)) {
        throw std::runtime_error("control cache has trailing rows");
    }
    return roots;
}

void use_cached_control_root(
    const CachedControlRoot& root,
    ControlExactnessTotals& totals
) {
    ++totals.searches;
    ++totals.cached_root_results;
    totals.full_window_fallbacks += root.exactness.full_window_fallbacks;
    totals.range_conflicts += root.exactness.range_conflicts;
    totals.unresolved_ranges += root.exactness.unresolved_ranges;
}
#endif

chess::SearchResult run_candidate(
    CandidateSearcher& searcher,
    const chess::Position& position,
    const Options& options,
    CandidateRepetitionTotals& repetition_totals
) {
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    searcher.clear_selective_stats();
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
    // Production UCI enters the history-aware overload. A history containing
    // only the root is the truthful context for an isolated dataset sample:
    // it enables path-local repetition and the 50-move rule without
    // manufacturing earlier occurrences.
    const std::array<chess::HashKey, 1> game_history{
        position.zobrist_key};
    chess::SearchResult result;
    if (options.candidate_time_ms > 0) {
        result = searcher.search_best_move(
            position,
            chess::SearchLimits{
                .max_depth = options.candidate_max_depth,
                .move_time = std::chrono::milliseconds{
                    options.candidate_time_ms},
            },
            game_history);
    } else {
        // Adaptive aspiration exists only in iterative deepening, so the
        // fixed-depth evaluator uses SearchLimits rather than the one-shot
        // integer-depth compatibility API.
        result = searcher.search_best_move(
            position,
            chess::SearchLimits{.max_depth = options.depth},
            game_history);
    }
    ++repetition_totals.history_aware_searches;
#else
    // Preserve V38--V41 evaluator search semantics.
    chess::SearchResult result = options.candidate_time_ms > 0
        ? searcher.search_best_move(
            position,
            chess::SearchLimits{
                .max_depth = options.candidate_max_depth,
                .move_time = std::chrono::milliseconds{
                    options.candidate_time_ms},
            })
        : searcher.search_best_move(position, options.depth);
#endif
    const auto& repetition = searcher.repetition_stats();
    repetition_totals.threefold_draws += repetition.threefold_draws;
    repetition_totals.search_cycle_draws += repetition.search_cycle_draws;
    repetition_totals.fifty_move_draws += repetition.fifty_move_draws;
    repetition_totals.tt_score_suppressions +=
        repetition.tt_score_suppressions;
    return result;
}

int strict_score_of_move(
    chess::NnueSearcherV36& control,
    const chess::Position& position,
    chess::Move move,
    int depth,
    ControlExactnessTotals& exactness_totals
) {
    chess::Position child = position;
    child.make_move(move);
    int score = -run_control(
        control, child, depth - 1, exactness_totals,
        ControlSearchRole::CandidateChild).score;
    // A standalone child search starts mate distance at ply zero, whereas the
    // same child is searched at ply one below the teacher root. Normalize the
    // strict child score back to the parent-root convention before comparing.
    return normalize_child_score_to_parent(score);
}

template<typename Value>
double percentile95(std::vector<Value> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1];
}

std::size_t phase_index(const chess::Position& position) {
    const std::size_t pieces =
        static_cast<std::size_t>(chess::popcount(position.occupancy()));
    return std::min<std::size_t>(
        (pieces - 1) / 4,
        chess::PhaseQuantizedNnueModel::PhaseCount - 1);
}

int position_absolute_ply(const chess::Position& position) {
    return 2 * (position.fullmove_number - 1)
        + (position.side_to_move == chess::Color::Black ? 1 : 0);
}

struct SampleMembership {
    bool ranking = false;
    bool safety = false;
};

SampleMembership sample_membership(
    int static_target_cp,
    int ranking_target_abs_cp,
    bool include_all_in_objective
) {
    const bool safety =
        std::abs(static_target_cp) >= ranking_target_abs_cp;
    return SampleMembership{
        include_all_in_objective || !safety,
        safety,
    };
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const std::vector<Sample> samples = load_samples(options);
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model: " + options.model);
        }
        model.set_neon_dotprod_enabled(true);
        chess::NnueSearcherV36 control(model, 64, 4);
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
        if (!options.write_control_cache.empty()) {
            write_control_cache(options, samples, control);
            return 0;
        }
        const std::vector<CachedControlRoot> cached_control_roots =
            options.control_cache.empty()
            ? std::vector<CachedControlRoot>{}
            : load_control_cache(options, samples);
        const bool uses_control_cache = !options.control_cache.empty();
#else
        constexpr bool uses_control_cache = false;
#endif
        CandidateSearcher candidate(
            model,
#if defined(CHESS_EVALUATE_NNUE_V43)
            options.tt_mb, options.tt_bucket_size,
#else
            64, 4,
#endif
            10, 14, 14'000,
            CandidateSearcher::MoveOrderingWeights{}, options.config);
        candidate.set_twofold_search_draw_enabled(
            options.twofold_search_draw_enabled);
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
        candidate.set_aspiration_config(options.aspiration_config);
#endif
#if defined(CHESS_EVALUATE_NNUE_V43)
        candidate.set_reuse_stale_tt_scores(
            options.reuse_stale_tt_scores);
        candidate.set_reuse_deeper_tt_scores(
            options.reuse_deeper_tt_scores);
        if (candidate.reuse_stale_tt_scores()
            || candidate.reuse_deeper_tt_scores()) {
            throw std::runtime_error(
                "V43 evaluator failed to apply locked TT score policy");
        }
        constexpr std::size_t BytesPerMib = 1024 * 1024;
        const std::size_t candidate_tt_bytes = candidate.tt_entry_count()
            * sizeof(chess::V43SingleBoundTranspositionTable::Entry);
        if (candidate_tt_bytes != options.tt_mb * BytesPerMib) {
            throw std::runtime_error(
                "V43 evaluator TT allocation does not match locked size");
        }
        if (!candidate.twofold_search_draw_enabled()) {
            throw std::runtime_error(
                "V43 evaluator failed to apply locked twofold policy");
        }
#endif

#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
        AtomicPositionDetails position_details(options.write_detail_jsonl);
        std::size_t position_detail_count = 0;
#endif

        std::uint64_t control_nodes = 0, candidate_nodes = 0;
        ControlExactnessTotals control_exactness;
        CandidateRepetitionTotals candidate_repetition;
        std::uint64_t control_us = 0, candidate_us = 0;
        std::uint64_t lmr_searches = 0, lmr_researches = 0;
        std::uint64_t null_searches = 0, null_cutoffs = 0;
        std::uint64_t reverse_futility_evaluations = 0;
        std::uint64_t reverse_futility_cutoffs = 0;
        std::uint64_t late_move_pruned_nodes = 0;
        std::uint64_t late_move_pruned_moves = 0;
        std::uint64_t qsearch_see_evaluations = 0;
        std::uint64_t qsearch_see_pruned_moves = 0;
        std::uint64_t main_search_see_evaluations = 0;
        std::uint64_t main_search_see_pruned_moves = 0;
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
        std::uint64_t aspiration_completed_iterations = 0;
        std::uint64_t aspiration_narrow_iterations = 0;
        std::uint64_t aspiration_narrow_attempts = 0;
        std::uint64_t aspiration_initial_window_successes = 0;
        std::uint64_t aspiration_fail_lows = 0;
        std::uint64_t aspiration_fail_highs = 0;
        std::uint64_t aspiration_reduced_depth_attempts = 0;
        std::uint64_t aspiration_accepted_reduced_depth_iterations = 0;
        std::uint64_t aspiration_accepted_narrow_nominal_depth_sum = 0;
        std::uint64_t aspiration_accepted_narrow_search_depth_sum = 0;
        int aspiration_max_accepted_depth_reduction = 0;
        std::uint64_t aspiration_full_window_fallbacks = 0;
        std::uint64_t aspiration_range_conflict_fallbacks = 0;
        std::uint64_t aspiration_unresolved_ranges = 0;
        std::uint64_t aspiration_retry_limit_fallbacks = 0;
        std::int64_t aspiration_initial_delta_sum_cp = 0;
        int aspiration_initial_delta_min_cp = 0;
        int aspiration_initial_delta_max_cp = 0;
        std::uint64_t aspiration_final_mean_score_count = 0;
        std::int64_t aspiration_final_mean_score_sum_cp = 0;
#endif
        std::uint64_t candidate_depth_sum = 0;
        std::uint64_t candidate_completed_depth_count = 0;
        int candidate_stopped = 0;
        int agreements = 0, ranking_agreements = 0, above100 = 0;
        int ranking_count = 0, safety_count = 0;
        int critical_mistakes = 0;
        int strict_best_score_violations = 0;
        int strict_best_score_max_excess_cp = 0;
        int win_to_draw = 0, win_to_loss = 0, self_mated = 0;
        int safety_critical_mistakes = 0;
        std::int64_t regret_sum = 0;
        double wdl_loss_sum = 0.0;
        std::vector<int> regrets;
        std::vector<double> wdl_losses;
        regrets.reserve(samples.size());
        wdl_losses.reserve(samples.size());
        const std::size_t explicit_ply_count =
            static_cast<std::size_t>(std::count_if(
                samples.begin(), samples.end(),
                [](const Sample& sample) { return sample.has_explicit_ply; }));
        const std::size_t explicit_ply_position_match_count =
            static_cast<std::size_t>(std::count_if(
                samples.begin(), samples.end(),
                [](const Sample& sample) {
                    return sample.has_explicit_ply
                        && sample.calibration_ply
                            == position_absolute_ply(sample.position);
                }));
        const auto [minimum_ply, maximum_ply] = std::minmax_element(
            samples.begin(), samples.end(),
            [](const Sample& left, const Sample& right) {
                return left.calibration_ply < right.calibration_ply;
            });
        const std::uint64_t calibration_ply_sum = std::accumulate(
            samples.begin(), samples.end(), std::uint64_t{0},
            [](std::uint64_t sum, const Sample& sample) {
                return sum + static_cast<std::uint64_t>(sample.calibration_ply);
            });

        for (std::size_t index = 0; index < samples.size(); ++index) {
            const int static_target_cp =
                model.evaluate_cp_rounded(samples[index].position);
            const SampleMembership membership = sample_membership(
                static_target_cp,
                options.ranking_target_abs_cp,
                options.include_all_in_objective);
            const bool ranking_sample = membership.ranking;
            const bool safety_sample = membership.safety;
            ranking_count += ranking_sample;
            safety_count += safety_sample;
            chess::SearchResult base, mutant;
            auto timed_base = [&] {
                const auto start = std::chrono::steady_clock::now();
                base = run_control(
                    control,
                    samples[index].position,
                    options.depth,
                    control_exactness,
                    ControlSearchRole::Root);
                control_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count();
            };
            auto timed_mutant = [&] {
                const auto start = std::chrono::steady_clock::now();
                mutant = run_candidate(
                    candidate,
                    samples[index].position,
                    options,
                    candidate_repetition);
                candidate_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count();
            };
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
            if (uses_control_cache) {
                base = cached_control_roots[index].result;
                use_cached_control_root(
                    cached_control_roots[index], control_exactness);
                timed_mutant();
            } else
#endif
            if ((index & 1U) == 0) { timed_base(); timed_mutant(); }
            else { timed_mutant(); timed_base(); }
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
            if (options.candidate_time_ms == 0
                && (mutant.stopped || mutant.depth != options.depth)) {
                const auto failed_aspiration = candidate.aspiration_stats();
                throw std::runtime_error(
                    std::string(AdaptiveEvaluatorVersion)
                    + " fixed-depth evaluation did not complete requested "
                    "iterative depth at sample " + std::to_string(index)
                    + " (requested=" + std::to_string(options.depth)
                    + ", actual=" + std::to_string(mutant.depth)
                    + ", stopped=" + std::to_string(mutant.stopped)
                    + ", fail_lows="
                    + std::to_string(failed_aspiration.fail_lows)
                    + ", fail_highs="
                    + std::to_string(failed_aspiration.fail_highs)
                    + ", full_window_fallbacks="
                    + std::to_string(
                        failed_aspiration.full_window_fallbacks)
                    + ", range_conflict_fallbacks="
                    + std::to_string(
                        failed_aspiration.range_conflict_fallbacks)
                    + ", unresolved_ranges="
                    + std::to_string(failed_aspiration.unresolved_ranges)
                    + ", retry_limit_fallbacks="
                    + std::to_string(
                        failed_aspiration.retry_limit_fallbacks)
                    + ")");
            }
#endif
            control_nodes += base.nodes;
            candidate_nodes += mutant.nodes;
            candidate_depth_sum += static_cast<std::uint64_t>(mutant.depth);
            candidate_stopped += mutant.stopped;
            candidate_completed_depth_count +=
                options.candidate_time_ms == 0
                && !mutant.stopped && mutant.depth == options.depth;
            const auto stats = candidate.selective_stats();
            lmr_searches += stats.lmr_searches;
            lmr_researches += stats.lmr_researches;
            null_searches += stats.null_move_searches;
            null_cutoffs += stats.null_move_cutoffs;
            reverse_futility_evaluations +=
                stats.reverse_futility_evaluations;
            reverse_futility_cutoffs += stats.reverse_futility_cutoffs;
            late_move_pruned_nodes += stats.late_move_pruned_nodes;
            late_move_pruned_moves += stats.late_move_pruned_moves;
            qsearch_see_evaluations += stats.qsearch_see_evaluations;
            qsearch_see_pruned_moves += stats.qsearch_see_pruned_moves;
            main_search_see_evaluations +=
                stats.main_search_see_evaluations;
            main_search_see_pruned_moves +=
                stats.main_search_see_pruned_moves;
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
            const auto aspiration = candidate.aspiration_stats();
            aspiration_completed_iterations +=
                aspiration.completed_iterations;
            const bool first_aggregate_narrow_iteration =
                aspiration_narrow_iterations == 0;
            aspiration_narrow_iterations += aspiration.narrow_iterations;
            aspiration_narrow_attempts += aspiration.narrow_attempts;
            aspiration_initial_window_successes +=
                aspiration.initial_window_successes;
            aspiration_fail_lows += aspiration.fail_lows;
            aspiration_fail_highs += aspiration.fail_highs;
            aspiration_reduced_depth_attempts +=
                aspiration.reduced_depth_attempts;
            aspiration_accepted_reduced_depth_iterations +=
                aspiration.accepted_reduced_depth_iterations;
            aspiration_accepted_narrow_nominal_depth_sum +=
                aspiration.accepted_narrow_nominal_depth_sum;
            aspiration_accepted_narrow_search_depth_sum +=
                aspiration.accepted_narrow_search_depth_sum;
            aspiration_max_accepted_depth_reduction = std::max(
                aspiration_max_accepted_depth_reduction,
                aspiration.max_accepted_depth_reduction);
            aspiration_full_window_fallbacks +=
                aspiration.full_window_fallbacks;
            aspiration_range_conflict_fallbacks +=
                aspiration.range_conflict_fallbacks;
            aspiration_unresolved_ranges += aspiration.unresolved_ranges;
            aspiration_retry_limit_fallbacks +=
                aspiration.retry_limit_fallbacks;
            aspiration_initial_delta_sum_cp +=
                aspiration.initial_delta_sum_cp;
            if (aspiration.narrow_iterations > 0) {
                if (first_aggregate_narrow_iteration) {
                    aspiration_initial_delta_min_cp =
                        aspiration.initial_delta_min_cp;
                    aspiration_initial_delta_max_cp =
                        aspiration.initial_delta_max_cp;
                } else {
                    aspiration_initial_delta_min_cp = std::min(
                        aspiration_initial_delta_min_cp,
                        aspiration.initial_delta_min_cp);
                    aspiration_initial_delta_max_cp = std::max(
                        aspiration_initial_delta_max_cp,
                        aspiration.initial_delta_max_cp);
                }
            }
            if (aspiration.has_final_mean_score) {
                ++aspiration_final_mean_score_count;
                aspiration_final_mean_score_sum_cp +=
                    aspiration.final_mean_score_cp;
            }
#endif

            int strict_candidate_score = base.score;
            if (base.best_move == mutant.best_move) {
                ++agreements;
            } else {
                strict_candidate_score = strict_score_of_move(
                    control,
                    samples[index].position,
                    mutant.best_move,
                    options.depth,
                    control_exactness);
            }
            if (strict_candidate_score > base.score) {
                ++strict_best_score_violations;
                strict_best_score_max_excess_cp = std::max(
                    strict_best_score_max_excess_cp,
                    strict_candidate_score - base.score);
                throw std::runtime_error(
                    "strict candidate move exceeds exact V36 teacher best at "
                    "sample " + std::to_string(index)
                    + " (teacher=" + std::to_string(base.score)
                    + ", candidate="
                    + std::to_string(strict_candidate_score) + ")");
            }
            const int regret = std::max(0, base.score - strict_candidate_score);
            const double wdl_loss = chess::wdl_calibration::expected_score_loss(
                base.score,
                strict_candidate_score,
                static_cast<int>(phase_index(samples[index].position)),
                samples[index].calibration_ply);
            if (ranking_sample) {
                regrets.push_back(regret);
                regret_sum += regret;
                wdl_losses.push_back(wdl_loss);
                wdl_loss_sum += wdl_loss;
                above100 += regret > 100;
                ranking_agreements += base.best_move == mutant.best_move;
            }
            const bool base_mate = evaluator_mate_score(base.score);
            const bool candidate_mate =
                evaluator_mate_score(strict_candidate_score);
            constexpr int WinningThresholdCp = 500;
            constexpr int DrawThresholdCp = 100;
            const bool base_winning = base.score >= WinningThresholdCp;
            const bool became_draw =
                base_winning
                && std::abs(strict_candidate_score) <= DrawThresholdCp;
            const bool became_loss =
                base_winning
                && strict_candidate_score < -DrawThresholdCp;
            const bool candidate_is_mated =
                strict_candidate_score <= -EvaluatorMateScoreThreshold;
            win_to_draw += became_draw;
            win_to_loss += became_loss;
            const bool newly_self_mated = candidate_is_mated
                && base.score > -DrawThresholdCp;
            self_mated += newly_self_mated;
            critical_mistakes +=
                became_draw || became_loss || newly_self_mated;
            safety_critical_mistakes += safety_sample
                && (became_draw || became_loss || newly_self_mated);
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
            if (position_details.enabled()) {
                std::ostream& detail = position_details.stream();
                detail << "{\"schema\":";
                write_json_string(detail, PositionDetailsSchema);
                detail
                    << ",\"index\":" << index
                    << ",\"global_index\":"
                    << static_cast<std::size_t>(options.offset) + index
                    << ",\"category\":";
                write_json_string(detail, samples[index].category);
                detail << ",\"sample_hash\":";
                write_json_string(detail, samples[index].hash);
                detail
                    << ",\"calibration_ply\":"
                    << samples[index].calibration_ply
                    << ",\"phase\":" << phase_index(samples[index].position)
                    << ",\"static_target_cp\":" << static_target_cp
                    << ",\"ranking_sample\":"
                    << (ranking_sample ? "true" : "false")
                    << ",\"safety_sample\":"
                    << (safety_sample ? "true" : "false")
                    << ",\"control_move\":";
                write_json_string(
                    detail, chess::move_to_string(base.best_move));
                detail << ",\"candidate_move\":";
                write_json_string(
                    detail, chess::move_to_string(mutant.best_move));
                detail
                    << ",\"control_score\":" << base.score
                    << ",\"candidate_strict_score\":"
                    << strict_candidate_score
                    << ",\"control_root_nodes\":" << base.nodes
                    << ",\"candidate_nodes\":" << mutant.nodes
                    << ",\"candidate_depth\":" << mutant.depth
                    << ",\"candidate_stopped\":"
                    << (mutant.stopped ? "true" : "false")
                    << ",\"move_match\":"
                    << (base.best_move == mutant.best_move ? "true" : "false")
                    << ",\"cp_regret\":" << regret
                    << ",\"wdl_loss\":" << wdl_loss
                    << ",\"win_to_draw\":"
                    << (became_draw ? "true" : "false")
                    << ",\"win_to_loss\":"
                    << (became_loss ? "true" : "false")
                    << ",\"self_mated\":"
                    << (newly_self_mated ? "true" : "false")
                    << ",\"critical_mistake\":"
                    << ((became_draw || became_loss || newly_self_mated)
                        ? "true" : "false")
                    << "}\n";
                if (!detail) {
                    throw std::runtime_error(
                        "failed while writing position detail at row "
                        + std::to_string(index));
                }
                ++position_detail_count;
            }
#endif
            if (options.detail_threshold > 0
                && regret >= options.detail_threshold) {
                std::cerr
                    << "detail"
                    << "\tindex=" << index
                    << "\tcategory=" << samples[index].category
                    << "\thash=" << samples[index].hash
                    << "\tpayload=" << samples[index].payload
                    << "\tcalibration_ply="
                    << samples[index].calibration_ply
                    << "\texplicit_ply="
                    << samples[index].has_explicit_ply
                    << "\tstatic_target_cp=" << static_target_cp
                    << "\tmetric_group="
                    << (ranking_sample ? "ranking" : "safety")
                    << "\tcontrol_move=" << chess::move_to_string(base.best_move)
                    << "\tcandidate_move=" << chess::move_to_string(mutant.best_move)
                    << "\tcontrol_score=" << base.score
                    << "\tcandidate_strict_score=" << strict_candidate_score
                    << "\tregret=" << regret
                    << "\twdl_loss=" << wdl_loss
                    << "\tcontrol_mate=" << base_mate
                    << "\tcandidate_mate=" << candidate_mate
                    << '\n';
            }
        }
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
        if (position_details.enabled()) {
            position_details.commit();
        }
        if (!options.write_detail_jsonl.empty()
            && (!position_details.committed()
                || position_detail_count != samples.size())) {
            throw std::runtime_error(
                "position-details atomic completion invariant failed");
        }
#endif
        const double count = static_cast<double>(samples.size());
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
        if (candidate_repetition.history_aware_searches != samples.size()) {
            throw std::runtime_error(
                std::string(AdaptiveEvaluatorVersion)
                + " evaluator bypassed history-aware candidate search");
        }
#endif
#if defined(CHESS_EVALUATE_NNUE_V43)
        if (options.candidate_time_ms == 0) {
            const std::uint64_t expected_iterations =
                static_cast<std::uint64_t>(samples.size())
                * static_cast<std::uint64_t>(options.depth);
            if (options.aspiration_config.enabled
                && aspiration_completed_iterations != expected_iterations) {
                throw std::runtime_error(
                    "V43 aspiration completed-iteration invariant failed");
            }
            if (candidate_stopped != 0
                || candidate_depth_sum != expected_iterations
                || candidate_completed_depth_count != samples.size()) {
                throw std::runtime_error(
                    "V43 fixed-depth completion invariant failed");
            }
        }
        if (!options.aspiration_config.enabled
            && (aspiration_completed_iterations != 0
                || aspiration_narrow_iterations != 0
                || aspiration_narrow_attempts != 0
                || aspiration_initial_window_successes != 0
                || aspiration_fail_lows != 0
                || aspiration_fail_highs != 0
                || aspiration_reduced_depth_attempts != 0
                || aspiration_accepted_reduced_depth_iterations != 0
                || aspiration_accepted_narrow_nominal_depth_sum != 0
                || aspiration_accepted_narrow_search_depth_sum != 0
                || aspiration_max_accepted_depth_reduction != 0
                || aspiration_full_window_fallbacks != 0
                || aspiration_range_conflict_fallbacks != 0
                || aspiration_unresolved_ranges != 0
                || aspiration_retry_limit_fallbacks != 0
                || aspiration_initial_delta_sum_cp != 0
                || aspiration_initial_delta_min_cp != 0
                || aspiration_initial_delta_max_cp != 0
                || aspiration_final_mean_score_count != 0
                || aspiration_final_mean_score_sum_cp != 0)) {
            throw std::runtime_error(
                "V43 disabled aspiration emitted nonzero telemetry");
        }
        if (aspiration_range_conflict_fallbacks != 0
            || aspiration_unresolved_ranges != 0) {
            throw std::runtime_error(
                "V43 scalar aspiration emitted impossible range telemetry");
        }
#endif
        const double ranking_count_double =
            static_cast<double>(ranking_count);
        auto selective_config_document = [](const auto& config) {
            std::ostringstream output;
            output << std::setprecision(
                std::numeric_limits<double>::max_digits10);
            write_selective_config_json(output, config);
            return output.str();
        };
        const std::string requested_selective_config_json =
            selective_config_document(options.config);
        const std::string effective_selective_config_json =
            selective_config_document(candidate.selective_config());
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
        auto aspiration_config_document = [](const auto& config) {
            std::ostringstream output;
            output << std::setprecision(
                std::numeric_limits<double>::max_digits10);
            write_aspiration_config_json(output, config);
            return output.str();
        };
        const std::string requested_aspiration_config_json =
            aspiration_config_document(options.aspiration_config);
        const std::string effective_aspiration_config_json =
            aspiration_config_document(candidate.aspiration_config());
#endif
        std::cout << std::setprecision(
            std::numeric_limits<double>::max_digits10)
            << "{\"count\":" << samples.size()
            << ",\"ranking_count\":" << ranking_count
            << ",\"safety_count\":" << safety_count
            << ",\"ranking_target_abs_cp\":" << options.ranking_target_abs_cp
            << ",\"dataset_ply_schema\":\""
            << (explicit_ply_count == samples.size()
                ? "category_hash_ply_payload_v1"
                : explicit_ply_count == 0
                    ? "legacy_payload_derived_v0"
                    : "mixed_explicit_and_legacy")
            << '"'
            << ",\"dataset_explicit_ply_count\":" << explicit_ply_count
            << ",\"dataset_explicit_ply_position_match_count\":"
            << explicit_ply_position_match_count
            << ",\"dataset_calibration_ply_min\":"
            << minimum_ply->calibration_ply
            << ",\"dataset_calibration_ply_max\":"
            << maximum_ply->calibration_ply
            << ",\"dataset_calibration_ply_mean\":"
            << static_cast<double>(calibration_ply_sum) / count
            << ",\"include_all_in_objective\":"
            << (options.include_all_in_objective ? "true" : "false")
            << ",\"heuristics_reset_per_sample\":true"
            << ",\"objective\":\"" << options.objective << '"'
            << ",\"wdl_formula\":\""
            << chess::wdl_calibration::formula << '"'
            << ",\"wdl_calibration_run\":\""
            << chess::wdl_calibration::calibration_run << '"'
            << ",\"depth\":" << options.depth
#if defined(CHESS_EVALUATE_NNUE_V43)
            << ",\"candidate_engine\":\"NnueSearcherV43\""
            << ",\"searcher_version\":43"
            << ",\"tt_mb\":" << options.tt_mb
            << ",\"tt_bucket_size\":" << options.tt_bucket_size
            << ",\"tt_layout\":\"scalar_single_bound_aos_4x16b_64b\""
            << ",\"reuse_stale_tt_scores\":"
            << (candidate.reuse_stale_tt_scores() ? "true" : "false")
            << ",\"reuse_deeper_tt_scores\":"
            << (candidate.reuse_deeper_tt_scores() ? "true" : "false")
            << ",\"tt_depth_policy\":\"Exact\""
            << ",\"tt_generation_policy\":\"CurrentOnly\""
#endif
            << ",\"candidate_time_ms\":" << options.candidate_time_ms
            << ",\"candidate_max_depth\":" << options.candidate_max_depth
            << ",\"control_root_source\":\""
            << (uses_control_cache ? "immutable_cache" : "live_search")
            << '"'
            << ",\"control_time_source\":\""
            << (uses_control_cache
                ? "not_measured_cached"
                : "measured_live_root")
            << '"'
            << ",\"control_cache_identity_sha256\":"
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
            << (uses_control_cache
                ? "\"" + options.control_cache_identity_sha256 + "\""
                : "null")
#else
            << "null"
#endif
            << ",\"twofold_search_draw_enabled\":"
            << (candidate.twofold_search_draw_enabled() ? "true" : "false")
            << ",\"candidate_repetition_history_aware_searches\":"
            << candidate_repetition.history_aware_searches
            << ",\"candidate_repetition_threefold_draws\":"
            << candidate_repetition.threefold_draws
            << ",\"candidate_repetition_search_cycle_draws\":"
            << candidate_repetition.search_cycle_draws
            << ",\"candidate_repetition_fifty_move_draws\":"
            << candidate_repetition.fifty_move_draws
            << ",\"candidate_repetition_tt_score_suppressions\":"
            << candidate_repetition.tt_score_suppressions
            << ",\"requested_selective_config\":"
            << requested_selective_config_json
            << ",\"effective_selective_config\":"
            << effective_selective_config_json
            << ",\"selective_config_applied_exactly\":"
            << (requested_selective_config_json
                    == effective_selective_config_json
                ? "true" : "false")
            // Backwards-compatible aggregate field consumed by V42/V43 tuners.
            << ",\"selective_config\":"
            << effective_selective_config_json
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
            << ",\"candidate_search_mode\":\"iterative_depth\""
            << ",\"aspiration_policy\":\""
            << (options.aspiration_config.enabled
                ? "adaptive_mean_score"
                : "legacy_fixed_50_cp")
            << '"'
            << ",\"requested_aspiration_config\":"
            << requested_aspiration_config_json
            << ",\"effective_aspiration_config\":"
            << effective_aspiration_config_json
            << ",\"aspiration_config_applied_exactly\":"
            << (requested_aspiration_config_json
                    == effective_aspiration_config_json
                ? "true" : "false")
            << ",\"aspiration_config\":"
            << effective_aspiration_config_json
            << ",\"detail_jsonl_schema\":\""
            << PositionDetailsSchema << '"'
            << ",\"detail_jsonl_written\":"
            << (options.write_detail_jsonl.empty() ? "false" : "true")
            << ",\"detail_jsonl_count\":" << position_detail_count
#endif
            << ",\"candidate_mean_depth\":"
            << static_cast<double>(candidate_depth_sum) / count
#if defined(CHESS_EVALUATE_NNUE_V43)
            << ",\"candidate_completed_depth_count\":"
            << candidate_completed_depth_count
#endif
            << ",\"candidate_stopped_pct\":"
            << 100.0 * candidate_stopped / count
            << ",\"control_nodes\":" << control_nodes
            << ",\"candidate_nodes\":" << candidate_nodes
            << ",\"control_exact_searches\":"
            << control_exactness.searches
            << ",\"control_live_root_searches\":"
            << control_exactness.live_root_searches
            << ",\"control_live_child_searches\":"
            << control_exactness.live_child_searches
            << ",\"control_cached_root_results\":"
            << control_exactness.cached_root_results
            << ",\"control_exact_full_window_fallbacks\":"
            << control_exactness.full_window_fallbacks
            << ",\"control_exact_range_conflicts\":"
            << control_exactness.range_conflicts
            << ",\"control_exact_unresolved_ranges\":"
            << control_exactness.unresolved_ranges
            << ",\"strict_best_score_violations\":"
            << strict_best_score_violations
            << ",\"strict_best_score_max_excess_cp\":"
            << strict_best_score_max_excess_cp
            << ",\"node_ratio\":" << static_cast<double>(candidate_nodes) / control_nodes
            << ",\"time_ratio\":";
        if (uses_control_cache) {
            std::cout << "null";
        } else {
            std::cout << static_cast<double>(candidate_us) / control_us;
        }
        std::cout
            << ",\"mean_root_regret\":"
            << (ranking_count > 0
                ? static_cast<double>(regret_sum) / ranking_count_double
                : 0.0)
            << ",\"p95_root_regret\":" << percentile95(regrets)
            << ",\"mean_wdl_loss\":"
            << (ranking_count > 0
                ? wdl_loss_sum / ranking_count_double
                : 0.0)
            << ",\"p95_wdl_loss\":" << percentile95(wdl_losses)
            << ",\"objective_loss\":"
            << (options.objective == "wdl"
                ? (ranking_count > 0
                    ? wdl_loss_sum / ranking_count_double
                    : 0.0)
                : (ranking_count > 0
                    ? static_cast<double>(regret_sum) / ranking_count_double
                    : 0.0))
            << ",\"above_100_cp_pct\":"
            << (ranking_count > 0
                ? 100.0 * above100 / ranking_count_double
                : 0.0)
            << ",\"critical_mistakes\":" << critical_mistakes
            << ",\"safety_critical_mistakes\":"
            << safety_critical_mistakes
            << ",\"win_to_draw\":" << win_to_draw
            << ",\"win_to_loss\":" << win_to_loss
            << ",\"self_mated\":" << self_mated
            << ",\"move_agreement_pct\":" << 100.0 * agreements / count
            << ",\"ranking_move_agreement_pct\":"
            << (ranking_count > 0
                ? 100.0 * ranking_agreements / ranking_count_double
                : 0.0)
            << ",\"lmr_searches\":" << lmr_searches
            << ",\"lmr_researches\":" << lmr_researches
            << ",\"null_move_searches\":" << null_searches
            << ",\"null_move_cutoffs\":" << null_cutoffs
            << ",\"reverse_futility_evaluations\":"
            << reverse_futility_evaluations
            << ",\"reverse_futility_cutoffs\":"
            << reverse_futility_cutoffs
            << ",\"late_move_pruned_nodes\":"
            << late_move_pruned_nodes
            << ",\"late_move_pruned_moves\":"
            << late_move_pruned_moves
            << ",\"qsearch_see_evaluations\":"
            << qsearch_see_evaluations
            << ",\"qsearch_see_pruned_moves\":"
            << qsearch_see_pruned_moves
            << ",\"main_search_see_evaluations\":"
            << main_search_see_evaluations
            << ",\"main_search_see_pruned_moves\":"
            << main_search_see_pruned_moves
#if defined(CHESS_EVALUATE_NNUE_V42) \
    || defined(CHESS_EVALUATE_NNUE_V43)
            << ",\"aspiration_completed_iterations\":"
            << aspiration_completed_iterations
            << ",\"aspiration_narrow_iterations\":"
            << aspiration_narrow_iterations
            << ",\"aspiration_narrow_attempts\":"
            << aspiration_narrow_attempts
            << ",\"aspiration_initial_window_successes\":"
            << aspiration_initial_window_successes
            << ",\"aspiration_fail_lows\":"
            << aspiration_fail_lows
            << ",\"aspiration_fail_highs\":"
            << aspiration_fail_highs
            << ",\"aspiration_reduced_depth_attempts\":"
            << aspiration_reduced_depth_attempts
            << ",\"aspiration_accepted_reduced_depth_iterations\":"
            << aspiration_accepted_reduced_depth_iterations
            << ",\"aspiration_accepted_narrow_nominal_depth_sum\":"
            << aspiration_accepted_narrow_nominal_depth_sum
            << ",\"aspiration_accepted_narrow_search_depth_sum\":"
            << aspiration_accepted_narrow_search_depth_sum
            << ",\"aspiration_accepted_depth_ratio\":"
            << (aspiration_accepted_narrow_nominal_depth_sum > 0
                ? static_cast<double>(
                    aspiration_accepted_narrow_search_depth_sum)
                    / static_cast<double>(
                        aspiration_accepted_narrow_nominal_depth_sum)
                : 1.0)
            << ",\"aspiration_max_accepted_depth_reduction\":"
            << aspiration_max_accepted_depth_reduction
            << ",\"aspiration_full_window_fallbacks\":"
            << aspiration_full_window_fallbacks
            << ",\"aspiration_range_conflict_fallbacks\":"
            << aspiration_range_conflict_fallbacks
            << ",\"aspiration_unresolved_ranges\":"
            << aspiration_unresolved_ranges
            << ",\"aspiration_retry_limit_fallbacks\":"
            << aspiration_retry_limit_fallbacks
            << ",\"aspiration_initial_delta_sum_cp\":"
            << aspiration_initial_delta_sum_cp
            << ",\"aspiration_initial_delta_min_cp\":"
            << aspiration_initial_delta_min_cp
            << ",\"aspiration_initial_delta_max_cp\":"
            << aspiration_initial_delta_max_cp
            << ",\"aspiration_final_mean_score_count\":"
            << aspiration_final_mean_score_count
            << ",\"aspiration_final_mean_score_sum_cp\":"
            << aspiration_final_mean_score_sum_cp
#endif
            << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
