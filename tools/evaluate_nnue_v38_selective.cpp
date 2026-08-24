#include "attacks.hpp"
#include "move.hpp"
#include "nnue_wdl_calibration.hpp"
#include "nnue_searcher_v36.hpp"
#if defined(CHESS_EVALUATE_NNUE_V40)
#include "nnue_searcher_v40.hpp"
#elif defined(CHESS_EVALUATE_NNUE_V39)
#include "nnue_searcher_v39.hpp"
#else
#include "nnue_searcher_v38.hpp"
#endif
#include "phase_quantized_nnue.hpp"
#include "position.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

#if defined(CHESS_EVALUATE_NNUE_V40)
using CandidateSearcher = chess::NnueSearcherV40;
#elif defined(CHESS_EVALUATE_NNUE_V39)
using CandidateSearcher = chess::NnueSearcherV39;
#else
using CandidateSearcher = chess::NnueSearcherV38;
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
    std::string objective = "cp";
    CandidateSearcher::SelectiveConfig config{};
};

struct Sample {
    std::string category;
    std::string hash;
    std::string payload;
    chess::Position position;
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
        const std::string payload = line.substr(second + 1);
        all.push_back(Sample{
            line.substr(0, first),
            line.substr(first + 1, second - first - 1),
            payload,
            position_from_payload(payload),
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
    try { return std::stoi(value); }
    catch (...) { throw std::runtime_error("bad integer for " + std::string(name)); }
}

double real(const char* value, std::string_view name) {
    try { return std::stod(value); }
    catch (...) { throw std::runtime_error("bad number for " + std::string(name)); }
}

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
        else if (arg == "--disable-lmr") options.config.enable_lmr = false;
        else if (arg == "--disable-null-move")
            options.config.enable_null_move = false;
        else if (arg == "--lmr-base") options.config.lmr_base = real(next(), arg);
        else if (arg == "--lmr-divisor") options.config.lmr_divisor = real(next(), arg);
        else if (arg == "--lmr-min-depth") options.config.lmr_min_depth = integer(next(), arg);
        else if (arg == "--lmr-min-move-index")
            options.config.lmr_min_move_index = static_cast<std::size_t>(integer(next(), arg));
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
                static_cast<std::size_t>(integer(next(), arg));
        else if (arg == "--late-move-pruning-depth-multiplier")
            options.config.late_move_pruning_depth_multiplier =
                static_cast<std::size_t>(integer(next(), arg));
        else if (arg == "--disable-qsearch-see-pruning")
            options.config.enable_qsearch_see_pruning = false;
        else if (arg == "--enable-qsearch-see-pruning")
            options.config.enable_qsearch_see_pruning = true;
        else if (arg == "--qsearch-see-threshold")
            options.config.qsearch_see_threshold = integer(next(), arg);
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
    if (options.objective != "cp" && options.objective != "wdl") {
        throw std::runtime_error("--objective must be cp or wdl");
    }
    return options;
}

chess::SearchResult run_control(
    chess::NnueSearcherV36& searcher, const chess::Position& position, int depth
) {
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    return searcher.search_best_move(position, depth);
}

chess::SearchResult run_candidate(
    CandidateSearcher& searcher,
    const chess::Position& position,
    const Options& options
) {
    searcher.clear_tt();
    searcher.clear_search_heuristics();
    searcher.clear_selective_stats();
    if (options.candidate_time_ms > 0) {
        return searcher.search_best_move(
            position,
            chess::SearchLimits{
                .max_depth = options.candidate_max_depth,
                .move_time = std::chrono::milliseconds{
                    options.candidate_time_ms},
            });
    }
    return searcher.search_best_move(position, options.depth);
}

int strict_score_of_move(
    chess::NnueSearcherV36& control,
    const chess::Position& position,
    chess::Move move,
    int depth
) {
    chess::Position child = position;
    child.make_move(move);
    return -run_control(control, child, depth - 1).score;
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

int absolute_ply(const chess::Position& position) {
    return 2 * (position.fullmove_number - 1)
        + (position.side_to_move == chess::Color::Black ? 1 : 0);
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
        CandidateSearcher candidate(
            model, 64, 4, 10, 14, 14'000,
            CandidateSearcher::MoveOrderingWeights{}, options.config);

        std::uint64_t control_nodes = 0, candidate_nodes = 0;
        std::uint64_t control_us = 0, candidate_us = 0;
        std::uint64_t lmr_searches = 0, lmr_researches = 0;
        std::uint64_t null_searches = 0, null_cutoffs = 0;
        std::uint64_t reverse_futility_evaluations = 0;
        std::uint64_t reverse_futility_cutoffs = 0;
        std::uint64_t late_move_pruned_nodes = 0;
        std::uint64_t late_move_pruned_moves = 0;
        std::uint64_t qsearch_see_evaluations = 0;
        std::uint64_t qsearch_see_pruned_moves = 0;
        std::uint64_t candidate_depth_sum = 0;
        int candidate_stopped = 0;
        int agreements = 0, ranking_agreements = 0, above100 = 0;
        int ranking_count = 0, safety_count = 0;
        int critical_mistakes = 0;
        int win_to_draw = 0, win_to_loss = 0, self_mated = 0;
        int safety_critical_mistakes = 0;
        std::int64_t regret_sum = 0;
        double wdl_loss_sum = 0.0;
        std::vector<int> regrets;
        std::vector<double> wdl_losses;
        regrets.reserve(samples.size());
        wdl_losses.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index) {
            const int static_target_cp =
                model.evaluate_cp_rounded(samples[index].position);
            const bool ranking_sample =
                options.include_all_in_objective
                || std::abs(static_target_cp) < options.ranking_target_abs_cp;
            ranking_count += ranking_sample;
            safety_count += !ranking_sample;
            chess::SearchResult base, mutant;
            auto timed_base = [&] {
                const auto start = std::chrono::steady_clock::now();
                base = run_control(control, samples[index].position, options.depth);
                control_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count();
            };
            auto timed_mutant = [&] {
                const auto start = std::chrono::steady_clock::now();
                mutant = run_candidate(candidate, samples[index].position, options);
                candidate_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count();
            };
            if ((index & 1U) == 0) { timed_base(); timed_mutant(); }
            else { timed_mutant(); timed_base(); }
            control_nodes += base.nodes;
            candidate_nodes += mutant.nodes;
            candidate_depth_sum += static_cast<std::uint64_t>(mutant.depth);
            candidate_stopped += mutant.stopped;
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

            int strict_candidate_score = base.score;
            if (base.best_move == mutant.best_move) {
                ++agreements;
            } else {
                strict_candidate_score = strict_score_of_move(
                    control, samples[index].position, mutant.best_move, options.depth);
            }
            const int regret = std::max(0, base.score - strict_candidate_score);
            const double wdl_loss = chess::wdl_calibration::expected_score_loss(
                base.score,
                strict_candidate_score,
                static_cast<int>(phase_index(samples[index].position)),
                absolute_ply(samples[index].position));
            if (ranking_sample) {
                regrets.push_back(regret);
                regret_sum += regret;
                wdl_losses.push_back(wdl_loss);
                wdl_loss_sum += wdl_loss;
                above100 += regret > 100;
                ranking_agreements += base.best_move == mutant.best_move;
            }
            const bool base_mate = std::abs(base.score) >= chess::CheckmateScore - 256;
            const bool candidate_mate =
                std::abs(strict_candidate_score) >= chess::CheckmateScore - 256;
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
                strict_candidate_score <= -chess::CheckmateScore + 256;
            win_to_draw += became_draw;
            win_to_loss += became_loss;
            const bool newly_self_mated = candidate_is_mated
                && base.score > -DrawThresholdCp;
            self_mated += newly_self_mated;
            critical_mistakes +=
                became_draw || became_loss || newly_self_mated;
            safety_critical_mistakes += !ranking_sample
                && (became_draw || became_loss || newly_self_mated);
            if (options.detail_threshold > 0
                && regret >= options.detail_threshold) {
                std::cerr
                    << "detail"
                    << "\tindex=" << index
                    << "\tcategory=" << samples[index].category
                    << "\thash=" << samples[index].hash
                    << "\tpayload=" << samples[index].payload
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
        const double count = static_cast<double>(samples.size());
        const double ranking_count_double =
            static_cast<double>(ranking_count);
        std::cout
            << "{\"count\":" << samples.size()
            << ",\"ranking_count\":" << ranking_count
            << ",\"safety_count\":" << safety_count
            << ",\"ranking_target_abs_cp\":" << options.ranking_target_abs_cp
            << ",\"include_all_in_objective\":"
            << (options.include_all_in_objective ? "true" : "false")
            << ",\"heuristics_reset_per_sample\":true"
            << ",\"objective\":\"" << options.objective << '"'
            << ",\"wdl_formula\":\""
            << chess::wdl_calibration::formula << '"'
            << ",\"wdl_calibration_run\":\""
            << chess::wdl_calibration::calibration_run << '"'
            << ",\"depth\":" << options.depth
            << ",\"candidate_time_ms\":" << options.candidate_time_ms
            << ",\"candidate_max_depth\":" << options.candidate_max_depth
            << ",\"candidate_mean_depth\":"
            << static_cast<double>(candidate_depth_sum) / count
            << ",\"candidate_stopped_pct\":"
            << 100.0 * candidate_stopped / count
            << ",\"control_nodes\":" << control_nodes
            << ",\"candidate_nodes\":" << candidate_nodes
            << ",\"node_ratio\":" << static_cast<double>(candidate_nodes) / control_nodes
            << ",\"time_ratio\":" << static_cast<double>(candidate_us) / control_us
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
            << "}\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
