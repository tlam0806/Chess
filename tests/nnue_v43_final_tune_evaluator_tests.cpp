#define main nnue_v43_final_tune_evaluator_cli_main_for_test
#include "../tools/evaluate_nnue_v38_selective.cpp"
#undef main

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template<typename Function>
void require_throws(Function&& function, std::string_view needle) {
    try {
        function();
    } catch (const std::exception& error) {
        require(
            std::string_view(error.what()).find(needle)
                != std::string_view::npos,
            "exception did not contain expected diagnostic");
        return;
    }
    throw std::runtime_error("operation unexpectedly succeeded");
}

std::vector<std::string> base_arguments() {
    return {
        "evaluate_nnue_v43_selective",
        "--dataset", "unused.tsv",
        "--depth", "3",
        "--enable-twofold-search-draw",
        "--tt-mb", "64",
        "--tt-bucket-size", "4",
        "--disable-reuse-stale-tt-scores",
        "--disable-reuse-deeper-tt-scores",
        "--disable-adaptive-aspiration",
    };
}

Options parse_arguments(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) argv.push_back(argument.data());
    return parse(static_cast<int>(argv.size()), argv.data());
}

void append(
    std::vector<std::string>& arguments,
    std::initializer_list<std::string_view> suffix
) {
    for (const std::string_view value : suffix) {
        arguments.emplace_back(value);
    }
}

std::string read_all(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "failed to read test artifact");
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        throw std::runtime_error(
            "usage: nnue_v43_final_tune_evaluator_tests <nnue-model.bin>");
    }

    // All 21 selective fields must survive the CLI exactly. In particular,
    // size_t values are parsed as signed and range-checked before conversion.
    auto full = base_arguments();
    append(full, {
        "--enable-lmr", "--lmr-base", "0.55", "--lmr-divisor", "2.3",
        "--lmr-min-depth", "7", "--lmr-min-move-index", "8",
        "--enable-null-move", "--null-min-depth", "2",
        "--null-reduction", "4",
        "--enable-reverse-futility", "--reverse-futility-max-depth", "3",
        "--reverse-futility-base-margin", "125",
        "--reverse-futility-margin-per-depth", "225",
        "--enable-late-move-pruning", "--late-move-pruning-max-depth", "4",
        "--late-move-pruning-base", "5",
        "--late-move-pruning-depth-multiplier", "3",
        "--enable-qsearch-see-pruning", "--qsearch-see-threshold", "-125",
        "--enable-main-search-see-pruning",
        "--main-search-see-max-depth", "2",
        "--main-search-see-margin-per-depth", "250",
        // Disabled adaptive aspiration is exactly the V43 legacy fixed +/-50
        // SearchLimits path, but its dormant fields may still be arbitrary and
        // must round-trip for a later aspiration refresh.
        "--aspiration-max-researches", "11",
        "--aspiration-mean-score-clamp-cp", "2345",
    });
    const Options parsed = parse_arguments(full);
    require(parsed.config.enable_lmr, "LMR enable did not parse");
    require(parsed.config.lmr_base == 0.55, "LMR base did not parse");
    require(parsed.config.lmr_divisor == 2.3, "LMR divisor did not parse");
    require(parsed.config.lmr_min_depth == 7, "LMR depth did not parse");
    require(parsed.config.lmr_min_move_index == 8,
        "LMR move index did not parse");
    require(parsed.config.null_move_min_depth == 2,
        "NMP depth did not parse");
    require(parsed.config.null_move_reduction == 4,
        "NMP reduction did not parse");
    require(parsed.config.reverse_futility_base_margin == 125,
        "RFP base did not parse");
    require(parsed.config.reverse_futility_margin_per_depth == 225,
        "RFP slope did not parse");
    require(parsed.config.late_move_pruning_base == 5,
        "LMP base did not parse");
    require(parsed.config.late_move_pruning_depth_multiplier == 3,
        "LMP multiplier did not parse");
    require(parsed.config.qsearch_see_threshold == -125,
        "QSEE threshold did not parse");
    require(parsed.config.main_search_see_max_depth == 2,
        "main SEE depth did not parse");
    require(parsed.config.main_search_see_margin_per_depth == 250,
        "main SEE margin did not parse");
    require(!parsed.aspiration_config.enabled,
        "legacy fixed50 mode unexpectedly enabled adaptive aspiration");
    require(parsed.aspiration_config.max_researches == 11,
        "arbitrary aspiration retry count was rejected");
    require(parsed.aspiration_config.mean_score_clamp_cp == 2345,
        "arbitrary aspiration clamp was rejected");

    struct InvalidCase {
        std::string_view flag;
        std::string_view value;
        std::string_view diagnostic;
    };
    const std::vector<InvalidCase> invalid_cases{
        {"--lmr-base", "nan", "finite"},
        {"--lmr-base", "-0.1", "safe range"},
        {"--lmr-divisor", "inf", "finite"},
        {"--lmr-divisor", "0", "> 0"},
        {"--lmr-min-depth", "0", "safe range"},
        {"--lmr-min-move-index", "-1", "must be in"},
        {"--null-min-depth", "0", "safe range"},
        {"--null-reduction", "0", "safe range"},
        {"--reverse-futility-max-depth", "0", "safe range"},
        {"--reverse-futility-base-margin", "-1", "safe range"},
        {"--reverse-futility-margin-per-depth", "-1", "safe range"},
        {"--late-move-pruning-max-depth", "0", "safe range"},
        {"--late-move-pruning-base", "-1", "must be in"},
        {"--late-move-pruning-depth-multiplier", "-1", "must be in"},
        {"--qsearch-see-threshold", "1000001", "safe range"},
        {"--main-search-see-max-depth", "0", "safe range"},
        {"--main-search-see-margin-per-depth", "-1", "safe range"},
        {"--tt-mb", "-1", "must be in"},
        {"--tt-bucket-size", "-1", "must be in"},
    };
    for (const InvalidCase& invalid : invalid_cases) {
        auto arguments = base_arguments();
        arguments.emplace_back(invalid.flag);
        arguments.emplace_back(invalid.value);
        require_throws(
            [&] { (void)parse_arguments(arguments); }, invalid.diagnostic);
    }

    // Integer parsing is whole-token strict; "3junk" must not silently mean 3.
    {
        auto arguments = base_arguments();
        append(arguments, {"--lmr-min-depth", "3junk"});
        require_throws(
            [&] { (void)parse_arguments(arguments); }, "bad integer");
    }

    // Adaptive refresh is not artificially locked to the first experiment's
    // retry/clamp values; the TT/repetition locks remain mandatory.
    {
        auto arguments = base_arguments();
        append(arguments, {
            "--enable-adaptive-aspiration",
            "--aspiration-min-depth", "4",
            "--aspiration-delta-base-cp", "67",
            "--aspiration-delta-divisor", "32123",
            "--aspiration-expansion-factor-per-mille", "2190",
            "--aspiration-max-fail-high-reductions", "3",
            "--aspiration-mean-score-new-weight-per-mille", "417",
            "--aspiration-max-researches", "9",
            "--aspiration-mean-score-clamp-cp", "2222",
        });
        const Options adaptive = parse_arguments(arguments);
        require(adaptive.aspiration_config.enabled,
            "adaptive refresh was not enabled");
        require(adaptive.aspiration_config.max_researches == 9,
            "adaptive retry count did not round-trip");
        require(adaptive.aspiration_config.mean_score_clamp_cp == 2222,
            "adaptive clamp did not round-trip");
    }

    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path()
        / ("nnue-v43-final-evaluator-test-" + std::to_string(nonce));
    std::filesystem::create_directories(directory);
    try {
        const std::filesystem::path dataset = directory / "dataset.tsv";
        {
            std::ofstream output(dataset);
            output
                << "phase7\trow\\\"0\t0\t"
                << "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
                   "RNBQKBNR w KQkq - 0 1\n";
        }
        const std::filesystem::path details = directory / "details.jsonl";
        std::vector<std::string> run{
            "evaluate_nnue_v43_selective",
            "--dataset", dataset.string(),
            "--model", argv[1],
            "--count", "1",
            "--depth", "2",
            "--objective", "wdl",
            "--include-all-in-objective",
            "--enable-twofold-search-draw",
            "--tt-mb", "64",
            "--tt-bucket-size", "4",
            "--disable-reuse-stale-tt-scores",
            "--disable-reuse-deeper-tt-scores",
            "--disable-adaptive-aspiration",
            "--aspiration-max-researches", "11",
            "--aspiration-mean-score-clamp-cp", "2345",
            "--write-detail-jsonl", details.string(),
        };
        std::vector<char*> run_argv;
        for (std::string& argument : run) run_argv.push_back(argument.data());
        std::ostringstream summary;
        std::ostringstream errors;
        std::streambuf* saved_output = std::cout.rdbuf(summary.rdbuf());
        std::streambuf* saved_error = std::cerr.rdbuf(errors.rdbuf());
        const int exit_code = nnue_v43_final_tune_evaluator_cli_main_for_test(
            static_cast<int>(run_argv.size()), run_argv.data());
        std::cout.rdbuf(saved_output);
        std::cerr.rdbuf(saved_error);
        require(exit_code == 0,
            "V43 detail run failed: " + errors.str());
        require(summary.str().find("\"requested_selective_config\":{")
                != std::string::npos,
            "summary omitted requested selective config");
        require(summary.str().find("\"effective_selective_config\":{")
                != std::string::npos,
            "summary omitted effective selective config");
        require(summary.str().find("\"selective_config_applied_exactly\":true")
                != std::string::npos,
            "effective selective config changed");
        require(summary.str().find("\"aspiration_config_applied_exactly\":true")
                != std::string::npos,
            "effective aspiration config changed");
        require(summary.str().find(
                "\"aspiration_policy\":\"legacy_fixed_50_cp\"")
                != std::string::npos,
            "adaptive-off evaluator did not identify legacy fixed50 policy");
        require(summary.str().find("\"detail_jsonl_count\":1")
                != std::string::npos,
            "summary has wrong detail row count");
        const std::string detail_text = read_all(details);
        require(detail_text.find(
                "\"schema\":\"nnue_selective_position_details_jsonl_v1\"")
                != std::string::npos,
            "detail row omitted schema");
        require(detail_text.find("\"sample_hash\":\"row\\\\\\\"0\"")
                != std::string::npos,
            "detail JSON string escaping is invalid");
        require(detail_text.find("\"control_root_nodes\":")
                != std::string::npos,
            "detail row omitted paired control nodes");
        require(detail_text.find("\"candidate_nodes\":")
                != std::string::npos,
            "detail row omitted candidate nodes");
        require(detail_text.find("\"wdl_loss\":") != std::string::npos,
            "detail row omitted WDL loss");
        require(static_cast<std::size_t>(std::count(
                detail_text.begin(), detail_text.end(), '\n')) == 1,
            "detail JSONL has the wrong row count");
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            require(
                entry.path().filename().string().find("details.jsonl.tmp-")
                    == std::string::npos,
                "atomic writer left a temporary file");
        }

        // A failed/incomplete writer must publish nothing.
        const std::filesystem::path incomplete = directory / "incomplete.jsonl";
        {
            AtomicPositionDetails writer(incomplete.string());
            writer.stream() << "partial\n";
        }
        require(!std::filesystem::exists(incomplete),
            "incomplete sidecar became visible");

        // No-clobber is part of the immutable artifact contract.
        require_throws(
            [&] { AtomicPositionDetails writer(details.string()); },
            "refusing to overwrite");
        require(read_all(details) == detail_text,
            "no-clobber attempt changed detail artifact");
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
        throw;
    }
    std::filesystem::remove_all(directory);
    return 0;
}
