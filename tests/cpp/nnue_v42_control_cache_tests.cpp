#define main nnue_v42_evaluator_cli_main_for_test
#include "../../tools/analyze/evaluate_nnue_v38_selective.cpp"
#undef main

#include <cassert>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "failed to read test cache");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) lines.push_back(line);
    return lines;
}

void write_lines(
    const std::filesystem::path& path,
    const std::vector<std::string>& lines
) {
    std::ofstream output(path);
    require(static_cast<bool>(output), "failed to write test cache");
    for (const std::string& line : lines) output << line << '\n';
    output.close();
    require(static_cast<bool>(output), "failed to close test cache");
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

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        throw std::runtime_error(
            "usage: nnue_v42_control_cache_tests <nnue-model.bin>");
    }

    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path()
        / ("nnue-v42-control-cache-test-" + std::to_string(nonce));
    std::filesystem::create_directories(directory);
    try {
        // Including every sample in the objective must not erase the separate
        // high-|static eval| safety diagnostics.
        constexpr std::array<int, 5> StaticTargetsCp{
            0, 1499, 1500, -1500, 2300,
        };
        int ranking_count = 0;
        int safety_count = 0;
        for (const int target_cp : StaticTargetsCp) {
            const SampleMembership membership = sample_membership(
                target_cp, 1500, true);
            ranking_count += membership.ranking;
            safety_count += membership.safety;
        }
        require(ranking_count == static_cast<int>(StaticTargetsCp.size()),
            "all-position objective excluded a sample");
        require(safety_count == 3,
            "all-position objective erased safety diagnostics");
        require(
            normalize_child_score_to_parent(chess::CheckmateScore)
                == chess::CheckmateScore - 1,
            "positive child mate score was not shifted to parent ply");
        require(
            normalize_child_score_to_parent(-chess::CheckmateScore)
                == -chess::CheckmateScore + 1,
            "negative child mate score was not shifted to parent ply");
        require(
            normalize_child_score_to_parent(EvaluatorMateScoreThreshold)
                == EvaluatorMateScoreThreshold - 1,
            "canonical positive mate threshold was not normalized");
        require(
            normalize_child_score_to_parent(
                EvaluatorMateScoreThreshold - 1)
                == EvaluatorMateScoreThreshold - 1,
            "non-mate score was incorrectly normalized");

        const std::filesystem::path dataset = directory / "dataset.tsv";
        {
            std::ofstream output(dataset);
            output
                << "phase7\trow0\t0\t"
                << "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
                   "RNBQKBNR w KQkq - 0 1\n"
                << "phase7\trow1\t1\t"
                << "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/"
                   "RNBQKBNR b KQkq e3 0 1\n"
                << "phase0\trow2\t78\t"
                << "7k/8/8/8/8/8/8/KQ6 w - - 0 40\n"
                << "phase0\trow3\t79\t"
                << "7k/8/8/8/8/8/8/KQ6 b - - 0 40\n";
        }

#if defined(CHESS_EVALUATE_NNUE_V43)
        // The tuner control is V43 with adaptive aspiration disabled. Its
        // search still must complete every requested iterative depth, while
        // all adaptive-only telemetry deliberately remains zero.
        std::vector<std::string> baseline_arguments{
            "evaluate_nnue_v43_selective",
            "--dataset", dataset.string(),
            "--model", argv[1],
            "--count", "1",
            "--depth", "3",
            "--objective", "wdl",
            "--include-all-in-objective",
            "--enable-twofold-search-draw",
            "--tt-mb", "64",
            "--tt-bucket-size", "4",
            "--disable-reuse-stale-tt-scores",
            "--disable-reuse-deeper-tt-scores",
            "--disable-adaptive-aspiration",
            "--aspiration-max-researches", "6",
            "--aspiration-mean-score-clamp-cp", "1500",
        };
        std::vector<char*> baseline_argv;
        baseline_argv.reserve(baseline_arguments.size());
        for (std::string& argument : baseline_arguments) {
            baseline_argv.push_back(argument.data());
        }
        std::ostringstream baseline_output;
        std::ostringstream baseline_error;
        std::streambuf* saved_output = std::cout.rdbuf(
            baseline_output.rdbuf());
        std::streambuf* saved_error = std::cerr.rdbuf(
            baseline_error.rdbuf());
        const int baseline_exit = nnue_v42_evaluator_cli_main_for_test(
            static_cast<int>(baseline_argv.size()), baseline_argv.data());
        std::cout.rdbuf(saved_output);
        std::cerr.rdbuf(saved_error);
        require(
            baseline_exit == 0,
            "V43 adaptive-off evaluator baseline failed: "
                + baseline_error.str());
        require(
            baseline_output.str().find(
                "\"candidate_completed_depth_count\":1")
                != std::string::npos,
            "V43 adaptive-off baseline did not complete requested depth");
        require(
            baseline_output.str().find(
                "\"aspiration_completed_iterations\":0")
                != std::string::npos,
            "V43 adaptive-off baseline emitted completed telemetry");
        require(
            baseline_output.str().find(
                "\"aspiration_narrow_attempts\":0")
                != std::string::npos,
            "V43 adaptive-off baseline emitted narrow telemetry");
#endif

        Options options;
        options.dataset = dataset.string();
        options.model = argv[1];
        options.depth = 3;
        options.write_control_cache = (directory / "control.tsv").string();
        options.control_cache_identity_sha256 = std::string(64, 'a');
        options.control_cache_evaluator_sha256 = std::string(64, 'b');
        options.control_cache_model_sha256 = std::string(64, 'c');
        options.control_cache_dataset_sha256 = std::string(64, 'd');

        const std::vector<Sample> samples = load_samples(options);
        require(samples.size() == 4, "unexpected fixture sample count");
        chess::PhaseQuantizedNnueModel model;
        require(model.load(options.model), "failed to load NNUE model");
        model.set_neon_dotprod_enabled(true);
        chess::NnueSearcherV36 control(model, 64, 4);

        chess::Position mate_in_one;
        require(mate_in_one.set_fen(
            "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"),
            "failed to parse mate-in-one fixture");
        ControlExactnessTotals mate_totals;
        const chess::SearchResult mate_root = run_control(
            control, mate_in_one, 3, mate_totals, ControlSearchRole::Root);
        require(evaluator_mate_score(mate_root.score),
            "strict teacher missed mate-in-one fixture");
        const int mate_move_score = strict_score_of_move(
            control,
            mate_in_one,
            mate_root.best_move,
            3,
            mate_totals);
        require(mate_move_score == mate_root.score,
            "parent-ply mate normalization differs from exact teacher root");
        require(mate_totals.searches == 2,
            "mate root/child exact-search accounting is wrong");

        std::ostringstream discarded_summary;
        std::streambuf* old_output = std::cout.rdbuf(
            discarded_summary.rdbuf());
        write_control_cache(options, samples, control);
        std::cout.rdbuf(old_output);
        require(
            discarded_summary.str().find("\"control_cache_written\":true")
                != std::string::npos,
            "cache writer did not report success");
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            require(
                entry.path().filename().string().find("control.tsv.tmp-")
                    == std::string::npos,
                "atomic cache writer left a temporary file");
        }

        options.control_cache = options.write_control_cache;
        options.write_control_cache.clear();
        const std::vector<CachedControlRoot> cached =
            load_control_cache(options, samples);
        require(cached.size() == samples.size(), "cache count changed");

        // Root results and node counts must be exactly reusable.  Only wall
        // time is deliberately absent from the immutable cache.
        ControlExactnessTotals live_totals;
        for (std::size_t index = 0; index < samples.size(); ++index) {
            const chess::SearchResult live = run_control(
                control,
                samples[index].position,
                options.depth,
                live_totals,
                ControlSearchRole::Root);
            require(live.best_move == cached[index].result.best_move,
                "cached root move differs from live teacher");
            require(live.score == cached[index].result.score,
                "cached root score differs from live teacher");
            require(live.nodes == cached[index].result.nodes,
                "cached root nodes differ from live teacher");
            require(live.depth == cached[index].result.depth,
                "cached root depth differs from live teacher");
        }
        require(live_totals.live_root_searches == samples.size(),
            "live root counter is wrong");

        ControlExactnessTotals cached_totals;
        for (const CachedControlRoot& root : cached) {
            use_cached_control_root(root, cached_totals);
        }
        require(cached_totals.cached_root_results == samples.size(),
            "cached root counter is wrong");
        require(cached_totals.live_root_searches == 0,
            "cache unexpectedly ran a live root");

        // A move other than the cached teacher best must still execute a
        // fresh exact V36 depth-1 child search; child scores are never cached.
        chess::Move alternative{};
        for (const chess::Move move
             : chess::generate_legal_moves(samples[0].position)) {
            if (move != cached[0].result.best_move) {
                alternative = move;
                break;
            }
        }
        require(alternative.value != 0, "fixture lacks an alternate root move");
        (void)strict_score_of_move(
            control,
            samples[0].position,
            alternative,
            options.depth,
            cached_totals);
        require(cached_totals.live_child_searches == 1,
            "candidate child was not rescored live");
        require(cached_totals.live_root_searches == 0,
            "candidate child rescore accidentally ran a root role");

        const std::filesystem::path cache_path = options.control_cache;
        const std::string cache_text = [&] {
            std::ifstream input(cache_path);
            std::ostringstream text;
            text << input.rdbuf();
            return text.str();
        }();
        require(cache_text.find("time_") == std::string::npos,
            "control cache contains a timing field");
        require(cache_text.find("wall") == std::string::npos,
            "control cache contains wall time");

        require_throws([&] {
            Options overwrite = options;
            overwrite.write_control_cache = overwrite.control_cache;
            overwrite.control_cache.clear();
            write_control_cache(overwrite, samples, control);
        }, "refusing to overwrite");

        Options wrong_identity = options;
        wrong_identity.control_cache_identity_sha256 = std::string(64, 'e');
        require_throws([&] {
            (void)load_control_cache(wrong_identity, samples);
        }, "identity_sha256");

        const std::vector<std::string> original = read_lines(cache_path);
        require(original.size() == 9 + samples.size(),
            "unexpected cache line count");

        auto tampered = original;
        tampered[9].replace(tampered[9].find("row0"), 4, "xxxx");
        const std::filesystem::path tampered_path = directory / "tampered.tsv";
        write_lines(tampered_path, tampered);
        Options tampered_options = options;
        tampered_options.control_cache = tampered_path.string();
        require_throws([&] {
            (void)load_control_cache(tampered_options, samples);
        }, "identity/order mismatch at row 0");

        auto reordered = original;
        std::swap(reordered[9], reordered[10]);
        const std::filesystem::path reordered_path = directory / "reordered.tsv";
        write_lines(reordered_path, reordered);
        Options reordered_options = options;
        reordered_options.control_cache = reordered_path.string();
        require_throws([&] {
            (void)load_control_cache(reordered_options, samples);
        }, "identity/order mismatch at row 0");

        auto truncated = original;
        truncated.pop_back();
        const std::filesystem::path truncated_path = directory / "truncated.tsv";
        write_lines(truncated_path, truncated);
        Options truncated_options = options;
        truncated_options.control_cache = truncated_path.string();
        require_throws([&] {
            (void)load_control_cache(truncated_options, samples);
        }, "truncated control cache");

        std::filesystem::remove_all(directory);
        std::cout << "V42 immutable root cache parity and guards passed\n";
        return 0;
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
}
