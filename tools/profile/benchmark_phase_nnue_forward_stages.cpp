#include "phase_quantized_nnue.hpp"
#include "phase_quantized_nnue_stage_benchmark.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<std::string_view, 16> CorpusFens{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
    "rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9",
    "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12",
    "4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
    "8/8/2p5/3p4/3P4/2P5/8/4K1k1 w - - 0 1",
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
    "2r3k1/5ppp/1p2p3/p2pP3/P2P1P2/1P1B2P1/5K1P/2R5 b - - 0 28",
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    "4k3/ppppppp1/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/pppppppp/8/8/8/8/PPP5/4K3 w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPP4/4K3 w kq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/4K3 w kq - 0 1",
};

struct Options {
    std::filesystem::path model;
    std::filesystem::path output;
    std::string backend;
    std::size_t samples = 256;
    std::size_t warmup = 10'000;
    std::array<std::size_t, 4> iterations{
        4'000'000, 20'000'000, 100'000'000, 3'000'000};
    std::size_t repeats = 12;
    std::uint64_t seed = 20260825;
};

std::size_t parse_size(std::string_view text, std::string_view name) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(std::string(text), &consumed);
    if (consumed != text.size() || value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view flag = argv[index];
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + std::string(flag));
        }
        const std::string_view value = argv[++index];
        if (flag == "--model") {
            options.model = value;
        } else if (flag == "--backend") {
            options.backend = value;
        } else if (flag == "--output") {
            options.output = value;
        } else if (flag == "--samples") {
            options.samples = parse_size(value, flag);
        } else if (flag == "--warmup") {
            options.warmup = parse_size(value, flag);
        } else if (flag == "--iterations") {
            options.iterations.fill(parse_size(value, flag));
        } else if (flag == "--s1-iterations") {
            options.iterations[0] = parse_size(value, flag);
        } else if (flag == "--s2-iterations") {
            options.iterations[1] = parse_size(value, flag);
        } else if (flag == "--s3-iterations") {
            options.iterations[2] = parse_size(value, flag);
        } else if (flag == "--full-iterations") {
            options.iterations[3] = parse_size(value, flag);
        } else if (flag == "--repeats") {
            options.repeats = parse_size(value, flag);
        } else if (flag == "--seed") {
            options.seed = std::stoull(std::string(value), nullptr, 0);
        } else if (flag == "--profile") {
            if (value == "smoke") {
                options.samples = 16;
                options.warmup = 500;
                options.iterations.fill(2'000);
                options.repeats = 4;
            } else if (value == "full") {
                options.samples = 256;
                options.warmup = 10'000;
                options.iterations = {
                    4'000'000, 20'000'000, 100'000'000, 3'000'000};
                options.repeats = 12;
                options.seed = 20260825;
            } else {
                throw std::runtime_error("--profile must be smoke or full");
            }
        } else {
            throw std::runtime_error("unknown option: " + std::string(flag));
        }
    }
    if (options.model.empty()) {
        throw std::runtime_error("--model PATH is required");
    }
    if (options.backend != "neon" && options.backend != "vnni") {
        throw std::runtime_error("--backend must be neon or vnni");
    }
    return options;
}

std::vector<chess::Position> make_corpus(
    std::size_t count,
    std::uint64_t seed
) {
    std::vector<chess::Position> result;
    result.reserve(count);
    const std::size_t start = static_cast<std::size_t>(seed % CorpusFens.size());
    // The coprime stride keeps the corpus deterministic on every host while
    // visiting every aux/phase fixture before repeating one.
    for (std::size_t index = 0; index < count; ++index) {
        chess::Position position;
        const std::size_t fen_index =
            (start + index * 5) % CorpusFens.size();
        if (!position.set_fen(CorpusFens[fen_index])) {
            throw std::runtime_error("invalid built-in benchmark FEN");
        }
        result.push_back(position);
    }
    return result;
}

// Small self-contained SHA-256 keeps the benchmark artifact reproducible
// without adding an OpenSSL dependency to chess_core.
std::string sha256_file(const std::filesystem::path& path) {
    constexpr std::array<std::uint32_t, 64> K{
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open model for SHA-256");
    }
    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    const std::uint64_t bit_size = static_cast<std::uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56) bytes.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) {
        bytes.push_back(static_cast<std::uint8_t>(bit_size >> shift));
    }
    std::array<std::uint32_t, 8> hash{
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (std::size_t offset = 0; offset < bytes.size(); offset += 64) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            words[index] = (static_cast<std::uint32_t>(bytes[offset + index * 4]) << 24)
                | (static_cast<std::uint32_t>(bytes[offset + index * 4 + 1]) << 16)
                | (static_cast<std::uint32_t>(bytes[offset + index * 4 + 2]) << 8)
                | bytes[offset + index * 4 + 3];
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const std::uint32_t s0 = std::rotr(words[index - 15], 7)
                ^ std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const std::uint32_t s1 = std::rotr(words[index - 2], 17)
                ^ std::rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        auto state = hash;
        for (std::size_t index = 0; index < 64; ++index) {
            const std::uint32_t s1 = std::rotr(state[4], 6)
                ^ std::rotr(state[4], 11) ^ std::rotr(state[4], 25);
            const std::uint32_t choose =
                (state[4] & state[5]) ^ (~state[4] & state[6]);
            const std::uint32_t temp1 = state[7] + s1 + choose + K[index] + words[index];
            const std::uint32_t s0 = std::rotr(state[0], 2)
                ^ std::rotr(state[0], 13) ^ std::rotr(state[0], 22);
            const std::uint32_t majority = (state[0] & state[1])
                ^ (state[0] & state[2]) ^ (state[1] & state[2]);
            const std::uint32_t temp2 = s0 + majority;
            state = {temp1 + temp2, state[0], state[1], state[2],
                state[3] + temp1, state[4], state[5], state[6]};
        }
        for (std::size_t index = 0; index < hash.size(); ++index) {
            hash[index] += state[index];
        }
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::uint32_t value : hash) output << std::setw(8) << value;
    return output.str();
}

std::string hex64(std::uint64_t value) {
    std::ostringstream output;
    output << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

std::string render_json(
    const Options& options,
    std::string_view model_sha256,
    const chess::PhaseNnueStageBenchmarkResult& result
) {
    std::ostringstream json;
    json << std::setprecision(12)
         << "{\"schema_version\":2"
         << ",\"protocol\":\"phase_nnue_forward_stages_v2\""
         << ",\"status\":\"valid\""
         << ",\"backend_requested\":\"" << options.backend << "\""
         << ",\"kernel\":\"" << result.kernel << "\""
         << ",\"model_sha256\":\"" << model_sha256 << "\""
         << ",\"timing\":{"
         << "\"wall_clock\":\"std::chrono::steady_clock\""
         << ",\"process_cpu_clock\":\"std::clock\""
         << ",\"subtraction\":\"none\""
         << ",\"stage_order\":\"rotating\"}"
         << ",\"corpus\":{\"samples\":" << result.sample_count
         << ",\"seed\":" << options.seed
         << ",\"checksum\":\"" << hex64(result.corpus_checksum) << "\""
         << ",\"nonzero_aux_samples\":" << result.nonzero_aux_samples
         << ",\"phase_histogram\":[";
    for (std::size_t index = 0; index < result.phase_counts.size(); ++index) {
        if (index != 0) json << ',';
        json << result.phase_counts[index];
    }
    json << "]}"
         << ",\"correctness\":{\"status\":\""
         << (result.scalar_parity
                && result.stage_parity
                && result.timed_sink_parity ? "pass" : "fail")
         << "\",\"scalar_parity\":" << (result.scalar_parity ? "true" : "false")
         << ",\"stage_parity\":" << (result.stage_parity ? "true" : "false")
         << ",\"timed_sink_parity\":"
         << (result.timed_sink_parity ? "true" : "false")
         << ",\"checksum\":\"" << hex64(result.stages.back().checksum) << "\"}"
         << ",\"warmup_evaluations\":" << options.warmup
         << ",\"repeats\":" << options.repeats
         << ",\"stages\":{";
    for (std::size_t index = 0; index < result.stages.size(); ++index) {
        const auto& stage = result.stages[index];
        std::vector<std::uint64_t> sorted = stage.elapsed_ns;
        std::vector<std::uint64_t> sorted_cpu = stage.cpu_ns;
        std::sort(sorted.begin(), sorted.end());
        std::sort(sorted_cpu.begin(), sorted_cpu.end());
        const auto conventional_median = [](const auto& values) {
            const std::size_t middle = values.size() / 2;
            return values.size() % 2 == 0
                ? (static_cast<double>(values[middle - 1])
                    + static_cast<double>(values[middle])) / 2.0
                : static_cast<double>(values[middle]);
        };
        const double median = conventional_median(sorted);
        const double median_cpu = conventional_median(sorted_cpu);
        const std::uint64_t minimum = sorted.front();
        const std::uint64_t maximum = sorted.back();
        const double mean = static_cast<double>(std::accumulate(
            sorted.begin(), sorted.end(), std::uint64_t{0})) / sorted.size();
        if (index != 0) json << ',';
        json << '\"' << stage.name << "\":{"
             << "\"evaluations_per_repeat\":"
             << stage.evaluations_per_repeat
             << ",\"median_ns_per_evaluation\":"
             << median / stage.evaluations_per_repeat
             << ",\"mean_ns_per_evaluation\":"
             << mean / stage.evaluations_per_repeat
             << ",\"min_ns_per_evaluation\":"
             << static_cast<double>(minimum) / stage.evaluations_per_repeat
             << ",\"max_ns_per_evaluation\":"
             << static_cast<double>(maximum) / stage.evaluations_per_repeat
             << ",\"median_process_cpu_ns_per_evaluation\":"
             << median_cpu / stage.evaluations_per_repeat
             << ",\"median_cpu_wall_ratio\":"
             << (median == 0.0 ? 0.0 : median_cpu / median)
             << ",\"median_batch_ns\":" << median
             << ",\"elapsed_ns\":[";
        for (std::size_t repeat = 0;
             repeat < stage.elapsed_ns.size();
             ++repeat) {
            if (repeat != 0) json << ',';
            json << stage.elapsed_ns[repeat];
        }
        json << "]"
             << ",\"process_cpu_ns\":[";
        for (std::size_t repeat = 0; repeat < stage.cpu_ns.size(); ++repeat) {
            if (repeat != 0) json << ',';
            json << stage.cpu_ns[repeat];
        }
        json << ']'
             << ",\"checksum\":\"" << hex64(stage.checksum) << "\""
             << ",\"timed_sink\":\"" << hex64(stage.timed_sink) << "\"}";
    }
    json << "},\"batches\":[";
    for (std::size_t index = 0; index < result.batches.size(); ++index) {
        const auto& batch = result.batches[index];
        if (index != 0) json << ',';
        json << "{\"repeat\":" << batch.repeat
             << ",\"order\":" << batch.order
             << ",\"stage\":\"" << batch.stage << "\""
             << ",\"evaluations\":" << batch.evaluations
             << ",\"wall_ns\":" << batch.elapsed_ns
             << ",\"process_cpu_ns\":" << batch.cpu_ns
             << ",\"cpu_wall_ratio\":"
             << (batch.elapsed_ns == 0
                    ? 0.0
                    : static_cast<double>(batch.cpu_ns) / batch.elapsed_ns)
             << '}';
    }
    json << "]}";
    return json.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
#if defined(_WIN32)
        if (_putenv_s("CHESS_NNUE_BACKEND", options.backend.c_str()) != 0) {
            throw std::runtime_error("failed to set CHESS_NNUE_BACKEND");
        }
#else
        if (setenv("CHESS_NNUE_BACKEND", options.backend.c_str(), 1) != 0) {
            throw std::runtime_error("failed to set CHESS_NNUE_BACKEND");
        }
#endif
        chess::PhaseQuantizedNnueModel model;
        if (!model.load(options.model.string())) {
            throw std::runtime_error("failed to load model/backend");
        }
        const auto result = chess::benchmark_phase_nnue_forward_stages(
            model,
            make_corpus(options.samples, options.seed),
            options.warmup,
            options.iterations,
            options.repeats);
        const std::string json = render_json(
            options, sha256_file(options.model), result);
        if (!options.output.empty()) {
            std::ofstream output(options.output);
            if (!output) throw std::runtime_error("failed to open --output");
            output << json << '\n';
        }
        std::cout << json << '\n';
        return result.scalar_parity
                && result.stage_parity
                && result.timed_sink_parity
            ? 0
            : 1;
    } catch (const std::exception& error) {
        std::cout << "{\"schema_version\":2"
                  << ",\"protocol\":\"phase_nnue_forward_stages_v2\""
                  << ",\"status\":\"error\",\"message\":\""
                  << error.what() << "\"}\n";
        return 1;
    }
}
