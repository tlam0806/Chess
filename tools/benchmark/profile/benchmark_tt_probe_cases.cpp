#include "range_bucket_transposition_table.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

struct CaseConfig {
    std::string_view name;
    const chess::RangeBucketTranspositionTable* tt = nullptr;
    std::vector<chess::HashKey> keys;
    int probe_depth = 8;
    chess::ScoreRange window{-100, 100};
};

double ns_per_probe(std::uint64_t elapsed_ns, std::uint64_t probes) {
    if (probes == 0) {
        return 0.0;
    }
    return static_cast<double>(elapsed_ns) / static_cast<double>(probes);
}

std::uint64_t run_case(
    const CaseConfig& config,
    int repeats,
    volatile std::uint64_t& checksum
) {
    chess::ScoreRange stored_score;
    chess::MoveRange stored_move;
    bool score_available = false;
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (chess::HashKey key : config.keys) {
            chess::ScoreRange window = config.window;
            const bool hit = config.tt->probe(
                key,
                config.probe_depth,
                window,
                stored_score,
                stored_move,
                0,
                score_available);
            checksum += static_cast<std::uint64_t>(hit)
                + static_cast<std::uint64_t>(score_available)
                + static_cast<std::uint64_t>(window.lower + 1'000'000)
                + static_cast<std::uint64_t>(window.upper + 1'000'000)
                + stored_move.lower.value
                + stored_move.upper.value;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

std::vector<chess::HashKey> make_keys(
    std::size_t begin_bucket,
    std::size_t count,
    std::size_t bucket_count,
    std::uint64_t salt
) {
    std::vector<chess::HashKey> keys;
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t bucket = begin_bucket + i;
        keys.push_back(static_cast<chess::HashKey>(bucket_count * salt + bucket));
    }
    return keys;
}

} // namespace

int main(int argc, char** argv) {
    std::size_t tt_mb = 64;
    int repeats = 80;
    std::size_t key_count = 262'144;
    if (argc >= 2) {
        repeats = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        key_count = static_cast<std::size_t>(std::stoull(argv[2]));
    }

    chess::RangeBucketTranspositionTable exact_tt(tt_mb, 4);
    chess::RangeBucketTranspositionTable lower_tt(tt_mb, 4);
    const std::size_t bucket_count = exact_tt.bucket_count();
    const std::size_t half_bucket_count = bucket_count / 2;
    key_count = std::min(key_count, half_bucket_count);

    const std::vector<chess::HashKey> hit_keys = make_keys(0, key_count, bucket_count, 1);
    const std::vector<chess::HashKey> index_miss_keys = make_keys(0, key_count, bucket_count, 13);
    const std::vector<chess::HashKey> empty_miss_keys =
        make_keys(half_bucket_count, key_count, bucket_count, 1);

    chess::MoveRange move{};
    move.lower = chess::Move{0x1111};
    move.upper = chess::Move{0x2222};
    for (chess::HashKey key : hit_keys) {
        exact_tt.store(key, 8, chess::ScoreRange{42, 42}, move);
        lower_tt.store(key, 8, chess::ScoreRange{42, chess::Infinity}, move);
    }

    const std::vector<CaseConfig> cases{
        CaseConfig{"empty_miss", &exact_tt, empty_miss_keys, 8, chess::ScoreRange{-100, 100}},
        CaseConfig{"index_miss", &exact_tt, index_miss_keys, 8, chess::ScoreRange{-100, 100}},
        CaseConfig{"depth_miss", &exact_tt, hit_keys, 20, chess::ScoreRange{-100, 100}},
        CaseConfig{"exact_return_hit", &exact_tt, hit_keys, 8, chess::ScoreRange{-100, 100}},
        CaseConfig{"lower_return_hit", &lower_tt, hit_keys, 8, chess::ScoreRange{-100, 40}},
        CaseConfig{"lower_hit_no_return", &lower_tt, hit_keys, 8, chess::ScoreRange{-100, 100}},
    };

    volatile std::uint64_t checksum = 0;
    std::cout << "tt_mb=" << tt_mb
              << " bucket_size=" << exact_tt.bucket_size()
              << " bucket_count=" << bucket_count
              << " key_count=" << key_count
              << " repeats=" << repeats
              << '\n';

    const std::uint64_t probes_per_case =
        static_cast<std::uint64_t>(key_count) * static_cast<std::uint64_t>(repeats);
    for (const CaseConfig& config : cases) {
        const std::uint64_t elapsed_ns = run_case(config, repeats, checksum);
        std::cout << "case=" << config.name
                  << " probes=" << probes_per_case
                  << " elapsed_ns=" << elapsed_ns
                  << " ns_per_probe=" << ns_per_probe(elapsed_ns, probes_per_case)
                  << '\n';
    }
    std::cout << "checksum=" << checksum << '\n';
}
