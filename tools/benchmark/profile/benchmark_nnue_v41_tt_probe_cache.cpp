#include "lower_move_range_bucket_transposition_table.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <unistd.h>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

namespace {

constexpr std::size_t EntriesPerBucket = 4;
constexpr std::size_t LogicalBytesPerBucket =
    EntriesPerBucket
    * (sizeof(chess::HashKey)
       + sizeof(chess::LowerMoveRangeBucketTranspositionTable::TTValue));

enum class ProbeCase {
    Control,
    EmptyMiss,
    IndexMiss,
    HitSlot0,
    HitSlot1,
    HitSlot2,
    HitSlot3,
    DepthMiss,
};

ProbeCase parse_case(std::string_view name) {
    if (name == "control") {
        return ProbeCase::Control;
    }
    if (name == "empty_miss") {
        return ProbeCase::EmptyMiss;
    }
    if (name == "index_miss") {
        return ProbeCase::IndexMiss;
    }
    if (name == "hit_slot0") {
        return ProbeCase::HitSlot0;
    }
    if (name == "hit_slot1") {
        return ProbeCase::HitSlot1;
    }
    if (name == "hit_slot2") {
        return ProbeCase::HitSlot2;
    }
    if (name == "hit_slot3") {
        return ProbeCase::HitSlot3;
    }
    if (name == "depth_miss") {
        return ProbeCase::DepthMiss;
    }
    throw std::invalid_argument("unknown case: " + std::string(name));
}

int hit_slot(ProbeCase probe_case) {
    switch (probe_case) {
    case ProbeCase::HitSlot0:
        return 0;
    case ProbeCase::HitSlot1:
        return 1;
    case ProbeCase::HitSlot2:
        return 2;
    case ProbeCase::HitSlot3:
        return 3;
    case ProbeCase::DepthMiss:
        return 3;
    case ProbeCase::Control:
    case ProbeCase::EmptyMiss:
    case ProbeCase::IndexMiss:
        return -1;
    }
    return -1;
}

std::uint64_t run_control(
    const std::vector<chess::HashKey>& keys,
    std::uint64_t probe_count
) {
    std::uint64_t checksum = 0;
    std::size_t key_index = 0;
    for (std::uint64_t probe_index = 0; probe_index < probe_count; ++probe_index) {
        checksum += keys[key_index] & 0xffffu;
        ++key_index;
        if (key_index == keys.size()) {
            key_index = 0;
        }
    }
    return checksum;
}

chess::HashKey key_for(
    std::size_t bucket,
    std::size_t bucket_count,
    std::uint64_t tag
) {
    return static_cast<chess::HashKey>(bucket)
        + static_cast<chess::HashKey>(bucket_count) * tag;
}

std::uint64_t run_probes(
    const chess::LowerMoveRangeBucketTranspositionTable& tt,
    const std::vector<chess::HashKey>& keys,
    std::uint64_t probe_count,
    int depth
) {
    std::uint64_t checksum = 0;
    std::size_t key_index = 0;
    for (std::uint64_t probe_index = 0; probe_index < probe_count; ++probe_index) {
        chess::ScoreRange window{-100, 100};
        chess::ScoreRange stored_score{};
        chess::MoveRange stored_move{};
        bool score_available = false;
        const bool hit = tt.probe(
            keys[key_index],
            depth,
            window,
            stored_score,
            stored_move,
            0,
            score_available,
            chess::TTDepthPolicy::Exact);

        checksum += static_cast<std::uint64_t>(hit)
            + 3u * static_cast<std::uint64_t>(score_available)
            + 5u * static_cast<std::uint64_t>(window.lower + chess::Infinity)
            + 7u * static_cast<std::uint64_t>(window.upper + chess::Infinity)
            + 11u * stored_move.lower.value;

        ++key_index;
        if (key_index == keys.size()) {
            key_index = 0;
        }
    }
    return checksum;
}

} // namespace

int main(int argc, char** argv) {
#if defined(__APPLE__)
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    const std::string case_name = argc >= 2 ? argv[1] : "hit_slot3";
    const std::size_t working_set_kib = argc >= 3
        ? static_cast<std::size_t>(std::stoull(argv[2]))
        : 1024;
    const std::uint64_t probe_count = argc >= 4
        ? static_cast<std::uint64_t>(std::stoull(argv[3]))
        : 20'000'000;
    const std::size_t tt_mb = argc >= 5
        ? static_cast<std::size_t>(std::stoull(argv[4]))
        : 128;
    const int start_delay_ms = argc >= 6 ? std::stoi(argv[5]) : 0;
    const ProbeCase probe_case = parse_case(case_name);

    chess::LowerMoveRangeBucketTranspositionTable tt(tt_mb, EntriesPerBucket);
    const std::size_t requested_working_bytes = working_set_kib * 1024;
    const std::size_t max_working_buckets = std::max<std::size_t>(tt.bucket_count() / 2, 1);
    const std::size_t working_buckets = std::clamp<std::size_t>(
        requested_working_bytes / LogicalBytesPerBucket,
        1,
        max_working_buckets);

    const bool populate = probe_case != ProbeCase::Control
        && probe_case != ProbeCase::EmptyMiss;
    chess::MoveRange move{};
    move.lower = chess::Move{0x1111};
    move.upper = chess::Move{0x2222};
    if (populate) {
        for (std::size_t bucket = 0; bucket < working_buckets; ++bucket) {
            for (std::size_t slot = 0; slot < EntriesPerBucket; ++slot) {
                tt.store(
                    key_for(bucket, tt.bucket_count(), slot + 1),
                    8,
                    chess::ScoreRange{42, 42},
                    move);
            }
        }
    }

    std::vector<chess::HashKey> keys;
    keys.reserve(working_buckets);
    const int slot = hit_slot(probe_case);
    for (std::size_t i = 0; i < working_buckets; ++i) {
        const std::size_t bucket = probe_case == ProbeCase::EmptyMiss
            ? working_buckets + i
            : i;
        const std::uint64_t tag = probe_case == ProbeCase::IndexMiss
            ? 9
            : static_cast<std::uint64_t>(std::max(slot, 0) + 1);
        keys.push_back(key_for(bucket, tt.bucket_count(), tag));
    }
    std::mt19937_64 rng(0x41caceULL);
    std::shuffle(keys.begin(), keys.end(), rng);

    const int depth = probe_case == ProbeCase::DepthMiss ? 20 : 8;
    const std::uint64_t warmup_probes = std::min<std::uint64_t>(
        std::max<std::uint64_t>(keys.size(), 10'000),
        2'000'000);
    std::uint64_t checksum = probe_case == ProbeCase::Control
        ? run_control(keys, warmup_probes)
        : run_probes(tt, keys, warmup_probes, depth);

    if (start_delay_ms > 0) {
        std::cout << "ready_pid=" << getpid() << '\n' << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(start_delay_ms));
    }

    const auto start = std::chrono::steady_clock::now();
    checksum += probe_case == ProbeCase::Control
        ? run_control(keys, probe_count)
        : run_probes(tt, keys, probe_count, depth);
    const auto end = std::chrono::steady_clock::now();
    const std::uint64_t elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    const double ns_per_probe = probe_count == 0
        ? 0.0
        : static_cast<double>(elapsed_ns) / static_cast<double>(probe_count);

    std::cout << "case=" << case_name
              << " tt_mb=" << tt_mb
              << " bucket_size=" << tt.bucket_size()
              << " bucket_count=" << tt.bucket_count()
              << " tt_value_bytes=" << sizeof(chess::LowerMoveRangeBucketTranspositionTable::TTValue)
              << " logical_bytes_per_bucket=" << LogicalBytesPerBucket
              << " requested_working_set_kib=" << working_set_kib
              << " logical_working_set_bytes=" << working_buckets * LogicalBytesPerBucket
              << " working_buckets=" << working_buckets
              << " probes=" << probe_count
              << " elapsed_ns=" << elapsed_ns
              << " ns_per_probe=" << ns_per_probe
              << " probes_per_second="
              << (elapsed_ns == 0
                      ? 0.0
                      : static_cast<double>(probe_count) * 1'000'000'000.0
                          / static_cast<double>(elapsed_ns))
              << " checksum=" << checksum
              << '\n';
}
