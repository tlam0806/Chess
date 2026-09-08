#include "range_bucket_transposition_table.hpp"
#include "single_bound_bucket_transposition_table.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <typename TT>
struct ProbeCase {
    std::string_view name;
    const TT* tt = nullptr;
    std::vector<chess::HashKey> keys;
    int probe_depth = 8;
    chess::ScoreRange window{-100, 100};
};

struct StoreCase {
    std::string_view name;
    std::vector<chess::HashKey> keys;
    int depth = 8;
    chess::ScoreRange score{42, 42};
    chess::MoveRange move{chess::Move{0x1111}, chess::Move{0x2222}};
};

double ns_per_call(std::uint64_t elapsed_ns, std::uint64_t calls) {
    return calls == 0 ? 0.0 : static_cast<double>(elapsed_ns) / static_cast<double>(calls);
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

template <typename TT>
std::uint64_t run_probe_case(
    const ProbeCase<TT>& probe_case,
    int repeats,
    volatile std::uint64_t& checksum
) {
    chess::ScoreRange stored_score;
    chess::MoveRange stored_move;
    bool score_available = false;
    const auto start = Clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (chess::HashKey key : probe_case.keys) {
            chess::ScoreRange window = probe_case.window;
            const bool hit = probe_case.tt->probe(
                key,
                probe_case.probe_depth,
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
                + stored_move.upper.value
                + static_cast<std::uint64_t>(stored_score.lower + 1'000'000)
                + static_cast<std::uint64_t>(stored_score.upper + 1'000'000);
        }
    }
    const auto end = Clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

template <typename TT>
std::uint64_t run_store_case(
    const StoreCase& store_case,
    int repeats,
    std::size_t bucket_size,
    volatile std::uint64_t& checksum
) {
    TT tt(64, bucket_size);
    const auto start = Clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (chess::HashKey key : store_case.keys) {
            tt.store(key, store_case.depth, store_case.score, store_case.move);
        }
    }
    const auto end = Clock::now();
    checksum += tt.entry_count();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

template <typename TT>
std::uint64_t run_prefilled_store_case(
    const StoreCase& store_case,
    const std::vector<chess::HashKey>& prefill_keys,
    int prefill_depth,
    int repeats,
    std::size_t bucket_size,
    volatile std::uint64_t& checksum
) {
    TT tt(64, bucket_size);
    chess::MoveRange prefill_move{chess::Move{0x3333}, chess::Move{0x4444}};
    for (chess::HashKey key : prefill_keys) {
        tt.store(key, prefill_depth, chess::ScoreRange{12, 12}, prefill_move);
    }

    const auto start = Clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (chess::HashKey key : store_case.keys) {
            tt.store(key, store_case.depth, store_case.score, store_case.move);
        }
    }
    const auto end = Clock::now();
    checksum += tt.entry_count();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

template <typename TT>
void fill_tt(
    TT& tt,
    const std::vector<chess::HashKey>& hit_keys,
    chess::ScoreRange score,
    chess::MoveRange move,
    int depth = 8
) {
    for (chess::HashKey key : hit_keys) {
        tt.store(key, depth, score, move);
    }
}

template <typename TT>
void run_probe_suite(
    std::string_view label,
    const TT& exact_tt,
    const TT& lower_tt,
    const TT& upper_tt,
    const std::vector<chess::HashKey>& hit_keys,
    const std::vector<chess::HashKey>& index_miss_keys,
    const std::vector<chess::HashKey>& empty_miss_keys,
    int repeats,
    volatile std::uint64_t& checksum
) {
    const std::vector<ProbeCase<TT>> cases{
        ProbeCase<TT>{"empty_miss", &exact_tt, empty_miss_keys, 8, chess::ScoreRange{-100, 100}},
        ProbeCase<TT>{"index_collision", &exact_tt, index_miss_keys, 8, chess::ScoreRange{-100, 100}},
        ProbeCase<TT>{"depth_miss", &exact_tt, hit_keys, 20, chess::ScoreRange{-100, 100}},
        ProbeCase<TT>{"exact_return_hit", &exact_tt, hit_keys, 8, chess::ScoreRange{-100, 100}},
        ProbeCase<TT>{"lower_return_hit", &lower_tt, hit_keys, 8, chess::ScoreRange{-100, 40}},
        ProbeCase<TT>{"lower_hit_no_return", &lower_tt, hit_keys, 8, chess::ScoreRange{-100, 100}},
        ProbeCase<TT>{"upper_return_hit", &upper_tt, hit_keys, 8, chess::ScoreRange{60, 100}},
        ProbeCase<TT>{"upper_hit_no_return", &upper_tt, hit_keys, 8, chess::ScoreRange{-100, 100}},
    };

    const std::uint64_t calls =
        static_cast<std::uint64_t>(hit_keys.size()) * static_cast<std::uint64_t>(repeats);
    for (const ProbeCase<TT>& probe_case : cases) {
        const std::uint64_t elapsed_ns = run_probe_case(probe_case, repeats, checksum);
        std::cout << "version=" << label
                  << " fn=probe"
                  << " case=" << probe_case.name
                  << " calls=" << calls
                  << " elapsed_ns=" << elapsed_ns
                  << " ns_per_call=" << ns_per_call(elapsed_ns, calls)
                  << '\n';
    }
}

template <typename TT>
void run_store_suite(
    std::string_view label,
    const std::vector<chess::HashKey>& new_store_keys,
    const std::vector<chess::HashKey>& same_key_keys,
    const std::vector<chess::HashKey>& same_key_prefill_keys,
    const std::vector<chess::HashKey>& bucket_full_replace_keys,
    const std::vector<chess::HashKey>& bucket_full_replace_prefill_keys,
    const std::vector<chess::HashKey>& bucket_full_skip_keys,
    const std::vector<chess::HashKey>& bucket_full_skip_prefill_keys,
    int repeats,
    std::size_t bucket_size,
    volatile std::uint64_t& checksum
) {
    auto print_store = [&](const StoreCase& store_case, std::uint64_t elapsed_ns) {
        const std::uint64_t calls =
            static_cast<std::uint64_t>(store_case.keys.size()) * static_cast<std::uint64_t>(repeats);
        std::cout << "version=" << label
                  << " fn=store"
                  << " case=" << store_case.name
                  << " calls=" << calls
                  << " elapsed_ns=" << elapsed_ns
                  << " ns_per_call=" << ns_per_call(elapsed_ns, calls)
                  << '\n';
    };

    const StoreCase new_store{"new_store", new_store_keys, 8, chess::ScoreRange{42, 42}};
    print_store(new_store, run_store_case<TT>(new_store, repeats, bucket_size, checksum));

    const StoreCase same_exact{
        "same_key_update_exact",
        same_key_keys,
        8,
        chess::ScoreRange{50, 50}
    };
    print_store(
        same_exact,
        run_prefilled_store_case<TT>(
            same_exact, same_key_prefill_keys, 8, repeats, bucket_size, checksum));

    const StoreCase same_lower{
        "same_key_update_lower",
        same_key_keys,
        8,
        chess::ScoreRange{60, chess::Infinity}
    };
    print_store(
        same_lower,
        run_prefilled_store_case<TT>(
            same_lower, same_key_prefill_keys, 8, repeats, bucket_size, checksum));

    const StoreCase replace{
        "bucket_full_replace",
        bucket_full_replace_keys,
        9,
        chess::ScoreRange{42, 42}
    };
    print_store(
        replace,
        run_prefilled_store_case<TT>(
            replace, bucket_full_replace_prefill_keys, 8, repeats, bucket_size, checksum));

    const StoreCase skip{"bucket_full_skip", bucket_full_skip_keys, 1, chess::ScoreRange{42, 42}};
    print_store(
        skip,
        run_prefilled_store_case<TT>(
            skip, bucket_full_skip_prefill_keys, 9, repeats, bucket_size, checksum));
}

} // namespace

int main(int argc, char** argv) {
    int repeats = 40;
    std::size_t key_count = 131'072;
    std::size_t bucket_size = 4;
    if (argc >= 2) {
        repeats = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        key_count = static_cast<std::size_t>(std::stoull(argv[2]));
    }

    chess::RangeBucketTranspositionTable v32_exact(64, bucket_size);
    chess::RangeBucketTranspositionTable v32_lower(64, bucket_size);
    chess::RangeBucketTranspositionTable v32_upper(64, bucket_size);
    chess::SingleBoundBucketTranspositionTable v34_exact(64, bucket_size);
    chess::SingleBoundBucketTranspositionTable v34_lower(64, bucket_size);
    chess::SingleBoundBucketTranspositionTable v34_upper(64, bucket_size);

    const std::size_t bucket_count =
        std::min(v32_exact.bucket_count(), v34_exact.bucket_count());
    key_count = std::min(key_count, bucket_count / 4);

    const std::vector<chess::HashKey> hit_keys = make_keys(0, key_count, bucket_count, 1);
    const std::vector<chess::HashKey> index_miss_keys = make_keys(0, key_count, bucket_count, 13);
    const std::vector<chess::HashKey> empty_miss_keys =
        make_keys(bucket_count / 2, key_count, bucket_count, 1);
    const std::vector<chess::HashKey> same_key_keys =
        make_keys(0, key_count, bucket_count, 201);
    const std::vector<chess::HashKey> same_key_prefill_keys = same_key_keys;

    std::vector<chess::HashKey> new_store_keys;
    new_store_keys.reserve(key_count);
    for (std::size_t i = 0; i < key_count; ++i) {
        new_store_keys.push_back(static_cast<chess::HashKey>(bucket_count * 101 + i));
    }

    std::vector<chess::HashKey> bucket_full_replace_keys;
    std::vector<chess::HashKey> bucket_full_skip_keys;
    std::vector<chess::HashKey> bucket_full_replace_prefill_keys;
    std::vector<chess::HashKey> bucket_full_skip_prefill_keys;
    bucket_full_replace_keys.reserve(key_count);
    bucket_full_skip_keys.reserve(key_count);
    bucket_full_replace_prefill_keys.reserve(key_count * bucket_size);
    bucket_full_skip_prefill_keys.reserve(key_count * bucket_size);
    for (std::size_t i = 0; i < key_count; ++i) {
        for (std::size_t salt = 0; salt < bucket_size; ++salt) {
            bucket_full_replace_prefill_keys.push_back(bucket_count * (300 + salt) + i);
            bucket_full_skip_prefill_keys.push_back(bucket_count * (400 + salt) + i);
        }
        bucket_full_replace_keys.push_back(bucket_count * (300 + bucket_size) + i);
        bucket_full_skip_keys.push_back(bucket_count * (400 + bucket_size) + i);
    }

    chess::MoveRange move{};
    move.lower = chess::Move{0x1111};
    move.upper = chess::Move{0x2222};
    fill_tt(v32_exact, hit_keys, chess::ScoreRange{42, 42}, move);
    fill_tt(v32_lower, hit_keys, chess::ScoreRange{42, chess::Infinity}, move);
    fill_tt(v32_upper, hit_keys, chess::ScoreRange{-chess::Infinity, 42}, move);
    fill_tt(v34_exact, hit_keys, chess::ScoreRange{42, 42}, move);
    fill_tt(v34_lower, hit_keys, chess::ScoreRange{42, chess::Infinity}, move);
    fill_tt(v34_upper, hit_keys, chess::ScoreRange{-chess::Infinity, 42}, move);

    volatile std::uint64_t checksum = 0;
    std::cout << "bucket_size=" << bucket_size
              << " v32_bucket_count=" << v32_exact.bucket_count()
              << " v34_bucket_count=" << v34_exact.bucket_count()
              << " key_count=" << key_count
              << " repeats=" << repeats
              << '\n';

    run_probe_suite(
        "v32",
        v32_exact,
        v32_lower,
        v32_upper,
        hit_keys,
        index_miss_keys,
        empty_miss_keys,
        repeats,
        checksum);
    run_probe_suite(
        "v34",
        v34_exact,
        v34_lower,
        v34_upper,
        hit_keys,
        index_miss_keys,
        empty_miss_keys,
        repeats,
        checksum);
    run_store_suite<chess::RangeBucketTranspositionTable>(
        "v32",
        new_store_keys,
        same_key_keys,
        same_key_prefill_keys,
        bucket_full_replace_keys,
        bucket_full_replace_prefill_keys,
        bucket_full_skip_keys,
        bucket_full_skip_prefill_keys,
        repeats,
        bucket_size,
        checksum);
    run_store_suite<chess::SingleBoundBucketTranspositionTable>(
        "v34",
        new_store_keys,
        same_key_keys,
        same_key_prefill_keys,
        bucket_full_replace_keys,
        bucket_full_replace_prefill_keys,
        bucket_full_skip_keys,
        bucket_full_skip_prefill_keys,
        repeats,
        bucket_size,
        checksum);

    std::cout << "checksum=" << checksum << '\n';
}
