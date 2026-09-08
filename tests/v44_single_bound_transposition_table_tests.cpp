#include "v44_single_bound_transposition_table.hpp"

#include <cassert>
#include <cstdint>

using namespace chess;

namespace {

using TT = V44SingleBoundTranspositionTable;

constexpr HashKey Key = 0x123456789abcdef0ULL;
constexpr Move Hint{0x1234};

void test_layout_and_payload() {
    static_assert(sizeof(TT::Entry) == 16);
    static_assert(sizeof(TT::Bucket) == 64);
    static_assert(alignof(TT::Bucket) == 64);

    TT tt(1);
    assert(tt.entry_count() == tt.bucket_count() * 4);
    assert(tt.occupied_entry_count() == 0);
    assert(tt.memory_bytes() == tt.bucket_count() * 64);
    const TT::Bucket* bucket = tt.bucket_address(Key);
    assert(reinterpret_cast<std::uintptr_t>(bucket) % 64 == 0);

    assert(tt.store(Key, 8, 42, TT::Bound::Exact, Hint, 0));
    assert(tt.occupied_entry_count() == 1);
    bool found = false;
    for (const TT::Entry& entry : bucket->entries) {
        if (entry.payload != 0 && entry.key == Key) {
            // score(18) + move(16) + depth(8) + bound(2) occupy 44 bits.
            // The remaining payload bits carry no cross-search metadata.
            assert((entry.payload >> 44) == 0);
            found = true;
        }
    }
    assert(found);
}

void test_cutoff_only_probe() {
    TT tt(1);

    assert(tt.store(Key, 8, 42, TT::Bound::Lower, Hint, 0));
    TT::ProbeResult result = tt.probe(Key, 8, -100, 40, 0);
    assert(result.key_hit);
    assert(result.score_usable);
    assert(result.cutoff);
    assert(result.score == 42);
    assert(result.bound == TT::Bound::Lower);
    assert(result.move == Hint);

    result = tt.probe(Key, 8, -100, 100, 0);
    assert(result.score_usable);
    assert(!result.cutoff);

    assert(tt.store(Key, 8, -17, TT::Bound::Upper, Hint, 0));
    result = tt.probe(Key, 8, -10, 100, 0);
    assert(result.cutoff);
    assert(result.score == -17);
    assert(result.bound == TT::Bound::Upper);

    assert(tt.store(Key, 8, 7, TT::Bound::Exact, Hint, 0));
    result = tt.probe(Key, 8, -1, 1, 0);
    assert(result.cutoff);
    assert(result.score == 7);
    assert(result.bound == TT::Bound::Exact);
}

void test_depth_policy() {
    TT tt(1);
    assert(tt.store(Key, 8, 42, TT::Bound::Exact, Hint, 0));

    TT::ProbeResult result = tt.probe(
        Key, 7, -100, 100, 0, TT::DepthPolicy::AtLeast);
    assert(result.cutoff);

    result = tt.probe(Key, 7, -100, 100, 0, TT::DepthPolicy::Exact);
    assert(result.key_hit);
    assert(!result.score_usable);
    assert(!result.cutoff);
    assert(result.move == Hint);
}

void test_clear_invalidates_all_entries() {
    TT tt(1);
    const HashKey stride = static_cast<HashKey>(tt.bucket_count());
    for (int i = 0; i < 4; ++i) {
        assert(tt.store(
            Key + static_cast<HashKey>(i) * stride,
            6 + i,
            20 + i,
            TT::Bound::Exact,
            Hint,
            0));
    }

    tt.clear();
    assert(tt.occupied_entry_count() == 0);
    for (int i = 0; i < 4; ++i) {
        const auto probe = tt.probe(
            Key + static_cast<HashKey>(i) * stride,
            6 + i,
            -100,
            100,
            0);
        assert(!probe.key_hit);
    }
}

void test_same_key_replacement_priority() {
    TT tt(1);
    assert(tt.store(Key, 6, 20, TT::Bound::Lower, Hint, 0));
    assert(!tt.store(Key, 6, 10, TT::Bound::Lower, Hint, 0));
    assert(tt.inspect(Key)->score == 20);
    assert(tt.store(Key, 6, 30, TT::Bound::Lower, Hint, 0));
    assert(tt.inspect(Key)->score == 30);

    assert(tt.store(Key, 6, 25, TT::Bound::Exact, Hint, 0));
    assert(tt.inspect(Key)->bound == TT::Bound::Exact);
    assert(!tt.store(Key, 6, 40, TT::Bound::Lower, Hint, 0));
    assert(tt.inspect(Key)->score == 25);

    assert(tt.store(Key, 7, 40, TT::Bound::Lower, Hint, 0));
    assert(tt.inspect(Key)->depth == 7);
    assert(tt.inspect(Key)->score == 40);
    assert(!tt.store(Key, 6, 99, TT::Bound::Exact, Hint, 0));
}

void test_opposite_bounds_replace_without_merge() {
    TT tt(1);
    assert(tt.store(Key, 6, 151, TT::Bound::Upper, Hint, 0));
    assert(tt.store(Key, 6, 153, TT::Bound::Lower, Hint, 0));

    const auto entry = tt.inspect(Key);
    assert(entry.has_value());
    assert(entry->bound == TT::Bound::Lower);
    assert(entry->score == 153);
}

void test_move_only_rules() {
    TT tt(1);
    constexpr Move deeper_move{0x1111};
    constexpr Move shallow_move{0x2222};
    constexpr Move equal_move{0x3333};

    assert(!tt.store_move(Key, 9, Move{}));
    assert(tt.store_move(Key, 9, Hint));
    TT::ProbeResult result = tt.probe(Key, 1, -100, 100, 0);
    assert(result.key_hit);
    assert(result.move == Hint);
    assert(result.bound == TT::Bound::None);
    assert(!result.score_usable);
    assert(!result.cutoff);

    tt.clear();
    assert(tt.store(Key, 12, 37, TT::Bound::Exact, deeper_move, 0));
    assert(!tt.store_move(Key, 11, shallow_move));
    assert(tt.inspect(Key)->move == deeper_move);
    assert(tt.store_move(Key, 12, equal_move));
    const auto entry = tt.inspect(Key);
    assert(entry->move == equal_move);
    assert(entry->score == 37);
    assert(entry->depth == 12);
    assert(entry->bound == TT::Bound::Exact);
}

void test_mate_score_normalization() {
    TT tt(1);
    constexpr int store_ply = 3;
    constexpr int winning_score = CheckmateScore - 7;
    assert(tt.store(
        Key, 8, winning_score, TT::Bound::Exact, Hint, store_ply));
    assert(tt.probe(Key, 8, -Infinity, Infinity, 3).score
        == CheckmateScore - 7);
    assert(tt.probe(Key, 8, -Infinity, Infinity, 5).score
        == CheckmateScore - 9);

    tt.clear();
    constexpr int losing_score = -CheckmateScore + 7;
    assert(tt.store(
        Key, 8, losing_score, TT::Bound::Exact, Hint, store_ply));
    assert(tt.probe(Key, 8, -Infinity, Infinity, 5).score
        == -CheckmateScore + 9);
}

void test_full_hash_and_four_entry_bucket() {
    TT tt(1);
    const HashKey stride = static_cast<HashKey>(tt.bucket_count());
    const HashKey base = 17;

    for (int i = 0; i < 4; ++i) {
        const HashKey key = base + static_cast<HashKey>(i) * stride;
        assert(tt.store(key, i + 1, i, TT::Bound::Exact, Hint, 0));
    }
    for (int i = 0; i < 4; ++i) {
        const HashKey key = base + static_cast<HashKey>(i) * stride;
        assert(tt.probe(key, i + 1, -100, 100, 0).key_hit);
    }

    const HashKey move_only_key = base + 4 * stride;
    assert(!tt.store_move(move_only_key, 255, Hint));

    const HashKey strong_key = base + 5 * stride;
    assert(tt.store(strong_key, 20, 77, TT::Bound::Exact, Hint, 0));
    assert(tt.probe(strong_key, 20, -100, 100, 0).cutoff);
}

void test_zero_hash_and_invalid_depth() {
    TT tt(1);
    assert(tt.store(0, 4, 12, TT::Bound::Exact, Hint, 0));
    assert(tt.probe(0, 4, -100, 100, 0).score == 12);

    assert(!tt.store(Key, -1, 12, TT::Bound::Exact, Hint, 0));
    assert(!tt.store(Key, 256, 12, TT::Bound::Exact, Hint, 0));
    assert(!tt.probe(Key, -1, -100, 100, 0).key_hit);
    assert(!tt.probe(Key, 256, -100, 100, 0).key_hit);
}

} // namespace

int main() {
    test_layout_and_payload();
    test_cutoff_only_probe();
    test_depth_policy();
    test_clear_invalidates_all_entries();
    test_same_key_replacement_priority();
    test_opposite_bounds_replace_without_merge();
    test_move_only_rules();
    test_mate_score_normalization();
    test_full_hash_and_four_entry_bucket();
    test_zero_hash_and_invalid_depth();
    return 0;
}
