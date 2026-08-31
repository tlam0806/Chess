#include "v43_single_bound_transposition_table.hpp"

#include <cassert>
#include <cstdint>
#include <limits>

using namespace chess;

namespace {

using TT = V43SingleBoundTranspositionTable;

constexpr HashKey Key = 0x123456789abcdef0ULL;
constexpr Move Hint{0x1234};

void test_layout() {
    static_assert(sizeof(TT::Entry) == 16);
    static_assert(sizeof(TT::Bucket) == 64);
    static_assert(alignof(TT::Bucket) == 64);

    TT tt(1);
    assert(tt.entry_count() == tt.bucket_count() * 4);
    assert(tt.memory_bytes() == tt.bucket_count() * 64);
    assert(reinterpret_cast<std::uintptr_t>(tt.bucket_address(Key)) % 64 == 0);
}

void test_cutoff_only_probe() {
    TT tt(1);

    assert(tt.store(Key, 8, 42, TT::Bound::Lower, Hint, 0));
    TT::ProbeResult result = tt.probe(Key, 8, -100, 40, 0);
    assert(result.key_hit);
    assert(result.generation_match);
    assert(result.score_usable);
    assert(result.cutoff);
    assert(result.score == 42);
    assert(result.bound == TT::Bound::Lower);
    assert(result.move == Hint);

    // A usable bound which cannot cut is move-ordering information only.  In
    // particular, probe has no API through which it could narrow alpha/beta.
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

void test_generation_is_move_only_by_default() {
    TT tt(1);
    assert(tt.store(Key, 6, 31, TT::Bound::Exact, Hint, 0));
    tt.advance_generation();

    TT::ProbeResult result = tt.probe(Key, 6, -100, 100, 0);
    assert(result.key_hit);
    assert(!result.generation_match);
    assert(!result.score_usable);
    assert(!result.cutoff);
    assert(result.move == Hint);

    // This switch supports the explicitly generationless experiment without
    // requiring another table implementation.
    result = tt.probe(
        Key, 6, -100, 100, 0,
        TT::DepthPolicy::AtLeast, TT::GenerationPolicy::Any);
    assert(result.key_hit);
    assert(!result.generation_match);
    assert(result.score_usable);
    assert(result.cutoff);
    assert(result.score == 31);
}

void test_generation_wrap_is_constant_time_logical_clear() {
    TT tt(1);
    assert(tt.store(Key, 6, 31, TT::Bound::Exact, Hint, 0));

    for (std::uint32_t i = 0;
         i <= std::numeric_limits<std::uint16_t>::max();
         ++i) {
        tt.advance_generation();
    }

    assert(tt.current_generation() == 0);
    // The epoch salt changed, so even GenerationPolicy::Any cannot mistake an
    // entry from the previous 16-bit epoch for a current key.  No table clear
    // or memory-sized write is needed at wrap.
    TT::ProbeResult result = tt.probe(
        Key, 6, -100, 100, 0,
        TT::DepthPolicy::AtLeast, TT::GenerationPolicy::Any);
    assert(!result.key_hit);

    assert(tt.store(Key, 6, 32, TT::Bound::Exact, Hint, 0));
    result = tt.probe(Key, 6, -100, 100, 0);
    assert(result.cutoff);
    assert(result.score == 32);
}

void test_opposite_bounds_replace_without_merge() {
    TT tt(1);
    assert(tt.store(Key, 6, 151, TT::Bound::Upper, Hint, 0));
    assert(tt.store(Key, 6, 153, TT::Bound::Lower, Hint, 0));

    const std::optional<TT::DecodedEntry> entry = tt.inspect(Key);
    assert(entry.has_value());
    assert(entry->bound == TT::Bound::Lower);
    assert(entry->score == 153);

    TT::ProbeResult result = tt.probe(Key, 6, 150, 152, 0);
    assert(result.cutoff);
    assert(result.bound == TT::Bound::Lower);
    assert(result.score == 153);
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
    assert(tt.inspect(Key)->score == 40);
}

void test_shallow_move_only_does_not_poison_deeper_score_move() {
    TT tt(1);
    constexpr Move deeper_move{0x1111};
    constexpr Move shallow_move{0x2222};
    constexpr Move equally_deep_move{0x3333};

    assert(tt.store(Key, 12, 37, TT::Bound::Exact, deeper_move, 0));
    assert(!tt.store_move(Key, 11, shallow_move));
    std::optional<TT::DecodedEntry> entry = tt.inspect(Key);
    assert(entry.has_value());
    assert(entry->move == deeper_move);
    assert(entry->score == 37);
    assert(entry->depth == 12);
    assert(entry->bound == TT::Bound::Exact);

    // At equal depth a newer best-move hint is permitted, but it must not
    // alter the scored result's depth, score, or bound.
    assert(tt.store_move(Key, 12, equally_deep_move));
    entry = tt.inspect(Key);
    assert(entry->move == equally_deep_move);
    assert(entry->score == 37);
    assert(entry->depth == 12);
    assert(entry->bound == TT::Bound::Exact);
}

void test_move_only_entry() {
    TT tt(1);
    assert(!tt.store_move(Key, 9, Move{}));
    assert(tt.store_move(Key, 9, Hint));
    const TT::ProbeResult result = tt.probe(Key, 1, -100, 100, 0);
    assert(result.key_hit);
    assert(result.move == Hint);
    assert(result.bound == TT::Bound::None);
    assert(!result.score_usable);
    assert(!result.cutoff);
}

void test_mate_score_normalization() {
    TT tt(1);
    // Mate in 7 as viewed at ply 3 is stored root-independent as mate in 4
    // from ply 0, then restored for a different probing ply.
    constexpr int store_ply = 3;
    constexpr int stored_view = CheckmateScore - 7;
    assert(tt.store(Key, 8, stored_view, TT::Bound::Exact, Hint, store_ply));

    TT::ProbeResult result = tt.probe(Key, 8, -Infinity, Infinity, 3);
    assert(result.score == CheckmateScore - 7);
    result = tt.probe(Key, 8, -Infinity, Infinity, 5);
    assert(result.score == CheckmateScore - 9);

    tt.clear();
    constexpr int losing_view = -CheckmateScore + 7;
    assert(tt.store(Key, 8, losing_view, TT::Bound::Exact, Hint, store_ply));
    result = tt.probe(Key, 8, -Infinity, Infinity, 3);
    assert(result.score == -CheckmateScore + 7);
    result = tt.probe(Key, 8, -Infinity, Infinity, 5);
    assert(result.score == -CheckmateScore + 9);
}

void test_mate_threshold_and_packing_boundaries() {
    TT tt(1);
    constexpr int mate_threshold = CheckmateScore - 1024;

    assert(tt.store(Key, 8, mate_threshold, TT::Bound::Exact, Hint, 3));
    assert(tt.probe(Key, 8, -Infinity, Infinity, 5).score
        == mate_threshold - 2);

    tt.clear();
    assert(tt.store(Key, 8, mate_threshold - 1, TT::Bound::Exact, Hint, 3));
    assert(tt.probe(Key, 8, -Infinity, Infinity, 5).score
        == mate_threshold - 1);

    tt.clear();
    assert(tt.store(Key, 8, -mate_threshold, TT::Bound::Exact, Hint, 3));
    assert(tt.probe(Key, 8, -Infinity, Infinity, 5).score
        == -mate_threshold + 2);

    tt.clear();
    assert(tt.store(Key, 8, -mate_threshold + 1, TT::Bound::Exact, Hint, 3));
    assert(tt.probe(Key, 8, -Infinity, Infinity, 5).score
        == -mate_threshold + 1);

    // Normalizing an impossible mate score beyond the representable engine
    // range must fail instead of wrapping the packed 18-bit score.
    tt.clear();
    assert(!tt.store(Key, 8, CheckmateScore, TT::Bound::Exact, Hint, 1));
    assert(!tt.store(Key, 8, -CheckmateScore, TT::Bound::Exact, Hint, 1));
    assert(!tt.inspect(Key).has_value());

    // Both ends of the 18-bit packed domain round-trip at ply zero.  The
    // negative endpoint also proves that an encoded score of zero is not
    // mistaken for an empty entry (emptiness is the whole payload being zero).
    assert(tt.store(Key, 8, CheckmateScore, TT::Bound::Exact, Hint, 0));
    assert(tt.probe(Key, 8, -Infinity, Infinity, 0).score == CheckmateScore);
    tt.clear();
    assert(tt.store(Key, 8, -CheckmateScore, TT::Bound::Exact, Hint, 0));
    assert(tt.probe(Key, 8, -Infinity, Infinity, 0).score == -CheckmateScore);
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
        assert(tt.probe(key, 1, -100, 100, 0).key_hit);
    }

    // A move-only collision must not evict any current scored entry, even if
    // its nominal depth is much larger.
    const HashKey weak_key = base + 4 * stride;
    assert(!tt.store(weak_key, 255, 0, TT::Bound::None, Hint, 0));
    assert(!tt.probe(weak_key, 0, -100, 100, 0).key_hit);

    // A deeper exact result replaces the weakest slot.
    const HashKey strong_key = base + 5 * stride;
    assert(tt.store(strong_key, 20, 77, TT::Bound::Exact, Hint, 0));
    assert(tt.probe(strong_key, 20, -100, 100, 0).cutoff);
}

void test_zero_hash_is_supported() {
    TT tt(1);
    assert(tt.store(0, 4, 12, TT::Bound::Exact, Hint, 0));
    const TT::ProbeResult result = tt.probe(0, 4, -100, 100, 0);
    assert(result.key_hit);
    assert(result.cutoff);
    assert(result.score == 12);
}

void test_invalid_depth_and_stale_move_preservation() {
    TT tt(1);
    assert(!tt.store(Key, -1, 12, TT::Bound::Exact, Hint, 0));
    assert(!tt.store(Key, 256, 12, TT::Bound::Exact, Hint, 0));
    assert(!tt.probe(Key, -1, -100, 100, 0).key_hit);
    assert(!tt.probe(Key, 256, -100, 100, 0).key_hit);

    assert(tt.store(Key, 9, 12, TT::Bound::Exact, Hint, 0));
    tt.advance_generation();
    assert(tt.store(Key, 7, 19, TT::Bound::Lower, Move{}, 0));
    const std::optional<TT::DecodedEntry> entry = tt.inspect(Key);
    assert(entry.has_value());
    assert(entry->move == Hint);
    assert(entry->score == 19);
    assert(entry->depth == 7);
    assert(entry->bound == TT::Bound::Lower);
}

} // namespace

int main() {
    test_layout();
    test_cutoff_only_probe();
    test_depth_policy();
    test_generation_is_move_only_by_default();
    test_generation_wrap_is_constant_time_logical_clear();
    test_opposite_bounds_replace_without_merge();
    test_same_key_replacement_priority();
    test_shallow_move_only_does_not_poison_deeper_score_move();
    test_move_only_entry();
    test_mate_score_normalization();
    test_mate_threshold_and_packing_boundaries();
    test_full_hash_and_four_entry_bucket();
    test_zero_hash_is_supported();
    test_invalid_depth_and_stale_move_preservation();
    return 0;
}
