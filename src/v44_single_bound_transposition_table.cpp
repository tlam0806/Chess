#include "v44_single_bound_transposition_table.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

namespace chess {

namespace {

using Table = V44SingleBoundTranspositionTable;
using Bound = Table::Bound;
using Entry = Table::Entry;

constexpr std::size_t BytesPerMegabyte = 1024 * 1024;
constexpr int MateScoreThreshold = CheckmateScore - 1024;

constexpr unsigned ScoreBits = 18;
constexpr unsigned MoveBits = 16;
constexpr unsigned DepthBits = 8;
constexpr unsigned BoundBits = 2;

constexpr unsigned ScoreShift = 0;
constexpr unsigned MoveShift = ScoreShift + ScoreBits;
constexpr unsigned DepthShift = MoveShift + MoveBits;
constexpr unsigned BoundShift = DepthShift + DepthBits;
constexpr unsigned UsedBits = BoundShift + BoundBits;

static_assert(UsedBits <= 64);
static_assert(2 * CheckmateScore < (1 << ScoreBits));

constexpr std::uint64_t bit_mask(unsigned bits) {
    return (std::uint64_t{1} << bits) - 1;
}

constexpr std::uint64_t ScoreMask = bit_mask(ScoreBits);
constexpr std::uint64_t MoveMask = bit_mask(MoveBits);
constexpr std::uint64_t DepthMask = bit_mask(DepthBits);
constexpr std::uint64_t BoundMask = bit_mask(BoundBits);

std::size_t floor_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result <= value / 2) {
        result *= 2;
    }
    return result;
}

std::size_t bucket_count_from_megabytes(std::size_t megabytes) {
    const std::size_t requested_bytes =
        std::max<std::size_t>(megabytes, 1) * BytesPerMegabyte;
    const std::size_t requested_buckets =
        std::max<std::size_t>(requested_bytes / sizeof(Table::Bucket), 1);
    return floor_power_of_two(requested_buckets);
}

constexpr bool valid_depth(int depth) {
    return 0 <= depth && depth <= static_cast<int>(DepthMask);
}

constexpr bool valid_score(int score) {
    return -CheckmateScore <= score && score <= CheckmateScore;
}

int score_to_table(int score, int ply) {
    if (score >= MateScoreThreshold) {
        return score + ply;
    }
    if (score <= -MateScoreThreshold) {
        return score - ply;
    }
    return score;
}

int score_from_table(int score, int ply) {
    if (score >= MateScoreThreshold) {
        return score - ply;
    }
    if (score <= -MateScoreThreshold) {
        return score + ply;
    }
    return score;
}

std::uint64_t pack_payload(int score, Move move, int depth, Bound bound) {
    assert(valid_score(score));
    assert(valid_depth(depth));
    assert((static_cast<unsigned>(bound) & ~BoundMask) == 0);

    const std::uint64_t encoded_score =
        static_cast<std::uint64_t>(score + CheckmateScore);
    return ((encoded_score & ScoreMask) << ScoreShift)
        | ((static_cast<std::uint64_t>(move.value) & MoveMask) << MoveShift)
        | ((static_cast<std::uint64_t>(depth) & DepthMask) << DepthShift)
        | ((static_cast<std::uint64_t>(bound) & BoundMask) << BoundShift);
}

int unpack_score(std::uint64_t payload) {
    return static_cast<int>((payload >> ScoreShift) & ScoreMask)
        - CheckmateScore;
}

Move unpack_move(std::uint64_t payload) {
    return Move{static_cast<std::uint16_t>((payload >> MoveShift) & MoveMask)};
}

int unpack_depth(std::uint64_t payload) {
    return static_cast<int>((payload >> DepthShift) & DepthMask);
}

Bound unpack_bound(std::uint64_t payload) {
    return static_cast<Bound>((payload >> BoundShift) & BoundMask);
}

bool is_empty(const Entry& entry) {
    return entry.payload == 0;
}

bool depth_matches(int stored, int requested, Table::DepthPolicy policy) {
    if (policy == Table::DepthPolicy::Exact) {
        return stored == requested;
    }
    return stored >= requested;
}

int bound_priority(Bound bound) {
    switch (bound) {
    case Bound::Exact:
        return 2;
    case Bound::Lower:
    case Bound::Upper:
        return 1;
    case Bound::None:
        return 0;
    }
    return 0;
}

int replacement_priority(std::uint64_t payload) {
    return unpack_depth(payload) * 3 + bound_priority(unpack_bound(payload));
}

bool tighter_same_bound(Bound bound, int fresh_score, int old_score) {
    if (bound == Bound::Lower) {
        return fresh_score > old_score;
    }
    if (bound == Bound::Upper) {
        return fresh_score < old_score;
    }
    return true;
}

} // namespace

V44SingleBoundTranspositionTable::V44SingleBoundTranspositionTable(
    std::size_t megabytes
) : buckets_(bucket_count_from_megabytes(megabytes)),
    bucket_mask_(buckets_.size() - 1) {}

void V44SingleBoundTranspositionTable::clear() {
    std::fill(buckets_.begin(), buckets_.end(), Bucket{});
}

std::size_t V44SingleBoundTranspositionTable::entry_count() const noexcept {
    return buckets_.size() * 4;
}

std::size_t
V44SingleBoundTranspositionTable::occupied_entry_count() const noexcept {
    std::size_t result = 0;
    for (const Bucket& bucket : buckets_) {
        for (const Entry& entry : bucket.entries) {
            result += !is_empty(entry);
        }
    }
    return result;
}

std::size_t V44SingleBoundTranspositionTable::bucket_count() const noexcept {
    return buckets_.size();
}

std::size_t V44SingleBoundTranspositionTable::memory_bytes() const noexcept {
    return buckets_.size() * sizeof(Bucket);
}

std::size_t V44SingleBoundTranspositionTable::bucket_index(
    HashKey key
) const noexcept {
    return static_cast<std::size_t>(key) & bucket_mask_;
}

V44SingleBoundTranspositionTable::ProbeResult
V44SingleBoundTranspositionTable::probe(
    HashKey key,
    int depth,
    int alpha,
    int beta,
    int ply,
    DepthPolicy depth_policy
) const noexcept {
    ProbeResult result{};
    if (!valid_depth(depth)) {
        return result;
    }

    const Bucket& bucket = buckets_[bucket_index(key)];
    for (const Entry& entry : bucket.entries) {
        if (is_empty(entry) || entry.key != key) {
            continue;
        }

        result.key_hit = true;
        result.move = unpack_move(entry.payload);
        result.depth = unpack_depth(entry.payload);
        result.bound = unpack_bound(entry.payload);

        if (result.bound == Bound::None
            || !depth_matches(result.depth, depth, depth_policy)) {
            return result;
        }

        result.score_usable = true;
        result.score = score_from_table(unpack_score(entry.payload), ply);
        switch (result.bound) {
        case Bound::Exact:
            result.cutoff = true;
            break;
        case Bound::Lower:
            result.cutoff = result.score >= beta;
            break;
        case Bound::Upper:
            result.cutoff = result.score <= alpha;
            break;
        case Bound::None:
            break;
        }
        return result;
    }
    return result;
}

bool V44SingleBoundTranspositionTable::store(
    HashKey key,
    int depth,
    int score,
    Bound bound,
    Move move,
    int ply
) noexcept {
    if (!valid_depth(depth)) {
        return false;
    }
    if (bound == Bound::None) {
        if (move.value == 0) {
            return false;
        }
        score = 0;
    } else {
        score = score_to_table(score, ply);
        if (!valid_score(score)) {
            return false;
        }
    }

    Bucket& bucket = buckets_[bucket_index(key)];
    const std::uint64_t fresh_payload =
        pack_payload(score, move, depth, bound);

    Entry* empty = nullptr;
    Entry* weakest = nullptr;
    int weakest_priority = std::numeric_limits<int>::max();

    for (Entry& entry : bucket.entries) {
        if (is_empty(entry)) {
            if (empty == nullptr) {
                empty = &entry;
            }
            continue;
        }

        if (entry.key == key) {
            const int old_depth = unpack_depth(entry.payload);
            const Bound old_bound = unpack_bound(entry.payload);
            const Move old_move = unpack_move(entry.payload);

            if (bound == Bound::None) {
                if (move.value == 0 || old_move == move) {
                    return false;
                }
                if (old_bound != Bound::None && depth < old_depth) {
                    return false;
                }
                const int stored_depth = old_bound == Bound::None
                    ? std::max(depth, old_depth)
                    : old_depth;
                entry.payload = pack_payload(
                    unpack_score(entry.payload),
                    move,
                    stored_depth,
                    old_bound);
                return true;
            }

            if (old_bound == Bound::None || depth > old_depth) {
                entry.payload = fresh_payload;
                return true;
            }
            if (depth < old_depth) {
                return false;
            }

            if (old_bound == Bound::Exact && bound != Bound::Exact) {
                return false;
            }
            if (bound == Bound::Exact) {
                entry.payload = fresh_payload;
                return true;
            }
            if (old_bound != bound) {
                entry.payload = fresh_payload;
                return true;
            }

            if (tighter_same_bound(bound, score, unpack_score(entry.payload))) {
                entry.payload = fresh_payload;
                return true;
            }
            return false;
        }

        const int priority = replacement_priority(entry.payload);
        if (priority < weakest_priority) {
            weakest_priority = priority;
            weakest = &entry;
        }
    }

    Entry* target = empty;
    if (target == nullptr) {
        if (bound == Bound::None) {
            // A move-only observation cannot evict a scored result. It may
            // compete with another move-only observation in the bucket.
            weakest = nullptr;
            weakest_priority = std::numeric_limits<int>::max();
            for (Entry& entry : bucket.entries) {
                if (unpack_bound(entry.payload) != Bound::None) {
                    continue;
                }
                const int priority = replacement_priority(entry.payload);
                if (priority < weakest_priority) {
                    weakest_priority = priority;
                    weakest = &entry;
                }
            }
        }
        const int fresh_priority = depth * 3 + bound_priority(bound);
        if (weakest == nullptr || fresh_priority < weakest_priority) {
            return false;
        }
        target = weakest;
    }

    target->key = key;
    target->payload = fresh_payload;
    return true;
}

bool V44SingleBoundTranspositionTable::store_move(
    HashKey key,
    int depth,
    Move move
) noexcept {
    return store(key, depth, 0, Bound::None, move, 0);
}

std::optional<V44SingleBoundTranspositionTable::DecodedEntry>
V44SingleBoundTranspositionTable::inspect(HashKey key) const noexcept {
    const Bucket& bucket = buckets_[bucket_index(key)];
    for (const Entry& entry : bucket.entries) {
        if (!is_empty(entry) && entry.key == key) {
            return DecodedEntry{
                key,
                unpack_move(entry.payload),
                unpack_score(entry.payload),
                unpack_depth(entry.payload),
                unpack_bound(entry.payload)
            };
        }
    }
    return std::nullopt;
}

const V44SingleBoundTranspositionTable::Bucket*
V44SingleBoundTranspositionTable::bucket_address(HashKey key) const noexcept {
    return &buckets_[bucket_index(key)];
}

} // namespace chess
