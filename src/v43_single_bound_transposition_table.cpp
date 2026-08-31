#include "v43_single_bound_transposition_table.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

namespace chess {

namespace {

using Table = V43SingleBoundTranspositionTable;
using Bound = Table::Bound;
using Entry = Table::Entry;

constexpr std::size_t BytesPerMegabyte = 1024 * 1024;
constexpr int MateScoreThreshold = CheckmateScore - 1024;

constexpr unsigned ScoreBits = 18;
constexpr unsigned MoveBits = 16;
constexpr unsigned DepthBits = 8;
constexpr unsigned GenerationBits = 16;
constexpr unsigned BoundBits = 2;
constexpr unsigned EpochBits = 4;

constexpr unsigned ScoreShift = 0;
constexpr unsigned MoveShift = ScoreShift + ScoreBits;
constexpr unsigned DepthShift = MoveShift + MoveBits;
constexpr unsigned GenerationShift = DepthShift + DepthBits;
constexpr unsigned BoundShift = GenerationShift + GenerationBits;
constexpr unsigned EpochShift = BoundShift + BoundBits;

static_assert(EpochShift + EpochBits == 64);
static_assert(2 * CheckmateScore < (1 << ScoreBits));

constexpr std::uint64_t bit_mask(unsigned bits) {
    return (std::uint64_t{1} << bits) - 1;
}

constexpr std::uint64_t ScoreMask = bit_mask(ScoreBits);
constexpr std::uint64_t MoveMask = bit_mask(MoveBits);
constexpr std::uint64_t DepthMask = bit_mask(DepthBits);
constexpr std::uint64_t GenerationMask = bit_mask(GenerationBits);
constexpr std::uint64_t BoundMask = bit_mask(BoundBits);
constexpr std::uint64_t EpochMask = bit_mask(EpochBits);

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

std::uint64_t pack_payload(
    int score,
    Move move,
    int depth,
    std::uint16_t generation,
    std::uint8_t epoch,
    Bound bound
) {
    assert(valid_score(score));
    assert(valid_depth(depth));
    assert((static_cast<unsigned>(bound) & ~BoundMask) == 0);
    assert((epoch & ~EpochMask) == 0);

    const std::uint64_t encoded_score = static_cast<std::uint64_t>(score + CheckmateScore);
    return ((encoded_score & ScoreMask) << ScoreShift)
        | ((static_cast<std::uint64_t>(move.value) & MoveMask) << MoveShift)
        | ((static_cast<std::uint64_t>(depth) & DepthMask) << DepthShift)
        | ((static_cast<std::uint64_t>(generation) & GenerationMask) << GenerationShift)
        | ((static_cast<std::uint64_t>(bound) & BoundMask) << BoundShift)
        | ((static_cast<std::uint64_t>(epoch) & EpochMask) << EpochShift);
}

int unpack_score(std::uint64_t payload) {
    return static_cast<int>((payload >> ScoreShift) & ScoreMask) - CheckmateScore;
}

Move unpack_move(std::uint64_t payload) {
    return Move{static_cast<std::uint16_t>((payload >> MoveShift) & MoveMask)};
}

int unpack_depth(std::uint64_t payload) {
    return static_cast<int>((payload >> DepthShift) & DepthMask);
}

std::uint16_t unpack_generation(std::uint64_t payload) {
    return static_cast<std::uint16_t>((payload >> GenerationShift) & GenerationMask);
}

Bound unpack_bound(std::uint64_t payload) {
    return static_cast<Bound>((payload >> BoundShift) & BoundMask);
}

std::uint8_t unpack_epoch(std::uint64_t payload) {
    return static_cast<std::uint8_t>((payload >> EpochShift) & EpochMask);
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

bool same_generation(
    std::uint64_t payload,
    std::uint16_t generation,
    std::uint8_t epoch
) {
    return unpack_generation(payload) == generation && unpack_epoch(payload) == epoch;
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

// SplitMix64 finalizer.  It is used only once per 65,536 generations, not on
// the probe path.
HashKey next_epoch_salt(HashKey salt) {
    std::uint64_t value = salt + 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

} // namespace

V43SingleBoundTranspositionTable::V43SingleBoundTranspositionTable(
    std::size_t megabytes
) : buckets_(bucket_count_from_megabytes(megabytes)),
    bucket_mask_(buckets_.size() - 1) {}

void V43SingleBoundTranspositionTable::clear() {
    std::fill(buckets_.begin(), buckets_.end(), Bucket{});
    generation_ = 0;
    epoch_ = 0;
    epoch_key_salt_ = 0;
}

void V43SingleBoundTranspositionTable::advance_generation() noexcept {
    if (generation_ == std::numeric_limits<std::uint16_t>::max()) {
        generation_ = 0;
        epoch_ = static_cast<std::uint8_t>((epoch_ + 1) & EpochMask);
        epoch_key_salt_ = next_epoch_salt(epoch_key_salt_);
        return;
    }
    ++generation_;
}

std::size_t V43SingleBoundTranspositionTable::entry_count() const noexcept {
    return buckets_.size() * 4;
}

std::size_t V43SingleBoundTranspositionTable::bucket_count() const noexcept {
    return buckets_.size();
}

std::size_t V43SingleBoundTranspositionTable::memory_bytes() const noexcept {
    return buckets_.size() * sizeof(Bucket);
}

std::uint16_t V43SingleBoundTranspositionTable::current_generation() const noexcept {
    return generation_;
}

std::size_t V43SingleBoundTranspositionTable::bucket_index(HashKey key) const noexcept {
    return static_cast<std::size_t>(key) & bucket_mask_;
}

HashKey V43SingleBoundTranspositionTable::stored_key(HashKey key) const noexcept {
    return key ^ epoch_key_salt_;
}

V43SingleBoundTranspositionTable::ProbeResult
V43SingleBoundTranspositionTable::probe(
    HashKey key,
    int depth,
    int alpha,
    int beta,
    int ply,
    DepthPolicy depth_policy,
    GenerationPolicy generation_policy
) const noexcept {
    ProbeResult result{};
    if (!valid_depth(depth)) {
        return result;
    }

    const HashKey wanted_key = stored_key(key);
    const Bucket& bucket = buckets_[bucket_index(key)];
    for (const Entry& entry : bucket.entries) {
        if (is_empty(entry)) {
            continue;
        }
        if (entry.key != wanted_key) {
            continue;
        }

        result.key_hit = true;
        result.move = unpack_move(entry.payload);
        result.depth = unpack_depth(entry.payload);
        result.bound = unpack_bound(entry.payload);
        result.generation_match = same_generation(entry.payload, generation_, epoch_);

        if (result.bound == Bound::None
            || !depth_matches(result.depth, depth, depth_policy)
            || (!result.generation_match
                && generation_policy == GenerationPolicy::CurrentOnly)) {
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

bool V43SingleBoundTranspositionTable::store(
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

    const HashKey wanted_key = stored_key(key);
    Bucket& bucket = buckets_[bucket_index(key)];
    const std::uint64_t fresh_payload =
        pack_payload(score, move, depth, generation_, epoch_, bound);

    Entry* empty = nullptr;
    Entry* stale_victim = nullptr;
    Entry* weakest = nullptr;
    int stale_priority = std::numeric_limits<int>::max();
    int weakest_priority = std::numeric_limits<int>::max();

    for (Entry& entry : bucket.entries) {
        if (is_empty(entry)) {
            if (empty == nullptr) {
                empty = &entry;
            }
            continue;
        }

        if (entry.key == wanted_key) {
            const bool current = same_generation(entry.payload, generation_, epoch_);
            const int old_depth = unpack_depth(entry.payload);
            const Bound old_bound = unpack_bound(entry.payload);
            const Move old_move = unpack_move(entry.payload);

            if (!current) {
                const Move chosen_move = move.value != 0 ? move : old_move;
                entry.payload = pack_payload(
                    score, chosen_move, depth, generation_, epoch_, bound);
                return true;
            }

            if (bound == Bound::None) {
                if (move.value == 0 || old_move == move) {
                    return false;
                }
                // A move-only observation must not replace the best move
                // attached to a deeper scored result.  Keep the scored
                // result's depth because store_move has no score valid at the
                // requested depth.  For another move-only entry, retaining the
                // deepest observation improves its collision priority without
                // manufacturing a usable score.
                if (old_bound != Bound::None && depth < old_depth) {
                    return false;
                }
                const int stored_depth = old_bound == Bound::None
                    ? std::max(depth, old_depth)
                    : old_depth;
                entry.payload = pack_payload(
                    unpack_score(entry.payload), move, stored_depth,
                    generation_, epoch_, old_bound);
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
                // Opposite one-sided bounds replace; they are never merged.
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
        if (!same_generation(entry.payload, generation_, epoch_)
            && priority < stale_priority) {
            stale_priority = priority;
            stale_victim = &entry;
        }
        if (priority < weakest_priority) {
            weakest_priority = priority;
            weakest = &entry;
        }
    }

    Entry* target = empty != nullptr ? empty : stale_victim;
    if (target == nullptr) {
        if (bound == Bound::None) {
            // A current-generation score remains more useful than a move-only
            // hint regardless of the hint's nominal depth.  Move-only entries
            // may compete only with other move-only entries (stale entries
            // were handled above).
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

    target->key = wanted_key;
    target->payload = fresh_payload;
    return true;
}

bool V43SingleBoundTranspositionTable::store_move(
    HashKey key,
    int depth,
    Move move
) noexcept {
    return store(key, depth, 0, Bound::None, move, 0);
}

std::optional<V43SingleBoundTranspositionTable::DecodedEntry>
V43SingleBoundTranspositionTable::inspect(HashKey key) const noexcept {
    const HashKey wanted_key = stored_key(key);
    const Bucket& bucket = buckets_[bucket_index(key)];
    for (const Entry& entry : bucket.entries) {
        if (!is_empty(entry) && entry.key == wanted_key) {
            return DecodedEntry{
                key,
                unpack_move(entry.payload),
                unpack_score(entry.payload),
                unpack_depth(entry.payload),
                unpack_generation(entry.payload),
                unpack_epoch(entry.payload),
                unpack_bound(entry.payload)
            };
        }
    }
    return std::nullopt;
}

const V43SingleBoundTranspositionTable::Bucket*
V43SingleBoundTranspositionTable::bucket_address(HashKey key) const noexcept {
    return &buckets_[bucket_index(key)];
}

} // namespace chess
