#pragma once

#include "types.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace chess {

// Path-local repetition state. This is intentionally separate from the TT:
// entries in the TT are shared by unrelated branches, while this stack must
// follow make/unmake exactly.
class RepetitionStack {
public:
    static constexpr std::size_t Capacity = 256;

    struct Stats {
        std::uint64_t threefold_draws = 0;
        std::uint64_t search_cycle_draws = 0;
        std::uint64_t fifty_move_draws = 0;
        std::uint64_t tt_score_suppressions = 0;
    };

    void reset(
        std::span<const HashKey> game_history,
        HashKey current_key,
        int halfmove_clock,
        bool stop_on_search_cycle = false
    ) noexcept {
        size_ = 0;
        segment_begin_ = 0;
        twofold_positions_ = 0;
        root_index_ = Capacity;
        stop_on_search_cycle_ = stop_on_search_cycle;
        stats_ = {};

        // A pawn move or capture resets halfmove_clock and makes every older
        // position unreachable. Keep only the relevant, cache-friendly tail.
        const std::size_t reversible_positions = static_cast<std::size_t>(
            std::max(0, halfmove_clock)) + 1;
        const std::size_t available = std::min(
            game_history.size(),
            std::min(reversible_positions, Capacity - 128));
        const std::size_t begin = game_history.size() - available;
        for (std::size_t index = begin; index < game_history.size(); ++index) {
            push_impl(game_history[index], false, Capacity);
        }

        // Be defensive for FEN callers and APIs which pass no history or omit
        // the root position. UCI replay passes a history ending at current_key.
        if (size_ == 0 || keys_[size_ - 1] != current_key) {
            push_impl(
                current_key,
                halfmove_clock == 0,
                static_cast<std::size_t>(std::max(0, halfmove_clock)));
        }
        root_index_ = size_ - 1;
    }

    void push(
        HashKey key,
        bool irreversible,
        int halfmove_clock
    ) noexcept {
        push_impl(
            key,
            irreversible,
            static_cast<std::size_t>(std::max(0, halfmove_clock)));
    }

    void pop() noexcept {
        assert(size_ > 0);
        const std::size_t index = size_ - 1;
        if (prior_occurrences_[index] == IrreversibleMarker) {
            const std::uint32_t saved = saved_irreversible_state_[index];
            segment_begin_ = saved >> 16;
            twofold_positions_ = saved & 0xFFFFU;
        } else if ((prior_occurrences_[index] & PriorCountMask) == 1) {
            assert(twofold_positions_ > 0);
            --twofold_positions_;
        }
        --size_;
    }

    bool current_is_threefold() const noexcept {
        return size_ > 0
            && prior_occurrences_[size_ - 1] != IrreversibleMarker
            && (prior_occurrences_[size_ - 1] & PriorCountMask) >= 2;
    }

    // A second occurrence is sufficient only when the earlier occurrence is
    // at or below this search's root. Positions found solely in the real game
    // history still require a literal third occurrence.
    bool current_repeats_in_search_path() const noexcept {
        return size_ > 0
            && prior_occurrences_[size_ - 1] != IrreversibleMarker
            && (prior_occurrences_[size_ - 1] & SearchPathMarker) != 0;
    }

    bool has_twofold_position() const noexcept {
        return twofold_positions_ != 0;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    const Stats& stats() const noexcept {
        return stats_;
    }

    void record_threefold_draw() noexcept {
        ++stats_.threefold_draws;
    }

    void record_search_cycle_draw() noexcept {
        ++stats_.search_cycle_draws;
    }

    void record_fifty_move_draw() noexcept {
        ++stats_.fifty_move_draws;
    }

    void record_tt_score_suppression() noexcept {
        ++stats_.tt_score_suppressions;
    }

private:
    void push_impl(
        HashKey key,
        bool irreversible,
        std::size_t max_lookback
    ) noexcept {
        assert(size_ < Capacity);
        const std::size_t index = size_;
        if (irreversible) {
            keys_[index] = key;
            prior_occurrences_[index] = IrreversibleMarker;
            saved_irreversible_state_[index] =
                (static_cast<std::uint32_t>(segment_begin_) << 16)
                | static_cast<std::uint32_t>(twofold_positions_);
            segment_begin_ = index;
            twofold_positions_ = 0;
            ++size_;
            return;
        }

        std::uint8_t prior = 0;
        bool prior_in_search_path = false;
        if (max_lookback >= 4 && index >= 4) {
            // Identical positions have the same side to move, hence only
            // every second entry can match. A legal identical position needs
            // at least four plies, so index-2 is known impossible.
            const std::size_t earliest = std::max(
                segment_begin_,
                index - std::min(index, max_lookback));
            for (std::size_t previous = index - 4;; previous -= 2) {
                if (previous < earliest) {
                    break;
                }
                if (keys_[previous] == key) {
                    ++prior;
                    const bool match_in_search_path =
                        previous >= root_index_;
                    prior_in_search_path |= match_in_search_path;
                    if (prior == 2
                        || (stop_on_search_cycle_
                            && match_in_search_path)) {
                        break;
                    }
                }
                if (previous < earliest + 2) {
                    break;
                }
            }
        }

        keys_[index] = key;
        prior_occurrences_[index] = static_cast<std::uint8_t>(
            prior | (prior_in_search_path ? SearchPathMarker : 0));
        ++size_;
        if (prior == 1) {
            ++twofold_positions_;
        }
    }

    std::array<HashKey, Capacity> keys_{};
    // Packed previous segment begin (high 16 bits) and twofold count (low 16).
    // This array is touched only by irreversible pushes/pops.
    std::array<std::uint32_t, Capacity> saved_irreversible_state_{};
    // Low seven bits store the capped occurrence count; the high bit marks a
    // match at or below the current search root. 0xFF remains irreversible.
    std::array<std::uint8_t, Capacity> prior_occurrences_{};
    std::size_t size_ = 0;
    std::size_t segment_begin_ = 0;
    std::size_t twofold_positions_ = 0;
    std::size_t root_index_ = Capacity;
    bool stop_on_search_cycle_ = false;
    Stats stats_{};
    static constexpr std::uint8_t PriorCountMask = 0x7F;
    static constexpr std::uint8_t SearchPathMarker = 0x80;
    static constexpr std::uint8_t IrreversibleMarker = 0xFF;
};

class ScopedRepetitionPush {
public:
    ScopedRepetitionPush(
        RepetitionStack* stack,
        HashKey key,
        bool irreversible,
        int halfmove_clock
    ) noexcept
        : stack_(stack) {
        if (stack_ != nullptr) {
            stack_->push(key, irreversible, halfmove_clock);
        }
    }

    ~ScopedRepetitionPush() noexcept {
        if (stack_ != nullptr) {
            stack_->pop();
        }
    }

    ScopedRepetitionPush(const ScopedRepetitionPush&) = delete;
    ScopedRepetitionPush& operator=(const ScopedRepetitionPush&) = delete;

private:
    RepetitionStack* stack_ = nullptr;
};

} // namespace chess
