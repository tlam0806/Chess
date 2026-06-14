#include "transposition_table.hpp"

#include <algorithm>
#include <cstdlib>

namespace chess {

namespace {

constexpr std::size_t BytesPerMegabyte = 1024 * 1024;
constexpr int MateScoreThreshold = CheckmateScore - 1024;

std::size_t entry_count_from_megabytes(std::size_t megabytes) {
    const std::size_t bytes = std::max<std::size_t>(megabytes, 1) * BytesPerMegabyte;
    return std::max<std::size_t>(bytes / sizeof(TTEntry), 1);
}

int score_from_tt(int score, int ply) {
    if (score >= Infinity / 2 || score <= -Infinity / 2) {
        return score;
    }
    if (score >= MateScoreThreshold) {
        return score - ply;
    }
    if (score <= -MateScoreThreshold) {
        return score + ply;
    }
    return score;
}

} // namespace

TranspositionTable::TranspositionTable(std::size_t megabytes)
    : entries_(entry_count_from_megabytes(megabytes)) {
}

void TranspositionTable::clear() {
    std::fill(entries_.begin(), entries_.end(), TTEntry{});
}

void TranspositionTable::clear_stats() {
    stats_ = {};
}

std::size_t TranspositionTable::entry_count() const {
    return entries_.size();
}

const TranspositionTableStats& TranspositionTable::stats() const {
    return stats_;
}

bool TranspositionTable::probe(
    HashKey key,
    int depth,
    int& alpha,
    int& beta,
    int& score,
    Move& best_move
) const {
    return probe(key, depth, alpha, beta, score, best_move, TTProbeOptions{});
}

bool TranspositionTable::probe(
    HashKey key,
    int depth,
    int& alpha,
    int& beta,
    int& score,
    Move& best_move,
    TTProbeOptions options,
    TTBound* hit_bound
) const {
    ++stats_.probes;
    const TTEntry& entry = entries_[key % entries_.size()];

    if (!entry.valid) {
        ++stats_.empty_misses;
        return false;
    }

    if (entry.key != key) {
        ++stats_.index_collisions;
        return false;
    }

    ++stats_.key_hits;
    best_move = entry.best_move;
    if (best_move.value != 0) {
        ++stats_.move_hint_hits;
    }

    if (entry.depth < depth) {
        ++stats_.depth_misses;
        return false;
    }

    const int entry_score = options.normalize_mate_scores
        ? score_from_tt(entry.score, options.ply)
        : entry.score;
    if (!options.use_mate_scores && std::abs(entry_score) >= MateScoreThreshold) {
        return false;
    }

    if (entry.bound == TTBound::Exact) {
        ++stats_.exact_hits;
        if (!options.use_exact) {
            return false;
        }
        score = entry_score;
        if (hit_bound != nullptr) {
            *hit_bound = TTBound::Exact;
        }
        return true;
    }

    if (!options.use_bounds) {
        return false;
    }

    const int alpha_before = alpha;
    const int beta_before = beta;
    if (options.allow_bound_tighten
        && entry.bound == TTBound::Lower
        && (options.allow_bound_cutoff || entry_score < beta)) {
        ++stats_.lower_bound_hits;
        alpha = std::max(alpha, entry_score);
    } else if (options.allow_bound_tighten
        && entry.bound == TTBound::Upper
        && (options.allow_bound_cutoff || entry_score > alpha)) {
        ++stats_.upper_bound_hits;
        beta = std::min(beta, entry_score);
    } else if (entry.bound == TTBound::Lower) {
        ++stats_.lower_bound_hits;
    } else if (entry.bound == TTBound::Upper) {
        ++stats_.upper_bound_hits;
    }
    if (alpha != alpha_before || beta != beta_before) {
        ++stats_.bound_tightens;
    }

    const bool bound_cutoff =
        entry.bound == TTBound::Lower
            ? entry_score >= beta_before
            : entry_score <= alpha_before;
    if (options.allow_bound_cutoff && (alpha >= beta || bound_cutoff)) {
        ++stats_.bound_cutoffs;
        score = entry_score;
        if (hit_bound != nullptr) {
            *hit_bound = entry.bound;
        }
        return true;
    }

    return false;
}

void TranspositionTable::store(
    HashKey key,
    int depth,
    int score,
    TTBound bound,
    Move best_move
) {
    ++stats_.stores;
    TTEntry& entry = entries_[key % entries_.size()];
    if (!entry.valid || entry.key == key || depth >= entry.depth) {
        if (!entry.valid) {
            ++stats_.new_stores;
        } else if (entry.key == key) {
            ++stats_.same_key_updates;
        } else {
            ++stats_.replacement_collisions;
        }
        entry.key = key;
        entry.valid = true;
        entry.depth = depth;
        entry.score = score;
        entry.bound = bound;
        entry.best_move = best_move;
    } else {
        ++stats_.skipped_shallow_replacements;
    }
}

} // namespace chess
