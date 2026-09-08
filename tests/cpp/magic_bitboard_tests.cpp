#include "attacks.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include <utility>

using namespace chess;

namespace {

constexpr std::array<std::pair<int, int>, 4> RookDirections{{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}
}};

constexpr std::array<std::pair<int, int>, 4> BishopDirections{{
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
}};

template <std::size_t N>
Bitboard reference_attacks(Square square, Bitboard occupancy, const std::array<std::pair<int, int>, N>& directions) {
    Bitboard attacks = EmptyBB;
    for (auto [df, dr] : directions) {
        int file = file_of(square) + df;
        int rank = rank_of(square) + dr;
        while (is_valid_square(file, rank)) {
            const Bitboard square_bit = bit(make_square(file, rank));
            attacks |= square_bit;
            if (occupancy & square_bit) {
                break;
            }
            file += df;
            rank += dr;
        }
    }
    return attacks;
}

template <std::size_t N>
Bitboard relevant_mask(Square square, const std::array<std::pair<int, int>, N>& directions) {
    Bitboard mask = EmptyBB;
    for (auto [df, dr] : directions) {
        int file = file_of(square) + df;
        int rank = rank_of(square) + dr;
        while (is_valid_square(file, rank)) {
            const int next_file = file + df;
            const int next_rank = rank + dr;
            if (!is_valid_square(next_file, next_rank)) {
                break;
            }
            mask |= bit(make_square(file, rank));
            file = next_file;
            rank = next_rank;
        }
    }
    return mask;
}

template <typename AttackFn, std::size_t N>
void assert_all_relevant_subsets_match(
    AttackFn attack_fn,
    const std::array<std::pair<int, int>, N>& directions
) {
    for (Square square = 0; square < BoardSize; ++square) {
        const Bitboard mask = relevant_mask(square, directions);
        Bitboard subset = 0;
        do {
            assert(attack_fn(square, subset) == reference_attacks(square, subset, directions));
            subset = (subset - mask) & mask;
        } while (subset != 0);
    }
}

template <typename AttackFn, std::size_t N>
void assert_random_occupancies_match(
    AttackFn attack_fn,
    const std::array<std::pair<int, int>, N>& directions
) {
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    for (int sample = 0; sample < 20000; ++sample) {
        const auto square = static_cast<Square>(rng() % BoardSize);
        const Bitboard occupancy = rng();
        assert(attack_fn(square, occupancy) == reference_attacks(square, occupancy, directions));
    }
}

} // namespace

int main() {
    assert_all_relevant_subsets_match(rook_attacks, RookDirections);
    assert_all_relevant_subsets_match(bishop_attacks, BishopDirections);

    assert_random_occupancies_match(rook_attacks, RookDirections);
    assert_random_occupancies_match(bishop_attacks, BishopDirections);
}
