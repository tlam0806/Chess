#include "bitboard.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace chess;

namespace {

constexpr std::array<std::pair<int, int>, 4> RookDirections{{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}
}};

constexpr std::array<std::pair<int, int>, 4> BishopDirections{{
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
}};

struct MagicSet {
    std::array<Bitboard, 64> masks{};
    std::array<Bitboard, 64> magics{};
    std::array<int, 64> shifts{};
};

template <std::size_t N>
Bitboard sliding_attacks(Square square, Bitboard occupancy, const std::array<std::pair<int, int>, N>& directions) {
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

std::vector<Bitboard> blocker_subsets(Bitboard mask) {
    std::vector<Bitboard> subsets;
    subsets.reserve(static_cast<std::size_t>(1) << popcount(mask));

    Bitboard subset = 0;
    do {
        subsets.push_back(subset);
        subset = (subset - mask) & mask;
    } while (subset != 0);

    return subsets;
}

Bitboard random_sparse_u64(std::mt19937_64& rng) {
    return rng() & rng() & rng();
}

template <std::size_t N>
std::optional<Bitboard> try_find_magic(
    Square square,
    Bitboard mask,
    int shift,
    const std::array<std::pair<int, int>, N>& directions,
    std::mt19937_64& rng,
    int max_attempts
) {
    const std::vector<Bitboard> subsets = blocker_subsets(mask);
    std::vector<Bitboard> attacks;
    attacks.reserve(subsets.size());
    for (Bitboard subset : subsets) {
        attacks.push_back(sliding_attacks(square, subset, directions));
    }

    const std::size_t table_size = static_cast<std::size_t>(1) << (64 - shift);
    std::vector<Bitboard> used(table_size);
    std::vector<bool> occupied(table_size);

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        const Bitboard magic = random_sparse_u64(rng);
        if (popcount(mask * magic) < 6) {
            continue;
        }

        std::fill(used.begin(), used.end(), EmptyBB);
        std::fill(occupied.begin(), occupied.end(), false);

        bool ok = true;
        for (std::size_t i = 0; i < subsets.size(); ++i) {
            const std::size_t index = static_cast<std::size_t>((subsets[i] * magic) >> shift);
            if (!occupied[index]) {
                occupied[index] = true;
                used[index] = attacks[i];
            } else if (used[index] != attacks[i]) {
                ok = false;
                break;
            }
        }

        if (ok) {
            return magic;
        }
    }

    return std::nullopt;
}

template <std::size_t N>
MagicSet generate_magic_set(
    const char* name,
    const std::array<std::pair<int, int>, N>& directions,
    std::uint64_t seed
) {
    MagicSet set;
    std::mt19937_64 rng(seed);

    for (Square square = 0; square < BoardSize; ++square) {
        set.masks[square] = relevant_mask(square, directions);
        set.shifts[square] = 64 - popcount(set.masks[square]);

        const std::optional<Bitboard> magic =
            try_find_magic(square, set.masks[square], set.shifts[square], directions, rng, 50'000'000);
        if (!magic) {
            throw std::runtime_error(std::string("failed to find ") + name + " magic for square " + std::to_string(square));
        }
        set.magics[square] = *magic;

        std::cerr << name << " square=" << square
                  << " bits=" << (64 - set.shifts[square])
                  << " magic=0x" << std::hex << set.magics[square] << std::dec << '\n';
    }

    return set;
}

void print_bitboard_array(const char* name, const std::array<Bitboard, 64>& values) {
    std::cout << "constexpr std::array<Bitboard, 64> " << name << "{{\n";
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << "    0x" << std::hex << values[i] << "ULL" << std::dec;
        std::cout << (i + 1 == values.size() ? "\n" : ",\n");
    }
    std::cout << "}};\n\n";
}

void print_shift_array(const char* name, const std::array<int, 64>& values) {
    std::cout << "constexpr std::array<int, 64> " << name << "{{\n    ";
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << values[i];
        if (i + 1 != values.size()) {
            std::cout << ", ";
        }
        if ((i + 1) % 16 == 0 && i + 1 != values.size()) {
            std::cout << "\n    ";
        }
    }
    std::cout << "\n}};\n\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::uint64_t seed = 0xc0ffee1234ULL;
        if (argc == 3 && std::string(argv[1]) == "--seed") {
            seed = std::stoull(argv[2], nullptr, 0);
        } else if (argc != 1) {
            std::cerr << "Usage: generate_magic_bitboards [--seed N]\n";
            return 2;
        }

        const MagicSet rook = generate_magic_set("rook", RookDirections, seed);
        const MagicSet bishop = generate_magic_set("bishop", BishopDirections, seed ^ 0x9e3779b97f4a7c15ULL);

        print_bitboard_array("RookMasks", rook.masks);
        print_bitboard_array("RookMagics", rook.magics);
        print_shift_array("RookShifts", rook.shifts);
        print_bitboard_array("BishopMasks", bishop.masks);
        print_bitboard_array("BishopMagics", bishop.magics);
        print_shift_array("BishopShifts", bishop.shifts);
    } catch (const std::exception& error) {
        std::cerr << "generate_magic_bitboards: " << error.what() << '\n';
        return 1;
    }
}
