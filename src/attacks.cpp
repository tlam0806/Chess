#include "attacks.hpp"
#include <array>
#include <bit>


namespace chess {
    

    template<std::size_t N>
    consteval std::array<Bitboard, 64> make_attacks(std::array<std::pair<int, int>, N>&& offsets) {
        std::array<Bitboard, 64> fast_attack_table{};
        for (Square square = 0; square < BoardSize; square++) {
            int rank = rank_of(square);
            int file = file_of(square);
            Bitboard &attacked_squares = fast_attack_table[square];
            for (auto [d_file, d_rank] : offsets) {
                int new_file = file + d_file;
                int new_rank = rank + d_rank;
                if (is_valid_square(new_file, new_rank)) {
                    const Square attacked_square = make_square(new_file, new_rank);
                    attacked_squares |= bit(attacked_square);
                }
            }
        }
        return fast_attack_table;
    }

    constexpr auto KnightAttacks = make_attacks(std::array<std::pair<int, int>, 8>{{
        {-1, -2}, {-1, 2}, {1, -2}, {1, 2},
        {-2, -1}, {-2, 1}, {2, -1}, {2, 1}
    }});
    constexpr auto KingAttacks = make_attacks(std::array<std::pair<int, int>, 8>{{
        {-1, -1}, {-1,  0}, {-1, 1}, { 0, -1},
        { 0,  1}, {1,  -1}, { 1, 0}, { 1,  1}
    }});
    constexpr auto WhitePawnAttacks = make_attacks(std::array<std::pair<int, int>, 2>{{
        {-1, 1}, {1, 1}
    }});
    constexpr auto BlackPawnAttacks = make_attacks(std::array<std::pair<int, int>, 2>{{
        {-1, -1}, {1, -1}
    }});
    constexpr std:: array<std::pair<int, int>,  4> RookDirection{{
        {1, 0}, {-1, 0}, {0, -1}, {0, 1}
    }};
    constexpr std:: array<std::pair<int, int>,  4> BishopDirection{{
        {1, -1}, {1, 1}, {-1, -1}, {-1, 1}
    }};


    Bitboard knight_attacks(Square square) {
        return KnightAttacks[square];
    }

    Bitboard king_attacks(Square square) {
        return KingAttacks[square];
    }

    Bitboard pawn_attacks(Color color, Square square) {
        return color == Color::White ? WhitePawnAttacks[square] : BlackPawnAttacks[square];
    }

    Bitboard rook_attacks(Square square, Bitboard occupancy) {
        Bitboard attacked_squares{};
        for (auto [d_file, d_rank] : RookDirection) {
            int file = file_of(square) + d_file;
            int rank = rank_of(square) + d_rank;
            while (is_valid_square(file, rank)) {
                const Bitboard square_mask = bit(make_square(file, rank));
                attacked_squares |= square_mask;
                if (square_mask & occupancy) {
                    break;
                }
                file += d_file;
                rank += d_rank;
            }
        }
        return attacked_squares;
    }

    Bitboard bishop_attacks(Square square, Bitboard occupancy) {
        Bitboard attacked_squares{};
        for (auto [d_file, d_rank] : BishopDirection) {
            int file = file_of(square) + d_file;
            int rank = rank_of(square) + d_rank;
            while (is_valid_square(file, rank)) {
                const Bitboard square_mask = bit(make_square(file, rank));
                attacked_squares |= square_mask;
                if (square_mask & occupancy) {
                    break;
                }
                file += d_file;
                rank += d_rank;
            }
        }
        return attacked_squares;
    }

    Bitboard queen_attacks(Square square, Bitboard occupancy) {
        return bishop_attacks(square, occupancy) | rook_attacks(square, occupancy);
    }

    bool is_square_attacked(const Position& pos, Square square, Color by_color) {
        assert(is_valid_square(square));
        Color other_color = by_color == Color::Black ? Color::White : Color::Black;
        int color_index = static_cast<int>(by_color);
        const Bitboard occupancy = pos.occupancy();
        return (pawn_attacks(other_color, square) & pos.pieces[color_index][0])
                        || (knight_attacks(square) & pos.pieces[color_index][1])   
                        || (bishop_attacks(square, occupancy) & pos.pieces[color_index][2])  
                        || (rook_attacks(square, occupancy) & pos.pieces[color_index][3])  
                        || (queen_attacks(square, occupancy) & pos.pieces[color_index][4])  
                        || (king_attacks(square) & pos.pieces[color_index][5]);
    }

    Square king_square(const Position& pos, Color color) {
        const Bitboard king_board = pos.pieces[static_cast<int>(color)][static_cast<int>(PieceType::King)];
        assert(popcount(king_board) == 1);
        return std::countr_zero(king_board);
    }

    bool in_check(const Position& pos, Color king_color) {
        Color attack_color = opposite(king_color);
        return is_square_attacked(pos, king_square(pos, king_color), attack_color);
    }
}