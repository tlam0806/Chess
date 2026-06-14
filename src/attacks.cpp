#include "attacks.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <utility>


namespace chess {
    
namespace {

constexpr std::size_t RookTableSize = 4096;
constexpr std::size_t BishopTableSize = 512;

constexpr std::array<Bitboard, 64> RookMasks{{
    0x000101010101017eULL, 0x000202020202027cULL, 0x000404040404047aULL, 0x0008080808080876ULL,
    0x001010101010106eULL, 0x002020202020205eULL, 0x004040404040403eULL, 0x008080808080807eULL,
    0x0001010101017e00ULL, 0x0002020202027c00ULL, 0x0004040404047a00ULL, 0x0008080808087600ULL,
    0x0010101010106e00ULL, 0x0020202020205e00ULL, 0x0040404040403e00ULL, 0x0080808080807e00ULL,
    0x00010101017e0100ULL, 0x00020202027c0200ULL, 0x00040404047a0400ULL, 0x0008080808760800ULL,
    0x00101010106e1000ULL, 0x00202020205e2000ULL, 0x00404040403e4000ULL, 0x00808080807e8000ULL,
    0x000101017e010100ULL, 0x000202027c020200ULL, 0x000404047a040400ULL, 0x0008080876080800ULL,
    0x001010106e101000ULL, 0x002020205e202000ULL, 0x004040403e404000ULL, 0x008080807e808000ULL,
    0x0001017e01010100ULL, 0x0002027c02020200ULL, 0x0004047a04040400ULL, 0x0008087608080800ULL,
    0x0010106e10101000ULL, 0x0020205e20202000ULL, 0x0040403e40404000ULL, 0x0080807e80808000ULL,
    0x00017e0101010100ULL, 0x00027c0202020200ULL, 0x00047a0404040400ULL, 0x0008760808080800ULL,
    0x00106e1010101000ULL, 0x00205e2020202000ULL, 0x00403e4040404000ULL, 0x00807e8080808000ULL,
    0x007e010101010100ULL, 0x007c020202020200ULL, 0x007a040404040400ULL, 0x0076080808080800ULL,
    0x006e101010101000ULL, 0x005e202020202000ULL, 0x003e404040404000ULL, 0x007e808080808000ULL,
    0x7e01010101010100ULL, 0x7c02020202020200ULL, 0x7a04040404040400ULL, 0x7608080808080800ULL,
    0x6e10101010101000ULL, 0x5e20202020202000ULL, 0x3e40404040404000ULL, 0x7e80808080808000ULL,
}};

constexpr std::array<Bitboard, 64> RookMagics{{
    0x0080004000809020ULL, 0x1040001009c02000ULL, 0x4180092000100081ULL, 0x18800c6800801001ULL,
    0x02000810a0042200ULL, 0x0500020100040008ULL, 0x840000840a083310ULL, 0x0200020044088021ULL,
    0x800e80058040012cULL, 0x0000400040201000ULL, 0x4212002090814200ULL, 0x0452800800805000ULL,
    0x0001001105180100ULL, 0x88a200100c420038ULL, 0x801b000200030004ULL, 0x0411000080510002ULL,
    0x0000a88004804010ULL, 0x0000848040006000ULL, 0x8802430011002008ULL, 0x024a02001288a040ULL,
    0x80a4310005000800ULL, 0x0900808024001e00ULL, 0x90065c0001100208ULL, 0x0182420003008064ULL,
    0x8100400080008020ULL, 0x0000200880400080ULL, 0x0400802200409200ULL, 0x0000100100286100ULL,
    0x2088000900041100ULL, 0x20aa000600100408ULL, 0x4482000200041108ULL, 0x0203000900008042ULL,
    0x000080c0028000e0ULL, 0x1000814000802000ULL, 0x1014908602002040ULL, 0x0104800800801000ULL,
    0xa010822400800800ULL, 0x400a001002000804ULL, 0x0002000102000824ULL, 0x028202408200040dULL,
    0x1080004220004000ULL, 0x20205001a0014000ULL, 0x0502008012420020ULL, 0x9410008100080800ULL,
    0x00002c0801010011ULL, 0x1105000400230008ULL, 0x8000411012440008ULL, 0x1142008141020004ULL,
    0x40410080b2024200ULL, 0x20200a8023400080ULL, 0x0000100c20008080ULL, 0x0008001000088080ULL,
    0x4002045188010100ULL, 0x20c2009004080200ULL, 0x2442451008028400ULL, 0x4060390254029200ULL,
    0x1041002010438001ULL, 0x0000208051014003ULL, 0x2880102001090241ULL, 0x000110000c0900a1ULL,
    0x0042001008042002ULL, 0x0012001001840802ULL, 0x020008120100900cULL, 0x0004a08300422402ULL,
}};

constexpr std::array<int, 64> RookShifts{{
    52, 53, 53, 53, 53, 53, 53, 52, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 52, 53, 53, 53, 53, 53, 53, 52,
}};

constexpr std::array<Bitboard, 64> BishopMasks{{
    0x0040201008040200ULL, 0x0000402010080400ULL, 0x0000004020100a00ULL, 0x0000000040221400ULL,
    0x0000000002442800ULL, 0x0000000204085000ULL, 0x0000020408102000ULL, 0x0002040810204000ULL,
    0x0020100804020000ULL, 0x0040201008040000ULL, 0x00004020100a0000ULL, 0x0000004022140000ULL,
    0x0000000244280000ULL, 0x0000020408500000ULL, 0x0002040810200000ULL, 0x0004081020400000ULL,
    0x0010080402000200ULL, 0x0020100804000400ULL, 0x004020100a000a00ULL, 0x0000402214001400ULL,
    0x0000024428002800ULL, 0x0002040850005000ULL, 0x0004081020002000ULL, 0x0008102040004000ULL,
    0x0008040200020400ULL, 0x0010080400040800ULL, 0x0020100a000a1000ULL, 0x0040221400142200ULL,
    0x0002442800284400ULL, 0x0004085000500800ULL, 0x0008102000201000ULL, 0x0010204000402000ULL,
    0x0004020002040800ULL, 0x0008040004081000ULL, 0x00100a000a102000ULL, 0x0022140014224000ULL,
    0x0044280028440200ULL, 0x0008500050080400ULL, 0x0010200020100800ULL, 0x0020400040201000ULL,
    0x0002000204081000ULL, 0x0004000408102000ULL, 0x000a000a10204000ULL, 0x0014001422400000ULL,
    0x0028002844020000ULL, 0x0050005008040200ULL, 0x0020002010080400ULL, 0x0040004020100800ULL,
    0x0000020408102000ULL, 0x0000040810204000ULL, 0x00000a1020400000ULL, 0x0000142240000000ULL,
    0x0000284402000000ULL, 0x0000500804020000ULL, 0x0000201008040200ULL, 0x0000402010080400ULL,
    0x0002040810204000ULL, 0x0004081020400000ULL, 0x000a102040000000ULL, 0x0014224000000000ULL,
    0x0028440200000000ULL, 0x0050080402000000ULL, 0x0020100804020000ULL, 0x0040201008040200ULL,
}};

constexpr std::array<Bitboard, 64> BishopMagics{{
    0x0004100089140080ULL, 0x00230a0212020144ULL, 0x02b00c0042480008ULL, 0x8004042880020000ULL,
    0x00020210a0080c01ULL, 0x0006021005090480ULL, 0x80860806080c0000ULL, 0x1005840842101412ULL,
    0x0492905488184440ULL, 0x250020082100c091ULL, 0xa102080204003028ULL, 0x0e01021a06000002ULL,
    0x00a50310c0200000ULL, 0x108206904c200000ULL, 0x0d10040088080883ULL, 0x1001299444022000ULL,
    0x00080010200802d0ULL, 0x0104024805040406ULL, 0x0008000d00410204ULL, 0x00c1004804910120ULL,
    0x0084108202020800ULL, 0x0020400203100911ULL, 0x2041041084102208ULL, 0x0060800104014100ULL,
    0x8020040228101400ULL, 0x1001200108220401ULL, 0x408804060802c010ULL, 0x0020080001040418ULL,
    0x0001009005004000ULL, 0x0844430002008200ULL, 0x0007004404020880ULL, 0x0823050182128382ULL,
    0x200248c0060c5000ULL, 0x0244020304200410ULL, 0x0001220100080808ULL, 0x1400420280a80080ULL,
    0x0034002400820108ULL, 0x0002008100020440ULL, 0x0882040102040c80ULL, 0x40080840c5010108ULL,
    0x2084064240203000ULL, 0x20011c0920000440ULL, 0x0320292090000801ULL, 0x0808090451000800ULL,
    0x0020084100401400ULL, 0xc202129000804100ULL, 0x60822a0204000200ULL, 0x0015140080800200ULL,
    0x020100c220200b01ULL, 0x0000410c21200920ULL, 0x1200020042220200ULL, 0x0005000084042000ULL,
    0x08000010020e0880ULL, 0x3202202002029010ULL, 0x0820a00220831100ULL, 0x0848088805802420ULL,
    0x801a044402480200ULL, 0x8022009401080200ULL, 0x0008000084844100ULL, 0x0040020100840402ULL,
    0x160000c00810a402ULL, 0x1021004070220090ULL, 0x0008081004080050ULL, 0x80400a0086060040ULL,
}};

constexpr std::array<int, 64> BishopShifts{{
    58, 59, 59, 59, 59, 59, 59, 58, 59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59, 59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59, 59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59, 58, 59, 59, 59, 59, 59, 59, 58,
}};


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

    template <std::size_t N>
    Bitboard ray_attacks(Square square, Bitboard occupancy, const std::array<std::pair<int, int>, N>& directions) {
        Bitboard attacked_squares{};
        for (auto [d_file, d_rank] : directions) {
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

    template <std::size_t TableSize, std::size_t N>
    std::array<Bitboard, 64 * TableSize> make_slider_attack_table(
        const std::array<Bitboard, 64>& masks,
        const std::array<Bitboard, 64>& magics,
        const std::array<int, 64>& shifts,
        const std::array<std::pair<int, int>, N>& directions
    ) {
        std::array<Bitboard, 64 * TableSize> table{};

        for (Square square = 0; square < BoardSize; ++square) {
            const Bitboard mask = masks[square];
            Bitboard subset = 0;
            do {
                const std::size_t index = static_cast<std::size_t>((subset * magics[square]) >> shifts[square]);
                assert(index < TableSize);
                table[static_cast<std::size_t>(square) * TableSize + index] =
                    ray_attacks(square, subset, directions);
                subset = (subset - mask) & mask;
            } while (subset != 0);
        }

        return table;
    }

    const std::array<Bitboard, 64 * RookTableSize> RookAttackTable =
        make_slider_attack_table<RookTableSize>(RookMasks, RookMagics, RookShifts, RookDirection);

    const std::array<Bitboard, 64 * BishopTableSize> BishopAttackTable =
        make_slider_attack_table<BishopTableSize>(BishopMasks, BishopMagics, BishopShifts, BishopDirection);

} // namespace

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
        assert(is_valid_square(square));
        const std::size_t index =
            static_cast<std::size_t>(((occupancy & RookMasks[square]) * RookMagics[square]) >> RookShifts[square]);
        assert(index < RookTableSize);
        return RookAttackTable[static_cast<std::size_t>(square) * RookTableSize + index];
    }

    Bitboard bishop_attacks(Square square, Bitboard occupancy) {
        assert(is_valid_square(square));
        const std::size_t index =
            static_cast<std::size_t>(((occupancy & BishopMasks[square]) * BishopMagics[square]) >> BishopShifts[square]);
        assert(index < BishopTableSize);
        return BishopAttackTable[static_cast<std::size_t>(square) * BishopTableSize + index];
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
