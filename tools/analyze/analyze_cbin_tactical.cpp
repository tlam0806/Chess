#include "attacks.hpp"
#include "move.hpp"
#include "position.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace {

constexpr std::array<char, 8> Magic{'C', 'H', 'S', 'C', 'B', 'I', 'N', '1'};
constexpr std::size_t HeaderSize = 16;
constexpr std::size_t RecordSize = 36;

struct Stats {
    std::uint64_t total = 0;
    std::uint64_t in_check = 0;
    std::uint64_t no_legal_moves = 0;
    std::uint64_t has_capture = 0;
    std::uint64_t has_promotion = 0;
    std::uint64_t has_checking_move = 0;
    std::uint64_t legal_moves = 0;
    std::uint64_t legal_captures = 0;
    std::uint64_t legal_promotions = 0;
    std::uint64_t legal_checking_moves = 0;
    std::uint64_t abs_gt_1000 = 0;
    std::uint64_t abs_gt_2000 = 0;
    std::uint64_t abs_gt_10000 = 0;
    std::array<std::uint64_t, 13> target_buckets{};
};

char piece_char(unsigned char code) {
    static constexpr std::array<char, 13> chars{
        '.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'};
    return code < chars.size() ? chars[code] : '?';
}

unsigned char code_at(const unsigned char* board, int square) {
    const unsigned char byte = board[square / 2];
    return static_cast<unsigned char>((byte >> ((square % 2) * 4)) & 0x0fU);
}

std::string ep_square_from_aux(std::uint16_t aux) {
    if ((aux & (1U << 4)) == 0) {
        return "-";
    }
    for (int file = 0; file < 8; ++file) {
        if ((aux & (1U << (5 + file))) != 0) {
            std::string ep;
            ep.push_back(static_cast<char>('a' + file));
            ep.push_back('6');
            return ep;
        }
    }
    return "-";
}

std::string fen_from_record(const unsigned char* record) {
    const unsigned char* board = record;
    std::uint16_t aux = 0;
    std::memcpy(&aux, record + 32, sizeof(aux));

    std::ostringstream fen;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const int square = rank * 8 + file;
            const char piece = piece_char(code_at(board, square));
            if (piece == '.') {
                ++empty;
                continue;
            }
            if (empty != 0) {
                fen << empty;
                empty = 0;
            }
            fen << piece;
        }
        if (empty != 0) {
            fen << empty;
        }
        if (rank != 0) {
            fen << '/';
        }
    }

    std::string castling;
    if ((aux & (1U << 0)) != 0) {
        castling.push_back('K');
    }
    if ((aux & (1U << 1)) != 0) {
        castling.push_back('Q');
    }
    if ((aux & (1U << 2)) != 0) {
        castling.push_back('k');
    }
    if ((aux & (1U << 3)) != 0) {
        castling.push_back('q');
    }
    if (castling.empty()) {
        castling = "-";
    }

    fen << " w " << castling << ' ' << ep_square_from_aux(aux) << " 0 1";
    return fen.str();
}

std::int16_t target_from_record(const unsigned char* record) {
    std::int16_t target = 0;
    std::memcpy(&target, record + 34, sizeof(target));
    return target;
}

std::size_t target_bucket(int abs_target) {
    if (abs_target == 0) return 0;
    if (abs_target <= 50) return 1;
    if (abs_target <= 100) return 2;
    if (abs_target <= 200) return 3;
    if (abs_target <= 300) return 4;
    if (abs_target <= 500) return 5;
    if (abs_target <= 700) return 6;
    if (abs_target <= 1000) return 7;
    if (abs_target <= 1500) return 8;
    if (abs_target <= 2000) return 9;
    if (abs_target <= 5000) return 10;
    if (abs_target <= 10000) return 11;
    return 12;
}

double pct(std::uint64_t value, std::uint64_t total) {
    return total == 0 ? 0.0 : 100.0 * static_cast<double>(value) / static_cast<double>(total);
}

void print_count(const char* label, std::uint64_t value, std::uint64_t total) {
    std::cout << std::left << std::setw(28) << label
              << std::right << std::setw(12) << value
              << "  " << std::fixed << std::setprecision(4) << pct(value, total) << "%\n";
}

} // namespace

int main(int argc, char** argv) {
    std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    bool emit_tsv = false;
    if (argc >= 2) {
        if (std::string_view(argv[1]) == "--tsv") {
            emit_tsv = true;
        } else {
            limit = std::stoull(argv[1]);
        }
    }
    if (argc >= 3) {
        if (std::string_view(argv[2]) == "--tsv") {
            emit_tsv = true;
        } else {
            limit = std::stoull(argv[2]);
        }
    }

    std::array<unsigned char, HeaderSize> header{};
    std::cin.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (std::cin.gcount() != static_cast<std::streamsize>(header.size())
        || !std::equal(Magic.begin(), Magic.end(), header.begin())) {
        std::cerr << "bad cbin header\n";
        return 1;
    }

    if (emit_tsv) {
        std::cout
            << "idx\ttarget\tin_check\tlegal_moves\tlegal_captures\tlegal_promotions"
            << "\tlegal_checking_moves\thas_capture\thas_promotion\thas_checking_move\n";
    }

    Stats stats;
    std::array<unsigned char, RecordSize> record{};
    while (stats.total < limit) {
        std::cin.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(record.size()));
        if (std::cin.gcount() == 0) {
            break;
        }
        if (std::cin.gcount() != static_cast<std::streamsize>(record.size())) {
            std::cerr << "truncated record\n";
            return 1;
        }

        chess::Position pos;
        const std::string fen = fen_from_record(record.data());
        if (!pos.set_fen(fen)) {
            std::cerr << "failed to parse reconstructed FEN: " << fen << '\n';
            return 1;
        }

        const std::uint64_t sample_index = stats.total;
        ++stats.total;
        const int target = target_from_record(record.data());
        const int abs_target = target < 0 ? -target : target;
        stats.abs_gt_1000 += abs_target > 1000;
        stats.abs_gt_2000 += abs_target > 2000;
        stats.abs_gt_10000 += abs_target > 10000;
        ++stats.target_buckets[target_bucket(abs_target)];

        const bool checked = chess::in_check(pos, pos.side_to_move);
        stats.in_check += checked;

        chess::MoveList moves;
        chess::generate_legal_moves(pos, moves);
        stats.legal_moves += moves.size();
        stats.no_legal_moves += moves.empty();

        bool any_capture = false;
        bool any_promotion = false;
        bool any_checking = false;
        std::uint64_t sample_captures = 0;
        std::uint64_t sample_promotions = 0;
        std::uint64_t sample_checking_moves = 0;
        for (chess::Move move : moves) {
            const bool capture = chess::is_capture(move);
            const bool promotion = chess::promotion_piece(move) != chess::PieceType::None;
            any_capture = any_capture || capture;
            any_promotion = any_promotion || promotion;
            stats.legal_captures += capture;
            stats.legal_promotions += promotion;
            sample_captures += capture;
            sample_promotions += promotion;

            chess::Position next = pos;
            next.make_move(move);
            const bool checking = chess::in_check(next, next.side_to_move);
            any_checking = any_checking || checking;
            stats.legal_checking_moves += checking;
            sample_checking_moves += checking;
        }
        stats.has_capture += any_capture;
        stats.has_promotion += any_promotion;
        stats.has_checking_move += any_checking;

        if (emit_tsv) {
            std::cout << sample_index
                      << '\t' << target
                      << '\t' << checked
                      << '\t' << moves.size()
                      << '\t' << sample_captures
                      << '\t' << sample_promotions
                      << '\t' << sample_checking_moves
                      << '\t' << any_capture
                      << '\t' << any_promotion
                      << '\t' << any_checking
                      << '\n';
        }
    }

    if (emit_tsv) {
        return 0;
    }

    std::cout << "samples=" << stats.total << '\n';
    print_count("side_in_check", stats.in_check, stats.total);
    print_count("no_legal_moves", stats.no_legal_moves, stats.total);
    print_count("has_legal_capture", stats.has_capture, stats.total);
    print_count("has_legal_promotion", stats.has_promotion, stats.total);
    print_count("has_legal_checking_move", stats.has_checking_move, stats.total);
    print_count("abs_target_gt_1000", stats.abs_gt_1000, stats.total);
    print_count("abs_target_gt_2000", stats.abs_gt_2000, stats.total);
    print_count("abs_target_gt_10000", stats.abs_gt_10000, stats.total);

    const double total = static_cast<double>(stats.total);
    std::cout << "avg_legal_moves=" << static_cast<double>(stats.legal_moves) / total << '\n';
    std::cout << "avg_legal_captures=" << static_cast<double>(stats.legal_captures) / total << '\n';
    std::cout << "avg_legal_promotions=" << static_cast<double>(stats.legal_promotions) / total << '\n';
    std::cout << "avg_legal_checking_moves=" << static_cast<double>(stats.legal_checking_moves) / total << '\n';

    static constexpr std::array<const char*, 13> bucket_names{
        "0",
        "1..50",
        "51..100",
        "101..200",
        "201..300",
        "301..500",
        "501..700",
        "701..1000",
        "1001..1500",
        "1501..2000",
        "2001..5000",
        "5001..10000",
        ">10000",
    };
    std::cout << "target_abs_buckets:\n";
    for (std::size_t i = 0; i < bucket_names.size(); ++i) {
        print_count(bucket_names[i], stats.target_buckets[i], stats.total);
    }

    return 0;
}
