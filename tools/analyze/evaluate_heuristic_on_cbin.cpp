#include "evaluate.hpp"
#include "position.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
constexpr std::array<char, 8> Magic{'C','H','S','C','B','I','N','2'};
constexpr std::size_t HeaderSize = 16;
constexpr std::size_t RecordSize = 40;
constexpr std::array<double, 7> Edges{0, 100, 300, 600, 1000, 1600, 2000};

struct Bin { std::uint64_t count = 0; double error = 0; };

unsigned char code_at(const unsigned char* board, int square) {
    const unsigned char byte = board[square / 2];
    return static_cast<unsigned char>((byte >> ((square % 2) * 4)) & 0x0fU);
}

char piece_char(unsigned char code) {
    static constexpr std::array<char, 13> chars{
        '.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'};
    return code < chars.size() ? chars[code] : '?';
}

std::string ep_square(std::uint16_t aux) {
    if ((aux & (1U << 4)) == 0) return "-";
    for (int file = 0; file < 8; ++file) {
        if ((aux & (1U << (5 + file))) != 0) {
            return std::string{static_cast<char>('a' + file), '6'};
        }
    }
    return "-";
}

std::string fen_from_record(const unsigned char* record) {
    std::uint16_t aux = 0;
    std::memcpy(&aux, record + 32, sizeof(aux));
    std::ostringstream fen;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const char piece = piece_char(code_at(record, rank * 8 + file));
            if (piece == '.') { ++empty; continue; }
            if (empty) { fen << empty; empty = 0; }
            fen << piece;
        }
        if (empty) fen << empty;
        if (rank) fen << '/';
    }
    std::string castling;
    if (aux & (1U << 0)) castling += 'K';
    if (aux & (1U << 1)) castling += 'Q';
    if (aux & (1U << 2)) castling += 'k';
    if (aux & (1U << 3)) castling += 'q';
    if (castling.empty()) castling = "-";
    fen << " w " << castling << ' ' << ep_square(aux) << " 0 1";
    return fen.str();
}

double target_cp(const unsigned char* record) {
    std::int16_t raw = 0;
    std::memcpy(&raw, record + 34, sizeof(raw));
    return std::clamp(static_cast<double>(raw) * 100.0 / 208.0, -2000.0, 2000.0);
}
}

int main() {
    std::array<unsigned char, HeaderSize> header{};
    std::cin.read(reinterpret_cast<char*>(header.data()), HeaderSize);
    if (std::cin.gcount() != static_cast<std::streamsize>(HeaderSize)
        || !std::equal(Magic.begin(), Magic.end(), header.begin())) {
        std::cerr << "bad CHSCBIN2 header\n";
        return 1;
    }
    std::array<Bin, 6> bins{};
    Bin overall{};
    std::array<unsigned char, RecordSize> record{};
    while (true) {
        std::cin.read(reinterpret_cast<char*>(record.data()), RecordSize);
        if (std::cin.gcount() == 0) break;
        if (std::cin.gcount() != static_cast<std::streamsize>(RecordSize)) {
            std::cerr << "truncated record\n";
            return 1;
        }
        chess::Position pos;
        const std::string fen = fen_from_record(record.data());
        if (!pos.set_fen(fen)) {
            std::cerr << "failed FEN: " << fen << '\n';
            return 1;
        }
        const double target = target_cp(record.data());
        const double prediction = chess::evaluate_for_side_to_move(pos);
        const double error = std::abs(prediction - target);
        ++overall.count;
        overall.error += error;
        const double magnitude = std::abs(target);
        for (std::size_t i = 0; i < bins.size(); ++i) {
            const bool upper = i + 1 == bins.size() ? magnitude <= Edges[i + 1]
                                                    : magnitude < Edges[i + 1];
            if (magnitude >= Edges[i] && upper) {
                ++bins[i].count;
                bins[i].error += error;
                break;
            }
        }
    }
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "range,samples,cp_mae\n";
    for (std::size_t i = 0; i < bins.size(); ++i) {
        std::cout << static_cast<int>(Edges[i]) << '-' << static_cast<int>(Edges[i + 1])
                  << ',' << bins[i].count << ',' << bins[i].error / bins[i].count << '\n';
    }
    std::cout << "overall," << overall.count << ',' << overall.error / overall.count << '\n';
}
