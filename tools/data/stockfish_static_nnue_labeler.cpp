// Built against a pinned official Stockfish checkout by
// tools/data/build_stockfish_static_nnue_labeler.sh.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "attacks.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"

namespace {

using Stockfish::Eval::NNUE::AccumulatorCaches;
using Stockfish::Eval::NNUE::AccumulatorStack;
using Stockfish::Eval::NNUE::EvalFile;
using Stockfish::Eval::NNUE::Network;

constexpr std::array<char, 8> Magic{'C', 'H', 'S', 'C', 'B', 'I', 'N', '2'};
constexpr std::size_t HeaderSize = 16;
constexpr std::size_t RecordSize = 40;

enum class LabelComponent {
    Total,
    Psqt,
    Positional,
};

struct Args {
    std::string input = "-";
    std::string output = "-";
    std::uint64_t expected_records = 0;
    std::uint64_t progress_interval = 1'000'000;
    bool reject_in_check = true;
    LabelComponent label_component = LabelComponent::Total;
};

[[nodiscard]] LabelComponent parse_label_component(std::string_view text) {
    if (text == "total") return LabelComponent::Total;
    if (text == "psqt") return LabelComponent::Psqt;
    if (text == "positional") return LabelComponent::Positional;
    std::cerr << "bad --label-component value: " << text
              << " (expected total, psqt, or positional)\n";
    std::exit(2);
}

[[nodiscard]] std::string_view label_name(LabelComponent component) {
    switch (component) {
        case LabelComponent::Total: return "raw_psqt_plus_positional_side_to_move";
        case LabelComponent::Psqt: return "raw_psqt_side_to_move";
        case LabelComponent::Positional: return "raw_positional_side_to_move";
    }
    std::abort();
}

[[nodiscard]] std::uint64_t parse_u64(std::string_view text) {
    std::uint64_t value = 0;
    if (text.empty()) {
        std::cerr << "empty integer argument\n";
        std::exit(2);
    }
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            std::cerr << "bad integer argument: " << text << '\n';
            std::exit(2);
        }
        value = value * 10 + static_cast<unsigned>(ch - '0');
    }
    return value;
}

[[nodiscard]] Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto next = [&]() -> std::string_view {
            if (++i >= argc) {
                std::cerr << "missing value after " << arg << '\n';
                std::exit(2);
            }
            return argv[i];
        };
        if (arg == "--input") {
            args.input = std::string(next());
        } else if (arg == "--output") {
            args.output = std::string(next());
        } else if (arg == "--expected-records") {
            args.expected_records = parse_u64(next());
        } else if (arg == "--progress-interval") {
            args.progress_interval = parse_u64(next());
        } else if (arg == "--label-component") {
            args.label_component = parse_label_component(next());
        } else if (arg == "--allow-in-check") {
            args.reject_in_check = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: stockfish_static_nnue_labeler [--input PATH|-] [--output PATH|-]\n"
                << "       [--expected-records N] [--progress-interval N] [--allow-in-check]\n"
                << "       [--label-component total|psqt|positional]\n";
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            std::exit(2);
        }
    }
    return args;
}

[[nodiscard]] unsigned char piece_code(const unsigned char* board, int square) {
    const unsigned char byte = board[square / 2];
    return static_cast<unsigned char>((byte >> ((square % 2) * 4)) & 0x0fU);
}

[[nodiscard]] char piece_char(unsigned char code) {
    static constexpr std::array<char, 13> chars{
        '.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'};
    return code < chars.size() ? chars[code] : '?';
}

[[nodiscard]] std::string fen_from_record(const unsigned char* record) {
    std::uint16_t aux = 0;
    std::memcpy(&aux, record + 32, sizeof(aux));
    if ((aux >> 13) != 0) {
        return {};
    }

    std::ostringstream fen;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const char piece = piece_char(piece_code(record, rank * 8 + file));
            if (piece == '?') {
                return {};
            }
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
    if ((aux & (1U << 0)) != 0) castling += 'K';
    if ((aux & (1U << 1)) != 0) castling += 'Q';
    if ((aux & (1U << 2)) != 0) castling += 'k';
    if ((aux & (1U << 3)) != 0) castling += 'q';
    if (castling.empty()) castling = "-";

    std::string ep = "-";
    if ((aux & (1U << 4)) != 0) {
        for (int file = 0; file < 8; ++file) {
            if ((aux & (1U << (5 + file))) != 0) {
                ep = std::string{static_cast<char>('a' + file), '6'};
                break;
            }
        }
        if (ep == "-") {
            return {};
        }
    }

    // CBin stores every position normalized to the side-to-move perspective:
    // friendly pieces are white and the normalized side to move is white.
    fen << " w " << castling << ' ' << ep << " 0 1";
    return fen.str();
}

void write_i16_le(unsigned char* destination, std::int16_t value) {
    const std::uint16_t bits = static_cast<std::uint16_t>(value);
    destination[0] = static_cast<unsigned char>(bits & 0xffU);
    destination[1] = static_cast<unsigned char>((bits >> 8) & 0xffU);
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    std::ifstream input_file;
    std::ofstream output_file;
    std::istream* input = &std::cin;
    std::ostream* output = &std::cout;
    if (args.input != "-") {
        input_file.open(args.input, std::ios::binary);
        if (!input_file) {
            std::cerr << "failed to open input: " << args.input << '\n';
            return 1;
        }
        input = &input_file;
    }
    if (args.output != "-") {
        output_file.open(args.output, std::ios::binary);
        if (!output_file) {
            std::cerr << "failed to open output: " << args.output << '\n';
            return 1;
        }
        output = &output_file;
    }

    Stockfish::Attacks::init();
    Stockfish::Position::init();

    EvalFile eval_file;
    auto network = std::make_unique<Network>();
    network->load(std::filesystem::path{}, std::filesystem::path{}, eval_file);
    if (!eval_file.current.has_value() || network->get_content_hash() == 0) {
        std::cerr << "failed to load embedded Stockfish NNUE network\n";
        return 1;
    }
    auto caches = std::make_unique<AccumulatorCaches>(*network);
    auto stack = std::make_unique<AccumulatorStack>();

    std::array<unsigned char, HeaderSize> header{};
    input->read(reinterpret_cast<char*>(header.data()), HeaderSize);
    if (input->gcount() != static_cast<std::streamsize>(HeaderSize)
        || !std::equal(Magic.begin(), Magic.end(), header.begin())) {
        std::cerr << "input is not CHSCBIN2\n";
        return 1;
    }
    output->write(reinterpret_cast<const char*>(header.data()), HeaderSize);

    std::uint64_t records = 0;
    std::uint64_t rejected_in_check = 0;
    std::array<unsigned char, RecordSize> record{};
    while (true) {
        input->read(reinterpret_cast<char*>(record.data()), RecordSize);
        if (input->gcount() == 0) {
            break;
        }
        if (input->gcount() != static_cast<std::streamsize>(RecordSize)) {
            std::cerr << "truncated CHSCBIN2 record at index " << records << '\n';
            return 1;
        }

        const std::string fen = fen_from_record(record.data());
        if (fen.empty()) {
            std::cerr << "invalid packed board/aux at record " << records << '\n';
            return 1;
        }
        Stockfish::StateInfo state{};
        Stockfish::Position position;
        if (const auto error = position.set(fen, false, &state); error.has_value()) {
            std::cerr << "Stockfish rejected record " << records << ": " << error->what() << '\n';
            return 1;
        }
        if (position.checkers() && args.reject_in_check) {
            ++rejected_in_check;
            std::cerr << "in-check record reached labeler at index " << records << '\n';
            return 1;
        }

        stack->reset();
        const auto [psqt, positional] = network->evaluate(position, *stack, *caches);
        int raw_score = 0;
        switch (args.label_component) {
            case LabelComponent::Total:
                raw_score = static_cast<int>(psqt) + static_cast<int>(positional);
                break;
            case LabelComponent::Psqt: raw_score = static_cast<int>(psqt); break;
            case LabelComponent::Positional:
                raw_score = static_cast<int>(positional);
                break;
        }
        if (raw_score < std::numeric_limits<std::int16_t>::min()
            || raw_score > std::numeric_limits<std::int16_t>::max()) {
            std::cerr << label_name(args.label_component) << " score is outside int16 at record "
                      << records << ": " << raw_score << '\n';
            return 1;
        }
        write_i16_le(record.data() + 34, static_cast<std::int16_t>(raw_score));
        output->write(reinterpret_cast<const char*>(record.data()), RecordSize);
        if (!*output) {
            std::cerr << "failed writing output at record " << records << '\n';
            return 1;
        }

        ++records;
        if (args.progress_interval != 0 && records % args.progress_interval == 0) {
            std::cerr << "labeled=" << records << '\n';
        }
    }

    if (args.expected_records != 0 && records != args.expected_records) {
        std::cerr << "expected " << args.expected_records << " records, got " << records << '\n';
        return 1;
    }
    output->flush();
    if (!*output) {
        std::cerr << "failed flushing output\n";
        return 1;
    }
    std::cerr << "done labeled=" << records
              << " rejected_in_check=" << rejected_in_check
              << " network=" << eval_file.current->string()
              << " network_content_hash=" << network->get_content_hash()
              << " label=" << label_name(args.label_component) << '\n';
    return 0;
}
