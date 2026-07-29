#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "third_party/nnue_pytorch/binpack.h"

namespace {

constexpr std::array<char, 8> CompactMagic{'C', 'H', 'S', 'C', 'B', 'I', 'N', '2'};
constexpr std::uint16_t CompactVersion = 2;
constexpr std::uint16_t AuxFeatureCount = 13;
constexpr std::uint32_t TargetEncodingStockfishRaw = 1;
constexpr std::size_t CompactRecordSize = 40;
constexpr std::int16_t StockfishValueNone = 32002;

struct Args {
    std::string input = "-";
    std::string output = "-";
    std::uint64_t limit = 0;
    std::uint64_t progress_interval = 1'000'000;
    std::uint64_t dedup_expected_records = 0;
    std::uint64_t dedup_bits_per_record = 16;
    bool require_legal_move = false;
    bool require_limit_reached = false;
    bool deduplicate_positions = false;
    bool exclude_in_check = false;
};

[[nodiscard]] std::uint16_t read_be16(const unsigned char* data) {
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(data[0]) << 8) | data[1]);
}

[[nodiscard]] std::uint32_t read_le32(const unsigned char* data) {
    return static_cast<std::uint32_t>(data[0])
        | (static_cast<std::uint32_t>(data[1]) << 8)
        | (static_cast<std::uint32_t>(data[2]) << 16)
        | (static_cast<std::uint32_t>(data[3]) << 24);
}

void write_u16_le(std::ostream& out, std::uint16_t value) {
    const unsigned char bytes[2]{
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8) & 0xffu),
    };
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void write_i16_le(std::ostream& out, std::int16_t value) {
    write_u16_le(out, static_cast<std::uint16_t>(value));
}

void write_u32_le(std::ostream& out, std::uint32_t value) {
    const unsigned char bytes[4]{
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8) & 0xffu),
        static_cast<unsigned char>((value >> 16) & 0xffu),
        static_cast<unsigned char>((value >> 24) & 0xffu),
    };
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void write_compact_header(std::ostream& out) {
    out.write(CompactMagic.data(), static_cast<std::streamsize>(CompactMagic.size()));
    write_u16_le(out, CompactVersion);
    write_u16_le(out, AuxFeatureCount);
    write_u32_le(out, TargetEncodingStockfishRaw);
}

class StreamBinpackReader {
public:
    explicit StreamBinpackReader(std::istream& input)
        : input_(input) {
        is_end_ = !read_next_chunk();
    }

    [[nodiscard]] bool has_next() const {
        return !is_end_;
    }

    [[nodiscard]] binpack::TrainingDataEntry next() {
        if (is_end_) {
            throw std::runtime_error("attempted to read past end of binpack stream");
        }

        if (movelist_reader_.has_value()) {
            const auto entry = movelist_reader_->nextEntry();
            if (!movelist_reader_->hasNext()) {
                offset_ += movelist_reader_->numReadBytes();
                movelist_reader_.reset();
                fetch_next_chunk_if_needed();
            }
            return entry;
        }

        if (offset_ + sizeof(binpack::PackedTrainingDataEntry) + 2 > chunk_.size()) {
            throw std::runtime_error("truncated binpack chunk before packed entry");
        }

        binpack::PackedTrainingDataEntry packed{};
        std::memcpy(&packed, chunk_.data() + offset_, sizeof(packed));
        offset_ += sizeof(packed);

        const std::uint16_t num_plies = read_be16(chunk_.data() + offset_);
        offset_ += 2;

        const auto entry = binpack::unpackEntry(packed);
        if (num_plies > 0) {
            movelist_reader_.emplace(
                entry,
                reinterpret_cast<unsigned char*>(chunk_.data()) + offset_,
                num_plies);
        } else {
            fetch_next_chunk_if_needed();
        }
        return entry;
    }

private:
    std::istream& input_;
    std::vector<unsigned char> chunk_{};
    std::optional<binpack::PackedMoveScoreListReader> movelist_reader_{};
    std::size_t offset_ = 0;
    bool is_end_ = false;

    [[nodiscard]] bool read_exact(unsigned char* data, std::size_t size) {
        input_.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
        return input_.gcount() == static_cast<std::streamsize>(size);
    }

    [[nodiscard]] bool read_next_chunk() {
        unsigned char header[8]{};
        input_.read(reinterpret_cast<char*>(header), 1);
        if (input_.gcount() == 0) {
            return false;
        }
        if (input_.gcount() != 1 || !read_exact(header + 1, 7)) {
            throw std::runtime_error("truncated binpack chunk header");
        }
        if (header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P') {
            throw std::runtime_error("invalid binpack chunk magic");
        }

        const std::uint32_t size = read_le32(header + 4);
        if (size > binpack::maxChunkSize) {
            throw std::runtime_error("binpack chunk larger than supported maximum");
        }
        chunk_.assign(size, 0);
        if (size > 0 && !read_exact(chunk_.data(), size)) {
            throw std::runtime_error("truncated binpack chunk body");
        }
        offset_ = 0;
        return true;
    }

    void fetch_next_chunk_if_needed() {
        if (offset_ + sizeof(binpack::PackedTrainingDataEntry) + 2 <= chunk_.size()) {
            return;
        }
        is_end_ = !read_next_chunk();
    }
};

[[nodiscard]] std::vector<std::string_view> split_fields(std::string_view text) {
    std::vector<std::string_view> fields;
    std::size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && text[pos] == ' ') {
            ++pos;
        }
        const std::size_t begin = pos;
        while (pos < text.size() && text[pos] != ' ') {
            ++pos;
        }
        if (begin != pos) {
            fields.push_back(text.substr(begin, pos - begin));
        }
    }
    return fields;
}

[[nodiscard]] int piece_index_from_char(char piece) {
    switch (static_cast<char>(std::toupper(static_cast<unsigned char>(piece)))) {
    case 'P':
        return 0;
    case 'N':
        return 1;
    case 'B':
        return 2;
    case 'R':
        return 3;
    case 'Q':
        return 4;
    case 'K':
        return 5;
    default:
        throw std::runtime_error("bad FEN piece character");
    }
}

[[nodiscard]] int square_index(int file, int rank) {
    return rank * 8 + file;
}

[[nodiscard]] int relative_square(bool black_to_move, int square) {
    return black_to_move ? square ^ 56 : square;
}

void set_packed_square(std::array<unsigned char, 32>& board, int square, unsigned char code) {
    if (square < 0 || square >= 64) {
        throw std::runtime_error("compact square out of range");
    }
    const std::size_t byte_index = static_cast<std::size_t>(square / 2);
    const int shift = (square % 2) * 4;
    const unsigned char mask = static_cast<unsigned char>(0x0fU << shift);
    const unsigned char current = static_cast<unsigned char>((board[byte_index] >> shift) & 0x0fU);
    if (current != 0 && current != code) {
        throw std::runtime_error("two FEN pieces map to the same compact square");
    }
    board[byte_index] = static_cast<unsigned char>((board[byte_index] & ~mask) | (code << shift));
}

[[nodiscard]] int count_code(const std::array<unsigned char, 32>& board, unsigned char code) {
    int count = 0;
    for (unsigned char byte : board) {
        if ((byte & 0x0fU) == code) {
            ++count;
        }
        if (((byte >> 4) & 0x0fU) == code) {
            ++count;
        }
    }
    return count;
}

[[nodiscard]] std::uint16_t aux_from_fen_fields(
    std::string_view side_field,
    std::string_view castling_field,
    std::string_view ep_field) {
    if (side_field.size() != 1 || (side_field[0] != 'w' && side_field[0] != 'b')) {
        throw std::runtime_error("bad FEN side-to-move field");
    }
    const bool black_to_move = side_field[0] == 'b';
    std::uint16_t aux = 0;

    const auto has_castle = [&](char right) {
        return castling_field.find(right) != std::string_view::npos;
    };
    const bool white_ks = has_castle('K');
    const bool white_qs = has_castle('Q');
    const bool black_ks = has_castle('k');
    const bool black_qs = has_castle('q');

    if (!black_to_move) {
        aux |= static_cast<std::uint16_t>(white_ks) << 0;
        aux |= static_cast<std::uint16_t>(white_qs) << 1;
        aux |= static_cast<std::uint16_t>(black_ks) << 2;
        aux |= static_cast<std::uint16_t>(black_qs) << 3;
    } else {
        aux |= static_cast<std::uint16_t>(black_ks) << 0;
        aux |= static_cast<std::uint16_t>(black_qs) << 1;
        aux |= static_cast<std::uint16_t>(white_ks) << 2;
        aux |= static_cast<std::uint16_t>(white_qs) << 3;
    }

    if (ep_field != "-") {
        if (ep_field.size() != 2 || ep_field[0] < 'a' || ep_field[0] > 'h') {
            throw std::runtime_error("bad FEN en-passant field");
        }
        const int file = ep_field[0] - 'a';
        aux |= 1U << 4;
        aux |= static_cast<std::uint16_t>(1U << (5 + file));
    }
    return aux;
}

struct CompactRecord {
    std::array<unsigned char, 32> board{};
    std::uint16_t aux = 0;
    std::int16_t score = 0;
    std::uint16_t ply = 0;
    std::int16_t result = 0;
};

class BlockedBloomFilter {
public:
    BlockedBloomFilter(std::uint64_t expected_records, std::uint64_t bits_per_record) {
        if (expected_records == 0 || bits_per_record == 0) {
            throw std::runtime_error("dedup filter size must be non-zero");
        }
        if (expected_records > std::numeric_limits<std::uint64_t>::max() / bits_per_record) {
            throw std::runtime_error("dedup filter size overflow");
        }

        constexpr std::uint64_t bits_per_block = 512;
        const std::uint64_t requested_bits = expected_records * bits_per_record;
        const std::uint64_t requested_blocks =
            (requested_bits + bits_per_block - 1) / bits_per_block;
        if (requested_blocks > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("dedup filter exceeds addressable memory");
        }
        const std::size_t requested = static_cast<std::size_t>(requested_blocks);
        const std::size_t block_count = std::max<std::size_t>(requested, 1);
        blocks_.resize(block_count);
    }

    [[nodiscard]] bool test_and_set(const CompactRecord& record) {
        std::uint64_t hash1 = 0xcbf29ce484222325ULL;
        std::uint64_t hash2 = 0x9e3779b97f4a7c15ULL;
        for (unsigned char value : record.board) {
            hash1 = (hash1 ^ value) * 0x100000001b3ULL;
            hash2 = (hash2 ^ (static_cast<std::uint64_t>(value) + 0x9e3779b9ULL))
                * 0xbf58476d1ce4e5b9ULL;
        }
        const unsigned char aux_low = static_cast<unsigned char>(record.aux & 0xffU);
        const unsigned char aux_high = static_cast<unsigned char>((record.aux >> 8) & 0xffU);
        hash1 = (hash1 ^ aux_low) * 0x100000001b3ULL;
        hash1 = (hash1 ^ aux_high) * 0x100000001b3ULL;
        hash2 = (hash2 ^ (static_cast<std::uint64_t>(aux_low) + 0x9e3779b9ULL))
            * 0xbf58476d1ce4e5b9ULL;
        hash2 = (hash2 ^ (static_cast<std::uint64_t>(aux_high) + 0x9e3779b9ULL))
            * 0xbf58476d1ce4e5b9ULL;

        hash1 = mix(hash1);
        hash2 = mix(hash2);
        Block& block = blocks_[static_cast<std::size_t>(hash1 % blocks_.size())];

        bool present = true;
        for (std::size_t word = 0; word < block.size(); ++word) {
            const unsigned shift = static_cast<unsigned>(word * 8);
            const unsigned bit_index = static_cast<unsigned>((hash2 >> shift) & 63U);
            const std::uint64_t bit_mask = std::uint64_t{1} << bit_index;
            present = present && ((block[word] & bit_mask) != 0);
            block[word] |= bit_mask;
        }
        return present;
    }

    [[nodiscard]] std::uint64_t bytes() const {
        return static_cast<std::uint64_t>(blocks_.size()) * sizeof(Block);
    }

private:
    using Block = std::array<std::uint64_t, 8>;

    [[nodiscard]] static std::uint64_t mix(std::uint64_t value) {
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    std::vector<Block> blocks_{};
};

[[nodiscard]] CompactRecord record_from_entry(const binpack::TrainingDataEntry& entry) {
    const std::string fen_storage = entry.pos.fen();
    const std::string_view fen = fen_storage;
    const std::vector<std::string_view> fields = split_fields(fen);
    if (fields.size() < 4) {
        throw std::runtime_error("FEN has fewer than four fields");
    }

    CompactRecord record{};
    const bool black_to_move = fields[1] == "b";
    if (fields[1] != "w" && fields[1] != "b") {
        throw std::runtime_error("bad FEN side-to-move field");
    }

    int rank = 7;
    int file = 0;
    for (char ch : fields[0]) {
        if (ch == '/') {
            if (file != 8) {
                throw std::runtime_error("bad FEN rank width");
            }
            --rank;
            file = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(ch))) {
            file += ch - '0';
            if (file > 8) {
                throw std::runtime_error("bad FEN empty-square run");
            }
            continue;
        }

        if (rank < 0 || file >= 8) {
            throw std::runtime_error("too many FEN pieces");
        }
        const bool piece_is_white = std::isupper(static_cast<unsigned char>(ch)) != 0;
        const bool piece_is_friendly = black_to_move ? !piece_is_white : piece_is_white;
        const int piece_index = piece_index_from_char(ch);
        const int absolute_square = square_index(file, rank);
        const int compact_square = relative_square(black_to_move, absolute_square);
        const unsigned char code = static_cast<unsigned char>(
            piece_is_friendly ? piece_index + 1 : piece_index + 7);
        set_packed_square(record.board, compact_square, code);
        ++file;
    }
    if (rank != 0 || file != 8) {
        throw std::runtime_error("bad FEN board field");
    }
    if (count_code(record.board, 6) != 1 || count_code(record.board, 12) != 1) {
        throw std::runtime_error("compact board is missing exactly one friendly or enemy king");
    }

    record.aux = aux_from_fen_fields(fields[1], fields[2], fields[3]);
    if (entry.ply > 0x3FFFu) {
        throw std::runtime_error("binpack ply exceeds 14-bit range");
    }
    if (entry.result < -1 || entry.result > 1) {
        throw std::runtime_error("binpack result is not -1, 0, or 1");
    }
    record.score = entry.score;
    record.ply = entry.ply;
    record.result = entry.result;
    return record;
}

void write_record(std::ostream& out, const CompactRecord& record) {
    out.write(
        reinterpret_cast<const char*>(record.board.data()),
        static_cast<std::streamsize>(record.board.size()));
    write_u16_le(out, record.aux);
    write_i16_le(out, record.score);
    write_u16_le(out, record.ply);
    write_i16_le(out, record.result);
}

[[nodiscard]] std::uint64_t parse_u64(std::string_view text) {
    std::uint64_t value = 0;
    if (text.empty()) {
        throw std::runtime_error("empty integer argument");
    }
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            throw std::runtime_error("bad integer argument");
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            throw std::runtime_error("integer argument overflow");
        }
        value = value * 10 + digit;
    }
    return value;
}

[[nodiscard]] Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next_value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + std::string(name));
            }
            ++i;
            return argv[i];
        };

        if (arg == "--input") {
            args.input = std::string(next_value(arg));
        } else if (arg == "--output") {
            args.output = std::string(next_value(arg));
        } else if (arg == "--limit") {
            args.limit = parse_u64(next_value(arg));
        } else if (arg == "--progress-interval") {
            args.progress_interval = parse_u64(next_value(arg));
        } else if (arg == "--deduplicate-positions") {
            args.deduplicate_positions = true;
        } else if (arg == "--dedup-expected-records") {
            args.dedup_expected_records = parse_u64(next_value(arg));
        } else if (arg == "--dedup-bits-per-record") {
            args.dedup_bits_per_record = parse_u64(next_value(arg));
        } else if (arg == "--require-legal-move") {
            args.require_legal_move = true;
        } else if (arg == "--exclude-in-check") {
            args.exclude_in_check = true;
        } else if (arg == "--require-limit-reached") {
            args.require_limit_reached = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: robotmoon_binpack_to_cbin [--input PATH|-] [--output PATH|-]\n"
                << "                                  [--limit N] [--progress-interval N]\n"
                << "                                  [--deduplicate-positions]\n"
                << "                                  [--dedup-expected-records N]\n"
                << "                                  [--dedup-bits-per-record N]\n"
                << "                                  [--require-limit-reached]\n"
                << "                                  [--require-legal-move]\n"
                << "                                  [--exclude-in-check]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::uint64_t dedup_expected_records = args.dedup_expected_records != 0
            ? args.dedup_expected_records
            : args.limit;
        if (args.deduplicate_positions && dedup_expected_records == 0) {
            throw std::runtime_error(
                "deduplication requires --limit or --dedup-expected-records");
        }
        if (args.require_limit_reached && args.limit == 0) {
            throw std::runtime_error("--require-limit-reached requires --limit");
        }

        std::ifstream input_file;
        std::ofstream output_file;
        std::istream* input = &std::cin;
        std::ostream* output = &std::cout;

        if (args.input != "-") {
            input_file.open(args.input, std::ios::binary);
            if (!input_file) {
                throw std::runtime_error("failed to open input: " + args.input);
            }
            input = &input_file;
        }
        if (args.output != "-") {
            output_file.open(args.output, std::ios::binary);
            if (!output_file) {
                throw std::runtime_error("failed to open output: " + args.output);
            }
            output = &output_file;
        }

        StreamBinpackReader reader(*input);
        write_compact_header(*output);
        std::optional<BlockedBloomFilter> dedup_filter;
        if (args.deduplicate_positions) {
            dedup_filter.emplace(dedup_expected_records, args.dedup_bits_per_record);
            std::cerr << "dedup_filter_bytes=" << dedup_filter->bytes() << '\n';
        }

        std::uint64_t read = 0;
        std::uint64_t written = 0;
        std::uint64_t value_none_filtered = 0;
        std::uint64_t in_check_filtered = 0;
        std::uint64_t duplicate_or_bloom_filtered = 0;
        while (reader.has_next()) {
            const binpack::TrainingDataEntry entry = reader.next();
            ++read;
            if (entry.score == StockfishValueNone) {
                ++value_none_filtered;
                continue;
            }
            if (args.require_legal_move && !entry.isValid()) {
                throw std::runtime_error("illegal move in binpack entry " + std::to_string(read - 1));
            }
            if (args.exclude_in_check && entry.isInCheck()) {
                ++in_check_filtered;
                continue;
            }
            const CompactRecord record = record_from_entry(entry);
            if (dedup_filter.has_value() && dedup_filter->test_and_set(record)) {
                ++duplicate_or_bloom_filtered;
                continue;
            }
            write_record(*output, record);
            ++written;

            if (args.limit != 0 && written >= args.limit) {
                break;
            }
            if (args.progress_interval != 0 && written % args.progress_interval == 0) {
                std::cerr << "converted=" << written << '\n';
            }
        }

        if (args.require_limit_reached && written != args.limit) {
            throw std::runtime_error(
                "input ended after " + std::to_string(written)
                + " output records; required " + std::to_string(args.limit));
        }

        output->flush();
        if (!*output) {
            throw std::runtime_error("failed while writing compact output");
        }
        std::cerr << "done converted=" << written
                  << " read=" << read
                  << " value_none_filtered=" << value_none_filtered
                  << " in_check_filtered=" << in_check_filtered
                  << " duplicate_or_bloom_filtered=" << duplicate_or_bloom_filtered
                  << " bytes=" << (16 + written * CompactRecordSize)
                  << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
