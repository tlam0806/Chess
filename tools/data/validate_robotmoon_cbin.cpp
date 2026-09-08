#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<char, 8> CompactMagic{'C', 'H', 'S', 'C', 'B', 'I', 'N', '2'};
constexpr std::size_t HeaderSize = 16;
constexpr std::size_t RecordSize = 40;
constexpr std::size_t PositionKeySize = 34;
constexpr std::size_t BucketCount = 128;
constexpr std::int16_t StockfishValueNone = 32002;

struct Args {
    std::string input = "-";
    std::filesystem::path spool_directory;
    std::uint64_t expected_records = 0;
    std::uint64_t progress_interval = 10'000'000;
};

struct Fingerprint {
    std::uint64_t first = 0;
    std::uint64_t second = 0;

    auto operator<=>(const Fingerprint&) const = default;
};

static_assert(sizeof(Fingerprint) == 16);

[[nodiscard]] std::uint16_t read_u16_le(const unsigned char* data) {
    return static_cast<std::uint16_t>(data[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8);
}

[[nodiscard]] std::uint32_t read_u32_le(const unsigned char* data) {
    return static_cast<std::uint32_t>(data[0])
        | (static_cast<std::uint32_t>(data[1]) << 8)
        | (static_cast<std::uint32_t>(data[2]) << 16)
        | (static_cast<std::uint32_t>(data[3]) << 24);
}

[[nodiscard]] std::uint64_t mix(std::uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    return value ^ (value >> 33);
}

[[nodiscard]] Fingerprint fingerprint_position(const unsigned char* record) {
    std::uint64_t first = 0x84222325cbf29ce4ULL;
    std::uint64_t second = 0x243f6a8885a308d3ULL;
    for (std::size_t index = 0; index < PositionKeySize; ++index) {
        const std::uint64_t value = record[index];
        first = (first ^ value) * 0x100000001b3ULL;
        second ^= value + 0x9e3779b97f4a7c15ULL + (second << 6) + (second >> 2);
        second *= 0xd6e8feb86659fd93ULL;
    }
    return Fingerprint{mix(first), mix(second)};
}

[[nodiscard]] std::uint64_t parse_u64(std::string_view text) {
    if (text.empty()) {
        throw std::runtime_error("empty integer argument");
    }
    std::uint64_t value = 0;
    for (char character : text) {
        if (character < '0' || character > '9') {
            throw std::runtime_error("bad integer argument");
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            throw std::runtime_error("integer argument overflow");
        }
        value = value * 10 + digit;
    }
    return value;
}

[[nodiscard]] Args parse_args(int argc, char** argv) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next_value = [&](std::string_view name) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            ++index;
            return argv[index];
        };

        if (argument == "--input") {
            args.input = std::string(next_value(argument));
        } else if (argument == "--spool-directory") {
            args.spool_directory = next_value(argument);
        } else if (argument == "--expected-records") {
            args.expected_records = parse_u64(next_value(argument));
        } else if (argument == "--progress-interval") {
            args.progress_interval = parse_u64(next_value(argument));
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: validate_robotmoon_cbin [--input PATH|-] --spool-directory PATH\n"
                << "                                [--expected-records N]\n"
                << "                                [--progress-interval N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }
    if (args.spool_directory.empty()) {
        throw std::runtime_error("--spool-directory is required");
    }
    return args;
}

void validate_header(const std::array<unsigned char, HeaderSize>& header) {
    if (!std::equal(CompactMagic.begin(), CompactMagic.end(), header.begin())) {
        throw std::runtime_error("invalid CHSCBIN2 magic");
    }
    if (read_u16_le(header.data() + 8) != 2) {
        throw std::runtime_error("unsupported compact version");
    }
    if (read_u16_le(header.data() + 10) != 13) {
        throw std::runtime_error("unexpected auxiliary feature count");
    }
    if (read_u32_le(header.data() + 12) != 1) {
        throw std::runtime_error("unexpected target encoding");
    }
}

void validate_record(const std::array<unsigned char, RecordSize>& record, std::uint64_t index) {
    int friendly_kings = 0;
    int enemy_kings = 0;
    for (std::size_t byte_index = 0; byte_index < 32; ++byte_index) {
        const unsigned char byte = record[byte_index];
        const std::array<unsigned char, 2> codes{
            static_cast<unsigned char>(byte & 0x0fU),
            static_cast<unsigned char>((byte >> 4) & 0x0fU),
        };
        for (unsigned char code : codes) {
            if (code > 12) {
                throw std::runtime_error("piece code out of range at record " + std::to_string(index));
            }
            friendly_kings += code == 6 ? 1 : 0;
            enemy_kings += code == 12 ? 1 : 0;
        }
    }
    if (friendly_kings != 1 || enemy_kings != 1) {
        throw std::runtime_error("record does not contain exactly one king per side at "
            + std::to_string(index));
    }

    const std::uint16_t aux = read_u16_le(record.data() + 32);
    if ((aux & static_cast<std::uint16_t>(~0x1fffU)) != 0) {
        throw std::runtime_error("auxiliary bits out of range at record " + std::to_string(index));
    }
    const bool has_ep = (aux & (1U << 4)) != 0;
    const unsigned ep_file_count = std::popcount(static_cast<unsigned>((aux >> 5) & 0xffU));
    if ((has_ep && ep_file_count != 1) || (!has_ep && ep_file_count != 0)) {
        throw std::runtime_error("invalid en-passant auxiliary bits at record "
            + std::to_string(index));
    }

    const std::int16_t score = static_cast<std::int16_t>(read_u16_le(record.data() + 34));
    if (score == StockfishValueNone) {
        throw std::runtime_error("VALUE_NONE score at record " + std::to_string(index));
    }
    const std::uint16_t ply = read_u16_le(record.data() + 36);
    if (ply > 0x3fffU) {
        throw std::runtime_error("ply out of range at record " + std::to_string(index));
    }
    const std::int16_t result = static_cast<std::int16_t>(read_u16_le(record.data() + 38));
    if (result < -1 || result > 1) {
        throw std::runtime_error("result out of range at record " + std::to_string(index));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (std::filesystem::exists(args.spool_directory)
            && !std::filesystem::is_empty(args.spool_directory)) {
            throw std::runtime_error("spool directory is not empty");
        }
        std::filesystem::create_directories(args.spool_directory);

        std::ifstream input_file;
        std::istream* input = &std::cin;
        if (args.input != "-") {
            input_file.open(args.input, std::ios::binary);
            if (!input_file) {
                throw std::runtime_error("failed to open input: " + args.input);
            }
            input = &input_file;
        }

        std::array<unsigned char, HeaderSize> header{};
        input->read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
        if (input->gcount() != static_cast<std::streamsize>(header.size())) {
            throw std::runtime_error("truncated compact header");
        }
        validate_header(header);

        std::array<std::ofstream, BucketCount> buckets;
        std::array<std::filesystem::path, BucketCount> bucket_paths;
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket) {
            bucket_paths[bucket] = args.spool_directory
                / ("fingerprints_" + std::to_string(bucket) + ".bin");
            buckets[bucket].open(bucket_paths[bucket], std::ios::binary | std::ios::trunc);
            if (!buckets[bucket]) {
                throw std::runtime_error("failed to open fingerprint bucket");
            }
        }

        std::uint64_t records = 0;
        std::int16_t score_min = std::numeric_limits<std::int16_t>::max();
        std::int16_t score_max = std::numeric_limits<std::int16_t>::min();
        std::uint16_t ply_min = std::numeric_limits<std::uint16_t>::max();
        std::uint16_t ply_max = 0;
        std::array<std::uint64_t, 3> result_counts{};
        std::array<unsigned char, RecordSize> record{};
        while (true) {
            input->read(
                reinterpret_cast<char*>(record.data()),
                static_cast<std::streamsize>(record.size()));
            const std::streamsize bytes_read = input->gcount();
            if (bytes_read == 0) {
                break;
            }
            if (bytes_read != static_cast<std::streamsize>(record.size())) {
                throw std::runtime_error("truncated compact record at index "
                    + std::to_string(records));
            }

            validate_record(record, records);
            const std::int16_t score = static_cast<std::int16_t>(read_u16_le(record.data() + 34));
            const std::uint16_t ply = read_u16_le(record.data() + 36);
            const std::int16_t result = static_cast<std::int16_t>(read_u16_le(record.data() + 38));
            score_min = std::min(score_min, score);
            score_max = std::max(score_max, score);
            ply_min = std::min(ply_min, ply);
            ply_max = std::max(ply_max, ply);
            ++result_counts[static_cast<std::size_t>(result + 1)];

            const Fingerprint fingerprint = fingerprint_position(record.data());
            const std::size_t bucket = static_cast<std::size_t>(fingerprint.first)
                & (BucketCount - 1);
            buckets[bucket].write(
                reinterpret_cast<const char*>(&fingerprint),
                static_cast<std::streamsize>(sizeof(fingerprint)));
            if (!buckets[bucket]) {
                throw std::runtime_error("failed while writing fingerprint bucket");
            }

            ++records;
            if (args.progress_interval != 0 && records % args.progress_interval == 0) {
                std::cerr << "validated=" << records << '\n';
            }
        }

        if (args.expected_records != 0 && records != args.expected_records) {
            throw std::runtime_error("record count mismatch: got " + std::to_string(records)
                + ", expected " + std::to_string(args.expected_records));
        }
        for (std::ofstream& bucket : buckets) {
            bucket.close();
            if (!bucket) {
                throw std::runtime_error("failed while closing fingerprint bucket");
            }
        }

        std::uint64_t fingerprint_duplicates = 0;
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket) {
            const std::uintmax_t bytes = std::filesystem::file_size(bucket_paths[bucket]);
            if (bytes % sizeof(Fingerprint) != 0) {
                throw std::runtime_error("truncated fingerprint bucket");
            }
            const std::size_t count = static_cast<std::size_t>(bytes / sizeof(Fingerprint));
            std::vector<Fingerprint> fingerprints(count);
            std::ifstream bucket_input(bucket_paths[bucket], std::ios::binary);
            bucket_input.read(
                reinterpret_cast<char*>(fingerprints.data()),
                static_cast<std::streamsize>(bytes));
            if (!bucket_input && bytes != 0) {
                throw std::runtime_error("failed while reading fingerprint bucket");
            }
            std::sort(fingerprints.begin(), fingerprints.end());
            for (std::size_t index = 1; index < fingerprints.size(); ++index) {
                fingerprint_duplicates += fingerprints[index] == fingerprints[index - 1] ? 1 : 0;
            }
            std::filesystem::remove(bucket_paths[bucket]);
            std::cerr << "checked_bucket=" << (bucket + 1) << '/' << BucketCount << '\n';
        }
        std::filesystem::remove(args.spool_directory);

        std::cout
            << "{\"records\":" << records
            << ",\"score_min\":" << score_min
            << ",\"score_max\":" << score_max
            << ",\"ply_min\":" << ply_min
            << ",\"ply_max\":" << ply_max
            << ",\"result_counts\":{\"-1\":" << result_counts[0]
            << ",\"0\":" << result_counts[1]
            << ",\"1\":" << result_counts[2]
            << "},\"fingerprint_duplicates\":" << fingerprint_duplicates
            << "}\n";
        return fingerprint_duplicates == 0 ? 0 : 2;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
