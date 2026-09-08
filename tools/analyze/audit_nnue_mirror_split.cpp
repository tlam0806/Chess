#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::array<unsigned char, 8> CompactMagic{
    'C', 'H', 'S', 'C', 'B', 'I', 'N', '2'};
constexpr std::size_t HeaderSize = 16;
constexpr std::size_t RecordSize = 40;
constexpr std::size_t PositionKeySize = 34;
constexpr std::size_t PhaseCount = 8;

struct Args {
    std::filesystem::path train;
    std::filesystem::path validation;
    std::filesystem::path filtered_validation_output;
    std::uint64_t expected_train_records = 0;
    std::uint64_t expected_validation_records = 0;
    std::uint64_t progress_interval = 5'000'000;
};

struct Fingerprint {
    std::uint64_t first = 0;
    std::uint64_t second = 0;

    bool operator==(const Fingerprint&) const = default;
};

struct FingerprintHash {
    std::size_t operator()(const Fingerprint& value) const noexcept {
        const std::uint64_t mixed = value.first ^ std::rotl(value.second, 29);
        if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
            return static_cast<std::size_t>(mixed);
        }
        return static_cast<std::size_t>(mixed ^ (mixed >> 32));
    }
};

struct CorpusStats {
    std::uint64_t records = 0;
    std::array<std::uint64_t, PhaseCount> phase_counts{};
};

[[nodiscard]] std::uint16_t read_u16_le(const unsigned char* data) {
    return static_cast<std::uint16_t>(data[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8);
}

void write_u16_le(unsigned char* data, std::uint16_t value) {
    data[0] = static_cast<unsigned char>(value & 0xffU);
    data[1] = static_cast<unsigned char>((value >> 8) & 0xffU);
}

[[nodiscard]] std::uint32_t read_u32_le(const unsigned char* data) {
    return static_cast<std::uint32_t>(data[0])
        | (static_cast<std::uint32_t>(data[1]) << 8)
        | (static_cast<std::uint32_t>(data[2]) << 16)
        | (static_cast<std::uint32_t>(data[3]) << 24);
}

[[nodiscard]] std::uint64_t parse_u64(std::string_view text) {
    if (text.empty()) {
        throw std::runtime_error("empty integer argument");
    }
    std::uint64_t value = 0;
    for (const char character : text) {
        if (character < '0' || character > '9') {
            throw std::runtime_error("bad integer argument: " + std::string(text));
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
        const auto next = [&](std::string_view name) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++index];
        };
        if (argument == "--train") {
            args.train = next(argument);
        } else if (argument == "--validation") {
            args.validation = next(argument);
        } else if (argument == "--expected-train-records") {
            args.expected_train_records = parse_u64(next(argument));
        } else if (argument == "--expected-validation-records") {
            args.expected_validation_records = parse_u64(next(argument));
        } else if (argument == "--filtered-validation-output") {
            args.filtered_validation_output = next(argument);
        } else if (argument == "--progress-interval") {
            args.progress_interval = parse_u64(next(argument));
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: audit_nnue_mirror_split --train PATH --validation PATH\n"
                << "       [--expected-train-records N]\n"
                << "       [--expected-validation-records N]\n"
                << "       [--filtered-validation-output OUTPUT.cbin]\n"
                << "       [--progress-interval N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }
    if (args.train.empty() || args.validation.empty()) {
        throw std::runtime_error("--train and --validation are required");
    }
    return args;
}

[[nodiscard]] bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size()
        && text.substr(text.size() - suffix.size()) == suffix;
}

[[nodiscard]] std::vector<std::filesystem::path> compact_paths(
    const std::filesystem::path& input
) {
    if (!std::filesystem::exists(input)) {
        throw std::runtime_error("data path does not exist: " + input.string());
    }
    if (std::filesystem::is_regular_file(input)) {
        return {input};
    }
    if (!std::filesystem::is_directory(input)) {
        throw std::runtime_error("data path is not a file or directory: " + input.string());
    }
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(input)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (ends_with(name, ".cbin") || ends_with(name, ".cbin.zst")) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error("no CBin shards found in " + input.string());
    }
    return paths;
}

[[nodiscard]] std::string shell_quote(const std::filesystem::path& path) {
    std::string quoted = "'";
    for (const char character : path.string()) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    quoted += "'";
    return quoted;
}

class RecordReader {
public:
    explicit RecordReader(const std::filesystem::path& path) : path_(path) {
        const bool compressed = ends_with(path.string(), ".zst");
        if (compressed) {
            const std::string command = "zstd -q -dc -- " + shell_quote(path);
            stream_ = popen(command.c_str(), "r");
            pipe_ = true;
        } else {
            stream_ = std::fopen(path.string().c_str(), "rb");
        }
        if (stream_ == nullptr) {
            throw std::runtime_error("failed to open " + path.string());
        }
        std::array<unsigned char, HeaderSize> header{};
        if (std::fread(header.data(), 1, header.size(), stream_) != header.size()) {
            throw std::runtime_error("truncated CBin header in " + path.string());
        }
        validate_header(header);
    }

    RecordReader(const RecordReader&) = delete;
    RecordReader& operator=(const RecordReader&) = delete;

    ~RecordReader() {
        close(false);
    }

    [[nodiscard]] bool next(std::array<unsigned char, RecordSize>& record) {
        const std::size_t bytes = std::fread(record.data(), 1, record.size(), stream_);
        if (bytes == 0 && std::feof(stream_)) {
            return false;
        }
        if (bytes != record.size()) {
            throw std::runtime_error("truncated CBin record in " + path_.string());
        }
        return true;
    }

    void finish() {
        close(true);
    }

private:
    void validate_header(const std::array<unsigned char, HeaderSize>& header) const {
        if (!std::equal(CompactMagic.begin(), CompactMagic.end(), header.begin())) {
            throw std::runtime_error("invalid CHSCBIN2 magic in " + path_.string());
        }
        if (read_u16_le(header.data() + 8) != 2
            || read_u16_le(header.data() + 10) != 13
            || read_u32_le(header.data() + 12) != 1) {
            throw std::runtime_error("unsupported CBin header in " + path_.string());
        }
    }

    void close(bool check_status) {
        if (stream_ == nullptr) {
            return;
        }
        const int code = pipe_ ? pclose(stream_) : std::fclose(stream_);
        stream_ = nullptr;
        if (check_status && code != 0) {
            throw std::runtime_error("reader failed for " + path_.string());
        }
    }

    std::filesystem::path path_;
    std::FILE* stream_ = nullptr;
    bool pipe_ = false;
};

[[nodiscard]] unsigned piece_code(
    const unsigned char* board,
    std::size_t square
) {
    const unsigned char byte = board[square / 2];
    return square % 2 == 0 ? byte & 0x0fU : (byte >> 4) & 0x0fU;
}

void set_piece_code(
    unsigned char* board,
    std::size_t square,
    unsigned code
) {
    unsigned char& byte = board[square / 2];
    if (square % 2 == 0) {
        byte = static_cast<unsigned char>((byte & 0xf0U) | code);
    } else {
        byte = static_cast<unsigned char>((byte & 0x0fU) | (code << 4));
    }
}

[[nodiscard]] std::uint16_t mirrored_aux(std::uint16_t aux) {
    constexpr std::array<unsigned, 13> SourceBit{
        1, 0, 3, 2, 4, 12, 11, 10, 9, 8, 7, 6, 5};
    std::uint16_t result = 0;
    for (std::size_t destination = 0; destination < SourceBit.size(); ++destination) {
        result |= static_cast<std::uint16_t>(
            ((aux >> SourceBit[destination]) & 1U) << destination);
    }
    return result;
}

struct CanonicalPosition {
    std::array<unsigned char, PositionKeySize> key{};
    std::size_t piece_count = 0;
};

[[nodiscard]] CanonicalPosition canonical_position(
    const std::array<unsigned char, RecordSize>& record
) {
    CanonicalPosition canonical;
    std::copy_n(record.begin(), PositionKeySize, canonical.key.begin());
    int friendly_king = -1;
    int friendly_kings = 0;
    int enemy_kings = 0;
    for (std::size_t square = 0; square < 64; ++square) {
        const unsigned code = piece_code(record.data(), square);
        if (code != 0) {
            ++canonical.piece_count;
        }
        if (code == 6) {
            friendly_king = static_cast<int>(square);
            ++friendly_kings;
        } else if (code == 12) {
            ++enemy_kings;
        } else if (code > 12) {
            throw std::runtime_error("piece code outside 0..12");
        }
    }
    if (friendly_kings != 1 || enemy_kings != 1) {
        throw std::runtime_error("position does not contain exactly one king per side");
    }
    const std::uint16_t aux = read_u16_le(record.data() + 32);
    if ((aux & static_cast<std::uint16_t>(~0x1fffU)) != 0) {
        throw std::runtime_error("auxiliary bits outside 13-bit range");
    }
    if ((friendly_king & 7) >= 4) {
        std::fill_n(canonical.key.begin(), 32, 0);
        for (std::size_t square = 0; square < 64; ++square) {
            set_piece_code(
                canonical.key.data(),
                square ^ 7U,
                piece_code(record.data(), square));
        }
        write_u16_le(canonical.key.data() + 32, mirrored_aux(aux));
    }
    return canonical;
}

[[nodiscard]] std::uint64_t mix(std::uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    return value ^ (value >> 33);
}

[[nodiscard]] Fingerprint fingerprint(const CanonicalPosition& position) {
    std::uint64_t first = 0x84222325cbf29ce4ULL;
    std::uint64_t second = 0x243f6a8885a308d3ULL;
    for (const unsigned char byte : position.key) {
        const std::uint64_t value = byte;
        first = (first ^ value) * 0x100000001b3ULL;
        second ^= value + 0x9e3779b97f4a7c15ULL + (second << 6) + (second >> 2);
        second *= 0xd6e8feb86659fd93ULL;
    }
    return Fingerprint{mix(first), mix(second)};
}

[[nodiscard]] std::size_t phase_index(std::size_t pieces) {
    if (pieces == 0) {
        throw std::runtime_error("position has no pieces");
    }
    return std::min<std::size_t>((pieces - 1) / 4, PhaseCount - 1);
}

template<typename Visitor>
CorpusStats scan_corpus(
    const std::filesystem::path& root,
    std::uint64_t progress_interval,
    std::string_view label,
    Visitor&& visitor
) {
    CorpusStats stats;
    const std::vector<std::filesystem::path> paths = compact_paths(root);
    std::array<unsigned char, RecordSize> record{};
    for (const auto& path : paths) {
        RecordReader reader(path);
        while (reader.next(record)) {
            const CanonicalPosition canonical = canonical_position(record);
            ++stats.records;
            ++stats.phase_counts[phase_index(canonical.piece_count)];
            visitor(fingerprint(canonical));
            if (progress_interval != 0 && stats.records % progress_interval == 0) {
                std::cerr << "stage=" << label << " records=" << stats.records << '\n';
            }
        }
        reader.finish();
    }
    return stats;
}

struct FilterStats {
    CorpusStats corpus;
    std::uint64_t removed_canonical_duplicates = 0;
    std::uint64_t removed_cross_split_keys = 0;
};

FilterStats filter_validation(
    const std::filesystem::path& source,
    const std::filesystem::path& output,
    const std::unordered_set<Fingerprint, FingerprintHash>& overlap_keys,
    std::uint64_t progress_interval
) {
    if (std::filesystem::exists(output)) {
        throw std::runtime_error(
            "refusing to overwrite filtered validation output: " + output.string());
    }
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::FILE* destination = std::fopen(output.string().c_str(), "wb");
    if (destination == nullptr) {
        throw std::runtime_error(
            "failed to create filtered validation output: " + output.string());
    }
    const auto close_destination = [&]() {
        if (destination != nullptr) {
            std::fclose(destination);
            destination = nullptr;
        }
    };

    try {
        std::array<unsigned char, HeaderSize> header{};
        std::copy(CompactMagic.begin(), CompactMagic.end(), header.begin());
        write_u16_le(header.data() + 8, 2);
        write_u16_le(header.data() + 10, 13);
        header[12] = 1;
        if (std::fwrite(header.data(), 1, header.size(), destination) != header.size()) {
            throw std::runtime_error("failed to write filtered validation header");
        }

        FilterStats stats;
        std::unordered_set<Fingerprint, FingerprintHash> seen;
        seen.reserve(1'250'000);
        std::array<unsigned char, RecordSize> record{};
        for (const auto& path : compact_paths(source)) {
            RecordReader reader(path);
            while (reader.next(record)) {
                const CanonicalPosition canonical = canonical_position(record);
                const Fingerprint key = fingerprint(canonical);
                if (!seen.insert(key).second) {
                    ++stats.removed_canonical_duplicates;
                    continue;
                }
                if (overlap_keys.count(key) != 0) {
                    ++stats.removed_cross_split_keys;
                    continue;
                }
                if (std::fwrite(
                        record.data(), 1, record.size(), destination)
                    != record.size()) {
                    throw std::runtime_error("failed to write filtered validation record");
                }
                ++stats.corpus.records;
                ++stats.corpus.phase_counts[phase_index(canonical.piece_count)];
                if (progress_interval != 0
                    && stats.corpus.records % progress_interval == 0) {
                    std::cerr
                        << "stage=filter_validation records="
                        << stats.corpus.records << '\n';
                }
            }
            reader.finish();
        }
        if (std::fflush(destination) != 0 || std::ferror(destination) != 0) {
            throw std::runtime_error("failed to flush filtered validation output");
        }
        close_destination();
        return stats;
    } catch (...) {
        close_destination();
        throw;
    }
}

void require_expected(
    std::string_view label,
    std::uint64_t actual,
    std::uint64_t expected
) {
    if (expected != 0 && actual != expected) {
        throw std::runtime_error(
            std::string(label) + " record count mismatch: expected="
            + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

void print_phase_counts(const std::array<std::uint64_t, PhaseCount>& counts) {
    std::cout << '[';
    for (std::size_t index = 0; index < counts.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << counts[index];
    }
    std::cout << ']';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        std::unordered_set<Fingerprint, FingerprintHash> validation_keys;
        if (args.expected_validation_records != 0) {
            validation_keys.reserve(static_cast<std::size_t>(
                args.expected_validation_records * 5 / 4));
        }
        std::uint64_t validation_duplicates = 0;
        const CorpusStats validation = scan_corpus(
            args.validation,
            args.progress_interval,
            "validation",
            [&](const Fingerprint& value) {
                if (!validation_keys.insert(value).second) {
                    ++validation_duplicates;
                }
            });
        require_expected(
            "validation", validation.records, args.expected_validation_records);

        std::uint64_t cross_split_overlaps = 0;
        std::unordered_set<Fingerprint, FingerprintHash> overlap_keys;
        const CorpusStats train = scan_corpus(
            args.train,
            args.progress_interval,
            "train",
            [&](const Fingerprint& value) {
                if (validation_keys.count(value) != 0) {
                    ++cross_split_overlaps;
                    overlap_keys.insert(value);
                }
            });
        require_expected("train", train.records, args.expected_train_records);

        const bool all_phases_present = std::all_of(
            train.phase_counts.begin(), train.phase_counts.end(),
            [](std::uint64_t count) { return count != 0; })
            && std::all_of(
                validation.phase_counts.begin(), validation.phase_counts.end(),
                [](std::uint64_t count) { return count != 0; });
        const bool passed = validation_duplicates == 0
            && cross_split_overlaps == 0
            && all_phases_present;

        FilterStats filtered;
        bool filtered_passed = false;
        if (!args.filtered_validation_output.empty()) {
            filtered = filter_validation(
                args.validation,
                args.filtered_validation_output,
                overlap_keys,
                args.progress_interval);
            filtered_passed = filtered.corpus.records != 0
                && std::all_of(
                    filtered.corpus.phase_counts.begin(),
                    filtered.corpus.phase_counts.end(),
                    [](std::uint64_t count) { return count != 0; });
        }

        std::cout
            << "{\"event\":\"mirror_split_audit\",\"fingerprint_bits\":128"
            << ",\"train_records\":" << train.records
            << ",\"validation_records\":" << validation.records
            << ",\"validation_canonical_duplicates\":" << validation_duplicates
            << ",\"cross_split_overlaps\":" << cross_split_overlaps
            << ",\"all_phases_present\":" << (all_phases_present ? "true" : "false")
            << ",\"train_phase_counts\":";
        print_phase_counts(train.phase_counts);
        std::cout << ",\"validation_phase_counts\":";
        print_phase_counts(validation.phase_counts);
        if (!args.filtered_validation_output.empty()) {
            std::cout
                << ",\"unique_overlap_keys\":" << overlap_keys.size()
                << ",\"filtered_validation_records\":"
                << filtered.corpus.records
                << ",\"filtered_removed_canonical_duplicates\":"
                << filtered.removed_canonical_duplicates
                << ",\"filtered_removed_cross_split_keys\":"
                << filtered.removed_cross_split_keys
                << ",\"filtered_phase_counts\":";
            print_phase_counts(filtered.corpus.phase_counts);
        }
        const char* status = passed
            ? "pass"
            : (filtered_passed ? "filtered" : "fail");
        std::cout << ",\"status\":\"" << status << "\"}\n";
        return passed || filtered_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
