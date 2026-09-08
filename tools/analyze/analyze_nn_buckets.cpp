#include "attacks.hpp"
#include "board_encoder.hpp"
#include "move.hpp"
#include "nn_value.hpp"
#include "position.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model = "models/value_net_finetune_v7_mixed_old70k_material70k.bin";
    std::string input = "data/nn_mix_old5k_material5k_test_baseline.jsonl";
    int limit = 0;
};

struct Sample {
    chess::Position pos;
    int target = 0;
};

struct Bucket {
    std::string name;
    std::uint64_t count = 0;
    double abs_error_sum = 0.0;
    double signed_error_sum = 0.0;
    int max_abs_error = 0;

    explicit Bucket(std::string bucket_name)
        : name(std::move(bucket_name)) {
    }

    void add(int prediction, int target) {
        const int error = prediction - target;
        const int abs_error = std::abs(error);
        ++count;
        abs_error_sum += abs_error;
        signed_error_sum += error;
        max_abs_error = std::max(max_abs_error, abs_error);
    }

    void print() const {
        if (count == 0) {
            std::cout << name << " count=0\n";
            return;
        }
        std::cout << name
                  << " count=" << count
                  << " mae_cp=" << (abs_error_sum / static_cast<double>(count))
                  << " bias_cp=" << (signed_error_sum / static_cast<double>(count))
                  << " max_error_cp=" << max_abs_error << '\n';
    }
};

int parse_int(std::string_view value, std::string_view name) {
    try {
        std::size_t parsed = 0;
        const int result = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + std::string(name) + ": " + std::string(value));
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };

        if (arg == "--model") {
            options.model = std::string(require_value(arg));
        } else if (arg == "--input") {
            options.input = std::string(require_value(arg));
        } else if (arg == "--limit") {
            options.limit = parse_int(require_value(arg), arg);
        } else if (arg == "--help") {
            std::cout << "Usage: analyze_nn_buckets [--model path] [--input path] [--limit N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.limit < 0) {
        throw std::runtime_error("--limit must be non-negative");
    }
    return options;
}

std::vector<int> extract_int_array(const std::string& line, std::string_view key) {
    const std::size_t key_pos = line.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("sample is missing " + std::string(key));
    }

    const std::size_t open = line.find('[', key_pos);
    const std::size_t close = line.find(']', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("invalid array for " + std::string(key));
    }

    std::vector<int> values;
    std::size_t pos = open + 1;
    while (pos < close) {
        while (pos < close && (line[pos] == ' ' || line[pos] == ',')) {
            ++pos;
        }
        if (pos >= close) {
            break;
        }
        std::size_t end = pos;
        while (end < close && line[end] != ',') {
            ++end;
        }
        values.push_back(parse_int(std::string_view(line).substr(pos, end - pos), std::string(key)));
        pos = end + 1;
    }
    return values;
}

int extract_target(const std::string& line) {
    constexpr std::string_view key = "\"target\":";
    const std::size_t key_pos = line.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("sample is missing target");
    }
    std::size_t start = key_pos + key.size();
    std::size_t end = start;
    while (end < line.size() && line[end] != ',' && line[end] != '}') {
        ++end;
    }
    return parse_int(std::string_view(line).substr(start, end - start), "target");
}

bool decode_sample(const std::string& line, Sample& sample) {
    const std::vector<int> aux_values = extract_int_array(line, "\"aux\"");
    if (aux_values.size() != chess::AuxFeatureCount) {
        throw std::runtime_error("unexpected aux feature count");
    }
    if (aux_values[chess::HasEnPassant] != 0) {
        return false;
    }

    chess::Position pos;
    pos.clear();
    pos.side_to_move = chess::Color::White;
    pos.white_can_castle_kingside = aux_values[chess::FriendlyCanCastleKingside] != 0;
    pos.white_can_castle_queenside = aux_values[chess::FriendlyCanCastleQueenside] != 0;
    pos.black_can_castle_kingside = aux_values[chess::EnemyCanCastleKingside] != 0;
    pos.black_can_castle_queenside = aux_values[chess::EnemyCanCastleQueenside] != 0;

    std::array<bool, chess::EncodedFeatureCount> seen{};
    const std::vector<int> features = extract_int_array(line, "\"features\"");
    for (int raw_feature : features) {
        if (raw_feature < 0 || raw_feature >= chess::EncodedFeatureCount) {
            throw std::runtime_error("feature index out of range");
        }
        if (seen[static_cast<std::size_t>(raw_feature)]) {
            continue;
        }
        seen[static_cast<std::size_t>(raw_feature)] = true;

        chess::FeatureIndex index = static_cast<chess::FeatureIndex>(raw_feature);
        const chess::Square piece_square = static_cast<chess::Square>(index % chess::EncoderSquares);
        index /= chess::EncoderSquares;
        index /= chess::EncoderSquares; // King square is only context for this decode.
        const auto king_context = static_cast<chess::EncodedKingContext>(index % chess::EncoderKingContexts);
        index /= chess::EncoderKingContexts;
        const auto piece_side = static_cast<chess::EncodedPieceSide>(index % chess::EncoderPieceSides);
        index /= chess::EncoderPieceSides;
        const auto piece = static_cast<chess::PieceType>(index);

        if (king_context != chess::EncodedKingContext::FriendlyKing) {
            continue;
        }

        const chess::Color color = piece_side == chess::EncodedPieceSide::Friendly
            ? chess::Color::White
            : chess::Color::Black;
        if (!pos.is_empty(piece_square)) {
            throw std::runtime_error("decoded duplicate piece square");
        }
        pos.set_piece(color, piece, piece_square);
    }

    if (chess::popcount(pos.occupancy(chess::Color::White, chess::PieceType::King)) != 1
        || chess::popcount(pos.occupancy(chess::Color::Black, chess::PieceType::King)) != 1) {
        return false;
    }

    sample.pos = pos;
    sample.target = extract_target(line);
    return true;
}

int material_cp_white_minus_black(const chess::Position& pos) {
    constexpr int values[] = {100, 320, 330, 500, 900, 0};
    int score = 0;
    for (int piece = 0; piece < 6; ++piece) {
        score += values[piece] * chess::popcount(pos.pieces[static_cast<int>(chess::Color::White)][piece]);
        score -= values[piece] * chess::popcount(pos.pieces[static_cast<int>(chess::Color::Black)][piece]);
    }
    return score;
}

bool has_capture(const std::vector<chess::Move>& moves) {
    for (chess::Move move : moves) {
        if (chess::is_capture(move)) {
            return true;
        }
    }
    return false;
}

bool has_promotion(const std::vector<chess::Move>& moves) {
    for (chess::Move move : moves) {
        if (chess::promotion_piece(move) != chess::PieceType::None) {
            return true;
        }
    }
    return false;
}

bool has_check_move(const chess::Position& pos, const std::vector<chess::Move>& moves) {
    for (chess::Move move : moves) {
        chess::Position next = pos;
        next.make_move(move);
        if (chess::in_check(next, next.side_to_move)) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        chess::NnValueModel model;
        if (!model.load(options.model)) {
            throw std::runtime_error("failed to load model: " + options.model);
        }

        std::ifstream input(options.input);
        if (!input) {
            throw std::runtime_error("failed to open input: " + options.input);
        }

        Bucket all("all");
        Bucket quiet("quiet_no_capture_no_check_no_promotion");
        Bucket capture("capture_available");
        Bucket check("check_available");
        Bucket promotion("promotion_available");
        Bucket material_equalish("material_abs_lt_300");
        Bucket material_imbalanced("material_abs_ge_300");
        Bucket target_equalish("target_abs_lt_200");
        Bucket target_medium("target_abs_200_500");
        Bucket target_large("target_abs_ge_500");
        Bucket mate_like("target_abs_ge_5000");

        int decoded = 0;
        int skipped = 0;
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }
            if (options.limit > 0 && decoded >= options.limit) {
                break;
            }

            Sample sample;
            if (!decode_sample(line, sample)) {
                ++skipped;
                continue;
            }

            const int prediction = model.evaluate_cp_rounded(sample.pos);
            const int raw_target = sample.target;
            const int target = static_cast<int>(std::lround(std::clamp(
                static_cast<float>(raw_target),
                -model.target_clip(),
                model.target_clip())));
            const std::vector<chess::Move> moves = chess::generate_legal_moves(sample.pos);
            const bool capture_available = has_capture(moves);
            const bool check_available = has_check_move(sample.pos, moves);
            const bool promotion_available = has_promotion(moves);
            const int material_abs = std::abs(material_cp_white_minus_black(sample.pos));
            const int target_abs = std::abs(raw_target);

            all.add(prediction, target);
            if (!capture_available && !check_available && !promotion_available) {
                quiet.add(prediction, target);
            }
            if (capture_available) {
                capture.add(prediction, target);
            }
            if (check_available) {
                check.add(prediction, target);
            }
            if (promotion_available) {
                promotion.add(prediction, target);
            }
            if (material_abs < 300) {
                material_equalish.add(prediction, target);
            } else {
                material_imbalanced.add(prediction, target);
            }
            if (target_abs < 200) {
                target_equalish.add(prediction, target);
            } else if (target_abs < 500) {
                target_medium.add(prediction, target);
            } else {
                target_large.add(prediction, target);
            }
            if (target_abs >= 5000) {
                mate_like.add(prediction, target);
            }
            ++decoded;
        }

        std::cout << "model=" << options.model
                  << " input=" << options.input
                  << " decoded=" << decoded
                  << " skipped=" << skipped << '\n';
        all.print();
        quiet.print();
        capture.print();
        check.print();
        promotion.print();
        material_equalish.print();
        material_imbalanced.print();
        target_equalish.print();
        target_medium.print();
        target_large.print();
        mate_like.print();
    } catch (const std::exception& error) {
        std::cerr << "analyze_nn_buckets: " << error.what() << '\n';
        return 1;
    }
}
