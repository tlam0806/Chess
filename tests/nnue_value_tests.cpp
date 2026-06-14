#include "nnue_value.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "move.hpp"
#include "position.hpp"

namespace {

void skip_ws(const std::string& text, std::size_t& pos) {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n')) {
        ++pos;
    }
}

void expect(const std::string& text, std::size_t& pos, char ch) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != ch) {
        throw std::runtime_error("malformed fixture");
    }
    ++pos;
}

std::size_t find_after(const std::string& text, const std::string& needle) {
    const std::size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing field: " + needle);
    }
    return pos + needle.size();
}

std::uint32_t parse_uint(const std::string& text, std::size_t& pos) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] < '0' || text[pos] > '9') {
        throw std::runtime_error("expected uint");
    }
    std::uint32_t value = 0;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        value = value * 10 + static_cast<std::uint32_t>(text[pos] - '0');
        ++pos;
    }
    return value;
}

float parse_float(const std::string& text, std::size_t& pos) {
    skip_ws(text, pos);
    const char* start = text.c_str() + pos;
    char* end = nullptr;
    const float value = std::strtof(start, &end);
    if (end == start) {
        throw std::runtime_error("expected float");
    }
    pos += static_cast<std::size_t>(end - start);
    return value;
}

chess::EncodedPosition parse_encoded(const std::string& line, float& python_output) {
    chess::EncodedPosition encoded;

    std::size_t pos = find_after(line, "\"features\":[");
    skip_ws(line, pos);
    while (pos < line.size() && line[pos] != ']') {
        encoded.features.push_back(parse_uint(line, pos));
        skip_ws(line, pos);
        if (pos < line.size() && line[pos] == ',') {
            ++pos;
        }
    }
    expect(line, pos, ']');

    pos = find_after(line, "\"aux\":[");
    int aux_index = 0;
    skip_ws(line, pos);
    while (pos < line.size() && line[pos] != ']') {
        if (aux_index >= chess::AuxFeatureCount) {
            throw std::runtime_error("too many aux values");
        }
        encoded.aux[aux_index++] = static_cast<std::uint8_t>(parse_uint(line, pos));
        skip_ws(line, pos);
        if (pos < line.size() && line[pos] == ',') {
            ++pos;
        }
    }
    expect(line, pos, ']');
    if (aux_index != chess::AuxFeatureCount) {
        throw std::runtime_error("wrong aux length");
    }

    pos = find_after(line, "\"python_output\":");
    python_output = parse_float(line, pos);
    return encoded;
}

void assert_incremental_matches_full_recompute(const chess::NnueValueModel& model) {
    chess::Position pos;
    pos.set_startpos();

    chess::NnueAccumulator accumulator;
    accumulator.reset(model, pos);

    std::mt19937 rng(42);
    for (int ply = 0; ply < 200; ++ply) {
        const float full = model.predict_normalized(chess::encode_position(pos));
        const float incremental = accumulator.predict_normalized(pos);
        const float error = std::fabs(full - incremental);
        if (error > 1.0e-4F) {
            throw std::runtime_error(
                "incremental mismatch before ply " + std::to_string(ply)
                + ": full=" + std::to_string(full)
                + " incremental=" + std::to_string(incremental)
                + " error=" + std::to_string(error));
        }

        const std::vector<chess::Move> moves = chess::generate_legal_moves(pos);
        if (moves.empty()) {
            break;
        }

        std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
        const chess::Move move = moves[dist(rng)];
        const chess::Position before = pos;
        pos.make_move(move);
        const chess::NnueAccumulatorUndo undo = accumulator.make_move_with_undo(before, move, pos);

        const float child_full = model.predict_normalized(chess::encode_position(pos));
        const float child_incremental = accumulator.predict_normalized(pos);
        if (std::fabs(child_full - child_incremental) > 1.0e-4F) {
            throw std::runtime_error("incremental child mismatch");
        }

        accumulator.undo(undo);
        const float parent_after_undo = accumulator.predict_normalized(before);
        if (std::fabs(full - parent_after_undo) > 1.0e-4F) {
            throw std::runtime_error("incremental undo mismatch");
        }
        accumulator.make_move(before, move, pos);
    }
}

void assert_incremental_move_matches_full_recompute(
    const chess::NnueValueModel& model,
    std::string_view fen,
    chess::Move move
) {
    chess::Position pos;
    if (!pos.set_fen(fen)) {
        throw std::runtime_error("bad test fen");
    }

    chess::NnueAccumulator accumulator;
    accumulator.reset(model, pos);

    const chess::Position before = pos;
    pos.make_move(move);
    const chess::NnueAccumulatorUndo undo = accumulator.make_move_with_undo(before, move, pos);

    const float full = model.predict_normalized(chess::encode_position(pos));
    const float incremental = accumulator.predict_normalized(pos);
    const float error = std::fabs(full - incremental);
    if (error > 1.0e-4F) {
        throw std::runtime_error(
            "incremental scenario mismatch: full=" + std::to_string(full)
            + " incremental=" + std::to_string(incremental)
            + " error=" + std::to_string(error));
    }

    accumulator.undo(undo);
    const float parent_full = model.predict_normalized(chess::encode_position(before));
    const float parent_incremental = accumulator.predict_normalized(before);
    if (std::fabs(parent_full - parent_incremental) > 1.0e-4F) {
        throw std::runtime_error("incremental scenario undo mismatch");
    }
}

void assert_incremental_special_moves_match_full_recompute(const chess::NnueValueModel& model) {
    assert_incremental_move_matches_full_recompute(
        model,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        chess::make_move(chess::make_square(4, 0), chess::make_square(6, 0), chess::MoveFlag::KingCastle));
    assert_incremental_move_matches_full_recompute(
        model,
        "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
        chess::make_move(chess::make_square(4, 3), chess::make_square(3, 4), chess::MoveFlag::Capture));
    assert_incremental_move_matches_full_recompute(
        model,
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        chess::make_move(chess::make_square(4, 4), chess::make_square(3, 5), chess::MoveFlag::EnPassant));
    assert_incremental_move_matches_full_recompute(
        model,
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        chess::make_move(
            chess::make_square(0, 6),
            chess::make_square(0, 7),
            chess::MoveFlag::QueenPromotion));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: nnue_value_tests <model.bin> <compare.jsonl>\n";
        return 2;
    }

    chess::NnueValueModel model;
    if (!model.load(argv[1])) {
        std::cerr << "failed to load model: " << argv[1] << '\n';
        return 1;
    }
    if (model.feature_count() != chess::EncodedFeatureCount
        || model.aux_feature_count() != chess::AuxFeatureCount
        || model.hidden_size() != 256) {
        std::cerr << "unexpected model shape: features=" << model.feature_count()
                  << " aux=" << model.aux_feature_count()
                  << " hidden=" << model.hidden_size() << '\n';
        return 1;
    }

    std::ifstream input(argv[2]);
    if (!input) {
        std::cerr << "failed to open compare file: " << argv[2] << '\n';
        return 1;
    }

    std::string line;
    int samples = 0;
    float max_error = 0.0F;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        float python_output = 0.0F;
        const chess::EncodedPosition encoded = parse_encoded(line, python_output);
        const float cpp_output = model.predict_normalized(encoded);
        const float error = std::fabs(cpp_output - python_output);
        max_error = std::max(max_error, error);
        if (error > 1.0e-4F) {
            std::cerr << "sample " << samples << " mismatch: cpp=" << cpp_output
                      << " python=" << python_output << " error=" << error << '\n';
            return 1;
        }
        ++samples;
    }

    if (samples == 0) {
        std::cerr << "compare file has no samples\n";
        return 1;
    }
    try {
        assert_incremental_matches_full_recompute(model);
        assert_incremental_special_moves_match_full_recompute(model);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    std::cout << "checked " << samples << " samples, max_error=" << max_error << '\n';
}
