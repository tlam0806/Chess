#include "nn_value.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

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

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: nn_value_tests <model.bin> <compare.jsonl>\n";
        return 2;
    }

    chess::NnValueModel model;
    assert(model.load(argv[1]));
    assert(model.feature_count() == chess::EncodedFeatureCount);
    assert(model.aux_feature_count() == chess::AuxFeatureCount);
    assert(model.hidden_size() == 256);

    std::ifstream input(argv[2]);
    assert(input);

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

    assert(samples > 0);
    std::cout << "checked " << samples << " samples, max_error=" << max_error << '\n';
}
