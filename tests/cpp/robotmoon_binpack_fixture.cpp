#include <cstdint>
#include <iostream>
#include <string>

#include "../../tools/third_party/nnue_pytorch/binpack.h"

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: robotmoon_binpack_fixture OUTPUT.binpack [--append-in-check]\n";
        return 2;
    }
    const bool append_in_check = argc == 3 && std::string(argv[2]) == "--append-in-check";
    if (argc == 3 && !append_in_check) {
        std::cerr << "unknown option: " << argv[2] << '\n';
        return 2;
    }

    binpack::CompressedTrainingDataEntryWriter writer(argv[1], std::ios_base::trunc);

    binpack::TrainingDataEntry white{};
    white.pos = chess::Position::fromFen(
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1");
    white.move = chess::uci::uciToMove(white.pos, "e5d6");
    white.score = 208;
    white.ply = 10;
    white.result = 1;
    if (!white.isValid()) {
        std::cerr << "white fixture move is not legal\n";
        return 1;
    }
    writer.addTrainingDataEntry(white);

    binpack::TrainingDataEntry continuation{};
    continuation.pos = white.pos;
    continuation.pos.doMove(white.move);
    continuation.move = chess::uci::uciToMove(continuation.pos, "e8d7");
    continuation.score = -123;
    continuation.ply = 11;
    continuation.result = -1;
    if (!continuation.isValid() || !binpack::isContinuation(white, continuation)) {
        std::cerr << "continuation fixture is not a legal compressed continuation\n";
        return 1;
    }
    writer.addTrainingDataEntry(continuation);

    binpack::TrainingDataEntry black{};
    black.pos = chess::Position::fromFen(
        "r3k2r/8/8/8/3Pp3/8/8/R3K2R b KQkq d3 0 1");
    black.move = chess::uci::uciToMove(black.pos, "e4d3");
    black.score = -416;
    black.ply = 33;
    black.result = -1;
    if (!black.isValid()) {
        std::cerr << "black fixture move is not legal\n";
        return 1;
    }
    writer.addTrainingDataEntry(black);

    if (append_in_check) {
        binpack::TrainingDataEntry in_check{};
        in_check.pos = chess::Position::fromFen(
            "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
        in_check.move = chess::uci::uciToMove(in_check.pos, "e1f1");
        in_check.score = 100;
        in_check.ply = 20;
        in_check.result = -1;
        if (!in_check.isInCheck() || !in_check.isValid()) {
            std::cerr << "in-check fixture or evasion move is invalid\n";
            return 1;
        }
        writer.addTrainingDataEntry(in_check);
    }

    binpack::TrainingDataEntry value_none = white;
    value_none.score = 32002;
    value_none.ply = 80;
    value_none.result = 0;
    writer.addTrainingDataEntry(value_none);
    return 0;
}
