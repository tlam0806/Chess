#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct Sample {
    chess::Position pos;
    chess::Move move{};
    chess::PieceType moved_piece = chess::PieceType::None;
    chess::PieceType captured_piece = chess::PieceType::None;
};

chess::PieceType captured_piece_for(const chess::Position& pos, chess::Move move) {
    if (!chess::is_capture(move) || chess::move_flag(move) == chess::MoveFlag::EnPassant) {
        return chess::PieceType::None;
    }
    return pos.piece_type_on_occupied(chess::opposite(pos.side_to_move), chess::to_square(move));
}

std::vector<Sample> make_samples(int positions, int plies_per_position, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>(positions) * 32);

    chess::Position pos;
    pos.set_startpos();
    for (int ply = 0; ply < positions; ++ply) {
        chess::MoveList moves;
        chess::generate_legal_moves(pos, moves);
        if (moves.empty()) {
            pos.set_startpos();
            continue;
        }

        for (chess::Move move : moves) {
            samples.push_back(Sample{
                pos,
                move,
                pos.piece_type_on_occupied(pos.side_to_move, chess::from_square(move)),
                captured_piece_for(pos, move)
            });
        }

        for (int step = 0; step < plies_per_position; ++step) {
            chess::MoveList next_moves;
            chess::generate_legal_moves(pos, next_moves);
            if (next_moves.empty()) {
                pos.set_startpos();
                break;
            }
            std::uniform_int_distribution<std::size_t> dist(0, next_moves.size() - 1);
            pos.make_move(next_moves[dist(rng)]);
        }
    }

    return samples;
}

template <typename Fn>
std::uint64_t time_us(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

[[gnu::noinline]] void consume_position_by_value(chess::Position pos, volatile std::uint64_t& sink) {
    sink ^= pos.zobrist_key;
    sink += static_cast<std::uint64_t>(pos.eval_score);
}

} // namespace

int main(int argc, char** argv) {
    int positions = 2000;
    int repeats = 100;
    if (argc >= 2) {
        positions = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        repeats = std::stoi(argv[2]);
    }

    std::vector<Sample> samples = make_samples(positions, 1, 20260619);
    volatile std::uint64_t sink = 0;

    const std::uint64_t copy_make_us = time_us([&] {
        std::uint64_t local = 0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (const Sample& sample : samples) {
                chess::Position next = sample.pos;
                next.make_move(sample.move, sample.moved_piece, sample.captured_piece);
                local ^= next.zobrist_key;
                local += static_cast<std::uint64_t>(next.eval_score);
            }
        }
        sink ^= local;
    });

    const std::uint64_t copy_make_call_us = time_us([&] {
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (const Sample& sample : samples) {
                chess::Position next = sample.pos;
                next.make_move(sample.move, sample.moved_piece, sample.captured_piece);
                consume_position_by_value(next, sink);
            }
        }
    });

    const std::uint64_t make_unmake_us = time_us([&] {
        std::uint64_t local = 0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (Sample& sample : samples) {
                chess::UndoState undo;
                sample.pos.make_move(sample.move, sample.moved_piece, sample.captured_piece, undo);
                local ^= sample.pos.zobrist_key;
                local += static_cast<std::uint64_t>(sample.pos.eval_score);
                sample.pos.unmake_move(sample.move, undo);
                local ^= sample.pos.zobrist_key;
            }
        }
        sink ^= local;
    });

    const double ops = static_cast<double>(samples.size()) * repeats;
    std::cout << "samples=" << samples.size()
              << " repeats=" << repeats
              << " copy_make_us=" << copy_make_us
              << " copy_make_call_us=" << copy_make_call_us
              << " make_unmake_us=" << make_unmake_us
              << " copy_make_ns_per_op=" << (copy_make_us * 1000.0 / ops)
              << " copy_make_call_ns_per_op=" << (copy_make_call_us * 1000.0 / ops)
              << " make_unmake_ns_per_op=" << (make_unmake_us * 1000.0 / ops)
              << " copy_vs_unmake_time="
              << (make_unmake_us == 0 ? 0.0 : static_cast<double>(copy_make_us) / make_unmake_us)
              << " copy_call_vs_unmake_time="
              << (make_unmake_us == 0 ? 0.0 : static_cast<double>(copy_make_call_us) / make_unmake_us)
              << " sink=" << sink
              << '\n';
}
