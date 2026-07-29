#include "king_safety.hpp"
#include "move.hpp"
#include "position.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

struct Case {
    chess::Position after;
    chess::Move move{};
    chess::Color moved_color = chess::Color::White;
    chess::PieceType moved_piece = chess::PieceType::None;
    chess::PieceType captured_piece = chess::PieceType::None;
    chess::Square captured_square = chess::NoSquare;
};

bool has_both_kings(const chess::Position& pos) {
    const auto king = static_cast<int>(chess::PieceType::King);
    return chess::popcount(pos.pieces[static_cast<int>(chess::Color::White)][king]) == 1
        && chess::popcount(pos.pieces[static_cast<int>(chess::Color::Black)][king]) == 1;
}

chess::PieceType captured_piece_for_move(const chess::Position& pos, chess::Move move) {
    if (!chess::is_capture(move)) {
        return chess::PieceType::None;
    }
    if (chess::move_flag(move) == chess::MoveFlag::EnPassant) {
        return chess::PieceType::Pawn;
    }
    return pos.piece_type_on_occupied(chess::to_square(move));
}

chess::Square captured_square_for_move(const chess::Position& pos, chess::Move move) {
    if (!chess::is_capture(move)) {
        return chess::NoSquare;
    }
    if (chess::move_flag(move) != chess::MoveFlag::EnPassant) {
        return chess::to_square(move);
    }
    const int direction = pos.side_to_move == chess::Color::White ? -1 : 1;
    return chess::make_square(
        chess::file_of(chess::to_square(move)),
        chess::rank_of(chess::to_square(move)) + direction);
}

void add_cases_from_position(const chess::Position& pos, std::vector<Case>& cases) {
    chess::MoveList moves;
    chess::generate_legal_moves(pos, moves);
    for (const chess::Move move : moves) {
        Case item;
        item.move = move;
        item.moved_color = pos.side_to_move;
        item.moved_piece = pos.piece_type_on_occupied(chess::from_square(move));
        item.captured_piece = captured_piece_for_move(pos, move);
        item.captured_square = captured_square_for_move(pos, move);
        item.after = pos;
        item.after.make_move(item.move, item.moved_piece, item.captured_piece);
        cases.push_back(item);
    }
}

std::vector<chess::Position> make_positions(int random_positions, int max_random_plies) {
    std::vector<chess::Position> positions;
    auto add_fen = [&](std::string_view fen) {
        chess::Position pos;
        if (pos.set_fen(fen) && has_both_kings(pos)) {
            positions.push_back(pos);
        }
    };

    chess::Position start;
    start.set_startpos();
    positions.push_back(start);
    add_fen("rnb1kb1r/ppppqppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 1 4");
    add_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    add_fen("r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 9");
    add_fen("2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 4 12");
    add_fen("4r2k/5ppp/5P2/1p1pp3/3nP2P/1p1b4/rP1P1P2/R1BR2K1 w - - 0 23");

    std::mt19937 rng(20260703);
    chess::Position pos;
    pos.set_startpos();
    for (int i = 0; i < random_positions; ++i) {
        positions.push_back(pos);
        const int plies = 1 + static_cast<int>(rng() % static_cast<unsigned>(max_random_plies));
        for (int ply = 0; ply < plies; ++ply) {
            chess::MoveList moves;
            chess::generate_legal_moves(pos, moves);
            if (moves.empty()) {
                pos.set_startpos();
                break;
            }
            std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
            const chess::Move move = moves[dist(rng)];
            const chess::PieceType moved_piece = pos.piece_type_on_occupied(chess::from_square(move));
            const chess::PieceType captured_piece = captured_piece_for_move(pos, move);
            pos.make_move(move, moved_piece, captured_piece);
        }
    }
    return positions;
}

} // namespace

int main(int argc, char** argv) {
    int random_positions = 128;
    int rounds = 20000;
    if (argc >= 2) {
        random_positions = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        rounds = std::stoi(argv[2]);
    }

    std::vector<Case> cases;
    for (const chess::Position& pos : make_positions(random_positions, 12)) {
        add_cases_from_position(pos, cases);
    }

    std::uint64_t checksum = 0;
    for (const Case& item : cases) {
        chess::Position pos = item.after;
        chess::update_king_safety_after_move(
            pos,
            item.move,
            item.moved_color,
            item.moved_piece,
            item.captured_piece,
            item.captured_square);
        checksum ^= pos.king_checkers[0] + 3 * pos.king_checkers[1];
        checksum ^= pos.king_pinned[0] + 5 * pos.king_pinned[1];
        checksum ^= pos.king_block_masks[0] + 7 * pos.king_block_masks[1];
    }

    const auto start = std::chrono::steady_clock::now();
    std::uint64_t calls = 0;
    for (int round = 0; round < rounds; ++round) {
        for (const Case& item : cases) {
            chess::Position pos = item.after;
            chess::update_king_safety_after_move(
                pos,
                item.move,
                item.moved_color,
                item.moved_piece,
                item.captured_piece,
                item.captured_square);
            checksum += pos.king_checkers[0] ^ pos.king_checkers[1];
            checksum += pos.king_pinned[0] ^ pos.king_pinned[1];
            checksum += pos.king_block_masks[0] ^ pos.king_block_masks[1];
            ++calls;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "cases=" << cases.size()
              << " rounds=" << rounds
              << " calls=" << calls
              << " elapsed_us=" << elapsed_us
              << " ns_per_call=" << (calls == 0 ? 0.0 : static_cast<double>(elapsed_us) * 1000.0 / calls)
              << " checksum=" << checksum
              << '\n';
}
