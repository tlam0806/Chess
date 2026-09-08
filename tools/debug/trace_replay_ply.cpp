#include "attacks.hpp"
#include "evaluate.hpp"
#include "heuristic_searcher_v6.hpp"
#include "move.hpp"
#include "nn_searcher_v6.hpp"
#include "nn_value.hpp"
#include "position.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ReplayMove {
    int ply = 0;
    std::string move;
};

std::vector<ReplayMove> load_moves(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open replay: " + path);
    }

    std::vector<ReplayMove> moves;
    std::string line;
    bool in_moves = false;
    while (std::getline(input, line)) {
        if (line.empty()) {
            in_moves = true;
            continue;
        }
        if (!in_moves) {
            continue;
        }

        std::istringstream parser(line);
        int ply = 0;
        std::string side;
        std::string engine;
        std::string move;
        if (parser >> ply >> side >> engine >> move) {
            moves.push_back({ply, move});
        }
    }
    return moves;
}

chess::Move find_legal_move(const chess::Position& pos, const std::string& text) {
    for (chess::Move move : chess::generate_legal_moves(pos)) {
        if (chess::move_to_string(move) == text) {
            return move;
        }
    }
    throw std::runtime_error("illegal replay move: " + text);
}

int material_cp(const chess::Position& pos) {
    constexpr int values[] = {100, 320, 330, 500, 900, 0};
    int score = 0;
    for (int piece = 0; piece < 6; ++piece) {
        score += values[piece] * chess::popcount(pos.pieces[static_cast<int>(chess::Color::White)][piece]);
        score -= values[piece] * chess::popcount(pos.pieces[static_cast<int>(chess::Color::Black)][piece]);
    }
    return score;
}

std::string stm_name(chess::Color color) {
    return color == chess::Color::White ? "white" : "black";
}

struct MoveTrace {
    std::string move;
    int nn_score = 0;
    int nn_nodes = 0;
    int heuristic_score = 0;
    int heuristic_nodes = 0;
    int immediate_nn = 0;
    int immediate_heuristic = 0;
};

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: trace_replay_ply <model.bin> <replay.txt> <target-ply> <depth>\n";
            return 2;
        }

        const std::string model_path = argv[1];
        const std::string replay_path = argv[2];
        const int target_ply = std::atoi(argv[3]);
        const int depth = std::atoi(argv[4]);

        chess::NnValueModel model;
        if (!model.load(model_path)) {
            throw std::runtime_error("failed to load model: " + model_path);
        }

        const std::vector<ReplayMove> replay_moves = load_moves(replay_path);
        chess::Position pos;
        pos.set_startpos();
        for (const ReplayMove& replay_move : replay_moves) {
            if (replay_move.ply >= target_ply) {
                break;
            }
            pos.make_move(find_legal_move(pos, replay_move.move));
        }

        std::cout << "position before ply " << target_ply << '\n';
        std::cout << "side_to_move " << stm_name(pos.side_to_move) << '\n';
        std::cout << "material_cp_white_minus_black " << material_cp(pos) << '\n';
        std::cout << "nn_eval_stm " << model.evaluate_cp_rounded(pos) << '\n';
        std::cout << "heuristic_eval_stm " << chess::evaluate_for_side_to_move(pos) << "\n\n";
        pos.print(std::cout);
        std::cout << '\n';

        std::vector<MoveTrace> traces;
        for (chess::Move move : chess::generate_legal_moves(pos)) {
            chess::Position next = pos;
            next.make_move(move);

            chess::NnSearcherV6 nn_searcher(model);
            chess::HeuristicSearcherV6 heuristic_searcher;
            const chess::SearchResult nn_child = nn_searcher.search_best_move(next, depth - 1);
            const chess::SearchResult heuristic_child = heuristic_searcher.search_best_move(next, depth - 1);

            traces.push_back(MoveTrace{
                chess::move_to_string(move),
                -nn_child.score,
                static_cast<int>(nn_child.nodes),
                -heuristic_child.score,
                static_cast<int>(heuristic_child.nodes),
                -model.evaluate_cp_rounded(next),
                -chess::evaluate_for_side_to_move(next),
            });
        }

        std::stable_sort(traces.begin(), traces.end(), [](const MoveTrace& lhs, const MoveTrace& rhs) {
            return lhs.nn_score > rhs.nn_score;
        });

        std::cout << "root candidates by NN V6 depth " << depth << '\n';
        std::cout << "move nn_score nn_nodes heuristic_score heuristic_nodes immediate_nn immediate_heuristic\n";
        for (const MoveTrace& trace : traces) {
            std::cout << trace.move << ' '
                      << trace.nn_score << ' '
                      << trace.nn_nodes << ' '
                      << trace.heuristic_score << ' '
                      << trace.heuristic_nodes << ' '
                      << trace.immediate_nn << ' '
                      << trace.immediate_heuristic << '\n';
        }

        for (const ReplayMove& replay_move : replay_moves) {
            if (replay_move.ply == target_ply) {
                chess::Move selected = find_legal_move(pos, replay_move.move);
                chess::Position after_selected = pos;
                after_selected.make_move(selected);
                std::cout << "\nafter selected " << replay_move.move << '\n';
                std::cout << "side_to_move " << stm_name(after_selected.side_to_move) << '\n';
                std::cout << "material_cp_white_minus_black " << material_cp(after_selected) << '\n';
                std::cout << "nn_eval_stm " << model.evaluate_cp_rounded(after_selected) << '\n';
                std::cout << "heuristic_eval_stm " << chess::evaluate_for_side_to_move(after_selected) << '\n';
                chess::HeuristicSearcherV6 heuristic_searcher;
                chess::NnSearcherV6 nn_searcher(model);
                const chess::SearchResult heuristic_reply = heuristic_searcher.search_best_move(after_selected, depth);
                const chess::SearchResult nn_reply = nn_searcher.search_best_move(after_selected, depth);
                std::cout << "heuristic_reply_d" << depth << ' '
                          << chess::move_to_string(heuristic_reply.best_move)
                          << " score " << heuristic_reply.score
                          << " nodes " << heuristic_reply.nodes << '\n';
                std::cout << "nn_reply_d" << depth << ' '
                          << chess::move_to_string(nn_reply.best_move)
                          << " score " << nn_reply.score
                          << " nodes " << nn_reply.nodes << '\n';
                for (const auto& [label, reply] : {
                         std::pair{"after_heuristic_reply", heuristic_reply},
                         std::pair{"after_nn_reply", nn_reply}
                     }) {
                    chess::Position after_reply = after_selected;
                    after_reply.make_move(reply.best_move);
                    std::cout << label << ' ' << chess::move_to_string(reply.best_move) << '\n';
                    std::cout << "side_to_move " << stm_name(after_reply.side_to_move) << '\n';
                    std::cout << "material_cp_white_minus_black " << material_cp(after_reply) << '\n';
                    std::cout << "nn_eval_stm " << model.evaluate_cp_rounded(after_reply) << '\n';
                    std::cout << "heuristic_eval_stm " << chess::evaluate_for_side_to_move(after_reply) << '\n';
                }
                break;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "trace_replay_ply: " << error.what() << '\n';
        return 1;
    }
}
