#include "search_engine.hpp"

#include <cassert>
#include <string_view>

using namespace chess;

namespace {

void assert_searcher_matches_function(Searcher& searcher, const Position& pos, int depth) {
    const SearchResult from_class = searcher.search(pos, depth);
    const SearchResult from_named_method = searcher.search_best_move(pos, depth);

    assert(from_class.score == from_named_method.score);
    assert(from_class.nodes == from_named_method.nodes);
    assert(from_class.best_move == from_named_method.best_move);
}

void assert_searcher_v2_matches_v1(
    Searcher& searcher,
    HeuristicSearcher& baseline,
    const Position& pos,
    int depth
) {
    const SearchResult from_class = searcher.search(pos, depth);
    const SearchResult from_named_method = searcher.search_best_move(pos, depth);
    const SearchResult from_v1 = baseline.search_best_move(pos, depth);

    assert(from_class.score == from_v1.score);
    assert(from_class.best_move == from_v1.best_move);
    assert(from_class.nodes > 0);

    assert(from_named_method.score == from_v1.score);
    assert(from_named_method.best_move == from_v1.best_move);
    assert(from_named_method.nodes > 0);
}

} // namespace

int main() {
    HeuristicSearcher heuristic;
    HeuristicSearcherV2 heuristic_v2(1);

    assert(heuristic.name() == std::string_view("heuristic"));
    assert(heuristic_v2.name() == std::string_view("heuristic_v2"));
    assert(heuristic_v2.tt_entry_count() > 0);

    {
        Position pos;
        pos.set_startpos();

        Searcher& generic = heuristic;
        assert_searcher_matches_function(generic, pos, 3);

        Searcher& generic_v2 = heuristic_v2;
        heuristic_v2.clear_tt();
        assert_searcher_v2_matches_v1(generic_v2, heuristic, pos, 3);

        const SearchResult cold = generic_v2.search(pos, 3);
        const SearchResult warm = generic_v2.search(pos, 3);
        assert(cold.score == warm.score);
        assert(cold.best_move == warm.best_move);
        assert(warm.nodes <= cold.nodes);
    }

    {
        Position pos;
        assert(pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"));

        Searcher& generic = heuristic;
        assert_searcher_matches_function(generic, pos, 3);

        Searcher& generic_v2 = heuristic_v2;
        heuristic_v2.clear_tt();
        assert_searcher_v2_matches_v1(generic_v2, heuristic, pos, 3);
    }
}
