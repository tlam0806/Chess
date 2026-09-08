# V19 search optimization notes

This note documents the V19 search changes and the validation used before
committing them at that point in the project.

## Searcher layout

- Retained Fast searchers and early NN searchers live under
  `experiments/include/searchers/fast` and `experiments/src/searchers/fast`.
- Retained Strict milestone searchers live under
  `experiments/include/searchers/strict` and
  `experiments/src/searchers/strict`.
- `HeuristicSearcherV19` was the Strict searcher used for this performance work.

## Move generation and legality

- Move generation now uses fixed-size `MoveList` storage instead of repeatedly
  allocating `std::vector<Move>`.
- V19 orders pseudo-legal moves and filters legality while scoring the move.
- `king_safety` contains shared board-level helpers:
  - `make_king_safety_context`
  - `is_pseudo_move_legal`
  - `gives_check_fast`
- The king safety context caches:
  - current king square
  - checking pieces
  - pinned pieces
  - legal block/capture mask when the side to move is in check
- En passant and castling keep explicit fallback/special handling for correctness.

## Move scoring

- `ScoredMove` stores the moved and captured piece type so later search code can
  call `make_move(move, moved_piece, captured_piece)` without re-querying the
  board.
- V19 uses `gives_check_fast` instead of copying the position and calling
  `in_check` only to detect whether a move gives check.

## SEE

- `static_exchange_eval` now has an overload that accepts the known moving and
  captured piece:

```cpp
int static_exchange_eval(
    const Position& pos,
    Move move,
    PieceType moving_piece,
    PieceType captured_piece
);
```

- The old `static_exchange_eval(pos, move)` API is preserved and delegates to the
  new overload.
- V19 calls the overload to avoid repeated piece lookup before SEE and inside the
  copied position's `make_move`.

## Validation

Commands run:

```sh
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake --build build --target see_tests heuristic_searcher_v19_tests benchmark_v18_v19_bucket_tt -j 8
/opt/homebrew/Cellar/cmake/4.2.0/bin/ctest --test-dir build -R 'see_tests|heuristic_searcher_v19_tests|castling_tests|movegen_regression_tests|perft_tests' --output-on-failure
./build/benchmark_v18_v19_bucket_tt --samples 32 --depth 7 --iterative --seed 20260615 --bucket-size 1
```

Observed benchmark result for the final V19 state:

```text
v19_vs_v18_nodes=1
v19_vs_v18_time=0.464284
score_mismatches=0
move_mismatches=0
```
