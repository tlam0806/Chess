# Retained search experiments

This directory preserves historical search implementations that are still used
to reproduce architecture milestones, compare fixed-depth behavior, and run
regression tests. They are evidence-bearing project history rather than the
current engine line.

| Path | Contents |
|---|---|
| `include/searchers/fast`, `src/searchers/fast` | Early performance-oriented classical and neural searchers, including the relaxed V34 TT experiment. |
| `include/searchers/strict`, `src/searchers/strict` | Classical Strict V15-V35 and NNUE V36-V42 milestone implementations. |

The current V43 production searcher and V44 candidate remain under
`include/searchers/strict` and `src/searchers/strict`. Historical sources are
still compiled into `chess_core`, so existing tools, benchmarks, tests, and
target names continue to work unchanged. A later cleanup may make that history
an optional build, but this move intentionally does not change build behavior.
