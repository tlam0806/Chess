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
`include/searchers/strict` and `src/searchers/strict`. The default build compiles
only V43. Enable `CHESS_BUILD_EXPERIMENTS` to compile the retained history, V44,
research tools, benchmarks, and comparison tests while preserving their target
names:

```sh
cmake -S . -B build-experiments -DCHESS_BUILD_EXPERIMENTS=ON
```

The TT timing, probe-path, and cache-line profilers each require the matching
`CHESS_TT_PROFILE_MODE` described in [`tools/README.md`](../tools/README.md).
