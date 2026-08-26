# Classical search milestone revalidation

Date: 2026-08-25

Status: **historical evidence reconstructed with caveats**

## Why this report exists

The repository contains many classical searcher versions, but the durable
documentation was uneven. This report separates the claims that can still be
verified from the ones whose original inputs are gone.

## Historical evidence

The README records a five-seed iterative depth-8 V7/V8/V9 run:

| Searcher | Nodes | Time |
|---|---:|---:|
| V7 | 102,355,555 | 180,788 ms |
| V8 | 102,806,141 | 177,956 ms |
| V9 | 100,725,973 | 173,799 ms |

These totals have no retained raw per-seed rows, machine identity, alternating
schedule, or uncertainty estimate. They describe an early experiment, not a
current performance baseline.

The tracked V19 note reports this final V19/V18 benchmark:

```text
v19_vs_v18_nodes=1
v19_vs_v18_time=0.464284
score_mismatches=0
move_mismatches=0
```

The exact command and code survive, but its default corpus
`data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl` does not. An
exact reconstruction is therefore impossible from a clean clone. Replacing the
input silently would produce a different experiment, so this result remains
historical evidence rather than a rerun.

## Clean-snapshot reconstruction: V32 vs V35

Source and build:

- Git commit: `634b2d481925abc4b5ebf6da9129ade2edf3a24f`
- snapshot: clean `git archive`
- compiler: AppleClang 21.0.0
- CMake: 4.2.0
- build: `Release`, `CHESS_ENABLE_LTO=ON`
- machine: MacBook Air, Apple M4, 10 cores, 16 GiB

Command, repeated ten times:

```sh
build-lto/benchmark_v32_v35 7 20 --iterative
```

Each invocation uses six fixed positions plus 20 deterministic random-walk
positions (seed `20260619`). V35 is always timed before V32, so the schedule is
not balanced against drift.

### Results

| Metric | Result |
|---|---:|
| Positions per run | 26 |
| Runs | 10 |
| V35 nodes per run | 23,294,908 |
| V32 nodes per run | 23,294,908 |
| Score mismatches | 0 |
| Best-move mismatches | 0 |
| Median V35/V32 time ratio | 0.98314 |
| Pooled V35/V32 time ratio | 0.97119 |
| Per-run ratio range | 0.91338-1.00243 |

The strong conclusion is **tree parity on this suite**. The timing suggests a
small V35 advantage, but the fixed order, large drift across repetitions and
lack of a paired randomized schedule make it inappropriate to quote as a
promotion-quality speedup.

## Clean test gate

All 69 registered CTests were built from the clean snapshot. Results:

- 64 passed.
- `heuristic_searcher_v21_tests`, `v22`, and `v23` reproducibly failed their
  historical `cold.score == warm.score` assertion.
- `nn_search_tests` and `nn_value_tests` could not load the legacy ignored
  `value_net_stream_100k` model/data assets from a clean archive.
- All current production-line tests passed, including V36, V37, V39, V40,
  V41 repetition, phase-quantized parity, `search_tests`, perft, SEE, castling,
  move generation and make/unmake.

An all-target build also exposed a separate build-system issue: several
optional TT-stat profiling executables are included in the default target while
their required TT-stat symbols are disabled. Test targets and production engine
targets build successfully; the optional-profile failure is not counted as an
engine test failure.

## Conclusion

The classical line has credible correctness continuity, but not a continuous
promotion-quality performance history. V19's original large speed claim should
remain labeled historical until its exact corpus is recovered. For future
classical work, use a versioned position suite, balanced A/B order, raw rows,
source/binary hashes and a declared statistical summary.

The ten reconstruction logs are retained in the tracked
[revalidation raw bundle](evidence/milestone_revalidation_20260825.tar.gz),
SHA-256
`c483ccebc01455b32fc7c378c5cf79f3bc7330278e140fb77ac60c40dedbfc90`.
The bundle also contains the full 69-test output and the passing 11-test
production-line gate.
