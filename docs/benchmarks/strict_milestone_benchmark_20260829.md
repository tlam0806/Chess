# Strict architecture-milestone fixed-depth benchmark

Date: 2026-08-29

Status: **Validated current-tree reconstruction**

## Conclusion

The selected versions all returned the same fixed-depth score at depths 6, 7,
and 8 on the 25-position corpus. At the primary depth 7, V35 used **7.520% of
V15's elapsed time**: 20.334 seconds instead of 270.413 seconds, a **13.30x
speedup**.

The cohort deliberately samples architecture boundaries rather than several
similar late Strict versions. It is still a reconstruction: all retained
searchers were compiled together against current shared primitives, so these
numbers are not timings from their original historical binaries.

## Selected architecture boundaries

| Version | Why it is in the series |
|---|---|
| V15 | First range-storing, exact-depth Strict baseline. |
| V19 | Bucketed range TT, cached king-safety context, and legality filtered while ordering. |
| V23 | First main-search split into noisy and quiet generation stages; quiet moves are not generated after an earlier proof cutoff. |
| V25 | Scored move lists switch from heap-backed vectors to fixed-capacity raw storage. |
| V27 | The copied-child search path is replaced by in-place make/unmake with RAII restoration. |
| V29 | Dedicated legal generators feed four callback stages: promotion, good capture, quiet, and bad capture. |
| V30 | `ScoredMove` is packed from 16 to 8 bytes and the move-ordering/search hot paths are rewritten. |
| V35 | Final range-correct Strict endpoint, including the killer stage and compact range-preserving TT/state layout. |

These labels identify the boundary represented by each checkpoint; the elapsed
change between two rows includes every change accumulated between them and is
not an isolated one-feature A/B.

## Protocol

- Workload: direct fixed-depth search at depths 6, 7, and 8 on 25 embedded
  FENs; corpus identity `fnv1a64:4e71582cee758c77`.
- State: a new searcher, 64 MiB TT, history, killer, and counter state for every
  timed `(depth, round, leg, FEN, version)` tuple. Construction and destruction
  are outside the timer.
- Schedule: eight complete Williams-balanced rounds. Each round has a forward
  leg and its exact reverse; every version occupies every slot twice and every
  ordered predecessor pair is balanced.
- Correctness: each depth begins with an unmeasured full-corpus preflight.
  Timed rows must match the V15 score at the requested depth, return a legal
  move, report the requested depth, and not stop early.
- Primary metric: pooled wall time of the in-process fixed-depth
  `search_best_move` call. Nodes and pooled NPS are explanatory diagnostics,
  not the promotion metric.
- Statistics: 50,000-replicate paired percentile bootstrap over the eight
  complete rounds, seed `20260829`; a 400-block `(round, leg, FEN)` bootstrap
  is retained as a sensitivity analysis.
- Build: Apple M4, Apple clang 21.0.0, ARM64 Release `-O3 -DNDEBUG`, ThinLTO.

Each depth ran as a separate process and artifact, with its own preflight.

## Primary result: depth 7

Elapsed ratios below 1.0 are better. Intervals are paired complete-round 95%
bootstrap intervals.

| Version | Pooled time | Time / V15 [95% CI] | Time / previous [95% CI] | Nodes | Pooled NPS |
|---|---:|---:|---:|---:|---:|
| V15 | 270.413 s | 1.0000 [1.0000, 1.0000] | — | 412,761,712 | 1.526M |
| V19 | 85.520 s | 0.3163 [0.3154, 0.3170] | 0.3163 [0.3154, 0.3170] | 477,666,976 | 5.585M |
| V23 | 67.769 s | 0.2506 [0.2501, 0.2511] | 0.7924 [0.7906, 0.7941] | 485,712,720 | 7.167M |
| V25 | 52.051 s | 0.1925 [0.1919, 0.1930] | 0.7681 [0.7668, 0.7693] | 402,033,696 | 7.724M |
| V27 | 49.096 s | 0.1816 [0.1809, 0.1823] | 0.9432 [0.9406, 0.9459] | 402,033,696 | 8.189M |
| V29 | 44.477 s | 0.1645 [0.1638, 0.1649] | 0.9059 [0.9031, 0.9083] | 435,992,336 | 9.803M |
| V30 | 21.069 s | 0.0779 [0.0776, 0.0781] | 0.4737 [0.4728, 0.4746] | 421,056,912 | 19.985M |
| V35 | 20.334 s | 0.0752 [0.0749, 0.0754] | 0.9651 [0.9635, 0.9665] | 446,454,896 | 21.956M |

V15 to V19 is the largest early step. V29 to V30 is the largest later step,
cutting time by 52.63%. V25 and V27 visited exactly the same nodes at all three
depths; V27's 5.68% depth-7 reduction is therefore the cleanest per-node
throughput comparison in this selected series. V35 searched 6.03% more nodes
than V30 but still finished 3.49% sooner because its pooled NPS was higher.

## Depth sensitivity

The values below are speedup versus V15 at the same depth. Higher is better.

| Version | Depth 6 | Depth 7 | Depth 8 |
|---|---:|---:|---:|
| V15 | 1.000x | 1.000x | 1.000x |
| V19 | 3.293x | 3.162x | 3.851x |
| V23 | 4.577x | 3.990x | 4.950x |
| V25 | 5.665x | 5.195x | 6.398x |
| V27 | 5.975x | 5.508x | 6.755x |
| V29 | 6.976x | 6.080x | 8.227x |
| V30 | 13.380x | 12.835x | 15.666x |
| V35 | 14.857x | 13.299x | 17.670x |

The relative progression is consistent at all three depths. For every version
and depth, the second-half versus first-half pooled-time change stayed within
0.85%; no late thermal drift like the earlier depth-8 run is visible here.

## Correctness result

The three runs contain 9,600 timed searches in total:

- zero score mismatches;
- zero illegal moves;
- zero stopped searches;
- zero reported-depth mismatches;
- deterministic score and node count for every `(depth, version, FEN)` tuple.

There were 144 differing-move rows versus V15. Repetition across the balanced
schedule reduces these to nine `(depth, version, FEN)` signatures and six
unique forced-child cases. Fresh V15 searches of each alternate child at
depth minus one verified **6/6** as realizing the shared root score, with zero
failures. They are tied best moves, not Strict score failures.

## Evidence and provenance

The benchmark executable was built in a clean external tree from Git base
`eb3e23807d7a184a2e75d29458aaeeb0dd85578b`, with only the v3 benchmark
harness, analyzer, and analyzer test overlaid. It was not built from the dirty
workspace. The retained source archive is the exact build source; the engine
implementations nevertheless share current primitives, which is why the result
remains a current-tree reconstruction.

The tracked [evidence bundle](evidence/strict_milestones_20260829.tar.gz)
contains all raw rows, empty stderr logs, analysis, exact source snapshot,
binary, static library, build metadata, analyzer/tests, move-tie audit, and a
checksum manifest.

- Bundle size: 6,514,394 bytes.
- Bundle SHA-256:
  `d14f5a69dc13feaf07f00fac8c2091636e3bacba6ab4f25b01c2f4c850a8efb4`.
- Benchmark harness SHA-256:
  `c705f413e61d674b55cd1196853de79056773268cdb930f2d0c82856483f8a00`.
- Benchmark binary SHA-256:
  `51d248ee42005bf8bd28e9fe4175e1dc6ed733c9fc6f172608aa6c1cae90a7e0`.
- Analyzer SHA-256:
  `b2c5bb977c7702af0fcf8b24fb99096716714ecd974ce89a65fa4aa0b112b943`.
- Raw D6/D7/D8 SHA-256:
  `64368925...`, `0b493b92...`, and `994b5051...`.

The analyzer passed seven tests, and a byte-identical rerun reproduced the
tracked analysis outputs. NDJSON metadata identifies the v3 protocol and exact
cohort but does not itself embed the Git or binary hash; source-to-log binding
therefore relies on the retained run directory and checksum manifest.

## Interpretation limits

- This is one Apple M4 host and one 25-FEN corpus; the intervals are
  conditional timing intervals, not Elo or population-wide uncertainty.
- Equal fixed-depth scores on this corpus support the Strict contract for
  these cases; they are not a proof over every legal chess position.
- Fresh state is intentional for controlled comparison but differs from a
  persistent game process that carries search heuristics between moves.
- The architecture labels explain why checkpoints were selected; they do not
  causally isolate each feature from other accumulated changes.
