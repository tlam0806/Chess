# Strict milestone fixed-depth benchmark

Date: 2026-08-29

Status: **Validated current-tree reconstruction; depth-8 absolute timing is conditional**

## Conclusion

On the same current source snapshot, the retained Strict implementations all
returned the same score at depths 6, 7, and 8 on the 25-position corpus. The
final Strict V35 implementation needed only **6.866% of V15's depth-7 search
time**, a **14.56x speedup**, while returning the same fixed-depth scores.

The benchmark is a reconstruction of retained implementations linked against
the same current primitives. It is not a measurement of the original binary,
compiler, or machine used when each historical version was introduced.

## Protocol

- Versions: V15, V19, V24, V27, V29, V32, V33, and V35.
- Workload: direct fixed-depth search at depths 6, 7, and 8 on 25 embedded
  FENs; corpus identity `fnv1a64:4e71582cee758c77`.
- State: a new searcher, 64 MiB TT, history, killer, and counter state for every
  timed `(depth, round, leg, FEN, version)` tuple. Construction and destruction
  are outside the timer.
- Schedule: eight complete Williams-balanced rounds. Each round has a forward
  leg and its exact reverse. For every FEN, each version occupies every timing
  slot twice and every ordered predecessor pair is balanced.
- Correctness: each depth has an unmeasured full-corpus preflight. Timed rows
  must match the V15 score at the requested depth, return a legal move, report
  the requested depth, and not stop early.
- Primary metric: pooled `go` wall time. Nodes and pooled NPS are explanatory
  diagnostics, not the promotion metric.
- Statistics: 50,000-replicate paired percentile bootstrap over the eight
  complete rounds, seed `20260829`. A 400-block `(round, leg, FEN)` bootstrap
  is retained as a sensitivity analysis.
- Build: Apple M4, Apple clang 21.0.0, ARM64 Release `-O3 -DNDEBUG`, ThinLTO.

Each depth ran as a separate process and artifact. This preserves completed
depths if a later run is interrupted; it also means each file has its own
preflight immediately before its timed rows.

## Primary result: depth 7

Elapsed ratios below 1.0 are better. The interval is the paired complete-round
95% bootstrap interval.

| Version | Pooled time | Time / V15 [95% CI] | Time / previous [95% CI] | Nodes | Pooled NPS |
|---|---:|---:|---:|---:|---:|
| V15 | 268.322 s | 1.0000 [1.0000, 1.0000] | — | 412,761,712 | 1.538M |
| V19 | 83.615 s | 0.3116 [0.3114, 0.3119] | 0.3116 [0.3114, 0.3119] | 477,666,976 | 5.713M |
| V24 | 57.512 s | 0.2143 [0.2141, 0.2146] | 0.6878 [0.6869, 0.6888] | 393,023,536 | 6.834M |
| V27 | 47.003 s | 0.1752 [0.1751, 0.1753] | 0.8173 [0.8164, 0.8182] | 402,033,696 | 8.553M |
| V29 | 42.357 s | 0.1579 [0.1577, 0.1580] | 0.9012 [0.9003, 0.9021] | 435,992,336 | 10.293M |
| V32 | 18.373 s | 0.0685 [0.0684, 0.0686] | 0.4338 [0.4330, 0.4346] | 446,454,896 | 24.300M |
| V33 | 21.493 s | 0.0801 [0.0800, 0.0802] | 1.1698 [1.1676, 1.1726] | 489,489,456 | 22.774M |
| V35 | 18.424 s | 0.0687 [0.0686, 0.0688] | 0.8572 [0.8555, 0.8590] | 446,454,896 | 24.232M |

The largest steps in this selected series were V15 to V19, then V29 to V32.
V33 was a regression on this workload: it visited 9.64% more nodes than V32
and took 16.98% longer. V35 recovered that loss and returned to the same node
count as V32. V32 and V35 are close in raw time; this report does not claim a
standalone V35-over-V32 speed win.

## Depth sensitivity

The values below are speedup versus V15 at the same depth. Higher is better.

| Version | Depth 6 | Depth 7 | Depth 8 |
|---|---:|---:|---:|
| V15 | 1.000x | 1.000x | 1.000x |
| V19 | 3.315x | 3.209x | 3.756x |
| V24 | 4.981x | 4.666x | 5.059x |
| V27 | 6.186x | 5.709x | 6.782x |
| V29 | 7.172x | 6.335x | 8.234x |
| V32 | 16.175x | 14.604x | 18.381x |
| V33 | 13.869x | 12.484x | 15.925x |
| V35 | 15.925x | 14.564x | 18.295x |

Depth 7 is the primary timing result. Its complete-round totals were stable at
69.554–69.729 seconds, and the per-version second-half drift was between
-0.356% and +0.080%.

Depth 8 is a useful relative sensitivity check but not a canonical absolute
timing anchor. Rounds 0–5 averaged 334.124 seconds, round 6 was 1.86% slower,
and round 7 was 14.69% slower. Every version's second half slowed by
3.18–5.43%, consistent with thermal or frequency drift. Ratios relative to V15
were more stable, with relative half-run drift between -1.29% and +0.86%.

## Correctness result

The three runs contain 9,600 timed rows in total:

- zero score mismatches;
- zero illegal moves;
- zero stopped searches;
- zero reported-depth mismatches;
- deterministic score and node count for every `(depth, version, FEN)` tuple.

There was one best-move difference. On `tactical_castling`, V15 selected the
rook promotion `d7c8r`, while V32, V33, and V35 selected `d7c8q`. Fresh forced
child searches with V15 verified that both moves realize the shared root score
at depths 6, 7, and 8. It is therefore a verified tied move, not a Strict score
failure.

## Evidence and provenance

The exact run used Git HEAD `9fe6ac0c551007168da5c0d02c081d6439fdece3`
with a dirty working tree. The retained source snapshot, tracked diff, status,
build flags, executable, raw rows, analyzer, tests, tie verifier, and hashes are
stored together in the
[evidence bundle](evidence/strict_milestones_20260829.tar.gz).

- Bundle size: 874,544 bytes.
- Bundle SHA-256:
  `d12384e4e9c826830aea68f4ae6cafd3a8123ea12b3605c87646601eb17dc9ab`.
- Benchmark harness SHA-256:
  `21873304f65e4f8e330864043414472f3b51ed9b3a12772d5790510a58611285`.
- Benchmark binary SHA-256:
  `b4635b4dce9c576c326a099350ffd7ffa54dfce4ab13fe03f3ee1568b944bc7d`.
- Analyzer SHA-256:
  `4bae8ffe5240d25d08e6f23cd6a8b1c233de461cfa115fc377fad6eeaed6983e`.
- Raw D6/D7/D8 SHA-256:
  `09c78225...`, `515cfd12...`, and `44c8aa88...`.

The analyzer passed 7 tests, and an independent recomputation reproduced every
pooled total, ratio, drift value, and all 50,000 bootstrap interval endpoints.
The NDJSON metadata does not itself embed the Git or binary hash; the ex-post
source/binary provenance is chronologically consistent and fully retained, but
is not a cryptographic source-to-log binding.

## Interpretation limits

- This is one Apple M4 host and one 25-FEN corpus; the intervals are conditional
  timing intervals, not Elo or population-wide uncertainty.
- The retained implementations share current move generation, evaluation, and
  other common primitives. Do not label these bars authentic historical-release
  speed.
- Fresh state is intentional for controlled comparison but differs from a
  persistent game process that carries search heuristics between moves.
- Equal fixed-depth scores on this corpus support the Strict contract for these
  cases; they are not a proof over every legal chess position.
