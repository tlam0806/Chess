# NNUE V41 Heroku platform baseline — 2026-08-23

## Outcome

Status: **validated Heroku baseline**. This is not yet a Heroku/local speed
comparison; the matching local run must wait for an idle machine.

Follow-up: the matching canonical local run is now documented in
[V41 canonical local Mac versus Heroku](nnue_v41_local_vs_heroku_20260825.md).

The production V41 image was measured on three independent Heroku Basic
one-off dynos. All 1,080 observations completed, all fixed-depth search
signatures matched across rounds and dynos, and no timeout or movetime warning
was recorded. The sole run-level warning is that release V10 does not report
the selected NNUE kernel.

| Mode | Limit | Aggregate NPS | Descriptive 95% resampling interval | Game-like result |
|---|---:|---:|---:|---|
| Fixed depth | 7 | 199,671 | [190,422, 207,707] | — |
| Fixed depth | 8 | 203,023 | [191,432, 212,475] | — |
| Movetime | 1,000 ms | 237,948 | [228,361, 245,954] | Median depth 9; 222,208 median nodes |
| Movetime | 3,000 ms | 246,625 | [234,186, 256,428] | Median depth 10; 698,880 median nodes |

NPS from different modes should not be compared as if they were the same
workload. Fixed-depth is the deterministic cross-host comparison gate;
movetime describes the production-like amount of work completed under a wall
clock limit.

## Measured release

- Heroku app: `stormy-garden-92984`
- Release: `v10`; staged Git identifier
  `db2966a287cf8deb76f69515dc2892a1c77bf9da`
- Dyno: three sequential `Basic` one-off dynos; the production worker was not
  restarted or scaled
- Runtime CPU in all three dynos: Intel Xeon Platinum 8375C, `x86_64`
- Runtime memory cgroup limit: 536,870,912 bytes (512 MiB)
- Engine: `ChessNNUEV41`
- Engine SHA-256:
  `105f9bccb6e6d518232d75319e5ae9c590b639d8f5eb37302b73edc3870604c6`
- Model SHA-256:
  `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`
- Runtime config SHA-256:
  `43e9f2e3185e11a5dd58c50997a5a768645243e0d114feed3b59289294a0d22a`
- Benchmark spec SHA-256:
  `b297dfe39c869e2f3eca07175f57e7302c4c29257f04d75493a70701648ed67e`
- Effective full-run spec SHA-256:
  `e452c0e0adb6a82f9555f02a8e16c61d5380ab3ba16312f4983f4e4609b89e49`
- Harness SHA-256:
  `f6b0d01c4258dcffececcf7c5590940dd25cc554bedeca324563be164e288f4e`
- Heroku runner SHA-256:
  `18471a2a5f870373d4893d41ca11987394fd208531b6b1c489a4b98792dbe4f6`

The Heroku release was built from a staged dirty working tree. The remote
binary hash is authoritative, but source-level reproducibility remains partial
until V41 and its deployment manifest are committed.

The exact harness, runner and spec streamed for this run were snapshotted before
the first dyno started. All three dynos reported the same non-null hashes.

## Protocol

The reusable suite is defined in
[`benchmarks/uci_platform_v1.json`](../../benchmarks/uci_platform_v1.json), and
the driver is
[`tools/benchmark_uci_platform.py`](../../tools/benchmark_uci_platform.py).

- 12 canonical FENs spanning opening, middlegame, tactics and endgames
- Production V41 UCI options applied explicitly
- One unmeasured depth-5 warm-up pass on each dyno
- Fixed depth 7 and 8: 10 complete rounds per dyno
- Movetime 1,000 and 3,000 ms: 5 complete rounds per dyno
- Rotate-then-reverse position order and interleaved mode order to balance
  early/late drift
- Timer placed inside the dyno from sending `go` to receiving `bestmove`
- NPS calculated as `sum(nodes) / sum(elapsed)`, not the arithmetic mean of
  per-position NPS
- Descriptive 95% interval from hierarchical resampling of dynos and complete
  rounds, 10,000 replicates, seed `20260823`. With only three dynos this is
  conditional on the sampled hosts and fixed suite, not a Heroku fleet-wide
  confidence claim.

The engine process is restarted for every measured position. This is required
for the deployed V41 adapter: `position` and `ucinewgame` clear the TT but do
not clear all history/counter-history state. Reusing a process changed fixed-
depth nodes and even the selected move, invalidating parity. Process isolation
measures the existing production binary without modifying or redeploying it;
engine startup and model loading occur outside the timed interval.

Process isolation also means each measured search begins with fresh engine
state and relatively cold search-code/cache state. That is intentional for a
fair deterministic Heroku/local ratio, but it is not identical to a live game
where one engine process survives across moves. The movetime rows should
therefore be read as controlled game-like probes, not as a complete end-to-end
Lichess latency measurement.

## Integrity and variability

- 3/3 dynos reported `valid`
- 1,080/1,080 expected observations present
- 0 fixed-depth mismatches in `(depth, score, nodes, bestmove)`
- 0 movetime wall-clock warnings
- Engine, model, config, suite, harness and runner hashes matched across all
  dynos
- Per-dyno fixed-depth NPS coefficient of variation was 1.6%–8.4%
- Movetime p95 overshoot was 15.1 ms at 1 second and 14.7 ms at 3 seconds
- Release V10 did not emit `nnue_kernel`; backend selection remains unresolved

Per-dyno aggregate fixed-depth throughput:

| Dyno | Depth 7 NPS | Depth 8 NPS |
|---:|---:|---:|
| 1 | 204,545 | 206,505 |
| 2 | 190,397 | 190,957 |
| 3 | 204,764 | 212,885 |

## Artifacts

Raw artifacts are retained locally under
`logs/uci_platform_full_heroku_20260823T143157Z/` and intentionally remain out
of Git because they contain 1,080 per-position observations and console logs.

| Artifact | SHA-256 |
|---|---|
| `heroku-run-1.json` | `473c42701a9cfec0d8d14080bbaf29d082f319b705dda94a05933c33c4cdd2ff` |
| `heroku-run-2.json` | `c509295a1c4e91a120222cb28146bac86ef856bc7b871bd0e8aea3fb32811780` |
| `heroku-run-3.json` | `b73a7799519415b19e9f738e45fd1d38bf37969c6976fe114177b6d5958758fe` |
| `summary.json` | `9f5df6597390c8c205a2635969170d49c2025f4a2a332d5f6fe69125909660ce` |
| `harness.snapshot.py` | `f6b0d01c4258dcffececcf7c5590940dd25cc554bedeca324563be164e288f4e` |
| `runner.snapshot.py` | `18471a2a5f870373d4893d41ca11987394fd208531b6b1c489a4b98792dbe4f6` |
| `spec.snapshot.json` | `b297dfe39c869e2f3eca07175f57e7302c4c29257f04d75493a70701648ed67e` |

## Interpretation and next gate

The release establishes an aggregate Heroku baseline near 200–203k NPS at
fixed depth. It also reveals meaningful shared-host variance: dyno 2 averaged
about 190k NPS, while the other dynos were mostly near 205–213k and individual
rounds fell as low as roughly 162k. It does not, by itself, prove how much
hosting slows the engine.

The next gate is the identical full suite on an idle local machine. The local
result may be compared only if every fixed-depth signature matches this
baseline. The Heroku/local ratio should then be calculated by independently
resampling complete local and Heroku rounds. Because this Heroku image is
`x86_64` and does not report its selected NNUE kernel, any claim that it uses
the scalar path remains a source-based inference until kernel selection is
exposed in the UCI handshake.
