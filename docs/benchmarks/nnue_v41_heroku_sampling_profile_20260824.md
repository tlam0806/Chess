# NNUE V41 Heroku Sampling Profile — 2026-08-24

## Result

Status: **valid current-release measurement** on one Heroku Basic one-off
dyno. This is a hotspot profile of the deployed V41 binary, not a fleet-wide
Heroku performance claim.

Historical scope: “current release” here means scalar release v10 at the time
of measurement. After SIMD, the same hotspot fell from 89.15% to 24.50% of
sampled CPU; see the
[post-SIMD and LTO report](nnue_v41_post_simd_and_lto_20260825.md).

The binary was the one deployed in Heroku release v10. That release was built
from a staged dirty working tree rather than a clean repository commit, so the
profile is reproducible and auditable at the archived-binary level, not from a
clean source checkout.

The current x86 release spends **89.15% of sampled CPU time** inside
`PhaseQuantizedNnueModel::forward_positional_scalar`. The obvious next
optimization is therefore an x86 SIMD NNUE forward backend (VNNI first), then
a repeat of this profile. Optimizing general search code first would target a
much smaller part of the current runtime.

## Identity

| Artifact | SHA-256 |
|---|---|
| Deployed `uci_nnue_v41` | `105f9bccb6e6d518232d75319e5ae9c590b639d8f5eb37302b73edc3870604c6` |
| Production NNUE model | `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02` |
| Deployed V41 config | `43e9f2e3185e11a5dd58c50997a5a768645243e0d114feed3b59289294a0d22a` |
| Canonical suite | `b297dfe39c869e2f3eca07175f57e7302c4c29257f04d75493a70701648ed67e` |
| Remote profiling harness | `9c5bc7c0a4c7d4a01f85c394f85787b7efcbd0944661a2454e0cc2aaa217a8c4` |
| `libprofiler.so.0` | `d0267bebfdc0f35a8cc93dc28fa38aeb36edc57f96122ba20a671f53b0832936` |

Machine: Intel Xeon Platinum 8375C at 2.90 GHz, Heroku Basic one-off dyno
`worker.2762`. Linux exposed `perf_event_paranoid=4`, so kernel `perf` was not
available to the unprivileged dyno.

## Protocol

- Profiler: gperftools 2.16 `libprofiler`, injected with `LD_PRELOAD` as
  supported by the [gperftools CPU profiler documentation](https://gperftools.github.io/gperftools/cpuprofile.html).
- Timer: `ITIMER_PROF` at 100 Hz. It samples scheduled CPU time rather than
  elapsed wall time.
- Scope: `CPUPROFILESIGNAL=12` toggled sampling immediately before `go depth
  8` and immediately after `bestmove`. Model loading and UCI startup were not
  sampled.
- Workload: all 12 positions and all production UCI options from
  `benchmarks/uci_platform_v1.json`.
- Isolation: a fresh engine process for every position, avoiding the known UCI
  history-state leak between positions.
- Schedule: four A-B-B-A blocks, where A is a control with `libprofiler`
  preloaded but sampling disabled, and B enables sampling only during search.
- Total: 16 complete rounds, 192 fixed-depth observations, 96 raw profiles and
  eight profiled full-suite rounds.

Every round searched exactly **1,698,311 nodes**. All repeated tuples
`(depth, score type/value, nodes, bestmove)` were identical for each position,
and all 12 position signatures matched the earlier canonical Heroku depth-8
baseline.

The exact run harness hashed the deployed config but did not parse it back and
compare its options to the suite. For this result, the config, engine, model
and suite hashes all match the prior validated canonical run, and all 12 search
signatures cross-match it. The reusable harness was subsequently tightened to
hard-fail on config/suite option drift.

## Flat CPU hotspots

The merged profile contains 68.92 sampled CPU-seconds, equivalent to about
6,892 samples at 100 Hz.

| Function (self/flat time) | CPU time | Share |
|---|---:|---:|
| `PhaseQuantizedNnueModel::forward_positional_scalar` | 61.44 s | **89.15%** |
| `PhaseQuantizedNnueAccumulator::rebuild_perspective` | 0.98 s | 1.42% |
| `LowerMoveRangeBucketTranspositionTable::probe` | 0.78 s | 1.13% |
| `PhaseQuantizedNnueModel::evaluate` (self) | 0.72 s | 1.04% |
| `PhaseQuantizedNnueAccumulator::update_features` | 0.68 s | 0.99% |
| `NnueSearcherV38::negamax` (self) | 0.39 s | 0.57% |
| `update_king_safety_after_move` | 0.24 s | 0.35% |
| Legal non-promotion capture generation | 0.22 s | 0.32% |
| Accumulator `push_state` | 0.21 s | 0.30% |
| `rook_attacks` | 0.21 s | 0.30% |
| `NnueSearcherV38::quiescence` (self) | 0.20 s | 0.29% |
| Accumulator `make_move_with_undo` (self) | 0.20 s | 0.29% |

Across the eight profiled rounds, scalar-forward share ranged from 87.09% to
91.15%. A 10,000-replicate bootstrap over complete rounds gives a descriptive
95% interval of **88.10%–90.15%**, conditional on this dyno and fixed suite.

The depth-8 suite is deliberately compute-weighted: `queenside_pressure`
accounts for roughly 40% of sampled CPU time. Removing that position leaves
41.50 sampled seconds, with scalar forward still at **88.70%**. The main
conclusion is therefore not caused by that single FEN.

Known NNUE evaluation/state helpers together add roughly four percentage
points beyond the scalar forward itself, so clearly identifiable NNUE work is
about 93% of current flat CPU time. This grouping is descriptive; the raw
function table remains the source of truth.

## Sampler and host checks

| Metric | Control | Sampling enabled |
|---|---:|---:|
| Median round NPS | 199,123 | 201,160 |
| Median process CPU / wall | 1.0001 | 0.9988 |

The median-round NPS metric was 1.02% higher with sampling, while pooled NPS
was 0.12% lower. The four A-B-B-A block deltas were +0.41%, +1.44%, +3.10%
and -5.20%. Sampler overhead is therefore not distinguishable from host/run
noise. Profiled NPS must not replace the canonical throughput benchmark.

CPU/wall stayed near 1.0 in both modes. This one-off dyno was almost
continuously scheduled during the measured searches, so the run contains no
evidence of significant scheduler stalls or descheduling. This metric cannot
detect reduced CPU frequency. Other dynos and times may differ.

## Interpretation and next action

1. The immediate Heroku bottleneck is the scalar x86 NNUE forward, not the
   alpha-beta control flow, move generation or transposition table.
2. Implement bit-exact runtime-dispatched VNNI, retaining scalar as the oracle
   and fallback.
3. Gate the new backend with exact evaluation tests and the same fixed-depth
   signatures.
4. Benchmark scalar versus VNNI on the same dyno, then repeat this sampling
   profile. Only that post-VNNI profile can rank accumulator, TT, movegen and
   other search optimizations fairly.

## Limitations

- This is one Basic dyno and one fixed workload. Percentages are conditional,
  not a Heroku fleet confidence interval.
- The deployed release is optimized and lacks DWARF debug lines. Function
  symbols were preserved and resolved, but line-level output is not meaningful.
- The exact ELF is archived, but release v10 cannot currently be rebuilt from
  a clean Git commit because its source snapshot came from a dirty worktree.
- Linux shared-library binaries were not copied out of the dyno. External
  libc/libm frames may therefore be incompletely named; all engine symbols in
  the reported hotspots were resolved.
- CPU sampling cannot see time when Heroku does not schedule the dyno. The
  separate process-CPU/wall ratio covers that concern for this run only.

## Reproduction

The reusable pipeline is:

```sh
GOBIN=/tmp/chess-pprof-bin go install \
  github.com/google/pprof@v0.0.0-20260802141513-ef3492d7dac3
python3 tools/benchmark/run_heroku_searcher_profile.py \
  --blocks 4 \
  --pprof /tmp/chess-pprof-bin/pprof
```

The raw local artifact for this run is
`logs/nnue_v41_heroku_sampling_20260824T120239Z/`. It includes the exact ELF
binary, 96 raw profiles, observations, manifest, suite and harness snapshots,
the merged pprof profile, and complete flat reports. The `logs/` directory is
ignored by Git; keep or archive this artifact separately if the report must be
reproducible from a clean clone.
