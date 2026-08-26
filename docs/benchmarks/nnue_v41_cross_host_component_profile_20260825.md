# V41 LTO cross-host component profile — 2026-08-25

## Status and conclusion

**Lifecycle:** production diagnostic; no engine or deployment change.

**Evidence status:** integrity-validated relative-share diagnostic. Component
sample shares are usable, but absolute Mac CPU time, component `ns/node`, and
the profile-derived total host ratio are conditional because the local control
runs experienced severe thermal/frequency drift.

The useful answer is nevertheless clear:

1. **Transposition-table work is the largest relative x86 disadvantage.** It
   used 10.50% of sampled search CPU on Heroku versus 2.89% on the M4. The
   conditional same-run estimate is 5.80x slower per node, with a descriptive
   95% interval of 4.92-6.95x.
2. **NNUE accumulator work and NNUE forward dominate on both hosts.** The
   leaf-only classification assigns them 55.07% of Heroku samples and 51.70%
   of local samples. Caller-informed reassignment of local `_platform_memmove`
   raises the local total to 55.28%, so the exact split between forward and
   accumulator is compiler-symbol-boundary sensitive. They should be treated
   as a joint priority.
3. **Move generation and attacks are the next clear cross-host gap.** Their
   conditional estimate is 1.91x, interval 1.56-2.37x.
4. SEE and search-control differences are inconclusive. Runtime/library
   symbols are not directly comparable across macOS and Linux.

This is a cross-host CPU/ISA/compiler/environment comparison, **not a pure
measurement of “Heroku overhead.”**

## Question and metric

The earlier canonical benchmark established that the clean V41 LTO build
searched identical trees at about 4.94M NPS on the M4 and 1.58M NPS on Heroku
at depth 8. This run asks where scheduled search CPU time was spent on each
host.

Symbols were normalized into ten mutually exclusive semantic buckets. Raw
sample share answers “where does this host spend on-CPU time?” A secondary
estimate converts each share to CPU nanoseconds per node using the control
runs surrounding each profiled pair:

```text
component CPU ns/node = control CPU ns/node x component sample share
```

Because the local controls were unstable, the converted values are reported
only as a sensitivity analysis. The raw shares and hotspot ordering are the
primary result.

## Pinned workload

Both hosts used the production V41 LTO workload:

- source snapshot intended: clean `634b2d481925abc4b5ebf6da9129ade2edf3a24f`;
- engine: `ChessNNUEV41`;
- model SHA-256:
  `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`;
- suite SHA-256:
  `b297dfe39c869e2f3eca07175f57e7302c4c29257f04d75493a70701648ed67e`;
- the same 16 production UCI options;
- 12 canonical FENs at fixed depth 8;
- 16 complete `A-B-B-A` blocks per host, where `A` is an unprofiled control
  and `B` is a profiled round;
- one fresh engine process for every position and round;
- exactly 1,698,311 nodes in every complete round.

The local manifest did not embed the Git commit even though it used the clean
snapshot path. Binary-level provenance is authoritative: local ARM64 engine
SHA-256 `ecd723c23e845ecbf27660a25615945201a459499f32bb1a9d6bc46219a9b73b`,
Heroku x86-64 engine SHA-256
`2b9bedd073bc6d15464567cc9a716a1b7e0b3f7227ed7b293b9f9bb2b7ba8128`.

## Sampling protocol

### Local M4

The local run used Xcode Time Profiler at its 1 ms sampling interval. One
all-process recorder covered each adjacent `B-B` pair. Within that recorder,
24 fresh engine processes were created and searched sequentially. A sample was
accepted only when both conditions held:

- its process ID was one of the captured engine PIDs; and
- its resolved stack contained the exact
  `NnueSearcherV38::search_best_move_impl` search root.

This structurally removed model loading, UCI startup, idle time, and shutdown
from the search profile. The run used `--no-checkpoint`, as requested.

### Heroku

The Heroku LTO run used `gperftools` `ITIMER_PROF` sampling at 100 Hz. Each
fresh engine process toggled profiling immediately before `go depth 8` and
stopped at `bestmove`. The profiler measured scheduled CPU time, not time when
the dyno was descheduled.

The two profiler formats were mapped to the same exclusive semantic buckets.
Bootstrap intervals independently resampled the 16 complete ABBA blocks on
each host, 10,000 replicates, seed `20260825`. They are conditional on this
fixed suite and these two observed hosts.

## Integrity gates

| Gate | Result |
|---|---:|
| Observations | 768 local + 768 Heroku |
| Complete rounds | 64 per host: 32 control + 32 profile |
| Profiled engine processes | 384 per host |
| Cross-host signature mismatches | 0 of 12 positions |
| Nodes per complete round | 1,698,311 on both hosts |
| Local missing/duplicate target PID | 0 |
| Local PID/thread-owner mismatch | 0 |
| Local selected samples without a stack | 0 |
| Local selected search sample weight | 15.316 s |
| Local aggregate sampled/search-CPU coverage | 99.46% |
| Heroku profiler/control CPU ratio | 1.0027 |

One local `development` case in round 17 retained only 3 ms of 22.87 ms
process CPU after the strict root-stack filter. This was the only material
coverage outlier; at most about 0.13% of aggregate sample weight is missing,
and removing it changes any component share by no more than about 0.084
percentage point. The manifest therefore correctly records that the target
coverage gate was not universally met even though aggregate coverage is high.

All 12 fixed-depth `(depth, score, nodes, bestmove)` signatures were constant
across 1,536 observations and identical between ARM64 and x86-64.

## Primary result: sampled search-CPU composition

| Component | M4 raw share | Heroku raw share | Conditional Heroku/M4 | Conditional 95% interval | Conditional gap share | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| NNUE forward | 36.08% | 32.04% | 1.40x | 1.21-1.65x | 25.1% | Material on both hosts |
| NNUE accumulator | 15.62%* | 23.03% | 2.34x* | 2.02-2.79x* | 35.9%* | Leaf-only attribution understates local state-copy work |
| Transposition table | 2.89% | 10.50% | 5.80x | 4.92-6.95x | 23.6% | Clearest relative x86 disadvantage |
| Move generation/attacks | 9.08% | 11.01% | 1.91x | 1.56-2.37x | 14.2% | Clear next target |
| Make/unmake + king safety | 7.58% | 6.88% | 1.42x | 1.21-1.69x | 5.6% | Smaller but supported gap |
| Move ordering | 4.93% | 3.80% | 1.21x | 1.01-1.47x | 1.8% | Marginal |
| SEE | 5.56% | 3.95% | 1.11x | 0.90-1.37x | 1.1% | Inconclusive |
| Search control | 13.99% | 8.29% | 0.94x | 0.77-1.15x | -1.5% | Inconclusive |
| Runtime/library | 4.10% | 0.51% | Not comparable | — | — | Different OS/runtime symbols |
| Other | 0.17% | ~0% | Not meaningful | — | — | Negligible |

“Gap share” allocates the conditional total component-time difference. It is
not a canonical performance decomposition because its denominator inherits
the unstable local timing calibration.

### Compiler symbol-boundary sensitivity

The primary table is exactly reproducible from flat leaf symbols, but a leaf
does not always name the logical caller. On macOS, `_platform_memmove` accounts
for 549 ms, or 3.58 percentage points of all retained search samples. In the
two native traces kept for caller audit, 43 of 44 search-root `memmove` samples
were directly below `PhaseQuantizedNnueAccumulator::make_move_with_undo`, whose
`push_state()` copies the NNUE state.

A declared sensitivity run reclassified all local `_platform_memmove` leaves
from runtime to accumulator while leaving Heroku unchanged; Heroku had no
visible flat `memmove` leaf in its round reports. The result was:

- local accumulator share: 19.20% instead of 15.62%;
- accumulator conditional ratio: 1.91x instead of 2.34x, descriptive 95%
  interval 1.64-2.26x;
- accumulator conditional gap share: 29.8% instead of 35.9%, interval
  26.2-34.5%;
- local forward plus accumulator: 55.28%, essentially equal to Heroku's
  55.07%.

The tracked
[caller audit](evidence/nnue_v41_cross_host_memmove_caller_audit_20260825.json)
preserves both trace-tree hashes, exported-XML hashes, exact immediate-caller
counts, and tool identities. Compact exports for 14 of 16 local sessions
retain flat leaves but not full ancestor stacks, so the caller-based
reassignment cannot be proven sample by sample for the entire run. These are
therefore two declared attribution scenarios—15.62% leaf-only and 19.20% with
`_platform_memmove` reassigned—not lower and upper bounds. The retained trace
also contains one accumulator-called `memcpy` sample, and other compiler
boundary leakage is possible. TT remains the clearest relative disadvantage,
while accumulator/forward remains the dominant joint target.

The round-level bootstrap sensitivity produced the same qualitative findings:
TT, accumulator, NNUE forward, move generation, and make/unmake remained above
1.0 at the lower 95% bound; SEE and search control remained inconclusive.

## Why absolute timing is conditional

The local ABBA controls did not behave like a stable benchmark:

- block control CPU cost ranged from 209.2 to 518.0 ns/node;
- the block-level coefficient of variation was 29.8%;
- first-half control median was about 214 ns/node and second-half median about
  408 ns/node;
- profiled `B` rounds appeared 16.65% faster than surrounding controls, and the
  second `B` was faster than the first in 15 of 16 blocks.

A profiler cannot legitimately make the same engine 16-18% faster. The pattern
is consistent with thermal/frequency drift on a charging fanless MacBook Air
and possibly a scheduler/P-core selection effect, although no temperature or
frequency counter was captured. Bootstrap resampling cannot repair this
systematic drift.

The exact run harness compared signed “overhead” only against a positive upper
limit, so the manifest incorrectly labels this large negative distortion as a
passed overhead gate. That is a harness-status bug, not an artifact-integrity
failure: observations, samples, signatures, and timing values remain intact.
The current harness now gates on the absolute profile/control timing
distortion. The archived harness snapshot is retained so this distinction is
auditable.

The ABBA-calibrated totals were 339.94 local versus 537.48 Heroku CPU ns/node,
or 1.581x with a descriptive interval of 1.384-1.840x. Those values are
retained for sensitivity only and must not replace the clean canonical
fixed-depth result of 3.12-3.19x higher local NPS.

## Canonical wall-time-anchored component table

The closest available answer to “real time per component” combines the
unprofiled canonical depth-8 wall times with the caller-informed sample shares:

- M4 total: 4,936,086.9 NPS, or 202.59 ns/node and 344.06 ms for the canonical
  12-position round;
- Heroku total: 1,581,861.2 NPS, or 632.17 ns/node and 1,073.62 ms for the same
  exact 1,698,311-node round;
- component time: canonical total wall time multiplied by that host's sampled
  component share.

One `ns/node` is numerically equal to one millisecond per million nodes.

| Component | M4 ns/node | Heroku ns/node | Heroku/M4 | M4 ms/round | Heroku ms/round |
|---|---:|---:|---:|---:|---:|
| NNUE forward | 73.09 | 202.56 | 2.77x | 124.1 | 344.0 |
| NNUE accumulator + `memmove` | 38.90 | 145.57 | 3.74x | 66.1 | 247.2 |
| Transposition table | 5.86 | 66.38 | 11.33x | 10.0 | 112.7 |
| Move generation/attacks | 18.39 | 69.58 | 3.78x | 31.2 | 118.2 |
| Make/unmake + king safety | 15.36 | 43.49 | 2.83x | 26.1 | 73.9 |
| Search control | 28.35 | 52.41 | 1.85x | 48.1 | 89.0 |
| Move ordering | 9.99 | 24.03 | 2.41x | 17.0 | 40.8 |
| SEE | 11.27 | 24.95 | 2.21x | 19.1 | 42.4 |
| Runtime/library | 1.04 | 3.20 | 3.07x | 1.8 | 5.4 |
| Other | 0.34 | ~0 | Not meaningful | 0.6 | ~0 |
| **Total** | **202.59** | **632.17** | **3.12x** | **344.1** | **1,073.6** |

This table uses actual measured total wall time, but individual rows remain
sampling estimates rather than direct per-function timers. The component
shares and totals came from separate runs, and different native profilers were
used on each OS. No confidence interval is assigned to the hybrid rows.

Under this scaling, the estimated excess time per canonical round is about
220 ms in forward, 181 ms in accumulator state work, 103 ms in TT, and 87 ms
in move generation. Those four groups explain most of the observed 730 ms
round-level gap.

## Engineering decision

By recoverable wall time, the next optimization work should be measured in
this order:

1. **NNUE forward:** VNNI removed the old 89% scalar bottleneck, but forward is
   still 32.04% of Heroku CPU and the largest estimated absolute time gap.
2. **Accumulator state/update:** profile the roughly 1.1 KiB state copy,
   `update_features`, and rebuild paths; test explicit AVX2/AVX-512 kernels or
   a less copy-heavy layout.
3. **TT microarchitecture:** benchmark probe/store separately on x86, inspect
   cache-line alignment, bucket layout, and code generation. TT has the
   largest relative disadvantage: 10.50% of Heroku search CPU versus 2.89%
   locally.
4. **Move generation/attacks:** test x86-specific inlining, POPCNT/BMI paths,
   table/cache behavior, and targeted multiversioning.

Do not optimize SEE or generic search control first from this evidence; their
cross-host differences do not exclude parity.

Before using component `ns/node` for a promotion gate, rerun the local profile
as several short sessions after reaching thermal steady state. Require
two-sided profile/control equivalence within 5%, stable first-versus-second
half timing, and control CV no more than roughly 3-5%.

## Evidence and reproduction

The compact machine-readable result is
[the evidence manifest](evidence/nnue_v41_cross_host_component_profile_20260825.json).
The 426,914-byte tracked
[evidence bundle](evidence/nnue_v41_cross_host_profile_20260825.tar.gz) has
SHA-256
`573a9e64226e930ef2db0923eb457d5627592280379703c497a0f4cfd6587ccf`.
It contains both manifests and observations, all 16 compact local Xcode
exports, 32 per-round Heroku `pprof` reports, the exact local run harness,
suite/config snapshots, the analyzer and its unit tests, three analysis
variants plus the caller-informed `memmove` sensitivity, and the independent
local canonical anchor. It also includes the compact caller-audit record; the
Heroku LTO summary used for the wall-time table is included as well. The two
native trace trees remain local because they total about 56 MB.

Only two bulky macOS `.trace` bundles were retained locally; they are not in
the tracked archive. The compact exports for all 16 sessions are retained.
The ignored Heroku raw profiler files are represented in the tracked bundle by
their verified per-round flat reports.

Key artifact hashes:

| Artifact | SHA-256 |
|---|---|
| Local manifest | `07169bafe1a303c68934e8173ac5bc16e006e1d4805cfba8ae1a1efa9d704831` |
| Local observations | `5b236b75fb952db5b77f192bdeade6c5e8d13f3ffa58a76319e530f2c7130785` |
| Local normalized rounds | `14fb6cc6372e6b7f441bf8a620b01799b4d14215e9ca2d626ff18299c95a8720` |
| Exact local run harness snapshot | `828289cbb74f291abb5b738f2eaf0c113ffb8a8e830e55fce03d810007b10d72` |
| Heroku manifest | `ba19226bc989ea27431f4f9c7ee59b91f195960d7b90d7ed131bc59a40afee79` |
| Heroku observations | `d5846f253fffea6ae738db40f46e251c2e895bb18e28ac8bf84e02d2e1a4dadf` |
| ABBA-control analysis | `fbbd0d84be7c0f73ad8d926ed745f8e9fce187520242f43c0e4cf19fc7388ca1` |
| Profile-measured sensitivity | `70674689d799fbaee487091861aa8c79822c345b73239448561a117dca0d354d` |
| Complete-round sensitivity | `35cba5c9a71184bc9f1651b2760e6d1a99d493e5f2783fad2cbaa370366a1c7c` |
| Caller-informed `memmove` sensitivity | `9370962817c46ebfdb41e785b5c4704aa785dfea3a9de6d644846bf61e230353` |
| `memmove` caller audit | `3ec700b1f6213133eb0156fa500149b6c4740a5496c31366d3981faf9c1cd30e` |
| Independent local canonical run | `a0295f0e393947f3a18404b82c2a4a0999177a7f36ebedf690a1acdd1fc429a6` |
| Heroku LTO benchmark summary | `cf15a677f94c43eb451ee09f7223ee395387e3fe1e97c20edacdba8bff2db42b` |
| Cross-host analyzer snapshot | `4ecbf086edec1eecf3bc616143e4863b6babe694b2c4ad8f5a018683d22b9e40` |
| Analyzer unit tests | `85facf020d044a531eccddaca43439be039d6f866fcacce4c6b4386be60ee59a` |
