# Project milestone ledger

Updated: 2026-08-29

This is the chronological index for the chess-engine project. It separates two
questions which older notes often mixed together:

1. **Lifecycle**: was the change experimental, promoted, deployed, or rejected?
2. **Evidence quality**: was it validated, reconstructed later, or left
   incomplete?

The benchmark reports linked below contain the protocol, raw-artifact hashes,
limitations, and negative results. A lower loss, a faster offline search, and a
self-play promotion are deliberately treated as different kinds of evidence.
New promotion claims follow the
[benchmark protocol and evidence policy](../benchmarks/PROTOCOL.md).

## Status vocabulary

| Label | Meaning |
|---|---|
| **Validated** | The recorded protocol supports the stated conclusion and the important identity/correctness gates passed. |
| **Validated with caveats** | The result is useful, but a protocol or provenance limitation narrows the conclusion. |
| **Reconstructed** | Re-run later from a clean historical/current snapshot; useful for documentation, not evidence that the original run used the same machine. |
| **Incomplete** | A required dataset, artifact, strength match, or reproducible harness is missing. |
| **Inconclusive** | The experiment ran, but its confidence interval includes the neutral result. |
| **Rejected** | The candidate failed its declared acceptance test. |

## Timeline

| ID | Date | Milestone | Lifecycle | Evidence | Main conclusion |
|---|---|---|---|---|---|
| M01 | 2026-06-11 | Bitboard engine, legal move generation, make/unmake, FEN and perft | Historical foundation | Validated correctness | The legal chess core exists and is covered by targeted tests/perft. A speed benchmark is not required to establish this milestone. |
| M02 | 2026-06-13 to 2026-06-15 | Classical V7-V19: history/killer ordering, MoveList, magic bitboards, lazy legal filtering and SEE | Historical | Mixed | V19's original V18 comparison reported identical nodes/moves/scores and a 0.4643 time ratio, but the sampled input corpus was never retained. See the [classical revalidation](../benchmarks/classical_search_revalidation_20260825.md). |
| M03 | 2026-06-19 to 2026-07-09 | Classical V20-V35 search and TT experiments; V34 Fast/experimental branch | Historical baseline | Validated current-tree reconstruction with provenance caveats | V34's `AtLeast` TT reuse relaxes the fixed-depth contract and is classified as Fast/experimental. A balanced 2026-08-29 reconstruction sampled the architecture boundaries V15/V19/V23/V25/V27/V29/V30/V35 and found zero score mismatches at D6-D8; V35 used 7.520% of V15's D7 time (13.30x faster). The implementations share current primitives, so this is not authentic historical-release speed. See the [Strict milestone benchmark](../benchmarks/strict_milestone_benchmark_20260829.md). |
| M04 | 2026-07-29 | Phase-aware quantized NNUE and V36 search | Superseded historical baseline | Validated with stopping-rule caveat | V36 NNUE at depth 3 beat heuristic V35 at depth 4: 46-132-22, 56.0%, nominal paired 95% CI 52.75-59.25%. The ordinary interval was inspected every five pairs under `--stop-on-ci`, so it is directional promotion evidence rather than a sequentially valid confirmatory interval. See [the original result](../nnue_vs_heuristic_result.md). |
| M05 | 2026-07-29 to 2026-08-16 | V38 selective search and V39 Fast promotion | Historically promoted search baseline | Directional with protocol caveats | V39 Fast led Balanced and baseline7 under the project's paired early-stop protocol; a later all-four candidate lost directly to Fast. Repeated CI peeking and cross-game heuristic state prevent a confirmatory statistical claim. See [V39/V40 search milestones](../benchmarks/nnue_v39_v40_search_milestones.md). |
| M06 | 2026-08-17 | V40 QSEE at -75cp | Adopted in V41 for efficiency | Historical efficiency evidence with provenance caveats; inconclusive strength | Holdout nodes fell 15.50% and time 11.36%, but the raw tune summary does not retain complete source/binary/model/harness identity. V40 scored 52.0% vs V39 with CI 49.71-54.29%. Deployment establishes the engineering choice, not a conclusive strength win. |
| M07 | 2026-08-25 | V41 in-search repetition and 50-move rules | Production search | Validated tested correctness and throughput; explicit coverage gap | Rule tests pass and a clean paired rerun reduced nodes 2.85% at depth 7 and 5.50% at depth 8; per-node speed was within about 1.3% of V40. A dedicated null-move/repetition regression is still missing. See [V41 draw-rule validation](../benchmarks/nnue_v41_draw_rules_20260825.md). |
| M08 | 2026-08-23 to 2026-08-25 | Heroku diagnosis, x86 SIMD/VNNI, and LTO | Production runtime | Validated | Scalar Heroku was about 200k NPS and spent 89.15% of sampled CPU in NNUE forward. VNNI raised fixed-depth throughput to 1.47-1.56M NPS; LTO then added 6.15-6.77%. See the [runtime reports](../benchmarks/README.md). |
| M09 | 2026-08-25 | Canonical Mac/Heroku comparison and component profile | Operational baseline and diagnostic | Validated with caveats | On exact fixed-depth signatures, Apple M4 measured 4.71-4.94M NPS and Heroku LTO measured 1.48-1.58M NPS. Cross-host sampling found TT at 10.50% versus 2.89%; accumulator was 23.03% on Heroku versus 15.62% leaf-only or 19.20% after caller-informed `memmove` attribution on M4. The aggregate ratio combines hardware, ISA, compiler and hosting effects; local thermal drift makes component `ns/node` conditional. See the [component profile](../benchmarks/nnue_v41_cross_host_component_profile_20260825.md). |
| M10 | 2026-08-23 to 2026-08-25 | Larger F2 and horizontally mirrored F2M model experiments | Experimental | Mixed | A 500M F2 candidate was rejected against an older 200M reference. F2M improved sealed offline error at 50M, but the current 600-game epoch-5 self-play was inconclusive. No F2M model is promoted. See [model-scaling evidence](../benchmarks/nnue_model_scaling_20260825.md). |

## Source anchors

| Milestone | Git/model anchor |
|---|---|
| M01 | `a0a8ddb` initial engine |
| M02 | `b9d987a` V8/V9; `5198211` MoveList; `1f1d19e` magic bitboards; `f5fb2fb` V19 |
| M03 | `c68faa8` Strict optimization through `d08810d` V35 variants, with V34 retained as the Fast/experimental `AtLeast` branch |
| M04 | `aa1f5a2` quantized NNUE pipeline; `b6d5732` tracked V36/V35 match report |
| M05 | `a6b1697` V38 tuning through `9ae6160` V39 Fast commit |
| M06-M08 | `803aeb5` V41/SIMD production; `36b85a8` release report; `634b2d4` LTO production |
| M09 | clean `634b2d4` local rerun, engine SHA `ecd723c2...`; Heroku LTO SHA `2b9bedd0...` |
| M10 | production F2 SHA `a1a52891...`; F2M work remains uncommitted and is identified by checkpoint/binary hashes in its report |

## Current production snapshot

The last committed engine snapshot is `634b2d481925abc4b5ebf6da9129ade2edf3a24f`:

- UCI engine: `ChessNNUEV41`
- search: V41, including V40 QSEE `-75cp` and in-search draw rules
- model SHA-256: `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`
- Heroku x86 backend: `x86_avx512vnni_256`
- build policy: Release plus LTO
- observed Heroku release on 2026-08-25: `v12`

The retained production smoke identifies source label `release-634b2d4-lto`
and engine SHA-256
`2b9bedd073bc6d15464567cc9a716a1b7e0b3f7227ed7b293b9f9bb2b7ba8128`.
The smoke artifact does not itself retain the Heroku release number, so the
release observation and binary-level evidence are recorded separately. A
sanitized `deployment_observation` in the tracked evidence manifest preserves
the observed `v12` lifecycle fact without account credentials.

## Benchmark coverage and open gates

| Area | Current evidence | What remains before the next promotion |
|---|---|---|
| Core rules/move generation | The earlier clean audit passed 64 of 69 available CTests. The current tree fixes the V21-V23 TT-range intersection; all three targeted cold/warm tests now pass, and the 25-FEN Strict matrix has zero score mismatch/illegal/stopped results. | Re-run the full clean suite after committing the fixes. Two legacy NN tests require ignored model/data assets and are not clean-clone tests. |
| V19 historical speed | Original ratio and command documented | Exact rerun is blocked because `data/tactical_disagreement_depth7_8h_v7_test10k_baseline.jsonl` is missing. Do not silently substitute another corpus. |
| V41 vs V39 playing strength | V40 speed is positive; V40/V39 strength is inconclusive | A new paired match needs a harness that resets search heuristics per game and pins binary/model/book/config hashes. |
| Current F2 model vs predecessor | Current F2 is deployed; older `hs2x8_os128` beat heuristic V35 | Run a clean direct model match before claiming that the F2 replacement itself gained playing strength. |
| Cross-search TT reuse | Correctness design and uncommitted implementation exist | Use a persistent-UCI A/B; process-per-case benchmarks cannot measure reuse. Old-generation entries may be move hints, never score cutoffs. |
| F2M promotion | Offline 50M win; epoch-5 self-play inconclusive | Export a clean pinned epoch-8 artifact, pass Python/C++ parity, then run a provenance-complete paired match and inference/search benchmark. |
| Current production profile | The LTO release now has a signature-matched M4/Heroku component profile; relative hotspot shares are valid | Before using component `ns/node` or a profile-derived total slowdown as canonical, rerun local controls under stable thermal/frequency conditions. By estimated recoverable wall time, prioritize NNUE forward, accumulator update, TT, then move generation; TT remains the largest relative inefficiency. |

## Evidence retention

Most historical `logs/`, `reports/`, `models/`, and match files are ignored by
Git. The tracked
[evidence manifest](../benchmarks/evidence/milestone_revalidation_20260825.json)
preserves the claims, important totals, identities and SHA-256 values. The new
2026-08-25 reruns are also retained in the 36 KiB tracked
[raw bundle](../benchmarks/evidence/milestone_revalidation_20260825.tar.gz).
Older milestones still depend on local ignored artifacts; because a hash cannot
recover deleted bytes, those bundles should be copied to an immutable external
store or a project release.

The LTO cross-host component diagnostic is retained separately in a compact
[426,914-byte evidence bundle](../benchmarks/evidence/nnue_v41_cross_host_profile_20260825.tar.gz)
with SHA-256
`573a9e64226e930ef2db0923eb457d5627592280379703c497a0f4cfd6587ccf`.
It preserves normalized sampling evidence and compact exports, not every bulky
native profiler trace.

The current-tree Strict architecture-milestone reconstruction is retained in a
[6,514,394-byte evidence bundle](../benchmarks/evidence/strict_milestones_20260829.tar.gz)
with SHA-256
`d14f5a69dc13feaf07f00fac8c2091636e3bacba6ab4f25b01c2f4c850a8efb4`.
It includes all 9,600 timed rows, the exact clean source snapshot, executable
and library, build metadata, analyzer/tests, move-tie verification, and
checksums.
