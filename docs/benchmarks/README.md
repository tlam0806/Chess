# Benchmark reports

This directory contains curated, reviewable benchmark milestones. Raw
per-position observations and profiler artifacts remain under `logs/` and are
identified by hashes in each report.

New runs should follow the [benchmark protocol and evidence policy](PROTOCOL.md).

| Date | Report | Status | Main result |
|---|---|---|---|
| 2026-06-13 to 2026-07-09 | [Classical search revalidation](classical_search_revalidation_20260825.md) | Reconstructed with caveats | V32/V35 reproduced identical trees on 260 position-pairs; historical V19 exact rerun is blocked by a missing corpus. |
| 2026-08-16 to 2026-08-17 | [V39 Fast and V40 QSEE](nnue_v39_v40_search_milestones.md) | Historical promotion with caveats | Every direct score favored V39 Fast, but its early-stop intervals were not sequentially valid; V40 saved 15.50% nodes and scored an inconclusive 52.0%. |
| 2026-08-23 | [V41 Heroku platform baseline](nnue_v41_heroku_platform_20260823.md) | Validated production baseline | Scalar x86 release measured about 200–203k NPS at fixed depth. |
| 2026-08-24 | [V41 Heroku sampling profile](nnue_v41_heroku_sampling_profile_20260824.md) | Validated diagnostic | Scalar NNUE forward accounted for 89.15% of sampled search CPU time. |
| 2026-08-25 | [V41 x86 SIMD validation and deployment](nnue_v41_x86_simd_heroku_20260825.md) | Validated production release | VNNI reached 1.47–1.56M fixed-depth NPS, 6.08–6.42x over scalar; deployed as Heroku v11. |
| 2026-08-25 | [V41 draw-rule validation](nnue_v41_draw_rules_20260825.md) | Validated tested correctness and throughput; explicit coverage gap | V41 reduced nodes 2.85–5.50% versus V40 while keeping per-node throughput close; a dedicated null-move/repetition regression remains open. |
| 2026-08-25 | [Canonical local Mac vs Heroku](nnue_v41_local_vs_heroku_20260825.md) | Validated operational comparison | M4 measured 4.71–4.94M fixed-depth NPS, 3.12–3.19x the Heroku LTO point estimate on identical signatures. |
| 2026-08-25 | [Post-SIMD profile and LTO](nnue_v41_post_simd_and_lto_20260825.md) | Validated diagnostic and production optimization | VNNI forward fell to 24.50% of sampled CPU; LTO added 6.15–6.77% fixed-depth throughput. |
| 2026-08-25 | [V41 cross-host component sampling](nnue_v41_cross_host_component_profile_20260825.md) | Validated relative-share diagnostic; absolute timing conditional | TT was the clearest x86 disadvantage; accumulator/forward and move generation are the main next targets. Local thermal drift prevents a canonical component-time or host-slowdown claim. |
| 2026-08-25 | [V41 NNUE forward stages: M4 versus Heroku](nnue_v41_forward_stage_arm_vs_x86_20260825.md) | Validated isolated-stage diagnostic | Full kernel core was 79.95 ns on M4 versus 227.11 ns on Heroku (2.84x); S1 was the largest isolated absolute gap, while S3 was large only as a ratio. |
| 2026-08-25 | [NNUE model scaling](nnue_model_scaling_20260825.md) | Rejected/inconclusive candidates | 500M F2 lost to an older 200M reference; F2M won offline but the 600-game epoch-5 match was inconclusive. |
| 2026-08-29 | [Strict architecture-milestone fixed-depth benchmark](strict_milestone_benchmark_20260829.md) | Validated current-tree reconstruction | Architecture checkpoints V15/V19/V23/V25/V27/V29/V30/V35 had zero score mismatches at D6-D8; V35 used 7.520% of V15's depth-7 time (13.30x faster). |
| 2026-09-01 | [V43 tuning and production promotion](nnue_v43_production_promotion_20260901.md) | Deployed production release; incomplete strength confirmation | V43 won the staged race and led the partial final 55.238% over 210 complete pairs while using 95.607% of production nodes; the operator promoted it and stopped the planned 600-game final after 421 games. |

The status labels are deliberate:

- **Validated production baseline** measures an existing production release.
- **Validated diagnostic** identifies a bottleneck without changing the
  production engine.
- **Validated relative-share diagnostic** supports hotspot composition while
  leaving absolute component timing conditional when calibration controls are
  unstable.
- **Validated isolated-stage diagnostic** compares deliberately isolated
  kernel regions; those regions are not additive production-time shares.
- **Validated current-tree reconstruction** compares retained implementations
  built together against current shared primitives; it does not claim the
  original historical binaries had the measured speed.
- **Validated staging candidate** passed correctness and performance gates but
  is not evidence that production has already been upgraded.
- **Validated production release** passed the staging gates and a post-release
  provenance, backend and fixed-depth smoke check before the worker resumed.

The chronological view and the remaining promotion gaps are in the
[project milestone ledger](../milestones/README.md). Compact tracked results and
raw-artifact hashes from the 2026-08-25 audit are in
[the evidence manifest](evidence/milestone_revalidation_20260825.json). The
corresponding new raw observations are preserved in a small tracked
[compressed bundle](evidence/milestone_revalidation_20260825.tar.gz).
