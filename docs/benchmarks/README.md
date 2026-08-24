# Benchmark reports

This directory contains curated, reviewable benchmark milestones. Raw
per-position observations and profiler artifacts remain under `logs/` and are
identified by hashes in each report.

| Date | Report | Status | Main result |
|---|---|---|---|
| 2026-08-23 | [V41 Heroku platform baseline](nnue_v41_heroku_platform_20260823.md) | Validated production baseline | Scalar x86 release measured about 200–203k NPS at fixed depth. |
| 2026-08-24 | [V41 Heroku sampling profile](nnue_v41_heroku_sampling_profile_20260824.md) | Validated diagnostic | Scalar NNUE forward accounted for 89.15% of sampled search CPU time. |
| 2026-08-25 | [V41 x86 SIMD validation](nnue_v41_x86_simd_heroku_20260825.md) | Validated staging candidate; not deployed | VNNI reached 1.47–1.56M fixed-depth NPS, 6.08–6.42x over scalar. |

The status labels are deliberate:

- **Validated production baseline** measures an existing production release.
- **Validated diagnostic** identifies a bottleneck without changing the
  production engine.
- **Validated staging candidate** passed correctness and performance gates but
  is not evidence that production has already been upgraded.
