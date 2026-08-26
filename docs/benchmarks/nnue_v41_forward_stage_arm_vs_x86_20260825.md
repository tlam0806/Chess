# V41 NNUE forward-stage benchmark: Apple M4 versus Heroku x86

Date: 2026-08-25
Status: **validated isolated-stage diagnostic; not a PMU profile and not an
additive production-time decomposition**

## Result

The production-equivalent full kernel core took **79.947 ns/evaluation** on
Apple M4 and **227.108 ns/evaluation** on a Heroku Basic dyno, a **2.841x**
wall-time ratio. The largest isolated absolute difference was S1. S3 had the
largest ratio, but its absolute difference was only 6.90 ns/evaluation.

| Isolated region | M4 wall ns/eval | Heroku wall ns/eval | Heroku / M4 | Difference | Conditional 95% CI for ratio |
|---|---:|---:|---:|---:|---:|
| S1: input to hidden2 | 61.584 | 176.140 | 2.860x | +114.556 ns | 2.839-2.886x |
| S2: hidden2 to hidden3 | 11.462 | 34.788 | 3.035x | +23.326 ns | 2.999-3.066x |
| S3: hidden3 to output | 2.139 | 9.038 | 4.224x | +6.898 ns | 4.193-4.260x |
| Full kernel core | 79.947 | 227.108 | 2.841x | +147.160 ns | 2.756-2.858x |

Process-CPU time independently gave the same conclusion:

| Region | M4 CPU ns/eval | Heroku CPU ns/eval | Heroku / M4 | Conditional 95% CI |
|---|---:|---:|---:|---:|
| S1 | 61.416 | 176.096 | 2.867x | 2.847-2.893x |
| S2 | 11.437 | 34.773 | 3.040x | 3.006-3.073x |
| S3 | 2.132 | 9.035 | 4.238x | 4.206-4.270x |
| Full | 79.733 | 227.035 | 2.847x | 2.803-2.865x |

The Heroku CPU/wall medians were 0.9996-0.9998, so these three one-off runs
show no material descheduling during the measured batches.

## Region definitions

- **S1** includes auxiliary-feature addition, SCReLU activation, the 256x32
  byte dot product, bias/scale/clamp, and a complete hidden2 output store.
- **S2** includes the hidden2 low/high-byte split, both 32x32 byte dot
  products, bias recombination, scale/clamp, and a complete hidden3 store.
- **S3** includes the 32-to-1 output dot product, reduction, bias, and signed
  output division.
- **Full** calls the complete candidate kernel. Unlike isolated S1/S2, it may
  keep intermediate values in registers.

S1, S2, and S3 must not be summed or treated as shares of Full. The isolated
regions deliberately materialize their outputs so the optimizer cannot remove
unused lanes; this changes register and memory behavior relative to the fused
Full path.

## Protocol

- Source base: production commit `634b2d481925abc4b5ebf6da9129ade2edf3a24f`
  plus benchmark-only snapshot commit
  `d9648da134c52198715167ae2b20aeb411608f59`, tree
  `a015461078ce682caae0755198d01c8511196890`.
- Production model SHA-256:
  `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`.
- Both builds used Release optimization, LTO, and the compile-only stage
  benchmark gate. The normal production build keeps that gate off.
- M4 backend: `arm_neon_dotprod_i8mm`; local benchmark binary SHA-256:
  `f8221a2a49e3e6da273d04c51da1adb12bb032fb5529359b8015b3b0f26cb383`.
- Heroku backend: `x86_avx512vnni_256`; remote benchmark binary SHA-256:
  `7d8cc433ecd1109585c1d6bd597546fbcf10e26f8abf9cb8a589ec4aff918bc8`.
- Local host: MacBook Air `Mac16,12`, Apple M4, 4 performance plus 6
  efficiency cores, 16 GiB, macOS 26.5.1; Apple Clang 21.
- Heroku hosts: three Basic one-off dynos, all reporting Intel Xeon Platinum
  8375C at 2.90 GHz with AVX2, AVX-512 BW/VL, and AVX-512 VNNI.
- Deterministic corpus: 256 samples, seed `20260825`, all eight phase buckets,
  112 nonzero auxiliary states, corpus checksum `0xf5399956b2fa9ea8`.
- Three independent processes on the M4 and three independent Heroku dynos.
  Each stage used 12 batches in a balanced rotating order.
- Per-batch work: S1 4,000,000; S2 20,000,000; S3 100,000,000; Full
  3,000,000 evaluations, after 10,000 warmups per stage.
- Wall time used `std::chrono::steady_clock`; process CPU used `std::clock`;
  no empty-loop subtraction was applied.

The point estimate is the median of the three host/process medians. The
conditional intervals use 10,000 hierarchical percentile-bootstrap
replicates with seed `20260825`: independently resample the three clusters per
host, then resample 12 batches within each selected cluster while preserving
the four rotating order strata.

## Correctness and compiler gates

All six full runs passed:

- scalar output parity;
- per-stage output parity;
- expected timed-sink parity, so the measured path itself was exercised;
- identical corpus and stage checksums across ARM and x86;
- all 8,192 production NNUE parity positions on the clean M4 build.

The M4 disassembly contains the expected SUDOT paths and complete output
stores for isolated S1/S2. The retained Heroku artifact proves the VNNI
runtime backend and CPU capability, but does not contain the remote object
disassembly; therefore it does not support a deployed static VPDPBUSD
instruction-count claim.

## Interpretation and limits

S1 is the next forward-kernel target because it contributes by far the largest
isolated absolute cross-host difference. S2 is the second target. S3 is
4.22x slower on x86 but is too small in absolute time to be the primary
bottleneck.

This is a source-equivalent kernel-core diagnostic, not a direct timer inside
the rated production searcher. It combines CPU, ISA, compiler, LTO code
layout, and Heroku hardware effects; it is not pure "Heroku overhead". A prior
Basic-dyno probe recorded `perf_event_paranoid=4`, which blocked unprivileged
PMU access. This run therefore reports elapsed and process CPU time rather
than cycles, instructions, IPC, or cache-miss counters; it did not repeat that
PMU probe on these three dynos.

Only three clusters were available per host, and the three local clusters are
processes on one physical M4 rather than three separate Macs. The intervals
therefore describe uncertainty conditional on these executions and this fixed
corpus, not the full M4 or Heroku fleet.

## Evidence

- [M4 summary](../../logs/nnue_forward_stages_m4_v2_clean_20260825/summary.json)
- [Heroku full summary](../../logs/nnue_forward_stages_heroku_full_v2_20260825T1358Z/summary.json)
- [Heroku artifact hashes](../../logs/nnue_forward_stages_heroku_full_v2_20260825T1358Z/SHA256SUMS)
- [Compact tracked evidence](evidence/nnue_v41_forward_stage_arm_vs_x86_20260825.json)
- [Reproducible analyzer output](evidence/nnue_v41_forward_stage_arm_vs_x86_20260825.analysis.json)
- [Compact raw-evidence bundle](evidence/nnue_v41_forward_stage_arm_vs_x86_20260825.tar.gz),
  SHA-256 `0f6d2f1bd180dfd3693208b0675d5cb88f9179dd45a881356806b77b32dfca44`

The staging release was Heroku v3 (`Deploy cb0f0eeb`), image digest
`sha256:634e1bb0b0f26815c2b4385bc1288710b3d2a8225bf547bb30c0bd60ff443e92`.
Formation remained at zero outside one-off runs. The dedicated staging app
`chess-nnue-stage-0825-1323` was destroyed after the artifacts were retained;
the production bot was never touched.
