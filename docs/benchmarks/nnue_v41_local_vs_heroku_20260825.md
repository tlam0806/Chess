# V41 canonical local Mac versus Heroku

Date: 2026-08-25

Status: **validated operational comparison; not an isolated hosting-overhead test**

## Question

Historical local logs suggested roughly 5M NPS, while the first Heroku scalar
baseline was around 200k NPS. Those numbers used different engine versions and
protocols. This rerun uses the same production source, model, UCI options,
12-position suite and fixed-depth signatures as the LTO Heroku measurements.

## Local protocol

- source: clean commit `634b2d481925abc4b5ebf6da9129ade2edf3a24f`
- build: Release plus LTO, AppleClang 21.0.0
- host: MacBook Air, Apple M4, 10 cores, 16 GiB
- backend: `arm_neon_dotprod_i8mm`
- model SHA-256:
  `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`
- suite SHA-256:
  `b297dfe39c869e2f3eca07175f57e7302c4c29257f04d75493a70701648ed67e`
- process isolation: a fresh engine for every measured case
- fixed depth: 7 and 8, ten complete rounds
- movetime: 1,000 and 3,000 ms, five complete rounds
- position schedule: rotate the suite, then reverse it on the paired round
- uncertainty: complete-round percentile bootstrap, 10,000 replicates, seed
  `20260823`; intervals are conditional on this fixed suite and host run
- all production UCI options applied and config-validated
- no observations discarded

The run contained 360 observations and passed every integrity gate. All 24
fixed-depth `(depth, score, nodes, bestmove)` signatures matched every retained
Heroku LTO and non-LTO run exactly.

## Results

| Mode | Local aggregate NPS | Local conditional 95% CI | Heroku LTO aggregate NPS | Local/Heroku point ratio |
|---|---:|---:|---:|---:|
| depth 7 | 4,712,417 | [4,562,923, 4,828,347] | 1,479,502 | 3.185x |
| depth 8 | 4,936,087 | [4,771,244, 5,033,566] | 1,581,861 | 3.120x |
| movetime 1s | 5,770,137 | [5,758,140, 5,785,216] | 1,891,390 | 3.051x |
| movetime 3s | 5,781,606 | [5,770,772, 5,792,065] | 1,874,143 | 3.085x |

Fixed-depth round CV was 4.67% at depth 7 and 4.64% at depth 8. One slow round
was retained at each depth. Timed-search round CV was below 0.30%. Median
movetime overshoot was approximately 2 ms. The pre-run system load average was
3.73 on a 10-core machine, so “idle” means no known engine/training load rather
than a literally quiescent OS.

## Interpretation

The remembered local number was directionally correct: under the new canonical
protocol, the M4 produces about 4.7-4.9M NPS at fixed depth and 5.77-5.78M NPS
in the longer timed searches.

The 3.05-3.19x ratio must not be called pure “Heroku overhead.” It combines:

- Apple M4 versus a Heroku Xeon host;
- ARM I8MM/NEON versus AVX-512 VNNI;
- AppleClang versus GCC and different code generation outside the NN kernel;
- cache, branch predictor and memory-system differences;
- shared-dyno scheduling/environment effects.

An isolated hosting-cost experiment would require the same x86 binary on a
dedicated/control x86 machine and Heroku, ideally on comparable CPU generations.

## Evidence

- local raw JSON SHA-256:
  `a0295f0e393947f3a18404b82c2a4a0999177a7f36ebedf690a1acdd1fc429a6`
- local harness source SHA-256:
  `f6b0d01c4258dcffececcf7c5590940dd25cc554bedeca324563be164e288f4e`
- Heroku LTO summary SHA-256:
  `cf15a677f94c43eb451ee09f7223ee395387e3fe1e97c20edacdba8bff2db42b`

The local JSON did not embed its harness hash; the hash above was computed from
the exact workspace harness immediately after the run. This is adequate for
the current audit but should be embedded automatically in future artifacts.
The full and smoke JSON are retained in the tracked
[revalidation raw bundle](evidence/milestone_revalidation_20260825.tar.gz),
SHA-256
`c483ccebc01455b32fc7c378c5cf79f3bc7330278e140fb77ac60c40dedbfc90`.
