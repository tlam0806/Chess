# V41 post-SIMD profile and LTO promotion

Date: 2026-08-25

Status: **validated diagnostic and validated production optimization**

## Post-SIMD sampling profile

The original scalar Heroku profile found that NNUE forward consumed 89.15% of
sampled search CPU. After the AVX-512 VNNI backend was deployed, a barrier-safe
profile repeated the canonical depth-8 workload:

| Metric | Result |
|---|---:|
| Searches | 192 |
| Non-empty sampled profiles | 96 |
| Total sampled CPU time | 7.43 s |
| VNNI forward time | 1.82 s |
| VNNI forward share | 24.50% |
| Complete-round bootstrap 95% CI | 21.39-27.57% |
| Forward share without `queenside_pressure` | 22.48% |
| Median-NPS sampling slowdown | 1.19% |

The barrier run is the canonical artifact. An earlier non-barrier attempt could
race position setup/`isready` and is not used for the headline number. The
profile belongs to the non-LTO VNNI binary, so it is a historical diagnostic,
not an exact hotspot split for the final LTO release. The interval is
descriptive and conditional on only eight profiled rounds (743 total 100 Hz
samples); it is not a fleet-wide hotspot interval.

The profile interval uses a complete-round percentile bootstrap with 10,000
replicates and seed `20260824`.

## LTO A/B

Four Heroku Basic one-off dynos ran same-dyno non-LTO/LTO comparisons with
balanced A/B order. Fixed-depth signatures were identical and the two engine
identities remained stable across all runs.

| Mode | Non-LTO NPS | LTO NPS | Speedup | Paired ratio 95% CI |
|---|---:|---:|---:|---:|
| depth 7 | 1,393,779 | 1,479,502 | +6.15% | [1.0079, 1.1124] |
| depth 8 | 1,481,551 | 1,581,861 | +6.77% | [1.0302, 1.1057] |
| movetime 1s | 1,740,847 | 1,891,390 | +8.65% | [1.0606, 1.1118] |
| movetime 3s | 1,744,450 | 1,874,143 | +7.43% | [1.0510, 1.0984] |

Bootstrap method: resample dynos, then matched cases within dyno; 10,000
replicates, seed `20260825`. Every lower bound exceeds 1.0, so the declared LTO
promotion gate passed. With only four Basic one-off dynos, the intervals are
conditional on this workload and sample of hosts; they are not Heroku
fleet-wide performance guarantees.

Identities:

- non-LTO engine SHA-256:
  `29579ec5bceafc00fa77a4caef50d4ca9ad5a181123369047a00af6456a08527`
- LTO engine SHA-256:
  `2b9bedd073bc6d15464567cc9a716a1b7e0b3f7227ed7b293b9f9bb2b7ba8128`
- A/B source label: `clean-release-36b85a8-paired-lto`
- committed production source: `634b2d481925abc4b5ebf6da9129ade2edf3a24f`

The full A/B was made before the two-file LTO build change received commit
`634b2d4`; the human source label alone is not a cryptographic source identity.
More importantly, the A/B and production smoke have the exact same LTO engine
hash above. The production smoke used source label `release-634b2d4-lto`,
reported backend `x86_avx512vnni_256`, and reproduced the canonical
start-position depth-7 signature. Heroku `v12` was observed as current on
2026-08-25. Those source/backend fields are retained in the raw smoke run; the
aggregate smoke summary does not contain them, and neither artifact retains the
Heroku release number itself.

## Evidence

| Artifact | SHA-256 |
|---|---|
| barrier profile analysis | `31a0604924ae184720825ccd75ad062c7ec039ddcee7ccc752e8418d2c96ca46` |
| barrier manifest | `51fdf35dbb81779179722a4a84abb283075d0b20f14d1e247d45a11e2db48b20` |
| barrier raw profile archive | `12334f6c46bab0059f6e24c07740c567462cb796cbc8dd4edd01bfdfd8239505` |
| LTO full summary | `cf15a677f94c43eb451ee09f7223ee395387e3fe1e97c20edacdba8bff2db42b` |
| LTO request | `38249a1f3ae14156f240d1e937c6d59e28bd5b009756adce72c7af3e725022b5` |
| production smoke summary | `302e6ab693c8cedb5e007a3870a9847f0b267fcee8348d89fbe0f7fbb364badd` |
| production smoke raw run | `c389ef3d930838491b9166da64d88a38ecfe3d92ad494b2215647326601d612c` |
