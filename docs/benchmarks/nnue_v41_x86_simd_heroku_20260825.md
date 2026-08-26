# NNUE V41 x86 SIMD validation on Heroku — 2026-08-25

## Outcome

Status: **validated and deployed to production as Heroku release v11**.

Historical scope: this report covers the non-LTO v11 release. LTO was later
validated and deployed as v12; see the
[post-SIMD and LTO report](nnue_v41_post_simd_and_lto_20260825.md).

V41 now has x86 SIMD implementations for its quantized NNUE forward pass. A
single engine binary chooses the fastest supported backend when the model is
loaded. On every sampled Heroku Basic dyno, `auto` selected the 256-bit
AVX-512 VNNI backend, `x86_avx512vnni_256`.

This is not three independently deployed engines. Scalar and AVX2 were forced
only as correctness controls and performance fallbacks inside the same binary.
The intended x86 runtime order is VNNI, then exact AVX2, then scalar.

The full three-dyno validation completed 3,240 measured searches with no
warning or search-signature mismatch. VNNI improved fixed-depth full-search
throughput by about 6.1–6.4 times over the scalar x86 path.

The exact committed release was deployed after the benchmark gates passed.
Production auto-selected the same VNNI backend, reproduced the canonical
start-position depth-7 signature, and resumed as one healthy Basic worker.

| Mode | Limit | Scalar NPS | AVX2 NPS | VNNI NPS | AVX2/scalar | VNNI/scalar | VNNI/AVX2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed depth | 7 | 241,181 | 1,022,380 | 1,466,965 | 4.239x | 6.082x | 1.435x |
| Fixed depth | 8 | 242,843 | 1,052,906 | 1,558,436 | 4.336x | 6.417x | 1.480x |
| Movetime | 1,000 ms | 281,665 | 1,253,978 | 1,820,460 | 4.452x | 6.463x | 1.452x |
| Movetime | 3,000 ms | 293,769 | 1,270,176 | 1,825,760 | 4.324x | 6.215x | 1.437x |

The primary promotion measurements are fixed depth because they preserve an
identical search tree. Movetime results describe the amount of work completed
under a production-like wall-clock limit; they should not be compared directly
with fixed-depth NPS as if they were the same workload.

## Correctness gates

- 3/3 dynos reported `auto = x86_avx512vnni_256`.
- Scalar, AVX2 and VNNI each passed the 8,192-position NNUE parity fixture on
  every dyno: 73,728 backend-position checks in total.
- All 24 fixed-depth signatures (12 positions at depths 7 and 8) matched across
  every backend, round and dyno in `(depth, score, nodes, bestmove)`.
- The scalar signatures also matched the earlier production V41 Heroku
  baseline exactly.
- 3,240/3,240 expected measured observations were present.
- All engine, model, config, suite, harness and runner provenance values matched
  across the nine backend runs.
- Production UCI options matched the benchmark config exactly.
- No timeout, malformed result, movetime wall warning or integrity warning was
  recorded.

Targeted local validation also passed with the production model and parity
fixture under ARM auto, forced scalar and forced NEON. The staging CPU executed
the forced scalar, AVX2 and VNNI paths directly.

## Per-dyno promotion gate

All backends ran on the same one-off dyno before moving to the next dyno. The
order was rotated as a complete Latin square so no backend always ran first or
last.

| Dyno order | Depth | Scalar NPS | AVX2 NPS | VNNI NPS | VNNI/scalar | VNNI/AVX2 |
|---|---:|---:|---:|---:|---:|---:|
| Scalar → AVX2 → VNNI | 7 | 241,418 | 1,066,028 | 1,542,564 | 6.390x | 1.447x |
| Scalar → AVX2 → VNNI | 8 | 244,658 | 1,079,497 | 1,631,582 | 6.669x | 1.511x |
| AVX2 → VNNI → Scalar | 7 | 232,950 | 1,024,828 | 1,462,736 | 6.279x | 1.427x |
| AVX2 → VNNI → Scalar | 8 | 232,604 | 1,055,867 | 1,536,839 | 6.607x | 1.456x |
| VNNI → Scalar → AVX2 | 7 | 249,759 | 979,917 | 1,402,294 | 5.615x | 1.431x |
| VNNI → Scalar → AVX2 | 8 | 252,069 | 1,024,790 | 1,511,901 | 5.998x | 1.475x |

VNNI beat scalar and AVX2 at both fixed depths on every sampled dyno. A paired
10,000-replicate bootstrap first resampled dynos and then matched cases within
each dyno:

| Depth | AVX2/scalar 95% interval | VNNI/scalar 95% interval | VNNI/AVX2 95% interval |
|---:|---:|---:|---:|
| 7 | [3.933, 4.461] | [5.635, 6.435] | [1.412, 1.455] |
| 8 | [4.071, 4.539] | [6.011, 6.711] | [1.451, 1.511] |

All point estimates and interval lower bounds pass the staging promotion
criterion of greater than 1.0. The archived `comparison.json` established
artifact validity but did not encode a speed threshold. The current
`full_promotion_gate` was applied to the retained raw observations after the
run and returned `pass` with no failure; future full runs now apply this gate
automatically. These intervals are descriptive for the three sampled Basic
dynos and fixed suite, not a guarantee for the entire Heroku fleet.

## Implementation

The accelerated code is isolated by instruction set rather than compiling the
whole engine for the build machine's CPU:

- `src/phase_quantized_nnue_forward_vnni.cpp`: 256-bit non-saturating
  `VPDPBUSD` implementation.
- `src/phase_quantized_nnue_forward_avx2.cpp`: exact AVX2 implementation,
  including unsigned-activation handling without `VPMADDUBSW` saturation.
- `src/phase_quantized_nnue_forward_backend.hpp` and
  `src/phase_quantized_nnue_forward_x86_impl.hpp`: private kernel interface and
  shared x86 implementation.
- `src/phase_quantized_nnue_forward.cpp`: baseline-safe CPU/OS feature dispatch
  and scalar/ARM implementations.

A local x86_64 cross-build disassembly found no YMM or ZMM instructions in the
common translation unit. Its VNNI object contained 72 non-saturating
`vpdpbusd` instructions, no `vpdpbusds`, and no ZMM references. Runtime feature
checks occur before an ISA-specific factory is called, so unsupported CPUs do
not execute an illegal instruction. The retained Heroku parity and backend
handshake artifacts validate the staged GCC binary's behavior; they do not
contain its object-level disassembly.

`CHESS_NNUE_BACKEND=auto|vnni|avx2|neon|scalar` exists for tests and
diagnostics. An explicitly requested unavailable or misspelled backend fails
model loading instead of silently falling back. Production should leave the
setting unset or use `auto`.

## Protocol

The reusable suite and base harness are documented in the earlier
[V41 Heroku platform baseline](nnue_v41_heroku_platform_20260823.md).

- Dedicated temporary staging app; production app and worker were not scaled,
  restarted or modified.
- Three independent Heroku Basic one-off dynos, all reporting Intel Xeon
  Platinum 8375C with AVX2, AVX-512 VL and AVX-512 VNNI.
- Same staging binary and production NNUE model for all backends.
- 12 canonical FENs with process-per-case isolation.
- One unmeasured depth-5 warm-up pass per backend and dyno.
- Fixed depth 7 and 8: 10 complete rounds per backend and dyno.
- Movetime 1,000 and 3,000 ms: 5 complete rounds per backend and dyno.
- Rotate-then-reverse position ordering and interleaved benchmark modes.
- Scalar/AVX2/VNNI, AVX2/VNNI/scalar and VNNI/scalar/AVX2 backend orders.
- Timer inside the dyno from sending `go` until receiving `bestmove`.
- Aggregate NPS calculated as `sum(nodes) / sum(elapsed time)`.
- No samples silently removed.

The measured searches ran from 2026-08-24 16:14 to 17:05 UTC, or 2026-08-25
00:14 to 01:05 in the project timezone (Asia/Singapore).

## Measured staging image

The app/release/image fields below were captured operationally while the
temporary app still existed; they are not embedded in the raw benchmark JSON.

- Temporary app: `chess-nnue-simd-0825-0008`
- Staging release: `v3`; formation remained `benchmark=0` outside one-off runs
- Container image digest:
  `sha256:1ca72f2742644f697ce4196051439712d0352d0dc82e05cd6efbdf6d8f1fe1ce`
- Staged source tree:
  `700ffab0eed6c471f1c778e5841d76a3e8ecb203`
- Engine SHA-256:
  `29579ec5bceafc00fa77a4caef50d4ca9ad5a181123369047a00af6456a08527`
- Model SHA-256:
  `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`
- Config SHA-256:
  `f0672851d58a0467211a8660a5306b4440c2b333b2cc2a9d01aafeb2e02887af`
- Benchmark spec SHA-256:
  `b297dfe39c869e2f3eca07175f57e7302c4c29257f04d75493a70701648ed67e`
- Effective full spec SHA-256:
  `e452c0e0adb6a82f9555f02a8e16c61d5380ab3ba16312f4983f4e4609b89e49`
- Harness SHA-256:
  `f6b0d01c4258dcffececcf7c5590940dd25cc554bedeca324563be164e288f4e`
- Composite runner SHA-256:
  `e7e2d782194d7c25250954574997a4245a2c7c657d6545960233c4baaa232824`

After the local artifacts were retained and checked, the temporary staging app
was destroyed. Production remained on release `v10` throughout the work.

The source came from a staged working-tree snapshot, not a clean main-repo
commit. The engine and model hashes are authoritative for this measurement,
and the implementation, tests and deployment pipeline were subsequently
committed together as `803aeb5a3de0031cff7916a6c143aa33869cd84a`.
The production GCC binary rebuilt from that commit has the same SHA-256 as the
measured staging binary.

## Production deployment

Deployment occurred only after the live game `jX9cIo0O` ended by checkmate.
The worker was scaled to zero before the new image was released, stayed at zero
during the one-off smoke checks, and was restored to one only after every gate
below passed.

- App: `stormy-garden-92984`
- Release: `v11` at 2026-08-24 17:29:57 UTC / 2026-08-25 01:29:57 SGT
- Previous rollback release: `v10`
- Source commit: `803aeb5a3de0031cff7916a6c143aa33869cd84a`
- Source tree: `c8851447e1641e44b557bd82e536bf41426a5dcb`
- `lichess-bot` commit: `ad4b56621bb0e6c52925212639fb73e8ce5ae451`
- Production image digest:
  `sha256:eb28b13219472e42fbb17cc8ff83ac2bf18c072c833fe0f7d7fb03f6ecb5788b`
- Production engine SHA-256:
  `29579ec5bceafc00fa77a4caef50d4ca9ad5a181123369047a00af6456a08527`
- Production model SHA-256:
  `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`
- Runtime handshake: `ChessNNUEV41`,
  `nnue_kernel=x86_avx512vnni_256`, `readyok`
- Fixed-depth smoke: start position, depth 7, score `25cp`, 33,514 nodes,
  best move `e2e4`
- Operational postflight: `worker.1`, Basic, release v11, state `up`; engine
  configuration passed, authenticated as `TrumCoVuaa`, awaiting challenges

The deploy tool now archives an explicit source commit and the tracked
`lichess-bot` commit instead of copying either dirty working tree. It embeds a
release manifest in the image and does not rewrite the existing production
token or restart the app before the code release.

## Raw artifacts

Raw artifacts remain local and ignored by Git under
`logs/nnue_simd_matrix_full_20260824T161435Z/`.

A compact machine-readable
[evidence manifest](evidence/nnue_v41_x86_simd_heroku_20260825.json) preserves
the headline performance values, post-run promotion result, all 24 canonical
fixed-depth signatures and hashes of the larger raw artifacts. The manifest
does not replace the full per-observation files, which should also be moved to
durable external storage before the local `logs/` directory is cleaned.
Its SHA-256 is
`ca6f85a4da63429eda787295821ed19a2298b3b676bc17421ae50cfd18689e01`.

| Artifact | SHA-256 |
|---|---|
| `heroku-matrix-run-1.json` | `28c776045f4790cda0f03cdff7e5fef04aacc1240901852e644e078da5314cb6` |
| `heroku-matrix-run-2.json` | `914e6da0ac7c6d85e06ae92b7b18a58011a6c39e2f6a696da10925f086e74994` |
| `heroku-matrix-run-3.json` | `8f80ce7c2e62e36ac74e866e50fdeb2356644748e619b699b87b020dbe1c94d8` |
| `comparison.json` | `0d74b373e12278ccfdcaa52bab64798dbf9fe3475dbb26825903f6f86a0b0a6c` |
| `request.json` | `a18fd230fa4e81cc4cc76741bebc6bab3e4b8b39c170550b2952b0832beea081` |
| `harness.snapshot.py` | `f6b0d01c4258dcffececcf7c5590940dd25cc554bedeca324563be164e288f4e` |
| `matrix_runner_local.snapshot.py` | `531c307f51628bb2aadef82e31f21645488ef6d192a32c70a2d28b8617ce7286` |
| `matrix_runner_remote.snapshot.py` | `cbe8d81c893a3ec53fa0cf3fd5218051172aa146610a6d315f4a67382e421f14` |
| `platform_aggregator.snapshot.py` | `d287f1d65630efaf37ad2ebb899bde9be00c1f3417df200b10c3bb0a6d68b040` |
| `comparator.snapshot.py` | `3bb10908910eaa148570498f5f11534d81dd5bf0600aa622792adbba74779ab7` |
| `spec.snapshot.json` | `b297dfe39c869e2f3eca07175f57e7302c4c29257f04d75493a70701648ed67e` |

## Interpretation and deployment gate

The previous production release spent about 89% of sampled search CPU time in
the scalar NNUE forward pass. The measured 6.1–6.4x full-search improvement is
therefore consistent with removing that bottleneck. It does not imply the
forward kernel itself is only six times faster; the rest of move generation,
make/unmake, accumulator maintenance, transposition-table work and search
control remains.

This result also does not reproduce the historical roughly 5M local NPS. The
validated Heroku VNNI range is about 1.47–1.83M NPS depending on benchmark
mode. Local and Heroku numbers should be compared only with the same V41 suite,
model, options and fixed-depth signatures.

The candidate passed correctness and performance promotion gates, then passed
the independent production provenance, VNNI handshake and fixed-depth smoke
gates documented above. Release v11 became active at the time of this report
and was later superseded by the LTO v12 release.
