# Benchmark protocol and evidence policy

Updated: 2026-08-25

This policy applies to new milestone claims. A benchmark can be useful without
meeting every promotion gate, but its report must say exactly what kind of
claim it supports.

## Keep claims separate

| Claim | Minimum evidence |
|---|---|
| Correctness | Targeted regression tests plus output/signature parity against a trusted implementation or snapshot. |
| Per-node speed | Identical search tree/signatures; paired timing on the same host; balanced order. |
| Search efficiency | Fixed-depth node and wall-time comparison; output signatures and tactical/safety gates. |
| Offline model quality | Sealed validation split, paired per-sample errors, model/data identities and an uncertainty estimate. |
| Playing strength | Color-reversed paired openings, independent game state, declared stopping rule and complete artifact identities. |
| Deployment | Staging parity/performance gate, release-image smoke, backend identity, rollback target and post-start verification. |

No one row implies another. In particular:

- lower loss is not playing strength;
- fewer nodes is not automatically stronger search;
- higher NPS is not automatically a stronger engine;
- a staging benchmark is not proof that production was deployed;
- a nominal confidence interval inspected repeatedly is not a valid sequential
  stopping boundary.

## Required manifest fields

Every serious run should preserve:

- UTC start/end time;
- full Git commit and dirty-diff/archive identity;
- compiler, version, build type, flags and LTO/PGO state;
- engine binary SHA-256;
- every model/checkpoint SHA-256;
- suite/opening-book/data-split SHA-256;
- config and exact effective engine options;
- harness/runner source SHA-256;
- host/CPU/architecture/backend and resource class;
- command/protocol, seeds, sample counts and stopping rule;
- raw observations, warnings and failed integrity gates;
- aggregation and confidence-interval method.

Do not dump the full process environment: it can contain credentials. Record a
whitelist of relevant metadata instead.

## Fixed-depth performance

1. Use a versioned position suite.
2. Warm up outside the measured rows.
3. Balance order (for example A-B-B-A or a Latin square across hosts).
4. Reset exactly the state declared by the protocol.
5. Require identical `(depth, score, nodes, bestmove)` before interpreting a
   speed ratio.
6. Aggregate NPS as `sum(nodes) / sum(time)`, not the arithmetic mean of
   per-position NPS.
7. Resample complete paired rounds/hosts; keep all observations and report
   outliers rather than deleting them silently.
8. Label confidence intervals conditional on the chosen suite and sampled
   hosts.

If the feature being measured is cross-search state (for example TT reuse), a
process-per-case harness is invalid. Use a persistent UCI process and a
realistic sequence of positions/moves.

## Self-play

Each color-reversed opening pair must start with independent engine state:

- clear TT and every history/killer/counter table, or create fresh searchers;
- pin engine/model/config/book/harness identities;
- preserve every game row, result reason, node/time totals and opening ID;
- define a fixed number of pairs before starting, or use a valid sequential
  boundary rather than repeatedly peeking at an ordinary 95% interval;
- report W-D-L, paired score, interval method and all early termination rules;
- never promote when the declared interval includes the neutral result.

Resume keys must validate identities, not only game/profile labels.

## Raw evidence retention

The report and compact tracked manifest are the review surface. The raw
artifact bundle is the source of truth. Because this repository ignores large
`logs/`, `reports/`, `models/` and match artifacts, a milestone is not durable
until the raw bundle is stored in an immutable external location or selected
compressed evidence is tracked/released. A SHA-256 proves identity only while
the bytes still exist.

## Lifecycle wording

Use explicit outcomes:

- `experimental`
- `historically promoted under project protocol`
- `validated staging candidate`
- `deployed production release`
- `inconclusive`
- `rejected`

If a later audit finds a protocol weakness, keep the historical lifecycle but
downgrade the evidence quality. Do not rewrite a past deployment as though its
original evidence was stronger than it was.
