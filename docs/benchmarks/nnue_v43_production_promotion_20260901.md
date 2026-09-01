# V43 single-bound search tuning and production promotion

Date: 2026-09-01

Status: **deployed production release, strength confirmation incomplete by
explicit operator decision**.

## Result

V43 replaced V41 as the production Lichess searcher. The promoted combined
configuration has canonical SHA-256:

```text
98b7732c9587da35554cc274a072a0a5b5f55902aaa77605e78c1ae13e88b4f2
```

The staged race selected that configuration first among 24 entrants and then
first in a three-engine round robin. Its final fixed-control match against the
previous production profile was stopped at the operator's request after 421 of
600 games because the operator had already authorized deployment.

The 210 complete color-reversed opening pairs contain 420 games:

| Metric | Partial final result |
|---|---:|
| Candidate W-D-L | 125-214-81 |
| Candidate score | 55.238% |
| Paired 95% interval | 52.059%-58.417% |
| Candidate/control nodes | 31.764B / 33.224B over all 421 saved games |
| Node ratio | 95.607% |
| Candidate/control search time ratio | 100.061% |
| Illegal games | 0 |

One additional saved game was the first color leg of the next pair and was a
draw. Including it gives 125-215-81 and 55.226%. It is excluded from the paired
interval above. The partial interval is descriptive evidence, not the
predeclared completed 600-game confirmation.

## Promoted search policy

Both race and production use the same phase-quantized NNUE model, twofold
in-search draw detection, a 64 MiB four-entry V43 TT, exact-depth TT score
reuse, and current-generation score cutoffs. Stale-generation and deeper-depth
score reuse are disabled.

| Mechanism | Promoted value |
|---|---|
| LMR | enabled; base `0.45`, divisor `2.9`, minimum depth `3`, minimum move `6` |
| Null move | enabled; minimum depth `2`, reduction `4` |
| RFP | enabled; maximum depth `4`, base margin `50cp`, per-depth margin `100cp` |
| LMP | enabled; maximum depth `3`, base `4`, depth multiplier `1` |
| QSEE | enabled; threshold `-25cp` |
| Main-search SEE | disabled; guard values depth `5`, margin `100cp` |
| Adaptive aspiration | enabled; `minDepth=2`, `base=68cp`, `divisor=33700`, `expansion=2290`, `maxFailHighReduction=1`, `meanWeight=370`, `maxResearches=6`, `clamp=1500cp` |

The model SHA-256 is:

```text
a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02
```

## Selection path

The offline all-prunes tuner fixed the Balanced aspiration policy while tuning
the selective policy. Equal-time self-play then used disjoint opening blocks
and fixed game counts:

| Stage | Protocol | Result for the eventual winner |
|---|---|---|
| R1 | 23 challengers versus production, 32 games each | 20/32, 62.5%; advanced in third place |
| R2 | eight survivors versus production, 96 new games each | 54.5/96, 56.771%; ranked first |
| R3 | top-three round robin, 100 games per pairing | 111.5/200, 55.75%; ranked first |
| Final | winner versus production, planned 600 games | interrupted at 421 games; paired result above |

R3 was decisive within that three-engine field: the winner scored 57.5% versus
`f5c7651f...` and 54.0% versus `0a5ca96d...`. The complete staged rounds plus
partial final contain 2,225 of the planned 2,404 games. No automatic early-stop
or automatic promotion rule was used.

## Deployment

The operator explicitly authorized promotion before the final match reached
600 games. The active V41 worker was allowed to finish game `ClwDoWdu` before
it was scaled to zero, so the deployment caused no game forfeit.

| Identity | Value |
|---|---|
| Local source commit | `de3b5ab7bacdd6bf96a7f130f5f9176d8bc4f256` |
| Source tree | `39ea31457eb746e4d4364f83a39b658aa36c1220` |
| Heroku release | `v17` |
| Heroku deploy ref | `12896c1d` |
| Container image digest | `sha256:fa10cbf8ebf79e399d83672d796edd4b423742488d2b4959c435a817ed7666a9` |
| lichess-bot source | `ad4b56621bb0e6c52925212639fb73e8ce5ae451` |

The release command reconstructed and verified the full promoted configuration,
checked the V43 engine identity, exercised normal and ponder/ponderhit UCI
paths, and replayed the `Ahm5kxQZ` tactical regression at depth 8. It exited 0
with:

```text
uci ponder and promoted-profile protocol passed
```

The Basic worker then started with `config-nnue-v43.yml`, reported `Engine
configuration OK`, connected as `TrumCoVuaa`, and resumed waiting for Lichess
challenges. Heroku release `v16` remains the rollback anchor.

## Evidence and limitations

The stopped run directory was
`logs/nnue_v43_selfplay_race_20260901_030146`. Important retained hashes are:

| Artifact | SHA-256 |
|---|---|
| Immutable race contract | `58c322db18c536c0521478490f292ef3790d2502d9650ec7fd9b0aaadccc7be6` |
| R3 summary | `b57587c94cdc5cd55fd456bdebe18910fff4a05b29e315fdd7bb78e4063a80e6` |
| R3 raw games | `3e233ec864eeede2bb6a6f5f40c45fbb4062f96acfd650468b8b8d00d7152720` |
| Partial final games | `4b4957f83bdbcf31a27bccb716619172324783cdc399c27eb5a1efa4d4b8e118` |
| Partial final manifest | `253412a078d76bff1df435c5dc1e34b6116c894812168c30d6d634ba61d465fb` |

The tracked compact evidence bundle preserves the contract, selections,
summaries, raw R3 games, partial final games and final manifest. It deliberately
omits the duplicate model and executable snapshots because their identities
are pinned and the production model remains retained separately. Bundle:
[`nnue_v43_production_promotion_20260901.tar.gz`](evidence/nnue_v43_production_promotion_20260901.tar.gz),
SHA-256
`03c9a684e4bcc2ec8bad7007bea4a747f12998afc877bc37c46208b42a05a7bc`.

The central limitation is explicit: lifecycle promotion is complete, but the
original 600-game confirmation protocol is incomplete. The 421-game partial
result favors V43 and its paired interval is above 50%, yet it must not be
reported as a completed 600-game validation.
