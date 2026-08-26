# NNUE model-scaling evidence: F2, 500M, and F2M

Date: 2026-08-25

Status: **mixed; no new model promoted**

## Production baseline

The production model remains the 200M F2 model:

```text
models/quantized_scale_grid/
old_score_huber200_lr_sweep_then_5ep_20260724_142758/
best/phase_quantized_nnue.bin
```

SHA-256:
`a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`.

Its training/validation selection lineage is recorded in the historical reports,
but there is no clean direct playing-strength match against the older validated
`hs2x8_os128` binary (`da8c716c...`). Therefore the project can claim that this
is the selected/deployed model, not that the model change itself has a proven
Elo gain over its predecessor.

## Ordinary 500M F2 candidate versus the older 200M reference

Status: **historically rejected under the project gate, with protocol and
provenance caveats**

The epoch-4 500M candidate played 600 games / 300 color-reversed pairs against
the older `old_score_huber200_top2_200m` reference, not the current production
model named above. Reference model SHA-256:
`da8c716c74e823655cbb9695846cb1006431f4483c2d61fd20aa24b3c69ee25d`.
Candidate binary SHA-256:
`0e00cf02a8549a2f55be0520ae225ac0a7936ee2eb63268a280b07a6d4b96fee`.

```text
candidate: 71 wins, 423 draws, 106 losses
score: 47.0833%
95% CI: [44.8346%, 49.3321%]
```

Under the project's historical gate, the upper nominal confidence bound below
50% rejected this candidate: more training data did not beat that 200M
reference in this run. This match is not a direct comparison with current
production F2, and the interval is not presented as independent confirmatory
evidence.

The run used the same historical gauntlet family that retained non-TT search
heuristics between games. The candidate/reference binaries and opening book
still survive and are hashable, so the negative project decision is worth
preserving. The run did not pin the engine, opening book, config and harness as
a complete identity set, so the nominal interval is not a clean modern Elo
proof.

## Controlled 50M F2 versus F2M trial

Status: **controlled offline architecture signal with provenance caveats**

All variants used the same raw positions, order, seed, optimizer and sealed
validation records.

Artifact identities:

| Variant | Checkpoint SHA-256 | Exported binary SHA-256 |
|---|---|---|
| F2 independent | `2b78942a7fac887311b09765f927ab6bdd5403c39996379569dadbd60c9edb4e` | `a033115efda0b946a9fb51208bbd2e60073920267ae4a6fd94621889ed094c1e` |
| F2M independent | `816ee34f2e183f6b7a8769524514989f30c1ae5db977234bedc93e1a08a8bdab` | `c6ea1039af21fdecdd2184ca09b94fe8a331b2478104c1a1bb64bb26b3720310` |
| F2M shared L1 | `6356f4936dad8262940fa58f1301fc56e115436af2bea7c995a95bbed204e476` | `c4b0111e52c5fa1b119f9f916a4fe0364b12c1551d9e9fc73ec53d1a5ed0d56a` |

| Model | Ranking CP MAE | Ranking WDL loss |
|---|---:|---:|
| F2 independent | 319.2366 | 0.02440781 |
| F2M independent | 313.0076 | 0.02306620 |
| F2M shared L1 | 312.8563 | 0.02301654 |

Relative to F2M independent, F2 had +6.2289cp paired ranking error, 95% block
bootstrap CI [+5.7196, +6.7292]. This supports the horizontal-mirror
architecture offline. Shared L1 improved by only 0.1513cp relative to
independent F2M, CI [-0.4655, +0.1701], so that comparison is inconclusive.

Offline error does not establish playing strength. The split audit and model
artifacts are hashed, but the F2M source changes and exact dirty diff were not
retained as a clean committed snapshot. The numerical signal is therefore
useful and controlled, but not fully source-reproducible from a clean clone.

## Epoch-5 self-play

Status: **inconclusive and not promotion-grade**

Two runs disagreed:

| Run | Games/pairs | F2M W-D-L | Score | 95% CI |
|---|---:|---:|---:|---:|
| early clean export | 94 / 47 | 35-37-22 | 56.91% | [50.13%, 63.70%] |
| current-code rerun | 600 / 300 | 150-311-139 | 50.92% | [48.39%, 53.44%] |

The longer rerun does not confirm the early signal. In addition, the current
gauntlet does not reset all search heuristics between games, its resume key does
not pin binary/model/book/config hashes, and repeated inspection of an ordinary
95% interval is not a confirmatory sequential test. Both results are therefore
exploratory; neither authorizes promotion.

## Next promotion gate

Before testing the epoch-8/next F2M model:

1. commit or freeze the exact F2M runtime/exporter and checkpoint identity;
2. export the binary model and pass Python/C++ parity;
3. make the gauntlet reset TT and all search heuristics per game;
4. record engine, candidate, baseline, opening book, config and harness hashes;
5. declare a fixed sample size or a valid sequential boundary before starting;
6. run paired inference/search throughput as well as self-play.

A separate optional historical gap is a clean current-F2 versus old
`hs2x8_os128` match. It is required only if the documentation wants to claim
that the production model replacement improved playing strength.

## Evidence

| Artifact | SHA-256 |
|---|---|
| 500M epoch-4 summary | `bbeeb2d924f052ecddad6ea5be74bb8a863d5fc7794a710a1f8a5d1cffb4c332` |
| controlled 50M report | `30cfa4f6f4f70355b7c3001d501ddb20c51af81855b3f4a1263eb44647a0b5ad` |
| controlled 50M comparison JSON | `3f06ca5a4bd37457dbf100a2e1f19ca8ec0a579717d486c1048b5bf13a79a4f3` |
| mirror-split audit JSON | `007f37b7c942977adf070ea36d1f700a088f71fd2f9ffcf8100d66ac4699f800` |
| early 94-game JSONL | `fdc90d718d30ee0c838140c2cba818fae41b4f07178037022249ce04cd380936` |
| 600-game JSONL | `2893405abad6f894e0652ea6bfa3d7c291d1ac901e9f7966580788bbd0bfecba` |
| 600-game runner log | `8fa840affce8d0f50f2db26cd6182b65d9ca33b7b60f8881ad12cc6cb07af15a` |
