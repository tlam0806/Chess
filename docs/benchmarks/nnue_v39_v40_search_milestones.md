# V39 Fast promotion and V40 QSEE experiment

Dates: 2026-08-16 to 2026-08-17

## V39 Fast

Status: **historically promoted; directional evidence with protocol caveats**

V39 Fast was selected through direct color-reversed, paired-opening self-play.
The project's early-stop confidence interval was the declared promotion rule;
it is not presented as a fixed-sample Elo interval.

| Match | Fast W-D-L | Fast score | Nominal 95% CI | Historical decision |
|---|---:|---:|---:|---|
| Fast vs Balanced | 21-59-10 | 56.11% | 50.05-62.17% | Fast favored |
| Fast vs baseline7 | 20-55-9 | 56.55% | 50.29-62.81% | Fast favored |
| Fast vs all-four winner | 21-59-10 | 56.11% | 50.05-62.17% | Keep Fast |

The final row is the reciprocal of the recorded all-four candidate result
(10-59-21, 43.89%, CI 37.83-49.95%). It is important negative evidence: the
offline/tournament winner did not survive its direct promotion match.

The intervals above were inspected repeatedly and the matches stopped as soon
as they barely excluded 50%; they are not sequentially valid confidence
intervals. The gauntlet also cleared TT but retained history/killer/counter
heuristics between games. Fast remains the version the project historically
promoted and every recorded direct score favors it, but these runs should be
read as directional engineering evidence, not a statistically confirmed
playing-strength theorem.

## V40 QSEE at -75cp

Status: **adopted in V41; historical efficiency evidence with provenance
caveats; inconclusive strength result**

The 6,000-position depth-8 holdout compared V39 Fast with QSEE disabled against
the selected `-75cp` threshold:

| Metric | QSEE off | QSEE -75 | Change |
|---|---:|---:|---:|
| Candidate nodes | 740,434,977 | 625,697,266 | -15.50% |
| Candidate/control time ratio | 0.0507945 | 0.0450266 | -11.36% |
| Mean WDL loss | 0.00501209 | 0.00512550 | +0.00011341 |
| Critical mistakes | 2 | 1 | -1 |

The direct 600-game match against V39 Fast was:

```text
V40: 116 wins, 392 draws, 92 losses
score: 52.00%
95% CI: [49.71%, 54.29%]
```

Because the interval includes 50%, V40 was **not independently proven stronger
than V39**. Its QSEE configuration was nevertheless adopted in V41 on the
measured efficiency evidence. Deployment establishes that engineering choice;
it must not be rewritten as a conclusive V40 playing-strength win.
This match used the same gauntlet family which retained non-TT search
heuristics across games, so even its nominal interval should not be treated as
a clean confirmatory test. The conclusion remains inconclusive either way.

The tune and match summaries retain the numerical results but not a complete
Git, binary, model, suite and harness identity set. V40 was subsequently folded
into committed V41 source. Accordingly, the efficiency result is useful
historical engineering evidence, not a fully reproducible benchmark under the
status vocabulary used by the current milestone ledger.

## Raw evidence

| Artifact | SHA-256 |
|---|---|
| `logs/nnue_v39_audit3_round_robin_ci_20260802_120129/summary.stopped.json` | `558778a834285cb1d18211830eee31b33e3a56e3795d5051129a1aa6728e5b7f` |
| `logs/nnue_v39_all4_winner_vs_fast_ci_20260816_153741/summary.json` | `145e38306e65b647594e38cad57b48ab98e9d8a634f9da0413a8da3f44b913bf` |
| `logs/nnue_v40_qsee_tune_20260817_005158/summary.json` | `debd8c0b38c83206899251e70c603a1057cbf06c8d96a6672c13e07ae7a490eb` |
| `logs/nnue_v40_qsee_vs_v39_fast_ci_20260817_084202/summary.json` | `31375140a34d786bc3298ab7cb81cc2485280a9b98008c7185efecdd92859cd9` |

These raw directories are local and Git-ignored. The tracked evidence manifest
retains the result fields and hashes.
