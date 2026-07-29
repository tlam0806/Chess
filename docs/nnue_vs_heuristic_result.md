# Verified NNUE vs heuristic result

## Conclusion

The phase-aware quantized NNUE model `hs2x8_os128`, used as the leaf evaluator
of `NnueSearcherV36`, beat `HeuristicSearcherV35` in a controlled paired
fixed-depth match. The NNUE result was above 50% even though it searched one
nominal ply less:

| Candidate | Opponent | Games | W-D-L | Points | Score | Paired-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| NNUE depth 3 | Heuristic V35 depth 4 | 200 | 46-132-22 | 112/200 | **56.0%** | **52.75%-59.25%** |

The match stopped after 100 opening pairs when the reported confidence interval
separated from 50%. Its raw decision was:

```text
ci model=new_huber200_200m_hs2x8_os128 pairs=100 games=200 score_rate=0.56 bootstrap95_lower=0.5275 bootstrap95_upper=0.5925
decision model=new_huber200_200m_hs2x8_os128 winner=new_huber200_200m_hs2x8_os128 pairs=100 games=200
summary model=new_huber200_200m_hs2x8_os128 W=46 D=132 L=22 points=112 score_rate=0.56 games=200 ci_decision=1 candidate_nodes=29046635 heuristic_nodes=79154582
```

The 56% score corresponds to approximately +41.9 score-equivalent Elo. Mapping
the endpoints of the score interval gives approximately +19.1 to +65.0 Elo.
This conversion is descriptive; it is not a separately fitted Elo rating.

## Model lineage

The winning network was:

- training run: `old_score_huber200_top2_200m_20260722_163841`;
- configuration: `hs2x8_os128`;
- architecture: shared transformer with eight phase-specific stacks;
- hidden quantization scales: `[2, 8]`;
- output quantization scale: `128`;
- objective: CP Huber loss with delta `200`;
- training data: one pass over the same 200M unique-position corpus used for
  the top-two comparison;
- checkpoint:
  `models/quantized_scale_grid/old_score_huber200_top2_200m_20260722_163841/hs2x8_os128/phase_component_best.pt`;
- exported runtime model:
  `models/quantized_scale_grid/old_score_huber200_top2_200m_20260722_163841/hs2x8_os128/phase_quantized_nnue.bin`.

On the held-out ranking split, this configuration recorded 286.1888 CP MAE and
a calibration slope of 0.6148. Those regression measurements selected the
checkpoint; the paired match above is the direct playing-strength evidence.

## Match protocol

- Search was fixed-depth and deterministic, without a time limit.
- The NNUE candidate used strict `NnueSearcherV36` at depth 3.
- The opponent was `HeuristicSearcherV35` at depth 4.
- The opening book contained positions filtered as balanced by Stockfish.
- Each of 100 openings was played twice with colors reversed.
- Openings were 10 plies long.
- Games ended on checkmate, stalemate, threefold repetition, the 50-move rule,
  or the 200-ply cap.
- Each engine used a 64 MiB transposition table.
- The paired bootstrap operated on the two-game opening-pair scores with
  100,000 resamples.
- Seed: `20260723`.

Total search work recorded by the harness:

| Engine | Nodes | Relative nodes |
|---|---:|---:|
| NNUE depth 3 | 29,046,635 | 36.70% |
| Heuristic V35 depth 4 | 79,154,582 | 100% |

The node ratio is contextual rather than an equal-work comparison because the
engines searched different depths. The important result is that the NNUE
candidate scored above 50% despite that one-ply handicap.

## Reproduction

The local model and data directories are intentionally ignored by Git. With
those artifacts present, build and run:

```bash
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake \
  --build build-release \
  --target phase_nnue_strict_paired_match \
  -j 8

./build-release/phase_nnue_strict_paired_match \
  --book data/stockfish_balanced_openings_10ply_1k_20260723.txt \
  --model new_huber200_200m_hs2x8_os128=models/quantized_scale_grid/old_score_huber200_top2_200m_20260722_163841/hs2x8_os128/phase_quantized_nnue.bin \
  --games 2000 \
  --candidate-depth 3 \
  --heuristic-depth 4 \
  --max-plies 200 \
  --tt-mb 64 \
  --seed 20260723 \
  --stop-on-ci \
  --min-pairs 30 \
  --check-every-pairs 5 \
  --bootstrap-samples 100000
```

The archived local evidence log was:

```text
logs/new_huber200_200m_vs_heuristic_d4_stockfish_balanced_until_ci_20260723.log
SHA-256: 8e74543127634383d4b3e214be5ad68371d6b36024303b037c952ab762646634
```

## Scope of the claim

This result supports the precise statement:

> The project trained a quantized NNUE that beat the hand-written heuristic
> searcher in the documented paired fixed-depth match.

It does not establish that NNUE wins at every depth, opening distribution, time
control, pruning configuration, or hardware setting. New engine defaults should
still be promoted using fresh paired matches or self-play under their intended
runtime conditions.
