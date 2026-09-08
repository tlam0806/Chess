# LMR/NMP adversarial selective-search pipeline

This pipeline tunes only LMR and null-move pruning on the strict V36-derived
V38 searcher. Reverse-futility and late-move pruning are explicitly disabled.

## Safety bank

`tools/data/build_nnue_selective_safety_bank.py` runs a fixed probe suite containing
LMR-only, NMP-only, and joint configurations against historical datasets. It
asks the strict V36 control to re-search every differing candidate move.

The miner records:

- confirmed win-to-draw, win-to-loss, and self-mate mistakes;
- near misses with root regret of at least 250 CP;
- the position hash and payload;
- mining depth, probe, control move/score, and candidate move/strict score.

Rows are deduplicated by position hash. Known regression rows supplied through
`--seed-tsv` always enter `core.tsv`; all mined rows are deterministically
hash-partitioned between:

- `core.tsv`, visible to mutation as a multi-depth hard gate;
- `sealed.tsv`, used only after mutation for adversarial selection.

`metadata.jsonl` preserves every observation and `manifest.json` records source
datasets, probes, depths, split sizes, and SHA-256 hashes.

## Candidate flow

```text
LMR-only / NMP-only / joint candidate
  -> core.tsv at depths 5, 6, 7, 8
  -> fixed 2,000-position tune gate at depth 6
  -> per-lineage Pareto frontier
  -> sealed.tsv adversarial selection at depth 7
  -> fresh normal selection at depth 7
  -> fresh holdout at depth 7
  -> at most eight spread frontier points at depth 8
```

A candidate is hard-rejected when any comparison reports a win-to-draw,
win-to-loss, or self-mate mistake. Mean regret is computed only for positions
whose direct NNUE static target has absolute value below 1500 CP; every
position remains eligible for the safety checks.

The objective remains a two-dimensional Pareto frontier:

```text
aggregate candidate/control node ratio
mean V36 root regret
```

P95 regret, percentage above 100 CP, move agreement, and prune counters are
reported but are not collapsed into a scalar fitness.

## Mutation space

LMR:

```text
base:            0.00 .. 1.00
divisor:         1.80 .. 3.50
minimum depth:   3 .. 8
minimum index:   2 .. 12
```

Null move:

```text
minimum depth:   3 .. 7
reduction:       1 .. 3
```

The initial population contains the strict control, LMR-only anchors,
NMP-only anchors, and joint anchors. Later mutations change one to three
parameters. Fifteen percent are random restarts; joint candidates can also
combine one LMR-only frontier parent with one NMP-only frontier parent.

## End-to-end runner

```bash
tools/tune/run_nnue_lmr_nmp_adversarial_8h.sh RUN_DIR
```

The runner first builds the historical safety bank, then creates a fresh
16,000-position `8k/4k/4k` tune/selection/holdout dataset excluding all
previous V38/V39 datasets, and finally launches the multi-stage tuner.
