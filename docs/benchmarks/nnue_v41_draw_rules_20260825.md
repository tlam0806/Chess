# V41 in-search draw-rule validation

Date: 2026-08-25

Status: **validated tested correctness and throughput; one explicit
null-move/repetition coverage gap; strength not measured**

## Change under test

V41 adds path-local threefold-repetition and 50-move handling to the V40 search
profile. The UCI adapter reconstructs the reversible real-game history; real
search moves extend it, while artificial null moves do not. Checkmate takes
precedence over a 50-move draw. When the current history makes a TT score
unsafe, V41 suppresses the score cutoff/store while retaining the move hint.

## Snapshot and protocol

- Git commit: `634b2d481925abc4b5ebf6da9129ade2edf3a24f`
- clean `git archive` snapshot
- Release plus LTO, AppleClang 21.0.0
- production model SHA-256:
  `a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`
- 12 canonical positions
- depth 7 and depth 8, 20 paired rounds each
- alternating V40/V41 measurement order by round and position
- TT and search heuristics reset for each position
- complete-round percentile bootstrap, 10,000 replicates; seed base
  `20260825`, incremented by input index (depth 7: `20260825`, depth 8:
  `20260826`)

The microbenchmark uses independent FENs. It measures the cost/tree effect of
the V41 logic, but it does not contain real repeated histories and does not
measure playing strength.

The raw benchmark format records per-round nodes and elapsed time, not score or
best move, and does not embed source/binary/model/host hashes. The clean
snapshot and model identity were controlled by the surrounding run procedure
and are preserved in the tracked evidence manifest. Future versions of this
benchmark should make those identities and output signatures self-contained.

## Results

| Depth | V40 nodes | V41 nodes | V41/V40 nodes | V41/V40 NPS (95% CI) | V41/V40 time (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 7 | 23,341,220 | 22,675,020 | 0.97146 | 0.98694 [0.97713, 0.99404] | 0.98432 [0.97728, 0.99420] |
| 8 | 35,335,200 | 33,392,420 | 0.94502 | 0.99746 [0.99361, 1.00235] | 0.94742 [0.94280, 0.95109] |

Interpretation:

- V41 searched 2.85% fewer nodes at depth 7 and 5.50% fewer at depth 8.
- Per-node throughput was about 1.31% lower at depth 7 and statistically
  indistinguishable in this conditional interval at depth 8.
- Total wall time fell 1.57% and 5.26%, respectively, because the smaller tree
  outweighed the added rule checks.

## Correctness gate

The clean-snapshot `search_tests` and
`nnue_searcher_v41_repetition_tests` passed. Coverage includes repetition,
50-move handling, checkmate precedence, TT-score suppression and pawn-move
halfmove-clock reset/undo. The implementation also prevents artificial null
moves from extending real-game repetition history, but this clean-snapshot gate
did not contain a dedicated null-move/repetition regression test.

The current uncommitted cross-search TT-generation work is intentionally not
included in this report. Its acceptance test needs a persistent UCI process;
this process-per-position benchmark cannot measure TT reuse across real moves.

## Evidence

| Artifact | SHA-256 |
|---|---|
| depth-7 raw log | `06df97224c89ce47a7f01e5cbb22c5e0a4d721cca1a9e047e8d59f57ff235a12` |
| depth-8 raw log | `cd7df202490a83583b752b3811fe46cdbe193594dd8e85f96d233ffb0d550f35` |
| analysis JSON | `8d3901e2525342abccac7c0b2321625bb39830e80154569da87723db3c688acc` |
| analyzer source | `cb8b1ea982202e9007e8d8b852ac2782935b4497fb295a0a42eda0adc1e3bdbd` |

The raw logs and generated analysis are retained in the tracked
[revalidation raw bundle](evidence/milestone_revalidation_20260825.tar.gz),
SHA-256
`c483ccebc01455b32fc7c378c5cf79f3bc7330278e140fb77ac60c40dedbfc90`.
