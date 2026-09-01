# Chess NNUE optimization context

Updated: 2026-09-01

This file is a handoff for continuing the current NNUE optimization work in a
new Codex tab. The relevant repository is:

```text
/Users/tunglamnguyen/Chess
```

The authoritative chronological status and benchmark limitations now live in
`docs/milestones/README.md` and `docs/benchmarks/README.md`. This file keeps the
lower-level optimization history.

## User intent and constraints

- Optimize the production NNUE implementation inside `Chess`.
- Do not modify the separate experimental `candidate` project/implementation
  unless explicitly requested.
- Preserve exact White/Black perspective semantics and incremental parity.
- Prefer readable C++ and verify generated assembly before introducing
  explicit intrinsics.
- Judge optimizations with fixed-node paired benchmarks, not one sequential
  before/after run.
- V43 is the deployed production searcher, using combined tuning config
  `98b7732c9587...`; V41 remains the previous production anchor and V42 is an
  experimental adaptive-aspiration branch. The operator explicitly authorized
  promotion before the frozen 600-game V43 confirmation finished. The race was
  then stopped at 421 games: 210 complete pairs scored 55.238%, paired CI95
  52.059%-58.417%, with a 95.607% aggregate node ratio. Heroku release `v17`
  deployed source `de3b5ab`; `v16` is the V41 rollback anchor.
- Preserve unrelated user changes when continuing optimization work.

## Current model and inference architecture

The production model is loaded through:

```text
include/phase_quantized_nnue.hpp
src/phase_quantized_nnue.cpp
src/phase_quantized_nnue_forward.cpp
src/searchers/strict/nnue_searcher_v36.cpp
```

The default model path is:

```text
models/quantized_scale_grid/
old_score_huber200_lr_sweep_then_5ep_20260724_142758/
best/phase_quantized_nnue.bin
```

Current default candidate scale tuple:

```text
hidden2_scale = 2
hidden3_scale = 8
output_scale  = 128
hidden_clip   = 181
SCReLU divisor = 128
```

Logical input:

```text
first 128 lanes  = side-to-move perspective accumulator
next 128 lanes   = opponent perspective accumulator
```

The first activation is:

```text
floor(clamp(x, 0, 181)^2 / 128)
```

L2 and L3 use clipped ReLU16 with their compile-time scales.

There are eight independently trained phase stacks. The selected phase is:

```text
phase = min((piece_count - 1) / 4, 7)
```

PSQT is maintained for both White and Black perspectives with eight buckets.
At evaluation:

```text
psqt =
    (psqt[side_to_move][phase] - psqt[opponent][phase])
    / (2 * psqt_scale)

score = positional + psqt
```

## Sparse piece feature layout

Each normal feature depends on:

```text
relative king square x piece type x friendly/enemy x relative piece square
```

`FeatureRow` is feature-major, aligned to 32 bytes, and exactly 160 bytes:

```cpp
struct alignas(32) FeatureRow {
    std::array<std::int8_t, 128> positional;
    std::array<std::int32_t, 8> psqt;
};
```

Both perspective accumulators have 128 `int32_t` lanes.

`update_features<AddedCount, RemovedCount>()` updates a single perspective.
For ordinary moves it is called once for White and once for Black.

Current make behavior:

```text
quiet:  <1,1> for White and Black
capture: <1,2> for White and Black
king move:
    incrementally update only the opposite perspective
    rebuild the moving king's perspective
```

The old reverse-update undo path has been removed; see the snapshot section.

## Rebuild optimization already implemented

`rebuild_perspective()` collects active feature rows, then processes the
accumulator in blocks of 16 lanes:

```text
load 16 bias values
add those 16 weights for every active row
store 16 accumulator values once
```

The code is ordinary C++, but Apple Clang keeps the 16 values in NEON
registers, unrolls the lane loop, and emits widening SIMD adds without spills.

Paired microbenchmark:

```text
legacy rebuild: 232.296 ns / perspective
block-16:       157.300 ns / perspective
speedup:        1.477x
```

In the full search sampling profile, rebuild fell from 9.31% to about 6.83%.

Microbenchmark source:

```text
tools/profile/benchmark_nnue_rebuild_perspective.cpp
```

## Historical snapshot-for-every-move undo

This design was implemented and benchmarked, but has now been superseded by
the per-ply state stack described below.

The old `PhaseQuantizedNnueUndo` contained:

```cpp
std::size_t piece_count_before;
std::array<std::array<std::int32_t, 128>, 2> accumulator_snapshot;
std::array<std::array<std::int32_t, 8>, 2> psqt_snapshot;
std::array<Square, 2> king_square_snapshot;
```

Every move saved the complete state before updating:

```cpp
undo.piece_count_before = piece_count_;
undo.accumulator_snapshot = accumulators_;
undo.psqt_snapshot = psqt_accumulators_;
undo.king_square_snapshot = king_squares_;
```

Undo restored that state instead of computing reverse feature rows. This
layout is historical and is no longer present in the current source.

Tests after this change:

```text
phase_quantized_nnue_tests: pass
nnue_searcher_v36_strict_tests: pass
Python/C++ parity: 8192 positions pass
search node counts and results unchanged
```

### Clean reverse-vs-snapshot benchmark

A separate reverse-undo binary was built in a temporary directory. Both
binaries passed the same parity and strict-search tests. They were benchmarked
in alternating A-B-B-A order on the same model, 10 fixed FENs, depth 6, NEON,
and 18,460,875 nodes per block.

Eight blocks per implementation:

```text
reverse total:  26,784,255 us
snapshot total: 26,005,650 us

reverse aggregate: 5.514M NPS
snapshot aggregate: 5.679M NPS
snapshot speedup:   2.994%
```

Four A-B-B-A block improvements:

```text
+3.67%
+1.12%
+0.05%
+7.32%
```

Block bootstrap 95% interval was approximately:

```text
[+0.59%, +5.72%]
```

The earlier sequential estimate of +8% was noise and must not be reused.
The defensible current estimate is about +3% NPS.

## Historical sampling profile after snapshot undo

Depth 6, 10 fixed FENs, NEON, 12,782 samples:

```text
Candidate forward                         24.88%
_platform_memmove                         9.93%
PhaseQuantizedNnueModel::evaluate wrapper 7.05%
rebuild_perspective                       6.86%
update_king_safety_after_move             4.63%
update_features total                     5.63%
make_move_with_undo self                  2.35%
NNUE undo self                            0.70%
```

Profile:

```text
logs/nnue_snapshot_all_depth6_sample_20260726.txt
```

Before snapshot undo, `update_features()` was 12.35%:

```text
<1,2> make capture: 4.19%
<2,1> undo capture: 4.03%
<1,1> make and undo quiet: 4.12%
```

After snapshot undo:

```text
<1,2> make capture remains
<1,1> contains only make quiet
<2,1> is gone
update_features total = 5.63%
```

`memmove` increased from about 3.46% to 9.93% because it now contains:

```text
roughly 3.5%: copy two accumulators into dense_input before forward
roughly 6.4%: save and restore move snapshots
```

The exact split is estimated from the before/after profiles.

## Per-ply NNUE state stack currently implemented

The old snapshot undo performed two copies per move:

```text
make: current state -> undo snapshot
undo: undo snapshot -> current state
```

The current design is a preallocated state stack:

```cpp
struct NnueState {
    std::array<PerspectiveAccumulator, 2> accumulators;
    std::array<PsqtAccumulator, 2> psqt;
    std::array<Square, 2> king_squares;
    std::size_t piece_count;
};
```

Current operation:

```text
make:
    states[ply + 1] = states[ply]
    increment ply
    update states[ply]

undo:
    decrement ply
```

Implementation details:

```text
PhaseQuantizedNnueUndo contains only ply_before (8 bytes).
states_ is a std::vector<NnueState>.
reset() reserves 256 states once without constructing/zeroing all of them.
push_state() copies the parent state into a child state once.
previously reached child slots are reused.
undo() only restores current_ply_ and performs no copy.
```

Relevant locations:

```text
include/phase_quantized_nnue.hpp:129
include/phase_quantized_nnue.hpp:172
src/phase_quantized_nnue.cpp:346
src/phase_quantized_nnue.cpp:368
src/phase_quantized_nnue.cpp:685
```

Validation:

```text
phase_quantized_nnue_tests: pass
nnue_searcher_v36_strict_tests: pass
incremental/full recompute parity: pass
node counts and search results unchanged
```

Clean alternating benchmark against snapshot save+restore, eight blocks each,
18,460,875 nodes per block:

```text
snapshot restore total: 25,528,501 us
state stack total:      24,721,640 us

snapshot restore: 5.785M aggregate NPS
state stack:      5.974M aggregate NPS
state speedup:    3.264%
```

All four A-B-B-A blocks favored the state stack:

```text
+0.95%
+4.09%
+2.35%
+5.74%
```

Block bootstrap 95% interval:

```text
[+1.64%, +4.91%]
```

Latest depth-6 sampling after the state stack:

```text
Candidate forward                         25.59%
PhaseQuantizedNnueModel::evaluate wrapper 7.93%
rebuild_perspective                       7.34%
_platform_memmove                         6.89%
update_features total                     8.06%
update_king_safety_after_move             4.01%
make_move_with_undo self                  1.56%
push_state self                           0.85%
undo self                                 0.11%
```

The absolute percentages vary with CPU frequency, but the important direct
change is:

```text
memmove: 9.93% -> 6.89%
undo:     0.70% -> 0.11%
```

Profile:

```text
logs/nnue_state_stack_depth6_sample_20260726.txt
```

## Direct two-accumulator forward currently implemented

The production NEON candidate no longer builds a temporary 256-lane
`dense_input`. `PhaseQuantizedNnueModel::evaluate()` passes the two 128-lane
arrays directly:

```cpp
forward_positional_candidate(
    accumulators[side_to_move],
    accumulators[opponent],
    aux,
    phase_index);
```

The candidate treats them as one logical input without copying:

```text
logical inputs 0..127   = STM accumulator
logical inputs 128..255 = opponent accumulator
```

For every 16-lane block it:

```text
loads four int32x4 accumulator registers
widens and adds each active aux int8 row in those registers
applies SCReLU
feeds the activation directly to the packed hidden2 dot product
```

The packed hidden2 weight pointer continues across both halves, so weight order
is unchanged. A no-aux template branch avoids the aux loop entirely when no aux
feature is active.

The old dense-input implementation remains as:

```text
forward_positional_scalar()
```

It is the non-NEON fallback and a Debug parity oracle. Debug builds construct the
old dense input and assert exact equality with the candidate result.

Validation:

```text
phase_quantized_nnue_tests: pass
nnue_searcher_v36_strict_tests: pass
candidate/scalar exact parity: pass
search node counts and best moves unchanged
```

Clean A-B-B-A benchmark against the state-stack binary before this change,
eight measurements each and 11,076,525 nodes per measurement:

```text
old dense-input path: 14,797,245 us, 5.988M aggregate NPS
direct two-acc path:  14,085,497 us, 6.291M aggregate NPS
speedup: 5.053%
```

Latest depth-6 sampling:

```text
Candidate forward 3939 / 12798 samples = 30.78%
_platform_memmove  544 / 12798 samples = 4.25%
```

`memmove` fell from 6.89% after the state-stack change to 4.25%; the remaining
copy is principally the one parent-to-child state copy in `push_state()`.

Profile:

```text
logs/nnue_direct_two_acc_depth6_sample_20260726.txt
```

## Combined aux table currently implemented

The direct two-accumulator kernel originally widened and added every active
`int8` aux row separately. Assembly already fused the second widening and add
into `SADDW`, but each active row still cost:

```text
1 vector load + 2 SSHLL + 4 SADDW
```

The current implementation precomputes all valid aux combinations when loading
the candidate kernel:

```text
16 castling-right combinations
x 9 en-passant states (none or files A-H)
= 144 combined rows
```

Each combined row has 256 `int16_t` values. Total storage is 73,728 bytes and
the on-disk model format is unchanged. The sum is safely inside `int16_t`
because at most six original `int8` rows are active.

For each 16-lane input block, current assembly uses:

```text
LDP q20, q21
SADDW
SADDW2
SADDW
SADDW2
```

There is no aux-row loop and no `SSHLL` in the candidate hot path. State zero
(no castling rights and no en passant) still uses the no-aux path.

Validation now explicitly checks all 144 aux states from both side-to-move
perspectives against the scalar evaluator, in addition to the incremental and
searcher tests.

Clean A-B-B-A benchmark against the per-row direct-forward binary, eight
measurements each and 11,076,525 nodes per measurement:

```text
per-row aux:   14,138,891 us, 6.267M aggregate NPS
combined aux:  13,790,339 us, 6.426M aggregate NPS
speedup: 2.528%
```

All four A-B-B-A blocks favored the combined table.

Latest profile:

```text
logs/nnue_combined_aux_depth6_sample_20260726.txt
```

## Two output accumulation chains currently implemented

The final 32-neuron output dot product now uses two independent `int64x2_t`
accumulators and combines them once at the end. Assembly keeps two separate
`SADALP` dependency chains and emits one final `ADD.2D`.

Validation:

```text
phase_quantized_nnue_tests: pass
nnue_searcher_v36_strict_tests: pass
```

The full-search effect is effectively neutral. Across sixteen alternating
measurements per binary, 18,460,875 nodes per measurement:

```text
single output sum: 6.4446M aggregate NPS
two output sums:   6.4364M aggregate NPS
change:           -0.126%
```

The change is retained because it was explicitly requested, but it should not
be described as an NPS improvement. The extra final vector add offsets the
benefit of the shorter dependency chains on this CPU.

## Aux features

There are 13 binary aux features:

```text
0  friendly can castle kingside
1  friendly can castle queenside
2  enemy can castle kingside
3  enemy can castle queenside
4  has en passant
5  en-passant file A
6  en-passant file B
7  en-passant file C
8  en-passant file D
9  en-passant file E
10 en-passant file F
11 en-passant file G
12 en-passant file H
```

Friendly/enemy is relative to `pos.side_to_move`. If en passant exists,
exactly `HasEnPassant` and one file feature are active. At most six rows can be
active at once: four castling rows plus two en-passant rows.

Each aux row has 256 `int8_t` weights and is added after choosing STM/opponent
accumulator order but before the first SCReLU.

Aux does not currently include:

```text
side-to-move bit
in-check
rule50
ply
phase
```

### Pending aux design question

The last open user question was whether aux should be incremental or evaluated
inside forward.

Current recommendation:

- Keep the aux weight vector out of the incremental accumulator state.
- Compute or maintain only a cheap 13-bit `AuxMask`.
- Fuse active aux rows into the first activation inside the direct-two-
  accumulator NEON forward.

Reasoning:

- Castling rights and en passant are global state rather than king-bucketed
  piece features.
- Friendly/enemy meaning swaps with side-to-move every ply.
- En passant is transient and can change every move.
- Maintaining a full 256-lane aux correction incrementally would add another
  1 KiB to every per-ply state and increase state-stack copy traffic.
- At most six aux rows are active, and often none after the opening.

If profiling later shows aux fusion itself is expensive, benchmark a stored
`AuxMask` first. Do not add a 256-lane incremental aux accumulator without a
paired performance result.

## Planned V38 LMR + null-move tuning dataset

The selective-search tuner must not evaluate all 200 million NNUE records.
That corpus is only the main source from which a much smaller, fixed and
reproducible search benchmark is sampled. Evaluating all 200M positions with
strict search would be far beyond the requested approximately two-hour budget.

The planned benchmark contains **8,000 unique positions total**, split before
any mutation is evaluated:

```text
tune pool:       4,000 positions
selection set:   2,000 positions
final holdout:   2,000 positions
total:           8,000 unique positions
```

The three splits must not overlap. A position must also occur at most once
inside a split. Save the sampling seed and a stable position hash so this can
be audited and reproduced.

Every split uses the requested category distribution:

```text
70% random positions from the NNUE 200M corpus
15% balanced opening/game positions
10% tactical, only-move, or mate-threat positions
 5% pawn endings or zugzwang-sensitive positions
```

Exact counts:

```text
                         tune  selection  holdout
NNUE random (70%)        2800       1400     1400
balanced (15%)            600        300      300
tactical (10%)            400        200      200
endgame (5%)              200        100      100
total                    4000       2000     2000
```

During mutation, one candidate is not run over all 4,000 tune positions.
Instead, evaluate it on a deterministic rotating subset of **200 positions**:

```text
140 NNUE random
 30 balanced
 20 tactical
 10 endgame
```

Rotate the subset across mutations so hundreds of candidates are not fitted to
the same tiny set, as happened in the previous 48-position move-order tune.
The control and mutant for one comparison must always receive exactly the same
positions, in the same order, with clean TT state and identical search limits.

Use the sets in three distinct roles:

```text
tune pool:
    fast mutation feedback only; rotating 200-position subsets

selection set:
    re-evaluate promising candidates from the tune Pareto frontier on all
    2,000 positions; only this result may promote a candidate

final holdout:
    run only on the small final selection frontier after tuning is finished;
    never use its result to continue mutation
```

The principal efficiency metric is aggregate node ratio:

```text
sum(candidate nodes) / sum(V36 control nodes)
```

Accuracy is measured using root regret under the strict control search:

```text
root regret =
    strict score of the V36 best move
    - strict score of the candidate move
```

Report at least mean root regret, p95 root regret, percentage above 100 CP,
mate mistakes, move agreement, time ratio, and LMR/null-move hit and cutoff
counts. A candidate with a mate mistake is rejected. Retain a Pareto frontier
instead of collapsing node ratio and root regret into one scalar fitness:
candidate A dominates B only if A is no worse in both aggregate node ratio and
mean root regret, and strictly better in at least one.

Relevant sources:

```text
main random corpus:
data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m

corpus manifest:
data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2.manifest.json

balanced opening lines:
data/stockfish_balanced_openings_10ply_1k_20260723.txt
```

There is currently no clean standalone tactical/endgame FEN file. Build those
categories from the corpus without overlap:

```text
tactical proxy:
    positions with a legal capture, check, or promotion; optionally refine
    with strict-search move-score spread to identify only-move positions

pawn-ending proxy:
    no queens, rooks, bishops, or knights, with at least one pawn
```

The CBIN2 corpus uses 40-byte records. Existing FEN reconstruction code can be
adapted from:

```text
tools/evaluate_heuristic_on_cbin.cpp
```

The two-hour target is a compute budget, not permission to reuse selection or
holdout positions during mutation. If search throughput is lower than
estimated, reduce the number of mutations before reducing or contaminating the
selection/holdout sets.

## Build, test, and benchmark commands

CMake is not on `PATH`; use:

```bash
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake
```

Build:

```bash
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake \
  --build build-release \
  --target phase_quantized_nnue_tests \
           nnue_searcher_v36_strict_tests \
           benchmark_nnue_v36_vs_heuristic_v35 \
  -j8
```

Focused tests:

```bash
ctest --test-dir build-release --output-on-failure \
  -R 'phase_quantized_nnue_tests|nnue_searcher_v36_strict_tests'
```

Stable fixed-node benchmark:

```bash
./build-release/benchmark_nnue_v36_vs_heuristic_v35 \
  --only nnue \
  --depth 6 \
  --fixed-count 10 \
  --repeats 10 \
  --nnue-kernel neon
```

For optimization comparisons, build two independent binaries and run several
alternating A-B-B-A blocks. Compare aggregate microseconds because node counts
are identical.

## Current adversarial LMR/NMP pipeline

The active replacement for the original selective-search tuner is documented
in:

```text
docs/nnue_selective_adversarial_pipeline.md
```

Its entrypoints are:

```text
tools/build_nnue_selective_safety_bank.py
tools/tune_nnue_lmr_nmp_adversarial.py
tools/run_nnue_lmr_nmp_adversarial_8h.sh
```

The pipeline starts from strict V36 and tunes LMR-only, NMP-only, and joint
lineages with RFP/LMP explicitly disabled. Historical critical and near-miss
positions are hash-deduplicated into a mutation-visible `core.tsv` and a sealed
`sealed.tsv`. Candidate flow is:

```text
core at depths 5/6/7/8
  -> fixed 2k tune set at depth 6
  -> sealed adversarial selection at depth 7
  -> fresh normal selection at depth 7
  -> fresh holdout at depth 7
  -> spread final frontier at depth 8
```

Do not merge the safety-bank rows into the mean-regret dataset. They are a hard
gate only. The end-to-end runner generates a new 16k `8k/4k/4k` dataset while
excluding every previous V38/V39 tune, selection, and holdout split.

## Current strongest V39 baseline: Fast

As of 2026-08-16, **Fast is the historically promoted V39 selective-search
baseline** and is the default `NnueSearcherV39::SelectiveConfig`. Use this
profile as the parent/control for future pruning experiments:

```text
LMR: enabled, base=0.45, divisor=2.9, min_depth=3, min_move_index=6
NMP: enabled, min_depth=2, reduction=3
RFP: enabled, max_depth=2, base_margin=175, margin_per_depth=275
LMP: enabled, max_depth=3, base=4, depth_multiplier=2

time-gauntlet profile:
fast_joint,0.45,2.9,3,6,2,3,1,2,175,275,1,3,4,2
```

Direct color-reversed self-play under the current paired-opening and early-stop
CI rule produced:

```text
Fast vs Balanced:         21W 59D 10L, 56.11%, CI95 [50.05%, 62.17%]
Fast vs baseline 7:       20W 55D  9L, 56.55%, CI95 [50.29%, 62.81%]
Fast vs all-four winner:  21W 59D 10L, 56.11%, CI95 [50.05%, 62.17%]
```

The last row is the reciprocal of the recorded `all4_winner` score (43.89%).
The matches repeatedly inspected an ordinary interval and stopped when it first
excluded 50%; the gauntlet also retained history/killer/counter heuristics
between games. Treat the intervals as the historical project promotion rule,
not sequentially valid confidence intervals or a fixed-sample Elo proof. All
recorded direct scores favor Fast, but the evidence is directional. The newer
offline four-prune winner must not replace Fast: it lost their direct match.

Relevant result files:

```text
logs/nnue_v39_audit3_round_robin_ci_20260802_120129/summary.stopped.json
logs/nnue_v39_all4_winner_vs_fast_ci_20260816_153741/summary.json
```

The historical adversarial LMR/NMP pipeline above remains useful as provenance,
but its strict-V36 seed and hard-gate policy are not the current runtime
baseline. New V39 tuning should start from Fast, keep it as an explicit control,
and require a direct paired-opening self-play win before changing the default.

## V40 QSEE: adopted in V41, strength inconclusive

V40 is the first experimental child of the historically promoted V39/Fast baseline. It
keeps all Fast parameters unchanged and enables quiescence static-exchange
evaluation pruning. The completed 2026-08-17 threshold tune selected
`qsearch_see_threshold = -75cp` as the balanced self-play candidate. On the
fresh 6k-position depth-8 holdout it reduced candidate nodes by 15.50% and
measured time by 11.36% versus QSEE off, while mean WDL loss moved from
0.00501209 to 0.00512550. Threshold zero saved slightly more nodes but was
slower in wall time and had higher WDL loss.

QSEE applies only to non-promotion captures in quiescence when the side to move
is not in check. A capture is skipped only when its SEE is below the configured
threshold and it does not give check. Check evasions, promotions, checking
captures, and en-passant captures at the current SEE value of zero are kept.
V38 and V39 expose the same config fields but leave QSEE disabled, preserving
V39/Fast as an unchanged control.

The one-dimensional tune evaluated the following full grid before Pareto
selection and fresh holdout:

```text
OFF, -600, -500, -400, -350, -300, -250, -225, -200,
-175, -150, -125, -100, -75, -50, -25, 0
```

Use `evaluate_nnue_v40_selective` for fixed-depth/fixed-time evaluation and
`nnue_v40_time_gauntlet` for paired-opening self-play. The direct V40/V39 match
finished 116W-392D-92L for V40, 52.0%, nominal CI [49.71%, 54.29%], so it did
not prove a playing-strength win. QSEE `-75cp` was nevertheless adopted in V41
on its measured efficiency evidence.

The retained V40 tune and match summaries do not include a complete
Git/binary/model/suite/harness identity set. Under the current milestone
vocabulary this is historical efficiency evidence with provenance caveats,
not a fully reproducible validated-performance artifact.

Relevant tune result:

```text
logs/nnue_v40_qsee_tune_20260817_005158/summary.json
```

## Historical production V41: in-search draw rules

V41 wraps the V40/QSEE `-75cp` search profile and moves draw handling into the
search tree. The UCI adapter replays the real game moves into a hash history;
each real search move then pushes its child hash onto a fixed-capacity,
path-local stack. Artificial null moves do not modify or query this stack.

At every real node V41 returns an exact score of zero for the third occurrence
or when `halfmove_clock >= 100`. Checkmate takes precedence over the 50-move
draw. Once any position has occurred twice on the active reversible path, TT
score cutoffs and TT stores are suppressed while the TT move hint remains
usable. Pawn moves, captures, and permanent castling-right changes delimit the
relevant history segment.

Implementing the 50-move rule exposed and fixed a core metadata bug: a normal
single-square pawn push uses `MoveFlag::Quiet`, but must reset
`halfmove_clock` to zero rather than incrementing it. Make/unmake regression
coverage now verifies the reset and restoration explicitly.

The clean-snapshot 2026-08-25 paired V40/V41 rerun with the production NNUE
model measured:

```text
depth 7: node ratio 97.1458%, aggregate NPS ratio 98.6937%, time ratio 98.4316%
depth 8: node ratio 94.5019%, aggregate NPS ratio 99.7464%, time ratio 94.7422%
```

See `docs/benchmarks/nnue_v41_draw_rules_20260825.md` for the protocol,
bootstrap intervals and limitations.

The former production Lichess deployment used `uci_nnue_v41`, the current F2
Huber model, and no external `AvoidDraw` options. Release v11 first deployed
the x86 SIMD engine from commit `803aeb5`. Release v12 then enabled LTO from commit
`634b2d4`; its retained smoke engine SHA-256 is
`2b9bedd073bc6d15464567cc9a716a1b7e0b3f7227ed7b293b9f9bb2b7ba8128`.
The smoke reported `x86_avx512vnni_256` and reproduced the canonical
start-position depth-7 signature (`25cp`, 33,514 nodes, `e2e4`). The full
four-dyno paired A/B measured +6.15% at depth 7 and +6.77% at depth 8, with
both ratio intervals above 1.0. Release v11 is the previous non-LTO runtime.
