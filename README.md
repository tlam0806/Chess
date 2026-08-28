# Chess

A learning-oriented C++20 chess engine built from bitboards upward. The project
now includes a long classical-search lineage, phase-aware quantized NNUE,
selective V41 search, a UCI adapter, and a documented Lichess deployment.

The project goal is not to clone Stockfish directly. It is a staged engine
project for learning the core systems work behind chess engines: bitboards,
legal move generation, perft, search, evaluation, neural evaluation, profiling,
and production deployment.

This README is the thematic project story. The tracked
[benchmark index](docs/benchmarks/README.md) retains the versioned reports,
artifact hashes, negative experiments, and promotion-gate details that have
already been packaged for the repository.

## How milestone claims are classified

The repository deliberately keeps old searchers, benchmark tools, and negative
experiments. Each milestone claim therefore has two independent labels:

| Dimension | Typical labels | Question answered |
|---|---|---|
| Lifecycle | experimental, historically promoted, deployed, rejected | What did the project do with the change? |
| Evidence | validated, validated with caveats, reconstructed, incomplete, inconclusive | How strongly do the retained artifacts support the claim? |

These labels prevent a newer version or one isolated metric from being mistaken
for a stronger engine: lower model loss, fewer nodes, higher NPS, and playing
strength are different results. Interpreting a new performance or strength
claim requires pinned artifacts, workload, correctness signatures, and stopping
rule.

## Four project milestones

The project is easier to understand as four engineering milestones than as a
list of every searcher version. The version numbers still matter for
reproducibility, but each milestone marks a step change substantial enough to
redefine how the engine was designed, searched, evaluated, or trained, rather
than another incremental version.

| Milestone | Main question | Result |
|---|---|---|
| 1. Correctness, simple search, heuristic evaluation | Can the engine represent chess correctly and return a defensible move? | A legal bitboard engine, perft/tests, plain alpha-beta, and a material-plus-piece-square evaluator. |
| 2. Range-correct Strict search and clean optimization | How much search work can be removed while preserving the same fixed-depth score or a mathematically valid alpha-beta bound? | Bound-correct TT, PVS/re-search, ordering, fixed move storage, magic attacks, lazy/staged legality, make/unmake, and cache/layout work, with per-position score/move/node gates where evidence survives. |
| 3. Neural evaluation becomes both stronger and practical | Can an NN evaluator justify its much higher cost? | The historical NNUE beat the heuristic engine while searching one ply less; incremental accumulators and SIMD kernels then made NN inference production-viable. |
| 4. Controlled selective or “dirty” pruning | How aggressively can the engine prune unlikely moves without causing tactical blunders? | LMR, null move, RFP, LMP, and QSEE were evaluated on large position sets, safety banks, Pareto frontiers, and finally paired self-play. |

This README uses those four thematic milestones. The version names remain in
the source tree and Git history for finer-grained reconstruction; the four
groups here describe the engineering story rather than replacing those source
anchors.

## Detailed milestone history

### 1. Correctness, simple search, and a heuristic evaluator

The first milestone was not speed. It was building a position representation
that could be trusted.

The initial engine established:

- bitboards with `a1 = 0` and `h8 = 63`;
- FEN parsing and complete position state;
- pawn, knight, king, sliding, and occupancy-dependent attacks;
- pseudo-legal and legal move generation;
- castling, en passant, promotion, check, and legal handling of pinned-piece
  cases;
- `make_move` and known-count perft validation;
- material and piece-square evaluation;
- plain negamax alpha-beta search.

At this stage search copied a `Position` for each child. In-place undo state and
RAII make/unmake guards came much later, during the V20-V27 optimization wave.
That distinction matters: the first milestone proved rules and state
transitions before trying to make them cheap.

The evaluator was intentionally understandable:

```text
score = material balance + piece-square terms
```

One chess point is represented as 100 centipawns (`cp`). The base material
values in the original evaluator were:

| Piece | Chess shorthand | Engine value |
|---|---:|---:|
| Pawn | 1 point | 100cp |
| Knight | 3.2 points | 320cp |
| Bishop | 3.3 points | 330cp |
| Rook | 5 points | 500cp |
| Queen | 9 points | 900cp |
| King | Not exchangeable | 0cp in static material |

The king does not need a capture value in normal evaluation because a legal
game ends at checkmate rather than with the king being captured. Checkmate is
therefore represented by the search as a separate terminal score, not as the
material value of a king.

Material alone would consider every square equivalent. The piece-square table
(`PST`) adds a small handcrafted bonus or penalty according to where each
piece stands. For example, a knight in the center can receive `+20cp`, while a
knight in a corner receives `-50cp`. The table is mirrored by color, so White
and Black receive the same preference from their own perspective.

Conceptually, the calculation was:

```text
piece contribution = base piece value + PST[piece][relative square]
white score        = sum(White contributions) - sum(Black contributions)
side-to-move score = white score if White moves, otherwise -white score
```

Thus winning an otherwise equal rook changes the score by roughly `+500cp`,
whereas improving a knight's square might change it by only a few dozen
centipawns. This evaluator did not directly understand mobility, pawn
structure, threats, long-term king safety, or game phase beyond what the fixed
tables happened to encode. Alpha-beta search supplied tactical correction by
looking ahead; the heuristic supplied the leaf score when that lookahead
stopped.

It was fast, deterministic, and expressive enough to be a useful teacher and
control. The simple alpha-beta tests compared its score with a small full
negamax reference suite and checked that the returned move realized that
score. They did not establish identical node counts or a unique best move when
several moves tied.

This correctness-first core remains the foundation under every later Strict,
NNUE, and selective searcher.

### 2. Range-correct Strict search and clean optimization

The project uses **Strict** for a correctness/debug line intended to make
fixed-depth comparisons and score bounds easier to reason about while data
structures and hot paths change. It is an engineering intent, not a claim that
every historical Strict version has already been proven behavior-identical.
“Clean pruning” is useful shorthand, but there are three technically different
things inside this milestone:

1. **Proven cutoffs:** alpha-beta and valid TT bounds can skip a subtree because
   the current window already proves it irrelevant.
2. **Work reduction without dropping moves:** move ordering, fixed move lists,
   magic attacks, cached state, and staged/lazy legality retain the legal move
   space and score contract while reducing implementation work. Ordering and
   proof cutoffs may still change which nodes are visited.
3. **Correct re-search protocols:** PVS and aspiration may search a narrow
   window first, but re-search when the narrow result is insufficient.

These are different from milestone 4, where a heuristic intentionally decides
that a move probably does not deserve a full search.

#### What was optimized in the Strict line

| Area | Evolution | Contract and caveat |
|---|---|---|
| Transposition table | Direct-mapped TT in V2; explicit bound behavior in V14; range-storing exact-depth Strict reuse in V15; bucketed range TT in V19; lower-only V33; range-preserving 12-byte V35 | Strict reuses a score only at the requested depth. V34's compact single-bound TT deliberately reused deeper entries and is therefore classified as a Fast experiment, not part of the Strict score contract. Replacement policy and the TT move can still change ordering, nodes, and tied best moves. |
| Move ordering | Tactical/static ordering, TT move, SEE-ranked captures, history, killers, counter-history, and staged scores | Ordering normally keeps every move and changes only when it is searched. It can dramatically change alpha-beta node count and tie-breaking, so node equality is not assumed. |
| `MoveList` | Heap-backed collections were replaced by fixed-capacity raw-storage management | Removes allocation and improves locality without changing the generated move set. The primitive was shared by both sides of the retained V18/V19 comparison, so that comparison cannot isolate its speedup. |
| Sliding attacks | Ray work was replaced by magic-bitboard lookup | Exhaustive relevant-blocker and random-occupancy tests support primitive equivalence. No isolated historical search-level speed A/B survives. |
| Lazy and staged generation | V19 generated the pseudo-legal list and filtered legality while scoring; V23+ split noisy/quiet stages; V28/V29 introduced dedicated legal callbacks | “Lazy legality filtering” is more precise for V19. The important invariant is that every legal move remains reachable until a proof-producing cutoff ends the node. |
| SEE | First used to order captures, then specialized to reuse already-known moved/captured pieces | SEE used only for ordering is clean. Using SEE to discard a capture is selective pruning and belongs to milestone 4. |
| King safety | V19 cached checkers, pins, king square, and block masks within a node; later position state gained persistent per-color caches | Avoids recomputation while retaining the same legal-move result. |
| Make/unmake | Early search copied positions; V26 used in-place undo state and V27 added RAII guards | Round-trip tests cover quiet moves, captures, en passant, promotions, castling, and random plies. |
| PVS and aspiration | Narrow scout/window searches plus required re-search | Correct re-search preserves the final score, although it normally changes nodes. |

V2-V13 were an exploratory search laboratory and already tried quiescence,
LMR, null move, PVS, history, and killers. Not all of those experiments were
behavior-preserving. V14 made bound behavior explicit; V15 introduced the
range-storing, exact-depth hand-written Strict line. V16-V33 and V35 then
concentrated the controlled TT, ordering, move-generation, state, and layout
work above. V34 branched into the Fast line to test a compact single-bound TT
with `AtLeast` depth reuse: a deeper result may accelerate the search, but it
is not guaranteed to equal the score requested at the shallower fixed depth.
Aggressive LMR/null-move tuning is treated separately in milestone 4.

#### What “exact behavior” means in this repository

A fair Strict comparison pins the corpus, depth mode, compiler, flags, TT size,
and starting heuristic state, then checks the result per position:

- score equality is the primary semantic gate;
- tied-optimal moves must be handled explicitly before claiming best-move
  equality;
- node equality is not part of the Strict contract; it is required only before
  claiming the same node count or using equal nodes to isolate per-node speed;
- timing is interpreted only after the score contract and workload relationship
  are known.

The main Strict benchmark series therefore excludes V34. It may appear as a
separate Fast/experimental comparison, but its time, nodes, and NPS describe a
different fixed-depth computation and cannot be presented as a clean Strict
speedup.

#### Measured Strict milestone progression

![Strict milestone depth-7 elapsed-time benchmark](docs/benchmarks/assets/strict-milestones-d7.svg)

| Version | Depth-7 time | Speedup vs V15 | Nodes | NPS |
|---|---:|---:|---:|---:|
| V15 | 268.322 s | 1.00× | 412,761,712 | 1.54M |
| V19 | 83.615 s | 3.21× | 477,666,976 | 5.71M |
| V24 | 57.512 s | 4.67× | 393,023,536 | 6.83M |
| V27 | 47.003 s | 5.71× | 402,033,696 | 8.55M |
| V29 | 42.357 s | 6.33× | 435,992,336 | 10.29M |
| V32 | 18.373 s | 14.60× | 446,454,896 | 24.30M |
| V33 | 21.493 s | 12.48× | 489,489,456 | 22.77M |
| V35 | 18.424 s | 14.56× | 446,454,896 | 24.23M |

- **Protocol:** 25 positions, direct depths 6–8, eight balanced rounds, fresh
  64 MiB searcher state per measurement on one Apple M4.
- **Correctness:** all 9,600 timed searches matched the reference score; there
  were no illegal moves, stopped searches, or depth failures. The only different
  move was force-verified as a tie.
- **Interpretation:** elapsed time is primary and NPS explains per-node cost.
  These are current-tree reconstruction results, not original historical-binary
  timings; depth 7 was stable, while depth 8 showed late thermal drift.

### 3. Neural evaluation becomes stronger and fast enough to use

#### Why I initially doubted the NN

I was initially skeptical that a neural evaluator was worth its cost. The
heuristic evaluator was extremely cheap, easy to debug, and already expressive
enough to describe material and many positional ideas. A neural evaluator is
called throughout the search tree, so a slow forward pass can erase any gain
from a better position score.

Early scalar inference reinforced that concern. The turning point was not just
training a model with lower loss. It was the combination of:

- a model that made better positional decisions;
- incremental features, so most moves update a few rows instead of rebuilding
  the whole input;
- quantized arithmetic;
- architecture-specific dot-product intrinsics.

The result was substantially stronger than I expected.

#### The phase-aware F2 NNUE

For each perspective, an active piece feature contains:

```text
piece type × friendly/enemy × perspective king square × piece square
```

The production F2 model stores 49,152 shared feature rows. Each row contributes
to a 128-lane accumulator for one king perspective. The two perspective
accumulators are concatenated, giving the dense network 256 values. Thirteen
auxiliary inputs represent castling and en-passant state without rebuilding a
second large accumulator.

```text
piece/king sparse features
        -> 128-lane STM accumulator
        -> 128-lane opponent accumulator
        -> concatenate to 256
        -> SCReLU
        -> 32
        -> 32
        -> 1 positional score
        + phase PSQT score
        -> centipawns
```

Other architectural details:

- eight material-phase dense heads;
- eight PSQT buckets accumulated alongside the positional rows;
- king moves rebuild only the affected king perspective;
- ordinary moves incrementally add/subtract feature rows;
- `int8` weights with wider accumulators and bit-exact scalar/SIMD parity;
- production scales `(hidden2=2, hidden3=8, output=128)`, clip `181`, and
  SCReLU divisor `128`.

The documented production artifact is 7,944,336 bytes with SHA-256
`a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`.

#### First match win over the heuristic engine

The historical `hs2x8_os128` NNUE was tested at depth 3 against
`HeuristicSearcherV35` at depth 4:

| Engine | Depth | W-D-L | Score | Nominal paired 95% CI | Nodes |
|---|---:|---:|---:|---:|---:|
| Historical NNUE | 3 | 46-132-22 | **56.0%** | 52.75%-59.25% | 29.0M |
| Heuristic V35 | 4 | 22-132-46 | 44.0% | 40.75%-47.25% | 79.2M |

That result changed the project direction: the NN won while searching one ply
less. The ordinary interval was inspected every five pairs and the run used an
early-stop rule, so it is directional historical promotion evidence, not a
modern sequential Elo proof. The winning model was also an older artifact, not
the current production model above.

See [the full NNUE-vs-heuristic result](docs/nnue_vs_heuristic_result.md).

#### Kernel work made the model operational

The hot mixed-sign dot product has a close native implementation on both
architectures:

```text
ARM: vsudotq_laneq_s32  -> SUDOT
x86: _mm256_dpbusd_epi32 -> VPDPBUSD
```

Both compute unsigned activation bytes × signed weight bytes into `int32`
accumulators. ARM uses a native-build NEON/I8MM kernel. x86 chooses at runtime
between AVX-512 VNNI, exact AVX2, and scalar fallbacks. Scalar/Python/ARM/x86
paths are required to produce the same integer evaluation.

On the matched three-dyno Heroku matrix, the production-style VNNI backend was
`6.08x` faster than scalar at depth 7 and `6.42x` faster at depth 8. LTO later
added another `6.15%` and `6.77%`. These are throughput results, separate from
the playing-strength result above.

See the [SIMD validation](docs/benchmarks/nnue_v41_x86_simd_heroku_20260825.md)
for the x86 implementation and parity gates.

### 4. Controlled selective (“dirty”) pruning

Strict search asks, “what result does this finite-depth search define?”
Selective search asks, “which work is unlikely to change the move enough to be
worth its cost?” The second question is deliberately approximate. In this
README, **dirty pruning** means a heuristic can skip or reduce work without a
mathematical guarantee that the finite-depth score remains identical.

#### Pruning mechanisms

| Mechanism | What it does | Main failure mode |
|---|---|---|
| LMR | Reduces only sufficiently late, quiet, non-capture, non-promotion, non-checking moves while not in check; a reduced score range that may exceed alpha is re-searched at full depth | A genuinely strong late move may look harmless at reduced depth and never earn the re-search. |
| Null-move pruning (NMP) | Gives the side to move a synthetic pass and makes a null-window search; it is disabled in check, inside another null search, and when the side has no non-pawn material | Zugzwang and positions where the obligation to move is itself harmful. This implementation has no verification search. |
| Reverse futility pruning (RFP) | At shallow non-PV nodes, outside check, null searches, and mate windows, cuts off when static evaluation exceeds beta by a depth-dependent margin; it is also disabled when the side to move has no non-pawn material | Tactical resources can invalidate the static margin. |
| Late-move pruning (LMP) | At shallow null-window nodes, skips ordinary quiet moves after `base + multiplier × depth²`; captures, promotions, and earlier priority stages have already been handled | A late quiet tactic or only move can be discarded. |
| Quiescence SEE pruning (QSEE) | Skips a non-checking, non-promotion capture only when `SEE < threshold`; checking captures, promotions, and evasions remain | A superficially losing sacrifice can be positionally or tactically correct. |

#### Evaluation was a pipeline, not one benchmark

The V38/V39 tuner compared each candidate with the finite-depth V36 Strict
control over thousands of positions. When a candidate selected a different
root move, that control re-searched the move and measured **root regret**:

```text
root regret = max(0,
                  strict score(best strict move)
                - strict score(candidate move))
```

This is disagreement with a pinned finite-depth teacher, not objective chess
error. V36 itself ends in a finite capture quiescence search, so a low regret
is useful selection evidence rather than proof that the move is best chess.

The active adversarial LMR/NMP pipeline used:

```text
historical safety-bank mining and deduplication
    -> fresh 16,000-position dataset (8k tune / 4k selection / 4k holdout)
    -> visible core safety gate at depths 5/6/7/8
    -> fixed 2,000-position gate drawn from the tune split at depth 6
    -> per-lineage node/regret Pareto frontier
    -> sealed adversarial selection at depth 7
    -> fresh 4,000-position selection and 4,000-position holdout
    -> at most eight spread points re-audited at depth 8
    -> later WDL and RFP/LMP selection
    -> paired, color-reversed self-play
```

Safety was not reduced to average CP loss:

- a Strict score of at least `+500cp` becoming `[-100,+100]` was win-to-draw;
- a Strict score of at least `+500cp` becoming less than `-100cp` was
  win-to-loss;
- a candidate move becoming mate-losing while the control was above `-100cp`
  was self-mate;
- any of those three events was a hard rejection, not an averaged penalty;
- six known critical rows plus 32 mined `>=250cp` near misses formed a
  38-position hash-deduplicated bank: 24 mutation-visible `core` rows and 14
  sealed rows; 41 configurations failed the core gate;
- mean regret was ranked only where the direct static NNUE target satisfied
  `|static_target_cp| < 1500`, but every position remained eligible for the
  hard safety checks;
- p95 regret, the fraction above `100cp`, move agreement, and prune counters
  were reported alongside the mean;
- tune, selection, and holdout sets were kept separate.

Those numerical hard thresholds belong to this adversarial CP-regret
pipeline. Later WDL/all-position experiments used different objectives, so the
README does not present them as universal definitions of chess safety.
The exact 38-row/41-rejection counts come from a locally retained historical
run; the tracked repository preserves the method and definitions more durably
than it preserves every raw intermediate artifact.

#### Depth-8 re-audit of selected node/regret trade-offs

The rows below were selected as spread points earlier in the pipeline and then
re-audited at depth 8 on 4,000 paired holdout positions. They are not all a
freshly recomputed depth-8 Pareto set: one can become dominated when the depth
changes. Control nodes vary slightly because each candidate was paired with
its own Strict run. Node totals and the critical gate use all 4,000 positions;
mean regret, p95, and above-100cp use the 3,743 ranking positions that passed
the static-target filter. The other 257 positions still participate in safety
checking. “Critical” is the hard self-mate/win-to-draw/win-to-loss gate.

| Candidate | Candidate / control nodes | Node ratio | Mean regret | P95 regret | Above 100cp | Critical |
|---|---:|---:|---:|---:|---:|---:|
| LMR `.45/2.65`, d5/i5 + NMP d3/r1 | 2.886B / 11.224B | 25.71% | 7.96cp | 46cp | 1.95% | 0 |
| LMR `.45/2.45`, d5/i3 | 3.517B / 11.242B | 31.29% | 6.90cp | 40cp | 1.68% | 0 |
| LMR `.50/2.55`, d5/i7 + NMP d5/r2 | 3.154B / 11.285B | 27.94% | 6.74cp | 40cp | 1.52% | 0 |
| LMR `.55/2.80`, d5/i8 + NMP d6/r2 | 3.884B / 11.309B | 34.35% | 5.55cp | 30cp | 1.26% | 0 |
| NMP d4/r1 | 7.257B / 11.190B | 64.85% | 1.55cp | 2cp | 0.27% | 0 |
| NMP d5/r1 | 8.951B / 11.224B | 79.75% | 0.93cp | 0cp | 0.08% | 0 |
| Strict control | 11.317B / 11.317B | 100.00% | 0.00cp | 0cp | 0.00% | 0 |

This table explains why there is no single “best pruning score.” The aggressive
end saves roughly three quarters of the nodes at the cost of more teacher
disagreement; the conservative end preserves the Strict decision more often
but saves less. The frontier process exists to keep those trade-offs visible,
while the final decision still belongs to direct play.

This table is directional historical evidence. Its raw summary is local and
Git-ignored, lacks a complete source/binary identity, and predates the tracked
per-sample heuristic-reset marker; independent heuristic reset for this run is
therefore not proven.

#### Selection and self-play

The historically promoted V39 `Fast` profile was:

```text
LMR: base=0.45, divisor=2.9, min depth=3, min move index=6
NMP: min depth=2, reduction=3
RFP: max depth=2, base margin=175, margin/depth=275
LMP: max depth=3, base=4, depth multiplier=2
```

The LMR move index is zero-based, so index `6` means the seventh searched move.
For LMP, the searched-move threshold is `4 + 2 × depth²`; captures,
promotions, and priority quiets can already have advanced that counter before
the ordinary quiet stage is considered for pruning.

Its final decision came from paired, color-reversed games rather than the
offline frontier alone:

| Match | Fast W-D-L | Fast score | Historical decision |
|---|---:|---:|---|
| Fast vs Balanced | 21-59-10 | 56.11% | Favor Fast |
| Fast vs baseline7 | 20-55-9 | 56.55% | Favor Fast |
| Fast vs later all-four offline winner | 21-59-10 | 56.11% | Reject the offline winner; keep Fast |

`baseline7` kept the tuned LMR/NMP base but disabled RFP and LMP, isolating the
newer pruning pair. The final row is reported from Fast's perspective; it is
the reciprocal of the raw all-four candidate result `10-59-21`.

Those matches repeatedly inspected an ordinary confidence interval and stopped
when it first excluded 50%; search heuristics also leaked between games. They
record the project's historical promotion decision, but are directional rather
than confirmatory statistics.

The all-four candidate is the useful warning here: it won the offline
selection and fixed-time holdout, then lost its direct match to Fast. The
Strict teacher, regret frontier, and safety bank are powerful filters; none
replaces self-play as the final promotion gate.

V40 then tuned QSEE over `OFF, -600, ... , -75, -50, -25, 0`. On a fresh
6,000-position depth-8 holdout, QSEE-off had the lowest WDL loss, `0` used the
fewest nodes, and the project chose `-75cp` as an interior frontier compromise.
At `-75cp`, nodes fell by `15.50%` and time by `11.36%`. The 600-game match
scored `52.0%` with CI `49.71%-54.29%`, so QSEE was adopted for measured
efficiency, not a proven strength win or a uniquely optimal scalar threshold.

V41 added in-search threefold and 50-move handling, plus TT-score suppression
when the score depends on reversible history. Those are correctness guardrails
around the selected search, not another pruning-strength claim.

See the [selective-search pipeline](docs/nnue_selective_adversarial_pipeline.md)
for the tracked candidate flow and safety-bank method.

### Evidence discipline and negative results

The milestone story intentionally retains results that did not become wins:

- V18/V19's historical timing ratio is not presented as exact-tree proof;
- the V39 all-four offline candidate lost its direct match to Fast;
- V40 QSEE improved efficiency but not conclusively playing strength;
- the ordinary 500M F2 candidate scored `47.08%` against the older 200M
  reference over 600 games and was rejected;
- horizontal mirroring improved controlled offline error, but the longer F2M
  self-play run scored `50.92%`, with CI `48.39%-53.44%`, and did not
  demonstrate a playing-strength improvement. No F2M model was promoted.

The model-scaling experiments remain documented in the
[model-scaling report](docs/benchmarks/nnue_model_scaling_20260825.md), but they
are not a project milestone because they did not improve the deployed engine.

## Current engine architecture

The production path now looks like this:

```text
Lichess / UCI client
        -> UCI adapter (ChessNNUEV41)
        -> V41 wrapper
        -> V40 QSEE + V39 selective search core
        -> move generation / make-unmake / TT / repetition stack
        -> incremental phase-aware quantized NNUE
        -> ARM-native NEON/I8MM or runtime-dispatched x86 VNNI/AVX2/scalar
```

Key implementation properties:

- C++20, bitboards and magic sliding attacks;
- legal noisy/quiet generators and staged move ordering;
- iterative deepening, alpha-beta/PVS, quiescence, null-move pruning, LMR,
  history, killer and counter-history heuristics;
- bucketed transposition tables;
- incremental NNUE accumulator state with phase and PSQT terms;
- UCI options for move overhead and selective-search controls;
- a containerized Lichess worker that starts a fresh UCI engine per game.

In the last documented committed UCI lifecycle, both `ucinewgame` and every
`position` command clear TT. `ucinewgame` does not perform a complete reset of
all learned search heuristics. Clearing on each `position` is safe for
history-dependent draw scores but prevents cross-move TT reuse. The working
tree contains an experimental generation-tag design in which older entries may
provide move hints but not score cutoffs; it is not a committed or benchmarked
production milestone.

### Last documented production snapshot

The last documented production snapshot is V41 at commit `634b2d4`; the
retained deployment observation records Heroku release v12 on 2026-08-25.
This README does not treat that observation as proof of live state on a later
date without a fresh provider query.

- V39 Fast selective-search baseline
- V40 quiescence SEE pruning at `-75cp`
- V41 in-search threefold-repetition and 50-move rules
- phase-aware quantized F2 NNUE model
- native-build ARM I8MM/NEON and runtime-dispatched x86 AVX2/AVX-512 VNNI
- Release plus LTO Heroku build

The documented production model is 7,944,336 bytes; its SHA-256 is
`a1a52891f95db9a1bacc48557325b0c5904da4b3c9f8a333eb2ec090b1bb9c02`.
The canonical same-signature benchmark measures 4.71-4.94M fixed-depth NPS on
the local Apple M4 and 1.48-1.58M NPS on the Heroku LTO release. This ratio is a
cross-platform operational comparison, not isolated hosting overhead.

## Repository guide

| Path | Purpose |
|---|---|
| `include/`, `src/` | Chess core, versioned searchers, TT implementations and NNUE runtime. |
| `tests/`, `tests_py/` | C++ correctness/parity tests and Python harness/analyzer tests. |
| `tools/` | Dataset export, tuning, UCI adapters, matches, benchmarks, profilers and evidence analyzers. |
| `nn/` | Training architectures, loaders, targets and quantization/export support. |
| `benchmarks/` | Versioned canonical position suites. |
| `deploy/` | Heroku/Lichess packaging and staging workflows. |
| `docs/benchmarks/` | Curated benchmark reports, protocol and compact evidence manifests. |

## Historical parallel branch: dense neural value evaluation

This section preserves the first dense value-network pipeline. It is useful
project history, but it does **not** describe the current production NNUE. The
production architecture is the phase-aware quantized `256 -> 32 -> 32 -> 1`
network described in the milestone and benchmark reports.

The prototype NN was a value model:

```text
position -> scalar score from side-to-move POV
```

It did not output policy/move probabilities.

### NN Input

The board is encoded as sparse feature indices instead of a dense tensor.

For each piece, the encoder records:

```text
piece type:       6 types
piece side:       friendly / enemy
king context:     relative to friendly king / enemy king
piece square:     64 squares
king square:      64 squares
```

Total sparse feature space:

```text
6 * 2 * 2 * 64 * 64 = 98304 features
```

The encoder also adds small auxiliary features:

- friendly castling rights
- enemy castling rights
- en passant availability
- en passant file

The position is encoded from side-to-move perspective, so the model sees:

```text
friendly pieces
enemy pieces
```

instead of fixed white/black ownership.

### NN Architecture

The prototype network was intentionally small:

```text
EmbeddingBag(98304 -> 256)
concat aux features
Linear -> ReLU -> Linear -> scalar
```

This was a pipeline test model, not the final NNUE.

### Training Target

The prototype model learned from the classical engine itself.

Dataset labels are generated by:

```cpp
search_best_move(position, depth).score
```

That score ultimately comes from:

```cpp
heuristic evaluate(position)
```

plus tactical correction from shallow alpha-beta search.

So that prototype learned:

```text
heuristic evaluation + shallow search behavior
```

It is not expected to exceed the teacher automatically. A stronger model must be accepted only after match testing against the previous model.

### Self-Play Training Loop

The intended promotion loop was:

```text
1. Freeze current model as model_old
2. Generate games using model_old + search
3. Train model_new
4. Match model_new vs model_old
5. Promote model_new only if it wins clearly
```

This prevents blindly replacing the engine with a model that only has lower training loss.

## Build

With CMake:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target \
  uci_nnue_v41 \
  search_tests \
  nnue_searcher_v41_repetition_tests \
  phase_quantized_nnue_tests \
  -j
```

On this machine, if `cmake` is not on `PATH`, use:

```sh
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake --build build --target \
  uci_nnue_v41 \
  search_tests \
  nnue_searcher_v41_repetition_tests \
  phase_quantized_nnue_tests \
  -j
```

Run tests:

```sh
ctest --test-dir build --output-on-failure \
  -R '^(search_tests|nnue_searcher_v41_repetition_tests|phase_quantized_nnue_tests)$'
```

The default all-target build currently also includes optional TT-stat profiling
executables whose link mode is inconsistent when TT stats are disabled. The
production and test targets above build normally; fixing the optional-profile
target wiring remains an explicit build-system cleanup item.

The V36-V41/phase-parity tests also require the production model and parity TSV
at the documented `models/quantized_scale_grid/.../best/` path. Those large
artifacts are currently ignored by Git, so a clean clone needs them restored by
hash before running the NNUE tests. The milestone manifest records the model
SHA-256; long-term artifact hosting is still required for one-command clean
reproduction.

Start the current UCI engine with its default model path:

```sh
build/uci_nnue_v41
```

Or pass a model explicitly:

```sh
build/uci_nnue_v41 \
  models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin
```

Example UCI session:

```text
uci
isready
ucinewgame
position startpos
go depth 8
quit
```

## Historical dense-model dataset export

Build the dataset exporter:

```sh
cmake --build build --target dataset_export -j
mkdir -p data
```

Generate game states with random opening plies, then engine-selected moves:

```sh
build/dataset_export \
  --games 1000 \
  --depth 2 \
  --random-plies 4 \
  --max-plies 60 \
  --progress-interval 100 \
  --seed 1001 \
  --output data/value_games.jsonl
```

## Historical dense-model training from saved JSONL

```sh
.venv/bin/python -m nn.train_value \
  --train data/value_train.jsonl \
  --val data/value_val.jsonl \
  --test data/value_test_baseline.jsonl \
  --epochs 10 \
  --batch-size 1024 \
  --device cpu \
  --output models/value_net.pt
```

The `--test` file is a held-out baseline. It is not used for training.

## Historical dense-model stream training

For large self-play runs, training data should not be fully written to disk.

Use streaming:

```sh
build/dataset_export \
  --games 100000 \
  --depth 2 \
  --random-plies 4 \
  --max-plies 60 \
  --progress-interval 1000 \
  --seed 1001 \
  --output /dev/stdout \
  | .venv/bin/python -m nn.train_stream_value \
      --val data/value_val_1k_stream.jsonl \
      --test data/value_test_baseline_1k_stream.jsonl \
      --batch-size 1024 \
      --device cpu \
      --output models/value_net_stream.pt \
      --log-interval 100000 \
      --eval-interval 1000000
```

This keeps train data out of disk. Only validation/test baseline files are stored.

## Historical dense-model export for C++ inference

PyTorch is only needed for training and exporting. Runtime inference in the engine uses a small C++ forward pass.

Export a trained checkpoint:

```sh
.venv/bin/python tools/export_value_net.py \
  --checkpoint models/value_net_stream_100k.pt \
  --output models/value_net_stream_100k.bin \
  --compare-input data/value_test_baseline_1k_stream.jsonl \
  --compare-output data/nn_value_compare_1000.jsonl \
  --compare-limit 1000
```

Build and run the C++ parity test:

```sh
cmake --build build --target nn_value_tests -j

build/nn_value_tests \
  models/value_net_stream_100k.bin \
  data/nn_value_compare_1000.jsonl
```

The test compares C++ inference against Python/PyTorch output on encoded samples.

## Historical dense-model search engine

Build the standalone NN search engine:

```sh
cmake --build build --target nn_engine -j
```

Search one position:

```sh
build/nn_engine \
  --model models/value_net_stream_100k.bin \
  --depth 3 \
  --go-once
```

Use a custom FEN:

```sh
build/nn_engine \
  --model models/value_net_stream_100k.bin \
  --depth 3 \
  --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" \
  --go-once
```

Without `--go-once`, it starts an interactive CLI where the NN engine plays Black.

## Current Direction

The next serious milestones are:

- benchmark a register-fused x86 VNNI forward candidate against the current
  kernel on the same dynos, with forced-backend parity before changing `auto`;
- finish and benchmark safe cross-search TT reuse with a persistent UCI process;
- make the self-play harness reset all search state and pin every artifact hash;
- run a provenance-complete V41-versus-V39 strength match if that historical
  comparison is still needed;
- re-run the full clean suite after the V21-V23 TT-range repair, and fix the
  optional TT-profile build wiring;
- archive selected raw benchmark bundles outside ignored local directories.

New models are promoted only after distinct parity, offline-quality,
throughput, and paired playing-strength gates.
