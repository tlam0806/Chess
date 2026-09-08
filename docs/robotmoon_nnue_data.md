# RobotMoon NNUE Data Pipeline

The production NNUE trainers consume CBin v2 records converted directly from
Stockfish binpack. Do not train the architecture sweep from the old JSONL or
CBin v1 files.

## CBin v2

Each file starts with the 16-byte header:

```text
magic             CHSCBIN2
format_version    uint16 little-endian = 2
aux_feature_count uint16 little-endian = 13
target_encoding   uint32 little-endian = 1 (Stockfish raw score)
```

Each 40-byte record contains:

```text
board       32 bytes, two four-bit piece codes per byte
aux_bits    uint16
score       int16 Stockfish raw score
ply         uint16
result      int16 in {-1, 0, 1}, from side-to-move perspective
```

For black-to-move positions, squares use the engine's relative-square mapping
`square ^ 56`. This flips ranks without mirroring files. Friendly/enemy piece
codes, castling rights and en-passant data are also side-to-move relative.

`VALUE_NONE` (`32002`) is discarded by the converter and rejected again by the
loaders. Scores are converted for reporting and WDL targets with:

```text
cp = clamp(100 * raw_score / 208, -2000, 2000)
```

The loss uses the ply-aware WDL model vendored with nnue-pytorch. It does not
clamp labels to an arbitrary training-only CP range.

## Conversion

Build the converter:

```sh
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHESS_BUILD_EXPERIMENTS=ON
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake --build build \
  --target robotmoon_binpack_to_cbin validate_robotmoon_cbin -j 4
```

Convert an uncompressed binpack:

```sh
build/robotmoon_binpack_to_cbin \
  --input data/external/robotmoon/source.binpack \
  --output data/robotmoon/source.cbin \
  --require-legal-move
```

For compressed input/output, use the streaming wrapper:

```sh
.venv/bin/python tools/data/robotmoon_binpack_zst_to_cbin.py \
  --input data/external/robotmoon/source.binpack.zst \
  --output data/robotmoon/source.cbin.zst \
  --converter build/robotmoon_binpack_to_cbin \
  --require-legal-move
```

CBin v1 files are intentionally rejected because they discarded `ply` and
`result`. They must be regenerated from the original binpack.

## Unique Corpus Build

For a training corpus, uniqueness is defined by the model input after
side-to-move normalization: the packed 32-byte board plus `aux_bits`. Score,
ply and result are labels and are not part of the uniqueness key. Color/rank
mirrors that produce the same normalized input are therefore one sample.

The large-corpus builder performs these stages atomically:

1. Filter `VALUE_NONE` and validate every legal source move.
2. Deduplicate normalized model inputs with a blocked Bloom filter.
3. Require the exact requested output count.
4. Independently scan every record and check 128-bit fingerprints for repeats.
5. Shard the output and compare the complete shard payload hash with the
   monolithic payload before publishing the shard directory.

```sh
.venv/bin/python tools/data/build_robotmoon_unique_corpus.py \
  --source data/external/robotmoon/source.binpack.zst \
  --output-prefix data/robotmoon_500m_unique_v2 \
  --records 500000000 \
  --records-per-shard 1000000 \
  --dedup-bits-per-record 16
```

Bloom false positives may discard a new input but cannot admit a repeated
input. The converter continues reading until the requested number of accepted
inputs is reached. The independent fingerprint pass verifies that the final
published corpus contains no repeated model input.

## Split And Model Selection

Train, validation and test membership is a deterministic CRC32 hash of the
canonical `board + aux_bits`. Duplicate positions therefore always enter the
same split, even across different shards.

Checkpoint selection and early stopping use validation WDL loss. CP MAE is a
diagnostic only. Test is evaluated once after loading the best validation
checkpoint and never participates in sweep ranking.

Every training log/checkpoint records training-pipeline version, target encoding,
split policy, WDL model and git provenance. Sweep tools reject legacy logs without
pipeline-v3 provenance. The training-pipeline version is independent from the
CBin v2 file-format version.

## Verification

The converter integration test writes both root and delta-compressed binpack
entries, including asymmetric white/black en-passant positions, then compares
all CBin fields and square orientation:

```sh
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake --build build \
  --target robotmoon_binpack_fixture robotmoon_binpack_to_cbin \
  validate_robotmoon_cbin -j 4
.venv/bin/python -m pytest tests/python/test_robotmoon_converter.py
```

## Stockfish Static-NNUE Distillation Labels

The static-teacher corpus excludes positions in check because Stockfish skips
static evaluation at those search nodes. Source records are also required to
contain a legal move, which excludes terminal positions. TT hits, optional
tablebase probes, repetition cutoffs and other search-state-dependent cases are
not position-intrinsic filters and are intentionally retained.

The teacher label is the raw Stockfish NNUE output

```text
raw_score = PSQT + positional
```

from the side-to-move perspective. It is recorded before UCI CP conversion,
optimism, rule-50 scaling, correction history and search. The CBin score remains
an `int16` raw Stockfish value; `100 * raw_score / 208` is only a reporting
conversion and is not applied by the labeler.

Build and smoke-test the pinned official Stockfish labeler:

```sh
tools/data/build_stockfish_static_nnue_labeler.sh
build/stockfish_static_nnue_labeler \
  --input eligible.cbin \
  --output labeled.cbin \
  --expected-records 1000000
```

The complete 200M collection, filtering, relabeling and independent duplicate
validation pipeline is:

```sh
tools/data/collect_stockfish_static_nnue_200m.sh
```

The final label manifest pins both the Stockfish commit and NNUE SHA-256.

## F2M Horizontal-Mirror Architecture

`F2M` is the production-compatible F2 phase architecture with HalfKP-style
file canonicalization. Each accumulator is oriented independently: if its own
king is on files e-h, both the king and every piece square use `square ^ 7`.
The king is therefore always encoded on files a-d. This reduces the shared
feature table from 49,152 to 24,576 rows without changing the 256-32-32 dense
network, eight phase stacks, PSQT buckets, or quantized arithmetic.

Castling kingside/queenside features and en-passant files are mirrored with
the side-to-move accumulator. Training/validation/test split keys use the same
canonical board and aux state, so horizontal twins cannot leak across splits.
The C++ accumulator stores a precomputed square XOR mask and king-row base per
perspective. Incremental updates never construct or mirror a `Position`.

An F2 checkpoint can initialize F2M by averaging horizontal weight pairs. The
optional symmetric F2 reference is useful for exact regression tests:

```sh
.venv/bin/python tools/train/convert_f2_to_f2m_checkpoint.py \
  --source SOURCE_F2.pt \
  --output F2M_INIT.pt \
  --symmetric-reference-output F2_SYMMETRIC_REFERENCE.pt
```

Train the phase model with the existing trainer, changing `--arch` to `F2M`.
Pass the converted checkpoint through `--resume-checkpoint` without
`--resume-optimizer` to warm-start with a fresh optimizer. Export uses the
same command as F2 and writes binary format version 2 automatically:

```sh
.venv/bin/python tools/train/export_phase_quantized_nnue.py \
  --checkpoint F2M_CHECKPOINT.pt \
  --output phase_quantized_nnue_f2m.bin
```

The runtime loader remains backward-compatible with F2 version-1 binaries.
Use `benchmark_nnue_horizontal_mirror` with a symmetric F2 binary and its F2M
fold to verify identical moves/scores/nodes and compare full-search NPS.
