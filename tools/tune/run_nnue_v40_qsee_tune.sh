#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v40_qsee_tune.sh RUN_DIR}"
DATASET_DIR="$RUN_DIR/dataset"
BALANCED_SOURCE="$RUN_DIR/stockfish_balanced_openings_10ply_2400.txt"
STOCKFISH_DIR="build/stockfish_qsee_upstream"
STOCKFISH="$STOCKFISH_DIR/src/stockfish"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

mkdir -p "$RUN_DIR"

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
fi

/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake \
  -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DCHESS_BUILD_EXPERIMENTS=ON
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake \
  --build build-release --target evaluate_nnue_v40_selective -j8

if [[ ! -f "$DATASET_DIR/manifest.txt" ]]; then
  if [[ ! -x "$STOCKFISH" ]]; then
    if [[ ! -f "$STOCKFISH_DIR/src/Makefile" ]]; then
      git clone --depth 1 \
        https://github.com/official-stockfish/Stockfish.git \
        "$STOCKFISH_DIR"
    fi
    make -C "$STOCKFISH_DIR/src" -j8 build ARCH=apple-silicon
  fi
  if [[ ! -f "$BALANCED_SOURCE" ]]; then
    .venv/bin/python tools/data/generate_stockfish_balanced_openings.py \
      --stockfish "$STOCKFISH" \
      --base-book data/opening_book_6plies.txt \
      --output "$BALANCED_SOURCE" \
      --manifest "$RUN_DIR/stockfish_balanced_openings_manifest.json" \
      --count 2400 \
      --seed 20260817
  fi
  .venv/bin/python tools/data/build_nnue_v38_tune_dataset.py \
    --corpus data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m \
    --balanced "$BALANCED_SOURCE" \
    --output-dir "$DATASET_DIR" \
    --scale 3 \
    --balanced-total 2400 \
    --exclude-dataset-dir logs/nnue_v38_selective_8h_v2_20260726_233500/dataset \
    --exclude-dataset-dir logs/nnue_v38_prune_tune_20260726_182434/dataset \
    --exclude-dataset-dir logs/nnue_v38_prune_stage2_20260726_203127/dataset \
    --exclude-dataset-dir logs/nnue_v39_pruning_8h_20260727_202350/dataset \
    --exclude-dataset-dir logs/nnue_v39_pruning_v2_8h_20260728_101928/dataset \
    --exclude-dataset-dir logs/nnue_lmr_nmp_adversarial_8h_20260728_172314/dataset \
    --exclude-dataset-dir logs/nnue_v39_config7_wdl_8h_20260802_021900/dataset \
    --exclude-dataset-dir logs/nnue_v39_all_prunes_8h_20260802_160001/dataset \
    --seed 20260817
fi

.venv/bin/python tools/tune/tune_nnue_v40_qsee.py \
  --binary build-release/evaluate_nnue_v40_selective \
  --dataset-dir "$DATASET_DIR" \
  --model "$MODEL" \
  --run-dir "$RUN_DIR" \
  --gate-size 3000 \
  --time-size 2000 \
  --gate-depth 6 \
  --selection-depth 7 \
  --holdout-depth 8 \
  --candidate-time-ms 20 \
  --workers 4 \
  --seed 20260817
