#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_lmr_nmp_adversarial_8h.sh RUN_DIR}"
SAFETY_BANK_DIR="$RUN_DIR/safety_bank"
DATASET_DIR="$RUN_DIR/dataset"
BALANCED_SOURCE="$RUN_DIR/stockfish_balanced_openings_10ply_1200.txt"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

mkdir -p "$RUN_DIR"

if [[ ! -f "$SAFETY_BANK_DIR/manifest.json" ]]; then
  .venv/bin/python tools/data/build_nnue_selective_safety_bank.py \
    --binary build-release/evaluate_nnue_v38_selective \
    --dataset logs/nnue_v39_pruning_v2_8h_20260728_101928/dataset/selection.tsv \
    --model "$MODEL" \
    --output-dir "$SAFETY_BANK_DIR" \
    --depth 6 \
    --seed-tsv data/nnue_v39_rfp_regression.tsv \
    --detail-threshold 250 \
    --core-fraction 0.70 \
    --max-core 100 \
    --max-sealed 500 \
    --seed 20260801 \
    --resume
fi

if [[ ! -f "$DATASET_DIR/manifest.txt" ]]; then
  if [[ ! -f "$BALANCED_SOURCE" ]]; then
    .venv/bin/python tools/data/generate_stockfish_balanced_openings.py \
      --stockfish build/stockfish_static_nnue_upstream/src/stockfish \
      --base-book data/opening_book_6plies.txt \
      --output "$BALANCED_SOURCE" \
      --manifest "$RUN_DIR/stockfish_balanced_openings_manifest.json" \
      --count 1200 \
      --seed 20260801
  fi
  .venv/bin/python tools/data/build_nnue_v38_tune_dataset.py \
    --corpus data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m \
    --balanced "$BALANCED_SOURCE" \
    --output-dir "$DATASET_DIR" \
    --scale 2 \
    --balanced-total 1000 \
    --exclude-dataset-dir logs/nnue_v38_selective_8h_v2_20260726_233500/dataset \
    --exclude-dataset-dir logs/nnue_v38_prune_tune_20260726_182434/dataset \
    --exclude-dataset-dir logs/nnue_v38_prune_stage2_20260726_203127/dataset \
    --exclude-dataset-dir logs/nnue_v39_pruning_8h_20260727_202350/dataset \
    --exclude-dataset-dir logs/nnue_v39_pruning_v2_8h_20260728_101928/dataset \
    --seed 20260801
fi

.venv/bin/python tools/tune/tune_nnue_lmr_nmp_adversarial.py \
  --binary build-release/evaluate_nnue_v38_selective \
  --dataset-dir "$DATASET_DIR" \
  --safety-bank-dir "$SAFETY_BANK_DIR" \
  --model "$MODEL" \
  --run-dir "$RUN_DIR" \
  --duration-sec 28800 \
  --mutation-duration-sec 9000 \
  --max-candidates 100 \
  --tune-size 2000 \
  --tune-depth 6 \
  --regression-depths 5 6 7 8 \
  --adversarial-depth 7 \
  --selection-depth 7 \
  --holdout-depth 7 \
  --final-depth 8 \
  --final-depth-cap 8 \
  --ranking-target-abs-cp 1500 \
  --seed 20260801
