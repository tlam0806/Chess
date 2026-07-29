#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v38_selective_8h.sh RUN_DIR}"
DATASET_DIR="$RUN_DIR/dataset"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

mkdir -p "$RUN_DIR"

if [[ ! -f "$DATASET_DIR/manifest.txt" ]]; then
  .venv/bin/python tools/build_nnue_v38_tune_dataset.py \
    --corpus data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m \
    --balanced data/stockfish_balanced_openings_10ply_1k_20260723.txt \
    --output-dir "$DATASET_DIR" \
    --scale 2 \
    --balanced-total 1000 \
    --exclude-dataset-dir logs/nnue_v38_prune_tune_20260726_182434/dataset \
    --exclude-dataset-dir logs/nnue_v38_prune_stage2_20260726_203127/dataset \
    --seed 20260728
fi

.venv/bin/python tools/tune_nnue_v38_selective.py \
  --binary build-release/evaluate_nnue_v38_selective \
  --dataset-dir "$DATASET_DIR" \
  --model "$MODEL" \
  --run-dir "$RUN_DIR" \
  --duration-sec 28800 \
  --mutation-duration-sec 19800 \
  --max-candidates 800 \
  --fixed-tune-size 1000 \
  --wide-mutations \
  --frontier-cap 12 \
  --ranking-target-abs-cp 1500 \
  --tune-depth 6 \
  --selection-depth 6 \
  --holdout-depth 7 \
  --seed 20260728
