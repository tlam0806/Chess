#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v38_selective_stage2.sh RUN_DIR}"
STAGE1_DIR="logs/nnue_v38_prune_tune_20260726_182434"
DATASET_DIR="$RUN_DIR/dataset"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

mkdir -p "$RUN_DIR"

.venv/bin/python tools/data/build_nnue_v38_tune_dataset.py \
  --corpus data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m \
  --balanced data/stockfish_balanced_openings_10ply_1k_20260723.txt \
  --output-dir "$DATASET_DIR" \
  --exclude-dataset-dir "$STAGE1_DIR/dataset" \
  --seed 20260727

.venv/bin/python tools/tune/tune_nnue_v38_selective.py \
  --binary build-release/evaluate_nnue_v38_selective \
  --dataset-dir "$DATASET_DIR" \
  --model "$MODEL" \
  --run-dir "$RUN_DIR" \
  --duration-sec 7200 \
  --max-candidates 180 \
  --stage2-multi-param \
  --exclude-log "$STAGE1_DIR/results.jsonl" \
  --tune-depth 6 \
  --selection-depth 6 \
  --holdout-depth 7 \
  --seed 20260727
