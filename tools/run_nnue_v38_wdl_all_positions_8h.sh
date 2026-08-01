#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v38_wdl_all_positions_8h.sh RUN_DIR}"
DATASET_DIR="logs/nnue_v38_selective_8h_v2_20260726_233500/dataset"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

mkdir -p "$RUN_DIR"

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
  --frontier-cap 0 \
  --ranking-target-abs-cp 1500 \
  --objective wdl \
  --include-all-in-objective \
  --allow-critical \
  --tune-depth 6 \
  --selection-depth 6 \
  --holdout-depth 7 \
  --seed 20260728

touch "$RUN_DIR/DONE"
