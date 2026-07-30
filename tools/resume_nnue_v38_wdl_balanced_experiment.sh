#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: resume_nnue_v38_wdl_balanced_experiment.sh RUN_DIR}"
DATASET_DIR="logs/nnue_v38_selective_8h_v2_20260726_233500/dataset"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

.venv/bin/python tools/resume_nnue_v38_selection_dedup_holdout.py \
  --tune-log "$RUN_DIR/results.jsonl" \
  --run-dir "$RUN_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --binary build-release/evaluate_nnue_v38_selective \
  --model "$MODEL" \
  --ranking-target-abs-cp 1500 \
  --objective wdl \
  --selection-depth 6 \
  --holdout-depth 7

tools/run_nnue_v38_wdl_balanced_match.sh "$RUN_DIR"
