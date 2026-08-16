#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v39_config7_wdl_8h.sh RUN_DIR}"
DATASET_DIR="$RUN_DIR/dataset"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

mkdir -p "$RUN_DIR"

if [[ ! -f "$DATASET_DIR/manifest.txt" ]]; then
  .venv/bin/python tools/build_nnue_v38_tune_dataset.py \
    --corpus data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m \
    --balanced logs/nnue_v39_pruning_8h_20260727_202350/stockfish_balanced_openings_10ply_1200.txt \
    --balanced logs/nnue_v39_pruning_v2_8h_20260728_101928/stockfish_balanced_openings_10ply_1200.txt \
    --balanced logs/nnue_lmr_nmp_adversarial_8h_20260728_172314/stockfish_balanced_openings_10ply_1200.txt \
    --output-dir "$DATASET_DIR" \
    --scale 2 \
    --exclude-dataset-dir logs/nnue_v38_selective_8h_v2_20260726_233500/dataset \
    --exclude-dataset-dir logs/nnue_v38_prune_tune_20260726_182434/dataset \
    --exclude-dataset-dir logs/nnue_v38_prune_stage2_20260726_203127/dataset \
    --exclude-dataset-dir logs/nnue_v39_pruning_8h_20260727_202350/dataset \
    --exclude-dataset-dir logs/nnue_v39_pruning_v2_8h_20260728_101928/dataset \
    --exclude-dataset-dir logs/nnue_lmr_nmp_adversarial_8h_20260728_172314/dataset \
    --seed 20260803
fi

.venv/bin/python tools/tune_nnue_v39_config7_wdl.py \
  --binary build-release/evaluate_nnue_v39_selective \
  --dataset-dir "$DATASET_DIR" \
  --model "$MODEL" \
  --run-dir "$RUN_DIR" \
  --duration-sec 28800 \
  --mutation-duration-sec 12600 \
  --max-candidates 180 \
  --tune-size 2000 \
  --selection-probe-size 500 \
  --audit-size 1000 \
  --tune-depth 6 \
  --selection-depth 7 \
  --holdout-depth 7 \
  --audit-depth 8 \
  --seed 20260803
