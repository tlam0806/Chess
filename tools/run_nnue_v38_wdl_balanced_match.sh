#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v38_wdl_balanced_match.sh RUN_DIR}"
DATASET_DIR="logs/nnue_v38_selective_8h_v2_20260726_233500/dataset"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"
BOOK="logs/nnue_lmr_nmp_adversarial_8h_20260728_172314/stockfish_balanced_openings_10ply_1200.txt"

mkdir -p "$RUN_DIR/match"

build-release/evaluate_nnue_v38_selective \
  --dataset "$DATASET_DIR/holdout.tsv" \
  --model "$MODEL" \
  --depth 7 \
  --ranking-target-abs-cp 1500 \
  --objective wdl \
  --lmr-base 0.45 \
  --lmr-divisor 2.45 \
  --lmr-min-depth 5 \
  --lmr-min-move-index 5 \
  --null-min-depth 3 \
  --null-reduction 2 \
  > "$RUN_DIR/balanced_old_holdout.json"

candidate_config="$(
  .venv/bin/python tools/select_nnue_wdl_balanced_candidate.py \
    --summary "$RUN_DIR/summary.json" \
    --old-result "$RUN_DIR/balanced_old_holdout.json" \
    --output "$RUN_DIR/balanced_candidate.json" \
    --node-tolerance 1.02
)"

build-release/nnue_v38_time_gauntlet \
  --book "$BOOK" \
  --model "$MODEL" \
  --output "$RUN_DIR/match/games.jsonl" \
  --openings 300 \
  --base-ms 10000 \
  --increment-ms 100 \
  --overhead-ms 20 \
  --max-plies 200 \
  --tt-mb 64 \
  --seed 20260727 \
  --balanced-rematch \
  --balanced-new-config "$candidate_config" \
  > "$RUN_DIR/match/stdout.log" \
  2> "$RUN_DIR/match/runner.log"

.venv/bin/python tools/summarize_nnue_v38_paired_match.py \
  --input "$RUN_DIR/match/games.jsonl" \
  --output "$RUN_DIR/match/summary.json"

touch "$RUN_DIR/DONE"
