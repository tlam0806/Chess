#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v39_audit3_round_robin_ci.sh RUN_DIR}"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"
BOOK="logs/nnue_lmr_nmp_adversarial_8h_20260728_172314/stockfish_balanced_openings_10ply_1200.txt"
MAX_PAIRS_PER_MATCHUP="${MAX_PAIRS_PER_MATCHUP:-300}"

mkdir -p "$RUN_DIR"

build-release/nnue_v39_time_gauntlet \
  --book "$BOOK" \
  --model "$MODEL" \
  --output "$RUN_DIR/games.jsonl" \
  --openings "$MAX_PAIRS_PER_MATCHUP" \
  --base-ms 10000 \
  --increment-ms 100 \
  --overhead-ms 20 \
  --max-plies 200 \
  --tt-mb 64 \
  --seed 20260804 \
  --stop-on-ci \
  --ci-min-pairs 40 \
  --profile fast_joint,0.45,2.9,3,6,2,3,1,2,175,275,1,3,4,2 \
  --profile balanced_joint,0.45,2.9,3,6,2,3,1,2,150,150,1,3,7,5 \
  --profile baseline7,0.45,2.9,3,6,2,3,0,2,150,150,0,3,7,5 \
  > "$RUN_DIR/stdout.log" \
  2> "$RUN_DIR/runner.log"

.venv/bin/python tools/summarize_nnue_round_robin.py \
  --input "$RUN_DIR/games.jsonl" \
  --output "$RUN_DIR/summary.json" \
  > "$RUN_DIR/summary.log"

touch "$RUN_DIR/DONE"
