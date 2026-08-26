#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v40_qsee_selfplay.sh RUN_DIR}"
MODEL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"
BOOK="logs/nnue_v40_qsee_tune_20260817_005158/stockfish_balanced_openings_10ply_2400.txt"
GAMES="$RUN_DIR/games.jsonl"

mkdir -p "$RUN_DIR"

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
fi

/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake \
  --build build-release --target nnue_v40_time_gauntlet -j8

build-release/nnue_v40_time_gauntlet \
  --book "$BOOK" \
  --model "$MODEL" \
  --output "$GAMES" \
  --openings 300 \
  --base-ms 10000 \
  --increment-ms 100 \
  --overhead-ms 20 \
  --max-plies 200 \
  --tt-mb 64 \
  --seed 20260818 \
  --stop-on-ci \
  --ci-min-pairs 40 \
  --profile v40_qsee,0.45,2.9,3,6,2,3,1,2,175,275,1,3,4,2,1,-75 \
  --profile v39_fast,0.45,2.9,3,6,2,3,1,2,175,275,1,3,4,2,0,-75 \
  > "$RUN_DIR/stdout.log" \
  2> "$RUN_DIR/runner.log"

.venv/bin/python tools/summarize_nnue_v38_paired_match.py \
  --input "$GAMES" \
  --output "$RUN_DIR/summary.json"

touch "$RUN_DIR/DONE"
