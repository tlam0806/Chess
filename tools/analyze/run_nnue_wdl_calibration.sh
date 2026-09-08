#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="${1:?usage: run_nnue_wdl_calibration.sh RUN_DIR}"
collector="$repo_dir/build-release/collect_nnue_wdl_calibration"
python="$repo_dir/.venv/bin/python"
book="$repo_dir/logs/nnue_lmr_nmp_adversarial_8h_20260728_172314/stockfish_balanced_openings_10ply_1200.txt"
model="$repo_dir/models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

mkdir -p "$run_dir"

"$collector" \
  --book "$book" \
  --model "$model" \
  --output "$run_dir/calibration_positions.jsonl" \
  --games 600 \
  --base-ms 10000 \
  --increment-ms 100 \
  --overhead-ms 20 \
  --max-plies 200 \
  --sample-every 4 \
  --samples-per-game 8 \
  --teacher-depth 8 \
  --tt-mb 64 \
  --seed 20260803

"$python" "$repo_dir/tools/analyze/fit_nnue_wdl_calibration.py" \
  --input "$run_dir/calibration_positions.jsonl" \
  --output-dir "$run_dir/fit" \
  --seed 20260803 \
  --steps 300

touch "$run_dir/DONE"
