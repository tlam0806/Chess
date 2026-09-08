#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:?usage: run_nnue_500m_epoch4_vs_200m_ci1.sh RUN_DIR}"
repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_dir"

new_checkpoint="models/quantized_scale_grid/old_score_huber200_500m_until_overfit_20260822_203621/quant_nnue_arch_F2_epoch4.pt"
new_phase_checkpoint="$run_dir/new500_epoch4_phase.pt"
new_model="$run_dir/new500_epoch4_phase_quantized_nnue.bin"
production_model="models/quantized_scale_grid/old_score_huber200_top2_200m_20260722_163841/hs2x8_os128/phase_quantized_nnue.bin"
book="logs/nnue_v40_qsee_tune_20260817_005158/stockfish_balanced_openings_10ply_2400.txt"
games="$run_dir/games.jsonl"
profile="0.45,2.9,3,6,2,3,1,2,175,275,1,3,4,2,1,-75"

mkdir -p "$run_dir"

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
fi

if [[ ! -f "$new_checkpoint" ]]; then
  echo "missing epoch-4 checkpoint: $new_checkpoint" >&2
  exit 2
fi
if [[ ! -f "$production_model" ]]; then
  echo "missing production model: $production_model" >&2
  exit 2
fi

if [[ ! -f "$new_model" ]]; then
  .venv/bin/python tools/train/convert_single_stack_to_phase_checkpoint.py \
    --input "$new_checkpoint" \
    --output "$new_phase_checkpoint"
  .venv/bin/python tools/train/export_phase_quantized_nnue.py \
    --checkpoint "$new_phase_checkpoint" \
    --output "$new_model"
fi

/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake \
  -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DCHESS_BUILD_EXPERIMENTS=ON
/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake \
  --build build-release --target nnue_v40_time_gauntlet -j8

build-release/nnue_v40_time_gauntlet \
  --book "$book" \
  --model "$new_model" \
  --opponent-model "$production_model" \
  --output "$games" \
  --openings 300 \
  --base-ms 10000 \
  --increment-ms 100 \
  --overhead-ms 20 \
  --max-plies 200 \
  --tt-mb 64 \
  --seed 20260823 \
  --ci-min-pairs 200 \
  --ci-max-width 0.01 \
  --profile "new500_epoch4,$profile" \
  --profile "production200,$profile" \
  > "$run_dir/stdout.log" \
  2> "$run_dir/runner.log"

.venv/bin/python tools/analyze/summarize_nnue_v38_paired_match.py \
  --input "$games" \
  --output "$run_dir/summary.json"

touch "$run_dir/DONE"
