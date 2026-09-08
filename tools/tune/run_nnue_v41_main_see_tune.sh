#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: run_nnue_v41_main_see_tune.sh RUN_DIR}"
DATASET_DIR="${NNUE_MAIN_SEE_DATASET_DIR:-logs/nnue_v40_qsee_tune_20260817_005158/dataset}"
MODEL="${NNUE_MAIN_SEE_MODEL:-models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin}"
CAFFEINATE_PID=""

mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"

finish() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$CAFFEINATE_PID" ]]; then
    kill "$CAFFEINATE_PID" 2>/dev/null || true
  fi
  if [[ $status -eq 0 ]]; then
    touch "$RUN_DIR/JOB_COMPLETE"
  else
    printf '%s\n' "$status" > "$RUN_DIR/JOB_FAILED"
  fi
  return "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
  CAFFEINATE_PID=$!
fi

for split in tune selection holdout; do
  if [[ ! -s "$DATASET_DIR/$split.tsv" ]]; then
    echo "missing dataset split: $DATASET_DIR/$split.tsv" >&2
    exit 1
  fi
done
if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DCHESS_BUILD_EXPERIMENTS=ON
cmake --build build-release --target evaluate_nnue_v41_selective -j8

.venv/bin/python tools/tune/tune_nnue_v41_main_see.py \
  --binary build-release/evaluate_nnue_v41_selective \
  --dataset-dir "$DATASET_DIR" \
  --model "$MODEL" \
  --run-dir "$RUN_DIR" \
  --gate-size 3000 \
  --time-size 2000 \
  --gate-depth 6 \
  --selection-depth 7 \
  --holdout-depth 8 \
  --candidate-time-ms 20 \
  --workers 4 \
  --seed 20260827
