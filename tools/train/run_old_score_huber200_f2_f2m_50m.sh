#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TAG="${TAG:-old_score_huber200_f2_f2m_50m_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="models/quantized_scale_grid/$TAG"
F2_DIR="$RUN_ROOT/f2"
F2M_DIR="$RUN_ROOT/f2m"
PREFLIGHT_DIR="$RUN_ROOT/preflight"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
REPORT="reports/$TAG.md"
COMPARISON_JSON="$RUN_ROOT/comparison.json"
TRAIN_DATA="${TRAIN_DATA:-data/robotmoon_old_score_train_50m_v1}"
EVAL_DATA="${EVAL_DATA:-data/robotmoon_old_score_val_mirror_safe_v1}"
EVAL_MANIFEST="${EVAL_MANIFEST:-data/robotmoon_old_score_val_mirror_safe_v1.manifest.json}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_component_best.pt}"
BUILD_DIR="${BUILD_DIR:-build-release}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000000}"
SELECTION_SAMPLES="${SELECTION_SAMPLES:-499149}"
RANKING_SAMPLES="${RANKING_SAMPLES:-499149}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
WORKERS="${WORKERS:-0}"
TORCH_THREADS="${TORCH_THREADS:-8}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-20260720}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-32768}"
SMOKE_EVAL_SAMPLES="${SMOKE_EVAL_SAMPLES:-8192}"
SLEEP_ON_EXIT="${SLEEP_ON_EXIT:-1}"

mkdir -p "$RUN_ROOT" "$PREFLIGHT_DIR" logs reports
printf 'state=preflight tag=%s updated_at=%s\n' \
  "$TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

CAFFEINATE_PID=""
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
  CAFFEINATE_PID=$!
fi

on_exit() {
  local code=$?
  if [[ -n "$CAFFEINATE_PID" ]]; then
    kill "$CAFFEINATE_PID" 2>/dev/null || true
  fi
  if [[ $code -ne 0 ]]; then
    printf 'state=failed exit_code=%s updated_at=%s\n' \
      "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  fi
  if [[ "$SLEEP_ON_EXIT" == "1" ]] && command -v pmset >/dev/null 2>&1; then
    pmset sleepnow >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

for required in \
  .venv/bin/python \
  tools/train/train_phase_component_nnue.py \
  tools/train/export_phase_quantized_nnue.py \
  tools/train/convert_f2_to_f2m_checkpoint.py \
  tools/train/verify_phase_training_checkpoint.py \
  tools/analyze/report_f2_f2m_50m.py \
  "$REFERENCE_CHECKPOINT" \
  "$EVAL_MANIFEST"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done
if [[ ! -d "$TRAIN_DATA" || ! -d "$EVAL_DATA" ]]; then
  echo "missing train or validation directory" >&2
  exit 1
fi
if (( TRAIN_SAMPLES != 50000000 )); then
  echo "paired 50M runner requires TRAIN_SAMPLES=50000000" >&2
  exit 1
fi
if (( SELECTION_SAMPLES <= 0 || RANKING_SAMPLES <= 0 \
      || SELECTION_SAMPLES + RANKING_SAMPLES > 998298 )); then
  echo "selection/ranking windows exceed sealed mirror-safe validation" >&2
  exit 1
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" \
  --target audit_nnue_mirror_split phase_quantized_nnue_tests \
    benchmark_nnue_horizontal_mirror \
  -j 4 >/dev/null

"$BUILD_DIR/audit_nnue_mirror_split" \
  --train "$TRAIN_DATA" \
  --validation "$EVAL_DATA" \
  --expected-train-records 50000000 \
  --expected-validation-records 998298 \
  --progress-interval 10000000 \
  > "$PREFLIGHT_DIR/mirror_split_audit.json" \
  2> "$PREFLIGHT_DIR/mirror_split_audit.log"
grep -q '"status":"pass"' "$PREFLIGHT_DIR/mirror_split_audit.json"

.venv/bin/python -m pytest -q \
  tests/python/test_compact_board_data_zstd.py \
  tests/python/test_nnue_horizontal_mirror.py \
  tests/python/test_phase_component_nnue.py \
  -k 'not global_cap_with_multiple_workers' \
  > "$PREFLIGHT_DIR/python_tests.log" 2>&1

run_training() {
  local architecture="$1"
  local output_dir="$2"
  local train_samples="$3"
  local selection_samples="$4"
  local ranking_samples="$5"
  local batch_size="$6"
  local warmup_steps="$7"
  local progress_batches="$8"
  local log="$9"
  local resume_args=()

  if (( train_samples == TRAIN_SAMPLES )); then
    resume_args=(--checkpoint-samples 5000000 --auto-resume)
  fi

  if [[ -f "$output_dir/summary.json" ]] \
    && grep -q '"event": "training_complete"' "$output_dir/summary.json"; then
    return
  fi
  if [[ -d "$output_dir" \
        && -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" \
        && ! -f "$output_dir/phase_component_in_progress.pt" ]]; then
    echo "refusing to reuse incomplete output directory: $output_dir" >&2
    exit 1
  fi
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_phase_component_nnue.py \
    --total-data "$TRAIN_DATA" \
    --eval-total-data "$EVAL_DATA" \
    --output-dir "$output_dir" \
    --arch "$architecture" \
    --activation screlu_relu16_all \
    --phase-stacks 8 \
    --objective total \
    --loss-type huber \
    --huber-delta 200 \
    --hidden-scales 2 8 \
    --output-scale 128 \
    --train-samples "$train_samples" \
    --selection-samples "$selection_samples" \
    --ranking-samples "$ranking_samples" \
    --batch-size "$batch_size" \
    --workers "$WORKERS" \
    --torch-threads "$TORCH_THREADS" \
    --shuffle-block-size 250000 \
    --lr 0.0005 \
    --psqt-lr 0.001 \
    --min-lr 0.00005 \
    --lr-warmup-steps "$warmup_steps" \
    --weight-decay 0 \
    --saturation-batches 20 \
    --progress-batches "$progress_batches" \
    --device "$DEVICE" \
    "${resume_args[@]}" \
    --seed "$SEED" > "$log" 2>&1
  grep -q '"event":"training_complete"' "$log"
}

export_and_verify() {
  local architecture="$1"
  local directory="$2"
  local parity_samples="$3"
  local prefix="$4"
  .venv/bin/python tools/train/verify_phase_training_checkpoint.py \
    --checkpoint "$directory/phase_component_best.pt" \
    --arch "$architecture" \
    --expected-train-samples "$5" \
    > "${prefix}_checkpoint.json"
  .venv/bin/python tools/train/export_phase_quantized_nnue.py \
    --checkpoint "$directory/phase_component_best.pt" \
    --output "$directory/phase_quantized_nnue.bin" \
    --parity-data "$EVAL_DATA" \
    --parity-output "$directory/phase_quantized_nnue_parity.tsv" \
    --parity-samples "$parity_samples" \
    > "${prefix}_export.log"
  "$BUILD_DIR/phase_quantized_nnue_tests" \
    "$directory/phase_quantized_nnue.bin" \
    "$directory/phase_quantized_nnue_parity.tsv" \
    > "${prefix}_cpp_parity.log"
}

for architecture in F2 F2M; do
  lower="$(printf '%s' "$architecture" | tr '[:upper:]' '[:lower:]')"
  smoke_dir="$PREFLIGHT_DIR/smoke_$lower"
  run_training \
    "$architecture" "$smoke_dir" "$SMOKE_SAMPLES" \
    "$SMOKE_EVAL_SAMPLES" "$SMOKE_EVAL_SAMPLES" 2048 4 0 \
    "$PREFLIGHT_DIR/smoke_$lower.log"
  export_and_verify \
    "$architecture" "$smoke_dir" 2048 \
    "$PREFLIGHT_DIR/smoke_$lower" "$SMOKE_SAMPLES"
done

.venv/bin/python tools/train/convert_f2_to_f2m_checkpoint.py \
  --source "$REFERENCE_CHECKPOINT" \
  --output "$PREFLIGHT_DIR/equal_f2m.pt" \
  --symmetric-reference-output "$PREFLIGHT_DIR/equal_f2.pt" \
  --fold average > "$PREFLIGHT_DIR/equal_conversion.log"
.venv/bin/python tools/train/export_phase_quantized_nnue.py \
  --checkpoint "$PREFLIGHT_DIR/equal_f2.pt" \
  --output "$PREFLIGHT_DIR/equal_f2.bin" \
  > "$PREFLIGHT_DIR/equal_f2_export.log"
.venv/bin/python tools/train/export_phase_quantized_nnue.py \
  --checkpoint "$PREFLIGHT_DIR/equal_f2m.pt" \
  --output "$PREFLIGHT_DIR/equal_f2m.bin" \
  > "$PREFLIGHT_DIR/equal_f2m_export.log"
"$BUILD_DIR/benchmark_nnue_horizontal_mirror" \
  "$PREFLIGHT_DIR/equal_f2.bin" "$PREFLIGHT_DIR/equal_f2m.bin" 6 5 \
  > "$PREFLIGHT_DIR/inference_benchmark.log"
grep -q 'result_mismatches=0 node_mismatches=0' \
  "$PREFLIGHT_DIR/inference_benchmark.log"
NPS_RATIO="$(sed -n 's/.*mirror_vs_reference_nps=\([^ ]*\).*/\1/p' \
  "$PREFLIGHT_DIR/inference_benchmark.log")"
awk -v ratio="$NPS_RATIO" 'BEGIN { if (ratio + 0 < 0.95) exit 1 }'

GIT_REVISION="$(git rev-parse HEAD)"
GIT_DIRTY_HASH="$(git status --porcelain=v1 | shasum -a 256 | awk '{print $1}')"
TRAIN_SHARD_SET_HASH="$(find -L "$TRAIN_DATA" -maxdepth 1 -type f -name '*.cbin.zst' \
  -print | sort | shasum -a 256 | awk '{print $1}')"
EVAL_SHA256="$(shasum -a 256 "$EVAL_DATA/part_00000.cbin.zst" | awk '{print $1}')"
{
  printf 'tag=%s\n' "$TAG"
  printf 'git_revision=%s\n' "$GIT_REVISION"
  printf 'git_dirty_status_sha256=%s\n' "$GIT_DIRTY_HASH"
  printf 'architecture_pair=F2,F2M; shared transformer plus 8 independent phase heads\n'
  printf 'phase_formula=clamp((piece_count - 1) // 4, 0, 7)\n'
  printf 'labels=original RobotMoon Stockfish search score\n'
  printf 'objective=Huber200 total score; target clamped to +/-2000cp\n'
  printf 'train_data=%s\n' "$TRAIN_DATA"
  printf 'train_shard_path_set_sha256=%s\n' "$TRAIN_SHARD_SET_HASH"
  printf 'train_samples_each=%s\n' "$TRAIN_SAMPLES"
  printf 'validation_data=%s\n' "$EVAL_DATA"
  printf 'validation_sha256=%s\n' "$EVAL_SHA256"
  printf 'selection_samples=%s\n' "$SELECTION_SAMPLES"
  printf 'ranking_samples=%s\n' "$RANKING_SAMPLES"
  printf 'seed=%s same_data_order=true training=from_scratch_sequential\n' "$SEED"
  printf 'fault_tolerance=atomic checkpoint every 5000000 samples; exact auto-resume\n'
  printf 'batch_size=%s lr=0.0005 min_lr=0.00005 warmup_steps=610\n' "$BATCH_SIZE"
  printf 'psqt_lr=0.001 hidden_scales=2,8 output_scale=128 weight_decay=0\n'
  printf 'device=%s workers=%s torch_threads=%s\n' "$DEVICE" "$WORKERS" "$TORCH_THREADS"
  printf 'mirror_split_audit=%s\n' "$PREFLIGHT_DIR/mirror_split_audit.json"
  printf 'inference_nps_ratio=%s\n' "$NPS_RATIO"
} > "$MANIFEST"

printf 'state=training architecture=F2 updated_at=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
run_training \
  F2 "$F2_DIR" "$TRAIN_SAMPLES" "$SELECTION_SAMPLES" "$RANKING_SAMPLES" \
  "$BATCH_SIZE" 610 250 "logs/${TAG}_f2.log"
export_and_verify F2 "$F2_DIR" 8192 "$RUN_ROOT/f2" "$TRAIN_SAMPLES"
printf 'state=trained architecture=F2 updated_at=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

printf 'state=training architecture=F2M updated_at=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
run_training \
  F2M "$F2M_DIR" "$TRAIN_SAMPLES" "$SELECTION_SAMPLES" "$RANKING_SAMPLES" \
  "$BATCH_SIZE" 610 250 "logs/${TAG}_f2m.log"
export_and_verify F2M "$F2M_DIR" 8192 "$RUN_ROOT/f2m" "$TRAIN_SAMPLES"
printf 'state=trained architecture=F2M updated_at=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

.venv/bin/python tools/analyze/report_f2_f2m_50m.py \
  --f2-dir "$F2_DIR" \
  --f2m-dir "$F2M_DIR" \
  --report "$REPORT" \
  --json "$COMPARISON_JSON" \
  > "$RUN_ROOT/comparison.log"

printf 'state=complete report=%s comparison=%s updated_at=%s\n' \
  "$REPORT" "$COMPARISON_JSON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

if [[ -n "$CAFFEINATE_PID" ]]; then
  kill "$CAFFEINATE_PID" 2>/dev/null || true
  CAFFEINATE_PID=""
fi
trap - EXIT
if [[ "$SLEEP_ON_EXIT" == "1" ]] && command -v pmset >/dev/null 2>&1; then
  pmset sleepnow >/dev/null 2>&1 || true
fi
