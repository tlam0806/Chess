#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BASE_TAG="${BASE_TAG:-old_score_huber200_f2_f2m_50m_20260823}"
RUN_ROOT="models/quantized_scale_grid/$BASE_TAG"
F2_DIR="$RUN_ROOT/f2"
F2M_DIR="$RUN_ROOT/f2m"
OUTPUT_DIR="$RUN_ROOT/f2m_shared_l1"
BASE_STATUS="logs/$BASE_TAG.status"
STATUS="logs/${BASE_TAG}_f2m_shared_l1_append.status"
LOG="logs/${BASE_TAG}_f2m_shared_l1.log"
REQUEST="$RUN_ROOT/append_f2m_shared_l1.requested"
READY="$RUN_ROOT/append_f2m_shared_l1.ready"
FAILED="$RUN_ROOT/append_f2m_shared_l1.failed"
MANIFEST="$RUN_ROOT/f2m_shared_l1.manifest"
REPORT="reports/$BASE_TAG.md"
COMPARISON_JSON="$RUN_ROOT/comparison.json"
COMPARISON_LOG="$RUN_ROOT/comparison.log"
TRAIN_DATA="${TRAIN_DATA:-data/robotmoon_old_score_train_50m_v1}"
EVAL_DATA="${EVAL_DATA:-data/robotmoon_old_score_val_mirror_safe_v1}"
BUILD_DIR="${BUILD_DIR:-build-release}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000000}"
SELECTION_SAMPLES="${SELECTION_SAMPLES:-499149}"
RANKING_SAMPLES="${RANKING_SAMPLES:-499149}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
WORKERS="${WORKERS:-0}"
TORCH_THREADS="${TORCH_THREADS:-8}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-20260720}"
POLL_SECONDS="${POLL_SECONDS:-10}"
SLEEP_ON_EXIT="${SLEEP_ON_EXIT:-1}"

mkdir -p "$RUN_ROOT" logs
rm -f "$READY" "$FAILED"
printf 'requested_at=%s architecture=F2M phase_layout=shared_first\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$REQUEST"
printf 'state=waiting_for_f2m base_tag=%s updated_at=%s\n' "$BASE_TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

CAFFEINATE_PID=""
SUCCESS=0
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
  CAFFEINATE_PID=$!
fi

on_exit() {
  local code=$?
  if [[ -n "$CAFFEINATE_PID" ]]; then
    kill "$CAFFEINATE_PID" 2>/dev/null || true
  fi
  if [[ $SUCCESS -ne 1 ]]; then
    printf 'exit_code=%s failed_at=%s\n' "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$FAILED"
    printf 'state=failed exit_code=%s updated_at=%s\n' "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  fi
}
trap on_exit EXIT

for required in .venv/bin/python tools/train/train_phase_component_nnue.py tools/train/export_phase_quantized_nnue.py tools/train/verify_phase_training_checkpoint.py tools/analyze/report_f2_f2m_50m.py; do
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
  echo "shared-L1 append requires TRAIN_SAMPLES=50000000" >&2
  exit 1
fi
if (( SELECTION_SAMPLES <= 0 || RANKING_SAMPLES <= 0 || SELECTION_SAMPLES + RANKING_SAMPLES > 998298 )); then
  echo "selection/ranking windows exceed sealed mirror-safe validation" >&2
  exit 1
fi

while ! grep -q '^state=trained architecture=F2M ' "$BASE_STATUS" 2>/dev/null; do
  if grep -q '^state=failed ' "$BASE_STATUS" 2>/dev/null; then
    echo "base paired job failed before F2M completed" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done
for completed in "$F2_DIR/summary.json" "$F2M_DIR/summary.json"; do
  if [[ ! -f "$completed" ]] || ! grep -q '"event": "training_complete"' "$completed"; then
    echo "missing completed base run: $completed" >&2
    exit 1
  fi
done

if [[ -f "$OUTPUT_DIR/summary.json" ]] && grep -q '"event": "training_complete"' "$OUTPUT_DIR/summary.json"; then
  printf 'state=reusing_complete_training updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
elif [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" && ! -f "$OUTPUT_DIR/phase_component_in_progress.pt" ]]; then
  echo "refusing to reuse incomplete output directory: $OUTPUT_DIR" >&2
  exit 1
else
  printf 'state=training architecture=F2M phase_layout=shared_first updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  printf '{"event":"process_resume","supervisor":"%s","updated_at":"%s"}\n' "${PROCESS_SUPERVISOR:-direct-shell}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_phase_component_nnue.py --total-data "$TRAIN_DATA" --eval-total-data "$EVAL_DATA" --output-dir "$OUTPUT_DIR" --arch F2M --activation screlu_relu16_all --phase-stacks 8 --phase-layout shared_first --objective total --loss-type huber --huber-delta 200 --hidden-scales 2 8 --output-scale 128 --train-samples "$TRAIN_SAMPLES" --selection-samples "$SELECTION_SAMPLES" --ranking-samples "$RANKING_SAMPLES" --batch-size "$BATCH_SIZE" --workers "$WORKERS" --torch-threads "$TORCH_THREADS" --shuffle-block-size 250000 --lr 0.0005 --psqt-lr 0.001 --min-lr 0.00005 --lr-warmup-steps 610 --weight-decay 0 --saturation-batches 20 --progress-batches 250 --checkpoint-samples 5000000 --auto-resume --device "$DEVICE" --seed "$SEED" >> "$LOG" 2>&1
  grep -q '"event":"training_complete"' "$LOG"
fi

printf 'state=verifying architecture=F2M phase_layout=shared_first updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
if [[ ! -x "$BUILD_DIR/phase_quantized_nnue_tests" ]]; then
  cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$BUILD_DIR" --target phase_quantized_nnue_tests -j 4 >/dev/null
fi
.venv/bin/python tools/train/verify_phase_training_checkpoint.py --checkpoint "$OUTPUT_DIR/phase_component_best.pt" --arch F2M --phase-layout shared_first --expected-train-samples "$TRAIN_SAMPLES" > "$RUN_ROOT/f2m_shared_l1_checkpoint.json"
.venv/bin/python tools/train/export_phase_quantized_nnue.py --checkpoint "$OUTPUT_DIR/phase_component_best.pt" --output "$OUTPUT_DIR/phase_quantized_nnue.bin" --parity-data "$EVAL_DATA" --parity-output "$OUTPUT_DIR/phase_quantized_nnue_parity.tsv" --parity-samples 8192 > "$RUN_ROOT/f2m_shared_l1_export.log"
"$BUILD_DIR/phase_quantized_nnue_tests" "$OUTPUT_DIR/phase_quantized_nnue.bin" "$OUTPUT_DIR/phase_quantized_nnue_parity.tsv" > "$RUN_ROOT/f2m_shared_l1_cpp_parity.log"

GIT_REVISION="$(git rev-parse HEAD)"
GIT_DIRTY_HASH="$(git status --porcelain=v1 | shasum -a 256 | awk '{print $1}')"
{
  printf 'base_tag=%s\n' "$BASE_TAG"
  printf 'git_revision=%s\n' "$GIT_REVISION"
  printf 'git_dirty_status_sha256=%s\n' "$GIT_DIRTY_HASH"
  printf 'architecture=F2M\n'
  printf 'phase_layout=shared_first; shared_256x32_then_8x_phase_32x32x1\n'
  printf 'export_layout=shared_layer_duplicated_into_existing_8_stack_binary\n'
  printf 'runtime_inference_change=none\n'
  printf 'train_data=%s\n' "$TRAIN_DATA"
  printf 'train_samples=%s\n' "$TRAIN_SAMPLES"
  printf 'validation_data=%s\n' "$EVAL_DATA"
  printf 'selection_samples=%s ranking_samples=%s\n' "$SELECTION_SAMPLES" "$RANKING_SAMPLES"
  printf 'seed=%s batch_size=%s workers=%s torch_threads=%s\n' "$SEED" "$BATCH_SIZE" "$WORKERS" "$TORCH_THREADS"
  printf 'fault_tolerance=atomic checkpoint every 5000000 samples; exact auto-resume\n'
  printf 'process_supervisor=%s\n' "${PROCESS_SUPERVISOR:-direct-shell}"
  printf 'objective=Huber200 total score; target clamped to +/-2000cp\n'
} > "$MANIFEST"

printf 'state=reporting architectures=F2,F2M,F2M_shared_L1 updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
.venv/bin/python tools/analyze/report_f2_f2m_50m.py \
  --f2-dir "$F2_DIR" \
  --f2m-dir "$F2M_DIR" \
  --shared-dir "$OUTPUT_DIR" \
  --report "$REPORT" \
  --json "$COMPARISON_JSON" > "$COMPARISON_LOG"

printf 'ready_at=%s architecture=F2M phase_layout=shared_first\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$READY"
printf 'state=complete ready=%s report=%s comparison=%s updated_at=%s\n' "$READY" "$REPORT" "$COMPARISON_JSON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
printf 'state=complete report=%s comparison=%s updated_at=%s\n' "$REPORT" "$COMPARISON_JSON" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE_STATUS"
SUCCESS=1
if [[ -n "$CAFFEINATE_PID" ]]; then
  kill "$CAFFEINATE_PID" 2>/dev/null || true
  CAFFEINATE_PID=""
fi
trap - EXIT
if [[ "$SLEEP_ON_EXIT" == "1" ]] && command -v pmset >/dev/null 2>&1; then
  pmset sleepnow >/dev/null 2>&1 || true
fi
