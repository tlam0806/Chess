#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRAIN_DATA="data/stockfish_static_nnue_psqtpos_5m_train_v1"
VAL_DATA="data/stockfish_static_nnue_psqtpos_1m_val_v1"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  TAG="${TAG:-psqt_relu16_scale_grid_smoke_$(date +%Y%m%d_%H%M%S)}"
  EPOCHS=1
  PATIENCE=2
  TRAIN_SAMPLES=20000
  SELECTION_SAMPLES=10000
  TRAIN_PROBE_SAMPLES=10000
  RANKING_SKIP=10000
  RANKING_SAMPLES=10000
  CALIBRATION_BATCHES=2
  WARMUP_STEPS=1
  BOOTSTRAP_REPLICATES=100
  PROGRESS_BATCHES=0
else
  TAG="${TAG:-psqt_relu16_scale_grid_2m_10ep_$(date +%Y%m%d_%H%M%S)}"
  EPOCHS=10
  PATIENCE=11
  TRAIN_SAMPLES=2000000
  SELECTION_SAMPLES=200000
  TRAIN_PROBE_SAMPLES=200000
  RANKING_SKIP=200000
  RANKING_SAMPLES=300000
  CALIBRATION_BATCHES=50
  WARMUP_STEPS=245
  BOOTSTRAP_REPLICATES=10000
  PROGRESS_BATCHES=100
fi

MODEL_ROOT="models/quantized_scale_grid/$TAG"
MANIFEST="logs/$TAG.manifest"
STATUS_FILE="logs/$TAG.status"
TRAIN_REPORT="reports/$TAG.md"
PAIRED_JSON="reports/${TAG}_paired_ranking.json"
PAIRED_REPORT="reports/${TAG}_paired_ranking.md"
REFERENCE_LABEL="relu16_hs16x64_os16"

CONFIGS=(
  "relu16_hs16x64_os16:screlu_relu16_all:16:64:16"
  "relu16_hs8x8_os256:screlu_relu16_all:8:8:256"
  "relu16_hs8x16_os128:screlu_relu16_all:8:16:128"
  "relu16_hs8x32_os64:screlu_relu16_all:8:32:64"
  "relu16_hs8x64_os32:screlu_relu16_all:8:64:32"
  "relu16_hs16x8_os128:screlu_relu16_all:16:8:128"
  "relu16_hs16x16_os64:screlu_relu16_all:16:16:64"
  "relu16_hs16x32_os32:screlu_relu16_all:16:32:32"
  "relu16_hs32x8_os64:screlu_relu16_all:32:8:64"
  "relu16_hs32x16_os32:screlu_relu16_all:32:16:32"
  "relu16_hs32x32_os16:screlu_relu16_all:32:32:16"
  "relu16_hs32x64_os8:screlu_relu16_all:32:64:8"
  "screlu_all_hs91x91_os4:screlu_all:91:91:4"
)

mkdir -p "$MODEL_ROOT" logs reports

on_exit() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    printf 'state=failed tag=%s exit_code=%s updated_at=%s\n' \
      "$TAG" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_FILE"
  fi
}
trap on_exit EXIT

python3 - "$TRAIN_DATA" "$VAL_DATA" "$REFERENCE_LABEL" "${CONFIGS[@]}" <<'PY'
import math
import os
import sys

train_dir, val_dir, reference, *specs = sys.argv[1:]
if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
    raise SystemExit("training or validation directory is missing")
if len(specs) != 13:
    raise SystemExit(f"expected 13 configs, got {len(specs)}")

labels = []
for spec in specs:
    label, activation, hs1_text, hs2_text, os_text = spec.split(":")
    hs1, hs2, output_scale = map(int, (hs1_text, hs2_text, os_text))
    labels.append(label)
    if min(hs1, hs2, output_scale) <= 0:
        raise SystemExit(f"non-positive scale in {spec}")
    if activation == "screlu_relu16_all":
        output_value_scale = 16_773_632.0 / (hs1 * hs2 * output_scale)
    elif activation == "screlu_all":
        output_value_scale = (181.0 * 181.0 / 128.0) * 16.0 / output_scale
    else:
        raise SystemExit(f"unexpected activation in {spec}")
    if not 900.0 <= output_value_scale <= 1100.0:
        raise SystemExit(
            f"output scale mismatch for {label}: value_scale={output_value_scale}"
        )
if len(labels) != len(set(labels)):
    raise SystemExit("duplicate config labels")
if reference not in labels:
    raise SystemExit("paired reference is absent from the grid")

train_inodes = {
    (os.stat(os.path.join(train_dir, name)).st_dev, os.stat(os.path.join(train_dir, name)).st_ino)
    for name in os.listdir(train_dir)
    if name.endswith(".cbin.zst")
}
val_inodes = {
    (os.stat(os.path.join(val_dir, name)).st_dev, os.stat(os.path.join(val_dir, name)).st_ino)
    for name in os.listdir(val_dir)
    if name.endswith(".cbin.zst")
}
if not train_inodes or not val_inodes:
    raise SystemExit("training or validation shards are missing")
if train_inodes & val_inodes:
    raise SystemExit("training and validation corpora share a shard inode")
print(
    f"grid_preflight=pass configs={len(specs)} "
    f"train_shards={len(train_inodes)} val_shards={len(val_inodes)}"
)
PY

for shard in "$TRAIN_DATA"/*.cbin.zst "$VAL_DATA"/*.cbin.zst; do
  /opt/homebrew/bin/zstd -q -t "$shard"
done

{
  printf 'tag=%s\n' "$TAG"
  printf 'smoke=%s\n' "$SMOKE"
  printf 'git_revision=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'train_data=%s\n' "$TRAIN_DATA"
  printf 'validation_data=%s\n' "$VAL_DATA"
  printf 'train_samples=%s\n' "$TRAIN_SAMPLES"
  printf 'selection_samples=%s\n' "$SELECTION_SAMPLES"
  printf 'ranking_skip=%s\n' "$RANKING_SKIP"
  printf 'ranking_samples=%s\n' "$RANKING_SAMPLES"
  printf 'epochs=%s\n' "$EPOCHS"
  printf 'base_lr=0.0005\n'
  printf 'psqt_lr=0.001\n'
  printf 'seed=20260720\n'
  printf 'reference=%s\n' "$REFERENCE_LABEL"
  printf 'configs=%s\n' "${CONFIGS[*]}"
} > "$MANIFEST"

printf 'state=running tag=%s updated_at=%s\n' \
  "$TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_FILE"

COMMON=(
  --arch F2
  --data "$TRAIN_DATA"
  --data-format cbin
  --train-data-all-records
  --eval-data "$VAL_DATA"
  --eval-data-format cbin
  --eval-data-all-records-for-val
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --batch-size 8192
  --lr 0.0005
  --psqt-lr 0.001
  --lr-schedule cosine
  --lr-warmup-steps "$WARMUP_STEPS"
  --min-lr 0.00005
  --weight-decay 0
  --device cpu
  --workers 4
  --eval-workers 2
  --torch-threads 8
  --train-max-samples "$TRAIN_SAMPLES"
  --val-max-samples "$SELECTION_SAMPLES"
  --train-probe-max-samples "$TRAIN_PROBE_SAMPLES"
  --calibration-max-batches "$CALIBRATION_BATCHES"
  --hidden-clip 181
  --screlu-divisor 128
  --quantization-convention scale_clean
  --feature-weight-scale 181
  --linear-weight-scale 64
  --output-weight-scale 16
  --psqt
  --psqt-weight-scale 16
  --screlu-init-fraction 0.25
  --screlu-first-bias-fraction 0.1
  --forward-mode quantized
  --loss-type cp_huber
  --cp-huber-delta 200
  --shuffle-block-size 250000
  --progress-batches "$PROGRESS_BATCHES"
  --eval-progress-batches 0
  --seed 20260720
  --log-initial-saturation
  --skip-final-test
)

summary_args=()
paired_args=()

for spec in "${CONFIGS[@]}"; do
  IFS=: read -r label activation hs1 hs2 output_scale <<< "$spec"
  output_dir="$MODEL_ROOT/$label"
  checkpoint="$output_dir/quant_nnue_arch_F2_best.pt"
  log="logs/${TAG}_${label}.log"
  mkdir -p "$output_dir"

  if [[ -f "$checkpoint" ]] && [[ -f "$log" ]] && grep -q '"event":"training_complete"' "$log"; then
    printf 'config=%s state=skipped_complete updated_at=%s\n' \
      "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_FILE"
  else
    printf 'config=%s state=running updated_at=%s\n' \
      "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_FILE"
    PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_quantized_nnue_architecture.py \
      --activation "$activation" \
      --fixed-hidden-scales "$hs1" "$hs2" \
      --fixed-output-scale "$output_scale" \
      --output-dir "$output_dir" \
      "${COMMON[@]}" > "$log" 2>&1
    [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
    grep -q '"event":"training_complete"' "$log" \
      || { echo "training did not complete: $label" >&2; exit 1; }
    printf 'config=%s state=complete updated_at=%s\n' \
      "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_FILE"
  fi

  summary_args+=(--log "${label}=${log}")
  if [[ "$label" == "$REFERENCE_LABEL" ]]; then
    paired_args+=(--reference "${label}=${checkpoint}")
  else
    paired_args+=(--candidate "${label}=${checkpoint}")
  fi
done

.venv/bin/python tools/analyze/summarize_nnue_activation_ablation.py \
  --tag "$TAG" \
  "${summary_args[@]}" \
  --description "F2 256-32-32 PSQT normalized master weights; NN LR 5e-4, PSQT LR 1e-3; CP Huber delta 200; output scales preserve approximately 1000 codes per normalized CP target unit." \
  --output "$TRAIN_REPORT"

PYTHONUNBUFFERED=1 .venv/bin/python tools/analyze/evaluate_nnue_paired_comparison.py \
  "${paired_args[@]}" \
  --data "$VAL_DATA" \
  --data-format cbin \
  --split all \
  --skip-samples "$RANKING_SKIP" \
  --max-samples "$RANKING_SAMPLES" \
  --batch-size 8192 \
  --workers 0 \
  --torch-threads 8 \
  --device cpu \
  --forward-mode quantized \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --bootstrap-block-size 512 \
  --confidence 0.95 \
  --output-json "$PAIRED_JSON" \
  --output-md "$PAIRED_REPORT" > "logs/${TAG}_paired.log" 2>&1

printf 'state=complete tag=%s train_report=%s paired_report=%s updated_at=%s\n' \
  "$TAG" "$TRAIN_REPORT" "$PAIRED_REPORT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_FILE"
trap - EXIT
