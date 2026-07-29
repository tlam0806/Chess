#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PARENT_TAG="${PARENT_TAG:-psqt_relu16_scale_grid_2m_10ep_20260720_0010}"
TAG="${TAG:-psqt_relu16_low_hs_extension_2m_10ep_$(date +%Y%m%d_%H%M%S)}"
TRAIN_DATA="data/stockfish_static_nnue_psqtpos_5m_train_v1"
VAL_DATA="data/stockfish_static_nnue_psqtpos_1m_val_v1"
PARENT_MODEL_ROOT="models/quantized_scale_grid/$PARENT_TAG"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS_FILE="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
TRAIN_REPORT="reports/$TAG.md"
PAIRED_JSON="reports/${TAG}_combined_paired_ranking.json"
PAIRED_REPORT="reports/${TAG}_combined_paired_ranking.md"

EPOCHS=10
TRAIN_SAMPLES=2000000
SELECTION_SAMPLES=200000
TRAIN_PROBE_SAMPLES=200000
RANKING_SKIP=200000
RANKING_SAMPLES=300000
REFERENCE_LABEL="relu16_hs32x8_os64"

NEW_CONFIGS=(
  "relu16_hs16x2_os512:16:2:512"
  "relu16_hs16x4_os256:16:4:256"
  "relu16_hs32x2_os256:32:2:256"
  "relu16_hs32x4_os128:32:4:128"
)

PARENT_LABELS=(
  relu16_hs16x64_os16
  relu16_hs8x8_os256
  relu16_hs8x16_os128
  relu16_hs8x32_os64
  relu16_hs8x64_os32
  relu16_hs16x8_os128
  relu16_hs16x16_os64
  relu16_hs16x32_os32
  relu16_hs32x8_os64
  relu16_hs32x16_os32
  relu16_hs32x32_os16
  relu16_hs32x64_os8
  screlu_all_hs91x91_os4
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

parent_status="logs/$PARENT_TAG.status"
if [[ ! -f "$parent_status" ]] || ! tail -1 "$parent_status" | grep -q '^state=complete '; then
  echo "parent grid is not complete: $parent_status" >&2
  exit 1
fi

python3 - "$TRAIN_DATA" "$VAL_DATA" "${NEW_CONFIGS[@]}" <<'PY'
import os
import sys

train_dir, val_dir, *specs = sys.argv[1:]
if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
    raise SystemExit("training or validation directory is missing")
if len(specs) != 4:
    raise SystemExit(f"expected 4 configs, got {len(specs)}")

labels = []
for spec in specs:
    label, hs1_text, hs2_text, os_text = spec.split(":")
    hs1, hs2, output_scale = map(int, (hs1_text, hs2_text, os_text))
    labels.append(label)
    if any(value <= 0 or value & (value - 1) for value in (hs1, hs2, output_scale)):
        raise SystemExit(f"all scales must be positive powers of two: {spec}")
    if hs1 * hs2 * output_scale != 16384:
        raise SystemExit(f"scale invariant failed: {spec}")

if len(labels) != len(set(labels)):
    raise SystemExit("duplicate config labels")

train_inodes = {
    (os.stat(os.path.join(train_dir, name)).st_dev,
     os.stat(os.path.join(train_dir, name)).st_ino)
    for name in os.listdir(train_dir) if name.endswith(".cbin.zst")
}
val_inodes = {
    (os.stat(os.path.join(val_dir, name)).st_dev,
     os.stat(os.path.join(val_dir, name)).st_ino)
    for name in os.listdir(val_dir) if name.endswith(".cbin.zst")
}
if not train_inodes or not val_inodes:
    raise SystemExit("training or validation shards are missing")
if train_inodes & val_inodes:
    raise SystemExit("training and validation corpora share a shard inode")

print(
    f"extension_preflight=pass configs={len(specs)} "
    f"train_shards={len(train_inodes)} val_shards={len(val_inodes)}"
)
PY

for label in "${PARENT_LABELS[@]}"; do
  checkpoint="$PARENT_MODEL_ROOT/$label/quant_nnue_arch_F2_best.pt"
  [[ -f "$checkpoint" ]] || { echo "missing parent checkpoint: $checkpoint" >&2; exit 1; }
done

for shard in "$TRAIN_DATA"/*.cbin.zst "$VAL_DATA"/*.cbin.zst; do
  /opt/homebrew/bin/zstd -q -t "$shard"
done

{
  printf 'tag=%s\n' "$TAG"
  printf 'parent_tag=%s\n' "$PARENT_TAG"
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
  printf 'new_configs=%s\n' "${NEW_CONFIGS[*]}"
} > "$MANIFEST"

printf 'state=running tag=%s parent_tag=%s updated_at=%s\n' \
  "$TAG" "$PARENT_TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_FILE"

COMMON=(
  --arch F2
  --activation screlu_relu16_all
  --data "$TRAIN_DATA"
  --data-format cbin
  --train-data-all-records
  --eval-data "$VAL_DATA"
  --eval-data-format cbin
  --eval-data-all-records-for-val
  --epochs "$EPOCHS"
  --patience 11
  --batch-size 8192
  --lr 0.0005
  --psqt-lr 0.001
  --lr-schedule cosine
  --lr-warmup-steps 245
  --min-lr 0.00005
  --weight-decay 0
  --device cpu
  --workers 4
  --eval-workers 2
  --torch-threads 8
  --train-max-samples "$TRAIN_SAMPLES"
  --val-max-samples "$SELECTION_SAMPLES"
  --train-probe-max-samples "$TRAIN_PROBE_SAMPLES"
  --calibration-max-batches 50
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
  --progress-batches 100
  --eval-progress-batches 0
  --seed 20260720
  --log-initial-saturation
  --skip-final-test
)

summary_args=()
new_paired_args=()

for spec in "${NEW_CONFIGS[@]}"; do
  IFS=: read -r label hs1 hs2 output_scale <<< "$spec"
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
    PYTHONUNBUFFERED=1 .venv/bin/python tools/train_quantized_nnue_architecture.py \
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
  new_paired_args+=(--candidate "${label}=${checkpoint}")
done

.venv/bin/python tools/summarize_nnue_activation_ablation.py \
  --tag "$TAG" \
  "${summary_args[@]}" \
  --description "Low-hs2 extension of the PSQT ReLU16 scale grid. Four power-of-two configs; same 2M repeated training positions, ten epochs, seed, optimizer, and held-out evaluation as the parent grid." \
  --output "$TRAIN_REPORT"

paired_args=()
for label in "${PARENT_LABELS[@]}"; do
  checkpoint="$PARENT_MODEL_ROOT/$label/quant_nnue_arch_F2_best.pt"
  if [[ "$label" == "$REFERENCE_LABEL" ]]; then
    paired_args+=(--reference "${label}=${checkpoint}")
  else
    paired_args+=(--candidate "${label}=${checkpoint}")
  fi
done
paired_args+=("${new_paired_args[@]}")

PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_nnue_paired_comparison.py \
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
  --bootstrap-replicates 10000 \
  --bootstrap-block-size 512 \
  --confidence 0.95 \
  --output-json "$PAIRED_JSON" \
  --output-md "$PAIRED_REPORT" > "logs/${TAG}_paired.log" 2>&1

printf 'state=complete tag=%s train_report=%s paired_report=%s updated_at=%s\n' \
  "$TAG" "$TRAIN_REPORT" "$PAIRED_REPORT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_FILE"
trap - EXIT
