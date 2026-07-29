#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:-psqt_relu16_top7_pending_50m_1ep_$(date +%Y%m%d_%H%M%S)}"
SOURCE_DATA="data/stockfish_static_nnue_psqtpos_200m_balanced_v1_shards_1m"
SOURCE_MANIFEST="data/robotmoon_balanced_cp_200m_sf_static_eligible_v1.manifest.json"
LABEL_MANIFEST="data/stockfish_static_nnue_psqtpos_200m_balanced_v1.manifest.json"
TRAIN_POOL="data/stockfish_static_nnue_psqtpos_199m_train_pool_v1"
VAL_DATA="data/stockfish_static_nnue_psqtpos_1m_val_v1"
VALIDATION_SHARD="part_00005.cbin.zst"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS_FILE="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
TRAIN_REPORT="reports/$TAG.md"
PAIRED_JSON="reports/${TAG}_paired_ranking.json"
PAIRED_REPORT="reports/${TAG}_paired_ranking.md"

TRAIN_SAMPLES=50000000
SELECTION_SAMPLES=500000
TRAIN_PROBE_SAMPLES=200000
RANKING_SKIP=500000
RANKING_SAMPLES=500000
SEED=20260720
REFERENCE_LABEL="relu16_hs32x8_os64"

CONFIGS=(
  "relu16_hs32x8_os64:32:8:64"
  "relu16_hs16x8_os128:16:8:128"
  "relu16_hs32x16_os32:32:16:32"
  "relu16_hs8x16_os128:8:16:128"
  "relu16_hs8x32_os64:8:32:64"
  "relu16_hs8x8_os256:8:8:256"
  "relu16_hs16x16_os64:16:16:64"
  "relu16_hs32x2_os256:32:2:256"
  "relu16_hs32x4_os128:32:4:128"
)

mkdir -p "$MODEL_ROOT" "$TRAIN_POOL" logs reports

on_exit() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    printf 'state=failed tag=%s exit_code=%s updated_at=%s\n' \
      "$TAG" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_FILE"
  fi
}
trap on_exit EXIT

PREFLIGHT_OUTPUT="$(python3 - "$SOURCE_MANIFEST" "$LABEL_MANIFEST" "$SOURCE_DATA" "$VAL_DATA" \
  "$VALIDATION_SHARD" "$TRAIN_POOL" "$SEED" "${CONFIGS[@]}" <<'PY'
import json
import os
import random
import sys
from pathlib import Path

source_manifest_path = Path(sys.argv[1])
label_manifest_path = Path(sys.argv[2])
source_dir = Path(sys.argv[3])
val_dir = Path(sys.argv[4])
validation_shard = sys.argv[5]
train_pool = Path(sys.argv[6])
seed = int(sys.argv[7])
specs = sys.argv[8:]

if len(specs) != 9:
    raise SystemExit(f"expected 9 configs, got {len(specs)}")

labels = []
for spec in specs:
    label, hs1_text, hs2_text, output_text = spec.split(":")
    hs1, hs2, output_scale = map(int, (hs1_text, hs2_text, output_text))
    labels.append(label)
    if any(value <= 0 or value & (value - 1) for value in (hs1, hs2, output_scale)):
        raise SystemExit(f"scales must be positive powers of two: {spec}")
    if hs1 * hs2 * output_scale != 16384:
        raise SystemExit(f"scale invariant failed: {spec}")
if len(labels) != len(set(labels)):
    raise SystemExit("duplicate config labels")

source_manifest = json.loads(source_manifest_path.read_text())
expected_bins = [
    (0, 100, 0.25),
    (100, 300, 0.25),
    (300, 600, 0.20),
    (600, 1000, 0.13),
    (1000, 1600, 0.08),
    (1600, 2000, 0.05),
    (2000, 5000, 0.04),
]
actual_bins = [
    (entry["min_abs_cp"], entry["max_abs_cp"], entry["ratio"])
    for entry in source_manifest["selection"]["bins"]
]
if actual_bins != expected_bins:
    raise SystemExit(f"unexpected source distribution: {actual_bins}")
if any(entry["repeated_records"] for entry in source_manifest["selection"]["bins"]):
    raise SystemExit("source manifest contains repeated records")

label_manifest = json.loads(label_manifest_path.read_text())
if label_manifest["records"] != 200_000_000 or label_manifest["shards"] != 200:
    raise SystemExit("labeled corpus does not contain the expected 200M records")

source_paths = sorted(source_dir.glob("part_*.cbin.zst"))
if len(source_paths) != 200:
    raise SystemExit(f"expected 200 source shards, found {len(source_paths)}")
validation_source = source_dir / validation_shard
validation_path = val_dir / validation_shard
if not validation_source.exists() or not validation_path.exists():
    raise SystemExit("validation shard is missing")
if not os.path.samefile(validation_source, validation_path):
    raise SystemExit("validation directory does not point to the expected source shard")

expected_pool = [path for path in source_paths if path.name != validation_shard]
for source in expected_pool:
    destination = train_pool / source.name
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise SystemExit(f"unexpected existing train-pool path: {destination}")
    else:
        os.link(source, destination)
pool_paths = sorted(train_pool.glob("part_*.cbin.zst"))
if len(pool_paths) != 199:
    raise SystemExit(f"expected 199 train-pool shards, found {len(pool_paths)}")
if (train_pool / validation_shard).exists():
    raise SystemExit("validation shard leaked into the training pool")

rng = random.Random(seed)
selected_paths = pool_paths.copy()
rng.shuffle(selected_paths)
selected_paths = selected_paths[:50]
print(
    f"preflight=pass configs={len(specs)} pool_shards={len(pool_paths)} "
    f"selected_shards={len(selected_paths)} selected_records=50000000"
)
for path in selected_paths:
    print(f"selected_shard={path.name}")
PY
 )"
printf '%s\n' "$PREFLIGHT_OUTPUT" | sed -n '/^preflight=/p'
SELECTED_SHARDS="$(printf '%s\n' "$PREFLIGHT_OUTPUT" | sed -n 's/^selected_shard=//p')"
[[ "$(printf '%s\n' "$SELECTED_SHARDS" | sed '/^$/d' | wc -l | tr -d ' ')" == "50" ]] \
  || { echo "preflight did not select exactly 50 shards" >&2; exit 1; }

while IFS= read -r name; do
  /opt/homebrew/bin/zstd -q -t "$TRAIN_POOL/$name"
done <<< "$SELECTED_SHARDS"
/opt/homebrew/bin/zstd -q -t "$VAL_DATA/$VALIDATION_SHARD"

{
  printf 'tag=%s\n' "$TAG"
  printf 'git_revision=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'source_data=%s\n' "$SOURCE_DATA"
  printf 'train_pool=%s\n' "$TRAIN_POOL"
  printf 'validation_data=%s\n' "$VAL_DATA"
  printf 'validation_shard=%s\n' "$VALIDATION_SHARD"
  printf 'train_samples=%s\n' "$TRAIN_SAMPLES"
  printf 'epochs=1\n'
  printf 'selection_samples=%s\n' "$SELECTION_SAMPLES"
  printf 'ranking_skip=%s\n' "$RANKING_SKIP"
  printf 'ranking_samples=%s\n' "$RANKING_SAMPLES"
  printf 'seed=%s\n' "$SEED"
  printf 'base_lr=0.0005\n'
  printf 'psqt_lr=0.001\n'
  printf 'lr_schedule=cosine\n'
  printf 'lr_warmup_steps=610\n'
  printf 'source_robotmoon_distribution=0:100:25%%,100:300:25%%,300:600:20%%,600:1000:13%%,1000:1600:8%%,1600:2000:5%%,2000:5000:4%%\n'
  printf 'selection_policy=random 50 complete 1M shards from 199-shard pool after excluding physical validation shard\n'
  printf 'selected_shards_begin\n'
  printf '%s\n' "$SELECTED_SHARDS"
  printf 'selected_shards_end\n'
  printf 'configs=%s\n' "${CONFIGS[*]}"
} > "$MANIFEST"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf 'preflight_only=complete tag=%s manifest=%s\n' "$TAG" "$MANIFEST"
  trap - EXIT
  exit 0
fi

printf 'state=running tag=%s updated_at=%s\n' \
  "$TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_FILE"

COMMON=(
  --arch F2
  --activation screlu_relu16_all
  --data "$TRAIN_POOL"
  --data-format cbin
  --train-data-all-records
  --eval-data "$VAL_DATA"
  --eval-data-format cbin
  --eval-data-all-records-for-val
  --epochs 1
  --patience 2
  --batch-size 8192
  --lr 0.0005
  --psqt-lr 0.001
  --lr-schedule cosine
  --lr-warmup-steps 610
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
  --progress-batches 250
  --eval-progress-batches 0
  --seed "$SEED"
  --log-initial-saturation
  --skip-final-test
)

summary_args=()
paired_args=()
for spec in "${CONFIGS[@]}"; do
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
  if [[ "$label" == "$REFERENCE_LABEL" ]]; then
    paired_args+=(--reference "${label}=${checkpoint}")
  else
    paired_args+=(--candidate "${label}=${checkpoint}")
  fi
done

.venv/bin/python tools/summarize_nnue_activation_ablation.py \
  --tag "$TAG" \
  "${summary_args[@]}" \
  --description "Nine PSQT ReLU16 configs trained from scratch for one epoch on the same random 50M unique-position subset. Physical validation shard excluded before sampling." \
  --output "$TRAIN_REPORT"

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
