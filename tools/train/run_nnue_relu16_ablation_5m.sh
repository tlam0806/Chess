#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${TAG:-nnue_relu16_ablation_5m_$(date +%Y%m%d_%H%M%S)}"
DATA="data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m"
VAL_DATA="data/robotmoon_val500k_balanced_target75u25c_v3"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
REPORT="reports/${TAG}.md"

mkdir -p "$MODEL_ROOT" logs reports

common=(
  --arch F2
  --data "$DATA"
  --data-format cbin
  --train-data-all-records
  --eval-data "$VAL_DATA"
  --eval-data-format cbin
  --eval-data-all-records-for-val
  --epochs 1
  --patience 1
  --batch-size 4096
  --lr 0.0005
  --lr-schedule constant
  --weight-decay 0
  --device cpu
  --workers 4
  --eval-workers 0
  --torch-threads 10
  --train-max-samples 5000000
  --val-max-samples 500000
  --test-max-samples 100000
  --train-probe-max-samples 500000
  --calibration-max-batches 50
  --hidden-clip 255
  --screlu-divisor 256
  --quantization-convention scale_clean
  --feature-weight-scale 255
  --linear-weight-scale 64
  --output-weight-scale 16
  --fixed-output-scale 16
  --screlu-init-fraction 0.25
  --screlu-first-bias-fraction 0.1
  --forward-mode quantized
  --loss-type cp_huber
  --cp-huber-delta 600
  --shuffle-block-size 1000000
  --progress-batches 250
  --eval-progress-batches 0
  --seed 20260719
  --skip-final-test
)

specs=("baseline:screlu_all:64:64")
for pair in "16:256" "24:192" "32:128" "48:96" "64:64"; do
  hs1="${pair%%:*}"
  hs2="${pair##*:}"
  specs+=("relu16_l2_hs${hs1}x${hs2}:screlu_relu16_screlu:${hs1}:${hs2}")
  specs+=("relu16_l2_l3_hs${hs1}x${hs2}:screlu_relu16_all:${hs1}:${hs2}")
done

summary_args=()
for spec in "${specs[@]}"; do
  IFS=: read -r label activation hs1 hs2 <<< "$spec"
  out="$MODEL_ROOT/$label"
  log="logs/${TAG}_${label}.log"
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_quantized_nnue_architecture.py \
    --activation "$activation" \
    --output-dir "$out" \
    "${common[@]}" \
    --fixed-hidden-scales "$hs1" "$hs2" 2>&1 | tee "$log"
  summary_args+=(--log "${label}=${log}")
done

.venv/bin/python tools/analyze/summarize_nnue_activation_ablation.py \
  --tag "$TAG" \
  "${summary_args[@]}" \
  --output "$REPORT"

echo "relu16_ablation_complete tag=$TAG report=$REPORT"
