#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:-nnue_capacity_20m_$(date +%Y%m%d_%H%M%S)}"
DATA="data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m"
VAL_DATA="data/robotmoon_val500k_balanced_target75u25c_v3"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
REPORT="reports/${TAG}.md"

mkdir -p "$MODEL_ROOT" logs reports

common=(
  --activation screlu_all
  --data "$DATA"
  --data-format cbin
  --train-data-all-records
  --eval-data "$VAL_DATA"
  --eval-data-format cbin
  --eval-data-all-records-for-val
  --epochs 4
  --patience 4
  --batch-size 4096
  --lr 0.0005
  --lr-schedule constant
  --weight-decay 0
  --device cpu
  --workers 4
  --eval-workers 0
  --torch-threads 10
  --train-max-samples 20000000
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
  --fixed-hidden-scales 64 64
  --fixed-output-scale 16
  --screlu-init-fraction 0.25
  --screlu-first-bias-fraction 0.1
  --forward-mode quantized
  --loss-type cp_huber
  --cp-huber-delta 600
  --shuffle-block-size 1000000
  --progress-batches 500
  --eval-progress-batches 0
  --seed 20260719
  --skip-final-test
)

for spec in "baseline:F2" "wide:F2_64"; do
  label="${spec%%:*}"
  arch="${spec##*:}"
  out="$MODEL_ROOT/$label"
  log="logs/${TAG}_${label}.log"
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train_quantized_nnue_architecture.py \
    --arch "$arch" \
    --output-dir "$out" \
    "${common[@]}" 2>&1 | tee "$log"
done

.venv/bin/python tools/summarize_nnue_capacity_ablation.py \
  --tag "$TAG" \
  --baseline-log "logs/${TAG}_baseline.log" \
  --wide-log "logs/${TAG}_wide.log" \
  --output "$REPORT"

echo "capacity_ablation_complete tag=$TAG report=$REPORT"
