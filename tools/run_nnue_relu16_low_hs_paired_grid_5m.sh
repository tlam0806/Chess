#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:-nnue_relu16_low_hs_paired_grid_2m_10ep_$(date +%Y%m%d_%H%M%S)}"
DATA="data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m"
VAL_DATA="data/robotmoon_val500k_balanced_target75u25c_v3"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
TRAIN_REPORT="reports/${TAG}.md"
PAIRED_JSON="reports/${TAG}_paired.json"
PAIRED_REPORT="reports/${TAG}_paired.md"
REFERENCE_LABEL="control_relu16_l2_l3_hs32x64_os32"
REFERENCE_CHECKPOINT="$MODEL_ROOT/$REFERENCE_LABEL/quant_nnue_arch_F2_best.pt"
REFERENCE_LOG="logs/${TAG}_${REFERENCE_LABEL}.log"
LEGACY_LABEL="legacy_screlu8_hs64x64_os16_5m_1ep"
LEGACY_CHECKPOINT="models/quantized_scale_grid/nnue_relu16_hsgrid_exact5m_20260719_012612/baseline/quant_nnue_arch_F2_best.pt"

mkdir -p "$MODEL_ROOT" logs reports
common=(
  --arch F2
  --activation screlu_relu16_all
  --data "$DATA"
  --data-format cbin
  --train-data-all-records
  --eval-data "$VAL_DATA"
  --eval-data-format cbin
  --eval-data-all-records-for-val
  --epochs 10
  --patience 10
  --batch-size 4096
  --lr 0.001
  --lr-schedule cosine
  --lr-warmup-steps 489
  --min-lr 0.00005
  --weight-decay 0
  --device cpu
  --workers 4
  --eval-workers 0
  --torch-threads 10
  --train-max-samples 2000000
  --val-max-samples 200000
  --test-max-samples 100000
  --train-probe-max-samples 200000
  --calibration-max-batches 50
  --hidden-clip 255
  --screlu-divisor 256
  --quantization-convention scale_clean
  --feature-weight-scale 255
  --linear-weight-scale 64
  --output-weight-scale 16
  --screlu-init-fraction 0.25
  --screlu-first-bias-fraction 0.1
  --forward-mode quantized
  --loss-type cp_huber
  --cp-huber-delta 600
  --shuffle-block-size 1000000
  --progress-batches 125
  --eval-progress-batches 0
  --seed 20260719
  --skip-final-test
)

PYTHONUNBUFFERED=1 .venv/bin/python tools/train_quantized_nnue_architecture.py \
  --output-dir "$MODEL_ROOT/$REFERENCE_LABEL" \
  --fixed-hidden-scales 32 64 \
  --fixed-output-scale 32 \
  "${common[@]}" 2>&1 | tee "$REFERENCE_LOG"

summary_args=(--log "${REFERENCE_LABEL}=${REFERENCE_LOG}")
paired_args=(
  --reference "${REFERENCE_LABEL}=${REFERENCE_CHECKPOINT}"
  --candidate "${LEGACY_LABEL}=${LEGACY_CHECKPOINT}"
)

# Preserve the real-valued output convention while lowering hidden divisors:
# output_scale = 65536 / (hidden_scale_1 * hidden_scale_2).
for spec in \
  "16:64:64" \
  "16:32:128" \
  "16:16:256" \
  "16:8:512" \
  "8:64:128" \
  "8:32:256" \
  "8:16:512" \
  "8:8:1024"; do
  IFS=: read -r hs1 hs2 os <<< "$spec"
  label="relu16_l2_l3_hs${hs1}x${hs2}_os${os}"
  log="logs/${TAG}_${label}.log"
  checkpoint="$MODEL_ROOT/$label/quant_nnue_arch_F2_best.pt"
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train_quantized_nnue_architecture.py \
    --output-dir "$MODEL_ROOT/$label" \
    --fixed-hidden-scales "$hs1" "$hs2" \
    --fixed-output-scale "$os" \
    "${common[@]}" 2>&1 | tee "$log"
  summary_args+=(--log "${label}=${log}")
  paired_args+=(--candidate "${label}=${checkpoint}")
done

.venv/bin/python tools/summarize_nnue_activation_ablation.py \
  --tag "$TAG" \
  "${summary_args[@]}" \
  --description "All branches use F2 256-32-32 with SCReLU8 -> ReLU16 -> ReLU16, 2M shuffled samples per epoch for 10 epochs, Huber delta=600, weight decay 0, peak LR 1e-3 with one-epoch warmup and cosine decay to 5e-5." \
  --output "$TRAIN_REPORT"

PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_nnue_paired_comparison.py \
  "${paired_args[@]}" \
  --data "$VAL_DATA" \
  --data-format cbin \
  --split all \
  --skip-samples 200000 \
  --max-samples 300000 \
  --batch-size 4096 \
  --workers 0 \
  --torch-threads 10 \
  --device cpu \
  --forward-mode quantized \
  --bootstrap-replicates 10000 \
  --bootstrap-block-size 512 \
  --confidence 0.95 \
  --output-json "$PAIRED_JSON" \
  --output-md "$PAIRED_REPORT" 2>&1 | tee "logs/${TAG}_paired.log"

echo "relu16_low_hs_paired_grid_complete tag=$TAG train_report=$TRAIN_REPORT paired_report=$PAIRED_REPORT"
