#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:?set TAG to the completed low-hidden-scale grid tag}"
VAL_DATA="data/robotmoon_val500k_balanced_target75u25c_v3"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
OUTPUT_JSON="reports/${TAG}_paired_holdout300k.json"
OUTPUT_MD="reports/${TAG}_paired_holdout300k.md"
REFERENCE_LABEL="control_relu16_l2_l3_hs32x64_os32"
LEGACY_LABEL="legacy_screlu8_hs64x64_os16_5m_1ep"
LEGACY_CHECKPOINT="models/quantized_scale_grid/nnue_relu16_hsgrid_exact5m_20260719_012612/baseline/quant_nnue_arch_F2_best.pt"

paired_args=(
  --reference "${REFERENCE_LABEL}=${MODEL_ROOT}/${REFERENCE_LABEL}/quant_nnue_arch_F2_best.pt"
  --candidate "${LEGACY_LABEL}=${LEGACY_CHECKPOINT}"
)
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
  paired_args+=(
    --candidate "${label}=${MODEL_ROOT}/${label}/quant_nnue_arch_F2_best.pt"
  )
done

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
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"

echo "paired_holdout_complete tag=$TAG output=$OUTPUT_MD"
