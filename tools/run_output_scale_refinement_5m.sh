#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
tag="${TAG:-os_refine_5m_c181_d128_hs32x16_$(date +%Y%m%d_%H%M%S)}"

exec "$python_bin" tools/run_quantized_scale_grid.py \
  --data data/robotmoon_balanced_cp_199m_rawmax5000_unique_v2_train_excluding_val_shard \
  --data-format cbin \
  --train-data-all-records \
  --eval-data data/robotmoon_val500k_balanced_target75u25c_v3 \
  --eval-data-format cbin \
  --architectures F2 \
  --activations screlu_all \
  --selected-scale-configs "32x16:10,32x16:12,32x16:14,32x16:16,32x16:18" \
  --hidden-clip 181 \
  --screlu-divisor 128 \
  --quantization-convention scale_clean \
  --feature-weight-scale 181 \
  --linear-weight-scale 64 \
  --output-weight-scale 16 \
  --screlu-init-fraction 0.25 \
  --screlu-first-bias-fraction 0.1 \
  --reject-initial-zero-rate 0.995 \
  --reject-initial-clip-rate 0.8 \
  --cp-huber-delta 200 \
  --jobs 5 \
  --device cuda \
  --epochs 1 \
  --patience 3 \
  --batch-size 4096 \
  --lr 0.0005 \
  --lr-schedule constant \
  --weight-decay 0.0001 \
  --workers 4 \
  --eval-workers 1 \
  --torch-threads 2 \
  --train-max-samples 5000000 \
  --val-max-samples 500000 \
  --test-max-samples 100000 \
  --calibration-max-batches 25 \
  --calibration-percentile 99.5 \
  --shuffle-block-size 250000 \
  --progress-batches 250 \
  --eval-progress-batches 0 \
  --log-initial-saturation \
  --skip-final-test \
  --output-dir models/output_scale_refine_5m \
  --log-dir logs \
  --tag "$tag" \
  --python "$python_bin"
