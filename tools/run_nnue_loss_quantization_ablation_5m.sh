#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
tag="${TAG:-nnue_loss_quant_ablation_fixed5m_$(date +%Y%m%d_%H%M%S)}"
epochs="${EPOCHS:-8}"
jobs=(float_cp_mse float_cp_huber quantized_cp_mse quantized_cp_huber)
mkdir -p logs "models/$tag"

run_branch() {
  local branch="$1"
  local forward_mode="${branch%%_*}"
  local loss_type="${branch#*_}"
  local output_dir="models/$tag/$branch"
  local log="logs/${tag}_${branch}.log"

  "$python_bin" tools/train_quantized_nnue_architecture.py \
    --arch F2 \
    --data data/robotmoon_ablation_fixed5m_unique_v1 \
    --data-format cbin \
    --train-data-all-records \
    --eval-data data/robotmoon_val500k_balanced_target75u25c_v3 \
    --eval-data-format cbin \
    --eval-data-all-records-for-val \
    --forward-mode "$forward_mode" \
    --loss-type "$loss_type" \
    --activation screlu_all \
    --hidden-clip 181 \
    --screlu-divisor 128 \
    --quantization-convention scale_clean \
    --feature-weight-scale 181 \
    --linear-weight-scale 64 \
    --output-weight-scale 16 \
    --fixed-hidden-scales 32 16 \
    --fixed-output-scale 16 \
    --screlu-init-fraction 0.25 \
    --screlu-first-bias-fraction 0.1 \
    --cp-huber-delta 200 \
    --epochs "$epochs" \
    --patience "$((epochs + 1))" \
    --batch-size 4096 \
    --lr 0.0005 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --device cpu \
    --workers 0 \
    --eval-workers 0 \
    --torch-threads 2 \
    --train-max-samples 5000000 \
    --val-max-samples 500000 \
    --calibration-max-batches 25 \
    --shuffle-block-size 250000 \
    --progress-batches 250 \
    --eval-progress-batches 0 \
    --log-initial-saturation \
    --skip-final-test \
    --seed 20260718 \
    --output-dir "$output_dir" \
    >"$log" 2>&1
}

pids=()
for branch in "${jobs[@]}"; do
  run_branch "$branch" &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '%s\n' "completed ${jobs[$index]}"
  else
    printf '%s\n' "failed ${jobs[$index]}"
    status=1
  fi
done

"$python_bin" tools/summarize_nnue_loss_quantization_ablation.py \
  --tag "$tag" \
  --log-dir logs \
  --output "reports/${tag}.md"

exit "$status"
