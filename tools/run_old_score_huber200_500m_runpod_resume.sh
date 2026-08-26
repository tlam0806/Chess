#!/usr/bin/env bash
set -euo pipefail

# Resume the from-scratch 500M run after the locally completed epoch 1.
# The RunPod network volume is mounted at /workspace, so checkpoints and logs
# survive Pod termination.

workspace="${WORKSPACE:-/workspace}"
repo_root="${REPO_ROOT:-$workspace/Chess}"
tag="${TAG:-old_score_huber200_500m_until_overfit_20260822_203621}"
data_dir="$repo_root/data/robotmoon_test80_2024_500m_unique_v2_shards_1m"
data_manifest="$repo_root/data/robotmoon_test80_2024_500m_unique_v2.manifest.json"
output_dir="$repo_root/models/quantized_scale_grid/$tag"
checkpoint="$output_dir/quant_nnue_arch_F2_current.pt"
log="$repo_root/logs/$tag.runpod.log"
status_file="$repo_root/logs/$tag.runpod.status"

cd "$repo_root"
mkdir -p "$output_dir" logs

if ! command -v zstd >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends zstd
fi

if [[ ! -d "$data_dir" || ! -f "$data_manifest" ]]; then
  echo "500M corpus is missing under $repo_root/data" >&2
  exit 1
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "epoch-1 resume checkpoint is missing: $checkpoint" >&2
  exit 1
fi

python3 - "$checkpoint" <<'PY'
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if int(payload.get("epoch", -1)) != 1:
    raise SystemExit(f"expected epoch-1 checkpoint, got epoch={payload.get('epoch')}")
print(
    "runpod_preflight=pass "
    f"gpu={torch.cuda.get_device_name(0)!r} "
    f"cuda={torch.version.cuda!r} "
    f"checkpoint_epoch={payload['epoch']}"
)
PY

cat > "$status_file" <<EOF
state=running tag=$tag host=$(hostname) pid=$$ updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

on_exit() {
  code=$?
  if [[ $code -ne 0 ]]; then
    echo "state=failed tag=$tag exit_code=$code updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
  fi
}
trap on_exit EXIT

# Keep batch size and per-epoch step count identical to epoch 1 so moving from
# CPU to CUDA changes throughput only, not the optimizer or LR trajectory.
PYTHONUNBUFFERED=1 python3 tools/train_quantized_nnue_architecture.py \
  --arch F2 \
  --data "$data_dir" \
  --data-format cbin \
  --output-dir "$output_dir" \
  --resume-checkpoint "$checkpoint" \
  --forward-mode quantized \
  --loss-type cp_huber \
  --cp-huber-delta 200 \
  --activation screlu_relu16_all \
  --hidden-clip 181 \
  --screlu-divisor 128 \
  --quantization-convention scale_clean \
  --feature-weight-scale 181 \
  --linear-weight-scale 64 \
  --output-weight-scale 16 \
  --psqt \
  --psqt-weight-scale 16 \
  --fixed-hidden-scales 2 8 \
  --fixed-output-scale 128 \
  --screlu-init-fraction 0.25 \
  --screlu-first-bias-fraction 0.1 \
  --log-initial-saturation \
  --epochs 12 \
  --patience 2 \
  --min-delta-loss 0.000001 \
  --batch-size 8192 \
  --lr 0.0005 \
  --psqt-lr 0.001 \
  --lr-schedule cosine \
  --lr-warmup-steps 2000 \
  --epoch-peak-lrs \
    0.000500 0.000100 0.000060 0.000040 \
    0.000025 0.0000175 0.00001225 0.000008575 \
    0.000006 0.000005 0.000005 0.000005 \
  --epoch-min-lrs \
    0.000050 0.000060 0.000040 0.000025 \
    0.0000175 0.00001225 0.000008575 0.000006 \
    0.000005 0.000005 0.000005 0.000005 \
  --lr-steps-per-epoch 59815 \
  --min-lr 0.000005 \
  --weight-decay 0 \
  --device cuda \
  --workers 10 \
  --eval-workers 10 \
  --torch-threads 2 \
  --train-max-samples 500000000 \
  --val-max-samples 5000000 \
  --test-max-samples 5000000 \
  --train-probe-max-samples 1000000 \
  --calibration-max-batches 50 \
  --shuffle-block-size 250000 \
  --progress-batches 500 \
  --eval-progress-batches 100 \
  --seed 20260822 >> "$log" 2>&1

echo "state=complete tag=$tag updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
trap - EXIT
