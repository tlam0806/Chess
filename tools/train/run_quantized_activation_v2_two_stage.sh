#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/workspace/Chess}"
data="${DATA:-${repo_root}/data/robotmoon_test80_2024_100m_v2_shards_1m}"
base_tag="${BASE_TAG:-quant_actgrid_v2_$(date +%Y%m%d_%H%M%S)}"
jobs="${JOBS:-4}"
train_workers="${TRAIN_WORKERS:-6}"
eval_workers="${EVAL_WORKERS:-2}"
batch_size="${BATCH_SIZE:-8192}"

cd "${repo_root}"
mkdir -p logs models/quantized_scale_grid

echo "BASE_TAG=${base_tag}"
echo "DATA=${data}"
echo "JOBS=${jobs}"
echo "TRAIN_WORKERS=${train_workers}"
echo "EVAL_WORKERS=${eval_workers}"
echo "BATCH_SIZE=${batch_size}"
echo "START=$(date -Is)"

run_grid() {
    local suffix="$1"
    local activation="$2"
    local hidden_scales="$3"
    local output_scales="$4"

    echo "RUN=${suffix} START=$(date -Is)"
    .venv/bin/python tools/train/run_quantized_scale_grid.py \
        --python .venv/bin/python \
        --data "${data}" \
        --data-format cbin \
        --architectures E2 F2 \
        --activations "${activation}" \
        --hidden-scales "${hidden_scales}" \
        --output-scales "${output_scales}" \
        --device cuda \
        --jobs "${jobs}" \
        --epochs 8 \
        --patience 4 \
        --batch-size "${batch_size}" \
        --lr 0.001 \
        --lr-drop-patience 2 \
        --lr-drop-factor 0.5 \
        --min-lr 0.00005 \
        --weight-decay 0.0001 \
        --workers "${train_workers}" \
        --eval-workers "${eval_workers}" \
        --train-max-samples 2000000 \
        --val-max-samples 100000 \
        --test-max-samples 100000 \
        --feature-weight-scale 255 \
        --linear-weight-scale 64 \
        --output-weight-scale 16 \
        --progress-batches 100 \
        --eval-progress-batches 0 \
        --skip-final-test \
        --resume \
        --tag "${base_tag}_${suffix}"
    echo "RUN=${suffix} END=$(date -Is)"
}

run_grid relu relu "64 128 256" "32 64"
run_grid screlu_first screlu_first "64 128 256" "8 16 32"
run_grid screlu_all screlu_all "8 16 32" "8 16 32"

echo "STAGE1_END=$(date -Is)"
.venv/bin/python -m tools.train.run_quantized_activation_stage2 \
    --stage1-tag "${base_tag}" \
    --tag "${base_tag}_stage2_5m" \
    --data "${data}" \
    --data-format cbin \
    --top-k 4 \
    --device cuda \
    --jobs "${jobs}" \
    --epochs 8 \
    --patience 4 \
    --batch-size "${batch_size}" \
    --lr 0.0005 \
    --lr-drop-patience 2 \
    --lr-drop-factor 0.5 \
    --min-lr 0.00005 \
    --weight-decay 0.0001 \
    --workers "${train_workers}" \
    --eval-workers "${eval_workers}" \
    --train-max-samples 5000000 \
    --val-max-samples 500000 \
    --test-max-samples 500000 \
    --feature-weight-scale 255 \
    --linear-weight-scale 64 \
    --output-weight-scale 16 \
    --progress-batches 100 \
    --eval-progress-batches 0

echo "END=$(date -Is)"
