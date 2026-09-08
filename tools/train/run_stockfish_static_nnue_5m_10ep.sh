#!/bin/zsh
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h:h}"
cd "$repo_root"

source_dir="data/stockfish_static_nnue_psqtpos_200m_balanced_v1_shards_1m"
train_dir="data/stockfish_static_nnue_psqtpos_5m_train_v1"
val_dir="data/stockfish_static_nnue_psqtpos_1m_val_v1"
tag="${TAG:-stockfish_static_psqtpos_c181d128_hs32x16_os16_5m_10ep_$(date +%Y%m%d_%H%M%S)}"
output_dir="models/$tag"
log="logs/$tag.log"
status_file="logs/$tag.status"

mkdir -p "$train_dir" "$val_dir" "$output_dir" logs

link_shard() {
  local index="$1" destination_dir="$2"
  local name
  name="part_$(printf '%05d' "$index").cbin.zst"
  local source="$source_dir/$name"
  local destination="$destination_dir/$name"
  [[ -f "$source" ]] || { print -u2 "missing source shard: $source"; exit 1; }
  if [[ -L "$destination" ]]; then
    /bin/unlink "$destination"
  fi
  if [[ -e "$destination" ]]; then
    [[ "$source" -ef "$destination" ]] \
      || { print -u2 "refusing to replace existing path: $destination"; exit 1; }
  else
    /bin/ln "$source" "$destination"
  fi
  /opt/homebrew/bin/zstd -q -t "$destination"
}

for index in 0 1 2 3 4; do
  link_shard "$index" "$train_dir"
done
link_shard 5 "$val_dir"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  print "subset_preflight=pass train_shards=5 validation_shards=1"
  exit 0
fi

print "state=running tag=$tag updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"

if .venv/bin/python tools/train/train_quantized_nnue_architecture.py \
    --arch F2 \
    --data "$train_dir" \
    --data-format cbin \
    --train-data-all-records \
    --eval-data "$val_dir" \
    --eval-data-format cbin \
    --eval-data-all-records-for-val \
    --forward-mode quantized \
    --loss-type cp_huber \
    --cp-huber-delta 200 \
    --activation screlu_all \
    --hidden-clip 181 \
    --screlu-divisor 128 \
    --quantization-convention scale_clean \
    --feature-weight-scale 181 \
    --linear-weight-scale 64 \
    --output-weight-scale 16 \
    --psqt \
    --psqt-weight-scale 16 \
    --fixed-hidden-scales 32 16 \
    --fixed-output-scale 16 \
    --screlu-init-fraction 0.25 \
    --screlu-first-bias-fraction 0.1 \
    --epochs 10 \
    --patience 11 \
    --batch-size 8192 \
    --lr 0.0005 \
    --lr-schedule cosine \
    --lr-warmup-steps 256 \
    --min-lr 0.00005 \
    --weight-decay 0 \
    --device cpu \
    --workers 4 \
    --eval-workers 2 \
    --torch-threads 4 \
    --train-max-samples 5000000 \
    --val-max-samples 500000 \
    --train-probe-max-samples 500000 \
    --calibration-max-batches 50 \
    --shuffle-block-size 250000 \
    --progress-batches 100 \
    --eval-progress-batches 0 \
    --log-initial-saturation \
    --skip-final-test \
    --seed 20260720 \
    --output-dir "$output_dir" >> "$log" 2>&1; then
  print "state=complete tag=$tag updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
else
  code=$?
  print "state=failed tag=$tag exit_code=$code updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
  exit "$code"
fi
