#!/bin/zsh
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h}"
cd "$repo_root"

train_data="data/stockfish_static_nnue_psqtpos_5m_train_v1"
val_data="data/stockfish_static_nnue_psqtpos_1m_val_v1"
tag="${TAG:-psqt_scale_clean_anchor_smoke_$(date +%Y%m%d_%H%M%S)}"
anchor_set="${ANCHOR_SET:-normalized_cp}"
root_output="models/quantized_scale_grid/$tag"
manifest="logs/$tag.manifest"
status_file="logs/$tag.status"

mkdir -p "$root_output" logs
[[ -d "$train_data" ]] || { print -u2 "missing train data: $train_data"; exit 1; }
[[ -d "$val_data" ]] || { print -u2 "missing validation data: $val_data"; exit 1; }

case "$anchor_set" in
  normalized_cp)
    # The trainer compares forward(...)/target_scale against CP/target_scale.
    # Therefore the clean output divisor is accumulator_scale/target_scale,
    # not the raw accumulator scale itself (target_scale is currently 1000).
    configs=(
      "screlu_all hs91x91_os4 91 91 4"
      "screlu_relu16_all hs16x64_os16 16 64 16"
    )
    ;;
  raw_dequant_control)
    configs=(
      "screlu_all hs91x91_os4096 91 91 4096"
      "screlu_relu16_all hs16x64_os16384 16 64 16384"
    )
    ;;
  *)
    print -u2 "unknown ANCHOR_SET: $anchor_set"
    exit 2
    ;;
esac

{
  print "tag=$tag"
  print "anchor_set=$anchor_set"
  print "train_data=$train_data"
  print "val_data=$val_data"
  print "train_samples=500000"
  print "val_samples=100000"
  print "epochs=2"
  print "seed=20260720"
  print "configs=${(j:,:)configs}"
} > "$manifest"

print "state=running tag=$tag updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"

for spec in "${configs[@]}"; do
  fields=(${=spec})
  activation="$fields[1]"
  name="$fields[2]"
  hs1="$fields[3]"
  hs2="$fields[4]"
  output_scale="$fields[5]"
  output_dir="$root_output/$name"
  log="logs/${tag}_${name}.log"
  mkdir -p "$output_dir"

  print "config=$name state=running updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
  .venv/bin/python tools/train_quantized_nnue_architecture.py \
    --arch F2 \
    --data "$train_data" \
    --data-format cbin \
    --train-data-all-records \
    --eval-data "$val_data" \
    --eval-data-format cbin \
    --eval-data-all-records-for-val \
    --forward-mode quantized \
    --loss-type cp_huber \
    --cp-huber-delta 200 \
    --activation "$activation" \
    --hidden-clip 181 \
    --screlu-divisor 128 \
    --quantization-convention scale_clean \
    --feature-weight-scale 181 \
    --linear-weight-scale 64 \
    --output-weight-scale 16 \
    --psqt \
    --psqt-weight-scale 16 \
    --fixed-hidden-scales "$hs1" "$hs2" \
    --fixed-output-scale "$output_scale" \
    --screlu-init-fraction 0.25 \
    --screlu-first-bias-fraction 0.1 \
    --epochs 2 \
    --patience 3 \
    --batch-size 8192 \
    --lr 0.0005 \
    --lr-schedule cosine \
    --lr-warmup-steps 64 \
    --min-lr 0.00005 \
    --weight-decay 0 \
    --device cpu \
    --workers 4 \
    --eval-workers 2 \
    --torch-threads 4 \
    --train-max-samples 500000 \
    --val-max-samples 100000 \
    --train-probe-max-samples 100000 \
    --calibration-max-batches 20 \
    --shuffle-block-size 250000 \
    --progress-batches 25 \
    --eval-progress-batches 0 \
    --log-initial-saturation \
    --skip-final-test \
    --seed 20260720 \
    --output-dir "$output_dir" > "$log" 2>&1

  print "config=$name state=complete updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
done

print "state=complete tag=$tag updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
