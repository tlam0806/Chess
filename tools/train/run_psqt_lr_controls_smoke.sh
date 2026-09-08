#!/bin/zsh
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h:h}"
cd "$repo_root"

train_data="data/stockfish_static_nnue_psqtpos_5m_train_v1"
val_data="data/stockfish_static_nnue_psqtpos_1m_val_v1"
tag="${TAG:-psqt_lr_controls_smoke_$(date +%Y%m%d_%H%M%S)}"
root_output="models/quantized_scale_grid/$tag"
manifest="logs/$tag.manifest"
status_file="logs/$tag.status"

mkdir -p "$root_output" logs
[[ -d "$train_data" ]] || { print -u2 "missing train data: $train_data"; exit 1; }
[[ -d "$val_data" ]] || { print -u2 "missing validation data: $val_data"; exit 1; }

configs=(
  "no_psqt no 0"
  "psqt_lr5e4 yes 0.0005"
  "psqt_lr1e3 yes 0.001"
)

{
  print "tag=$tag"
  print "architecture=F2"
  print "activation=screlu_relu16_all"
  print "hidden_scales=16 64"
  print "output_scale=16"
  print "train_samples=500000"
  print "val_samples=100000"
  print "epochs=2"
  print "base_lr=0.0005"
  print "seed=20260720"
  print "configs=${(j:,:)configs}"
} > "$manifest"

print "state=running tag=$tag updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"

for spec in "${configs[@]}"; do
  fields=(${=spec})
  name="$fields[1]"
  use_psqt="$fields[2]"
  psqt_lr="$fields[3]"
  output_dir="$root_output/$name"
  log="logs/${tag}_${name}.log"
  mkdir -p "$output_dir"

  if [[ "$use_psqt" == "yes" ]]; then
    psqt_args=(--psqt --psqt-weight-scale 16 --psqt-lr "$psqt_lr")
  else
    psqt_args=(--no-psqt)
  fi

  print "config=$name state=running updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
  .venv/bin/python tools/train/train_quantized_nnue_architecture.py \
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
    --activation screlu_relu16_all \
    --hidden-clip 181 \
    --screlu-divisor 128 \
    --quantization-convention scale_clean \
    --feature-weight-scale 181 \
    --linear-weight-scale 64 \
    --output-weight-scale 16 \
    "${psqt_args[@]}" \
    --fixed-hidden-scales 16 64 \
    --fixed-output-scale 16 \
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
