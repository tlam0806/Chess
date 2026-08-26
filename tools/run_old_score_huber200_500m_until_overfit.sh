#!/bin/zsh
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h}"
cd "$repo_root"

data_dir="data/robotmoon_test80_2024_500m_unique_v2_shards_1m"
data_manifest="data/robotmoon_test80_2024_500m_unique_v2.manifest.json"
tag="${TAG:-old_score_huber200_500m_until_overfit_$(date +%Y%m%d_%H%M%S)}"
output_dir="models/quantized_scale_grid/$tag"
log="logs/$tag.log"
status_file="logs/$tag.status"
run_manifest="logs/$tag.manifest"
report="reports/$tag.md"

epochs=12
batch_size=8192
# The CRC32 split retains 98% of 500M positions. The exact count differs from
# 490M by only sampling noise, so this reaches the cosine endpoint to one batch.
lr_steps_per_epoch=59815
epoch_peak_lrs=(
  0.000500 0.000100 0.000060 0.000040
  0.000025 0.0000175 0.00001225 0.000008575
  0.000006 0.000005 0.000005 0.000005
)
epoch_min_lrs=(
  0.000050 0.000060 0.000040 0.000025
  0.0000175 0.00001225 0.000008575 0.000006
  0.000005 0.000005 0.000005 0.000005
)

mkdir -p "$output_dir" logs reports
print "state=preflight tag=$tag updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"

caffeinate_pid=""
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
  caffeinate_pid=$!
fi
on_exit() {
  local code=$?
  if [[ -n "$caffeinate_pid" ]]; then
    kill "$caffeinate_pid" 2>/dev/null || true
  fi
  if [[ $code -ne 0 ]]; then
    print "state=failed tag=$tag exit_code=$code updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
  fi
}
trap on_exit EXIT

.venv/bin/python - "$data_dir" "$data_manifest" "$output_dir" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from nn.compact_board_data import compact_paths, validate_compact_dataset

data_dir, manifest_path, output_dir = map(Path, sys.argv[1:])
paths = validate_compact_dataset(data_dir)
if len(paths) != 500:
    raise SystemExit(f"expected 500 data shards, found {len(paths)}")
manifest = json.loads(manifest_path.read_text())
output = manifest["output"]
validation = manifest["validation"]
if output["records"] != 500_000_000 or output["shards"] != 500:
    raise SystemExit(f"unexpected corpus manifest output: {output}")
if validation["independent_128bit_fingerprint_duplicates"] != 0:
    raise SystemExit("500M corpus is not unique")
if manifest["format"]["target_encoding"] != "stockfish_raw_score_ply_result":
    raise SystemExit("unexpected target encoding")
if any(output_dir.iterdir()):
    raise SystemExit(f"output directory is not empty: {output_dir}")
free = shutil.disk_usage(Path.cwd()).free
if free < 2_000_000_000:
    raise SystemExit(f"insufficient free disk: {free} bytes")
print(
    "500m_preflight=pass "
    f"records={output['records']} shards={len(paths)} free_bytes={free}"
)
PY

{
  print "tag=$tag"
  print "training=from_scratch_model_and_optimizer"
  print "data=$data_dir"
  print "data_manifest=$data_manifest"
  print "labels=original RobotMoon Stockfish search score, CBin raw score units"
  print "split=crc32(board+aux) train=98% validation=1% sealed_test=1%"
  print "architecture=F2 shared transformer + 8 phase stacks + PSQT"
  print "activation=screlu_relu16_all hidden_clip=181 divisor=128"
  print "quantization=scale_clean feature_scale=181 linear_scale=64 output_weight_scale=16"
  print "fixed_scales=hidden[2,8] output=128 psqt_weight_scale=16"
  print "bias_initialization=hidden_fraction_0.25 first_fraction_0.1"
  print "objective=CP Huber delta=200 all target magnitudes"
  print "batch_size=$batch_size weight_decay=0 psqt_lr_multiplier=2"
  print "epochs_max=$epochs patience=2 min_delta_loss=0.000001"
  print "train_limit=500000000 (all records admitted by the 98% train split)"
  print "validation_limit=5000000 sealed_test_limit=5000000 train_probe=1000000"
  print "lr_schedule=independent cosine per epoch; warmup 2000 steps in epoch 1 only"
  print "lr_steps_per_epoch=$lr_steps_per_epoch"
  print "epoch_peak_lrs=${epoch_peak_lrs[*]}"
  print "epoch_min_lrs=${epoch_min_lrs[*]}"
  print "seed=20260822"
  print "promotion=quantized parity then paired self-play against current V40 production"
} > "$run_manifest"

print "state=running tag=$tag pid=$$ updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"

PYTHONUNBUFFERED=1 .venv/bin/python tools/train_quantized_nnue_architecture.py \
  --arch F2 \
  --data "$data_dir" \
  --data-format cbin \
  --output-dir "$output_dir" \
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
  --epochs "$epochs" \
  --patience 2 \
  --min-delta-loss 0.000001 \
  --batch-size "$batch_size" \
  --lr 0.0005 \
  --psqt-lr 0.001 \
  --lr-schedule cosine \
  --lr-warmup-steps 2000 \
  --epoch-peak-lrs "${epoch_peak_lrs[@]}" \
  --epoch-min-lrs "${epoch_min_lrs[@]}" \
  --lr-steps-per-epoch "$lr_steps_per_epoch" \
  --min-lr 0.000005 \
  --weight-decay 0 \
  --device cpu \
  --workers 4 \
  --eval-workers 2 \
  --torch-threads 4 \
  --train-max-samples 500000000 \
  --val-max-samples 5000000 \
  --test-max-samples 5000000 \
  --train-probe-max-samples 1000000 \
  --calibration-max-batches 50 \
  --shuffle-block-size 250000 \
  --progress-batches 500 \
  --eval-progress-batches 100 \
  --seed 20260822 >> "$log" 2>&1

.venv/bin/python - "$log" "$report" <<'PY'
import json
import sys
from pathlib import Path

log_path, report_path = map(Path, sys.argv[1:])
events = []
for line in log_path.read_text().splitlines():
    if not line.startswith("{"):
        continue
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass
epochs = [event for event in events if event.get("event") == "epoch"]
complete = next(
    (event for event in reversed(events) if event.get("event") == "training_complete"),
    None,
)
final_test = next(
    (event for event in reversed(events) if event.get("event") == "final_test"),
    None,
)
if complete is None or final_test is None:
    raise SystemExit("training log lacks completion or sealed-test event")
lines = [
    "# 500M from-scratch NNUE training",
    "",
    f"Selected epoch: `{complete['selected_epoch']}`.",
    "",
    "| Epoch | Train loss | Validation loss | Validation CP MAE | Train-probe CP MAE | LR | Best |",
    "|---:|---:|---:|---:|---:|---:|:---:|",
]
for event in epochs:
    lines.append(
        f"| {event['epoch']} | {event['train_loss']:.8f} | "
        f"{event['val_loss']:.8f} | {event['val_cp']:.4f} | "
        f"{event['train_probe_cp']:.4f} | {event['lr']:.8g} | "
        f"{'yes' if event['improved'] else 'no'} |"
    )
lines.extend([
    "",
    "## Sealed test",
    "",
    f"- Samples: {final_test['test_samples']}",
    f"- CP Huber loss: {final_test['test_loss']:.8f}",
    f"- CP MAE: {final_test['test_val_cp']:.4f}",
])
report_path.write_text("\n".join(lines) + "\n")
PY

print "state=complete tag=$tag report=$report updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"

if [[ -n "$caffeinate_pid" ]]; then
  kill "$caffeinate_pid" 2>/dev/null || true
  caffeinate_pid=""
fi
trap - EXIT
