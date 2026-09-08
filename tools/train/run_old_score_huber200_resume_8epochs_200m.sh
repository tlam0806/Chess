#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

TAG="${TAG:-old_score_huber200_resume_8epochs_200m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
BEST_DIR="$MODEL_ROOT/best"
BEST_CP_DIR="$MODEL_ROOT/best_cp_mae"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
REPORT="reports/$TAG.md"
HISTORY="$MODEL_ROOT/history.json"
TRAIN_DATA="data/robotmoon_old_score_train_200m_v1"
EVAL_DATA="data/robotmoon_old_score_val_1m_v1"
SOURCE_DIR="models/quantized_scale_grid/old_score_huber200_top2_200m_20260722_163841/hs2x8_os128"
SOURCE_CHECKPOINT="$SOURCE_DIR/phase_component_best.pt"

# Epoch 1 is the existing bs8192/lr5e-4 200M checkpoint. Each tuple below is:
# epoch:peak_lr:min_lr:warmup_steps:warmup_start_lr:seed
EPOCHS=(
  "2:0.000100:0.000080:500:0.000050:20260723"
  "3:0.000080:0.000060:0:0:20260724"
  "4:0.000060:0.000045:0:0:20260725"
  "5:0.000045:0.000035:0:0:20260726"
  "6:0.000035:0.000025:0:0:20260727"
  "7:0.000025:0.000018:0:0:20260728"
  "8:0.000018:0.000012:0:0:20260729"
)

mkdir -p "$MODEL_ROOT" logs reports
printf 'state=preflight tag=%s updated_at=%s\n' \
  "$TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

CAFFEINATE_PID=""
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
  CAFFEINATE_PID=$!
fi
on_exit() {
  local code=$?
  if [[ -n "$CAFFEINATE_PID" ]]; then
    kill "$CAFFEINATE_PID" 2>/dev/null || true
  fi
  if [[ $code -ne 0 ]]; then
    printf 'state=failed exit_code=%s updated_at=%s\n' \
      "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  fi
}
trap on_exit EXIT

.venv/bin/python - "$TRAIN_DATA" "$EVAL_DATA" "$SOURCE_CHECKPOINT" <<'PY'
import shutil
import sys
from pathlib import Path

import torch

from chess_nnue.compact_board_data import compact_paths

train, validation, checkpoint_path = map(Path, sys.argv[1:])
for path, expected_shards in ((train, 200), (validation, 1)):
    if not path.exists():
        raise SystemExit(f"missing data: {path}")
    actual = len(compact_paths(path))
    if actual != expected_shards:
        raise SystemExit(
            f"unexpected shard count: path={path} "
            f"expected={expected_shards} actual={actual}"
        )
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if "model_state" not in checkpoint or "optimizer_state" not in checkpoint:
    raise SystemExit("source checkpoint lacks model or optimizer state")
free = shutil.disk_usage(Path.cwd()).free
if free < 1_000_000_000:
    raise SystemExit(f"insufficient free disk for eight epochs: {free} bytes")
print(f"resume_8epoch_preflight=pass free_bytes={free}")
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'source_checkpoint=%s\n' "$SOURCE_CHECKPOINT"
  printf 'epochs=8 total; epoch1 source plus epochs2..8 resumed\n'
  printf 'samples_per_epoch=200000000 unique corpus, shuffled with a new seed\n'
  printf 'total_samples_after_completion=1600000000\n'
  printf 'resume=model_state+AdamW_optimizer_state\n'
  printf 'architecture=F2 shared transformer + 8 independent phase stacks\n'
  printf 'activation=screlu_relu16_all hidden_clip=181 divisor=128\n'
  printf 'scales=hidden_scale1=2 hidden_scale2=8 output_scale=128\n'
  printf 'objective=Huber200 total score, joint PSQT+positional, weight_decay=0\n'
  printf 'batch_size=8192 psqt_lr_multiplier=2\n'
  printf 'epoch_schedule=%s\n' "${EPOCHS[*]}"
  printf 'primary_best_metric=selection objective_loss (Huber200)\n'
  printf 'secondary_best_metric=selection CP MAE\n'
  printf 'no_early_stop=true\n'
} > "$MANIFEST"

printf 'state=evaluating_source epoch=1 updated_at=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

epoch1_dir="$MODEL_ROOT/epoch_01"
epoch1_log="logs/${TAG}_epoch_01.log"
if [[ ! -f "$epoch1_dir/summary.json" ]]; then
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_phase_component_nnue.py \
    --total-data "$TRAIN_DATA" \
    --eval-total-data "$EVAL_DATA" \
    --output-dir "$epoch1_dir" \
    --arch F2 \
    --activation screlu_relu16_all \
    --phase-stacks 8 \
    --objective total \
    --loss-type huber \
    --huber-delta 200 \
    --hidden-scales 2 8 \
    --output-scale 128 \
    --train-samples 8192 \
    --selection-samples 500000 \
    --ranking-samples 500000 \
    --batch-size 8192 \
    --workers 0 \
    --torch-threads 8 \
    --shuffle-block-size 8192 \
    --lr 0.0001 \
    --psqt-lr 0.0002 \
    --min-lr 0.00008 \
    --lr-warmup-steps 0 \
    --weight-decay 0 \
    --resume-checkpoint "$SOURCE_CHECKPOINT" \
    --resume-optimizer \
    --evaluate-only \
    --saturation-batches 20 \
    --device cpu \
    --seed 20260720 > "$epoch1_log" 2>&1
fi
grep -q '"event":"training_complete"' "$epoch1_log"

.venv/bin/python - \
  "$epoch1_dir" "$BEST_DIR" "$BEST_CP_DIR" "$HISTORY" <<'PY'
import json
import shutil
import sys
from pathlib import Path

epoch_dir, best_dir, best_cp_dir, history_path = map(Path, sys.argv[1:])
summary = json.loads((epoch_dir / "summary.json").read_text())
record = {
    "epoch": 1,
    "checkpoint": str(epoch_dir / "phase_component_best.pt"),
    "selection_huber": summary["selection"]["objective_loss"],
    "selection_cp_mae": summary["selection"]["cp_mae"],
    "ranking_huber": summary["ranking"]["objective_loss"],
    "ranking_cp_mae": summary["ranking"]["cp_mae"],
    "ranking_wdl": summary["ranking"]["wdl_score_only_loss"],
    "ranking_slope": summary["ranking"]["slope"],
}
state = {
    "epochs": [record],
    "best_huber_epoch": 1,
    "best_huber": record["selection_huber"],
    "best_cp_epoch": 1,
    "best_cp_mae": record["selection_cp_mae"],
}
for destination in (best_dir, best_cp_dir):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(epoch_dir / "phase_component_best.pt", destination / "phase_component_best.pt")
    shutil.copy2(epoch_dir / "ranking_errors.json", destination / "ranking_errors.json")
history_path.write_text(json.dumps(state, indent=2) + "\n")
print(json.dumps(state, separators=(",", ":")))
PY

resume_checkpoint="$epoch1_dir/phase_component_best.pt"
for spec in "${EPOCHS[@]}"; do
  IFS=: read -r epoch lr min_lr warmup warmup_start seed <<< "$spec"
  epoch_padded="$(printf '%02d' "$epoch")"
  output_dir="$MODEL_ROOT/epoch_$epoch_padded"
  log="logs/${TAG}_epoch_${epoch_padded}.log"
  psqt_lr="$(awk -v value="$lr" 'BEGIN { printf "%.9f", 2 * value }')"

  printf 'state=running epoch=%s lr=%s min_lr=%s seed=%s updated_at=%s\n' \
    "$epoch" "$lr" "$min_lr" "$seed" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

  if [[ ! -f "$output_dir/summary.json" ]]; then
    PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_phase_component_nnue.py \
      --total-data "$TRAIN_DATA" \
      --eval-total-data "$EVAL_DATA" \
      --output-dir "$output_dir" \
      --arch F2 \
      --activation screlu_relu16_all \
      --phase-stacks 8 \
      --objective total \
      --loss-type huber \
      --huber-delta 200 \
      --hidden-scales 2 8 \
      --output-scale 128 \
      --train-samples 200000000 \
      --selection-samples 500000 \
      --ranking-samples 500000 \
      --batch-size 8192 \
      --workers 4 \
      --torch-threads 8 \
      --shuffle-block-size 250000 \
      --lr "$lr" \
      --psqt-lr "$psqt_lr" \
      --min-lr "$min_lr" \
      --lr-warmup-steps "$warmup" \
      --warmup-start-lr "$warmup_start" \
      --weight-decay 0 \
      --resume-checkpoint "$resume_checkpoint" \
      --resume-optimizer \
      --saturation-batches 20 \
      --progress-batches 500 \
      --device cpu \
      --seed "$seed" > "$log" 2>&1
  fi
  grep -q '"event":"training_complete"' "$log"

  .venv/bin/python - \
    "$epoch" "$output_dir" "$BEST_DIR" "$BEST_CP_DIR" "$HISTORY" <<'PY'
import json
import shutil
import sys
from pathlib import Path

epoch = int(sys.argv[1])
epoch_dir, best_dir, best_cp_dir, history_path = map(Path, sys.argv[2:])
summary = json.loads((epoch_dir / "summary.json").read_text())
state = json.loads(history_path.read_text())
record = {
    "epoch": epoch,
    "checkpoint": str(epoch_dir / "phase_component_best.pt"),
    "selection_huber": summary["selection"]["objective_loss"],
    "selection_cp_mae": summary["selection"]["cp_mae"],
    "ranking_huber": summary["ranking"]["objective_loss"],
    "ranking_cp_mae": summary["ranking"]["cp_mae"],
    "ranking_wdl": summary["ranking"]["wdl_score_only_loss"],
    "ranking_slope": summary["ranking"]["slope"],
}
state["epochs"].append(record)
if record["selection_huber"] < state["best_huber"]:
    state["best_huber"] = record["selection_huber"]
    state["best_huber_epoch"] = epoch
    shutil.copy2(epoch_dir / "phase_component_best.pt", best_dir / "phase_component_best.pt")
    shutil.copy2(epoch_dir / "ranking_errors.json", best_dir / "ranking_errors.json")
if record["selection_cp_mae"] < state["best_cp_mae"]:
    state["best_cp_mae"] = record["selection_cp_mae"]
    state["best_cp_epoch"] = epoch
    shutil.copy2(epoch_dir / "phase_component_best.pt", best_cp_dir / "phase_component_best.pt")
    shutil.copy2(epoch_dir / "ranking_errors.json", best_cp_dir / "ranking_errors.json")
history_path.write_text(json.dumps(state, indent=2) + "\n")
print(json.dumps(state, separators=(",", ":")))
PY

  resume_checkpoint="$output_dir/phase_component_best.pt"
  printf 'state=epoch_complete epoch=%s updated_at=%s\n' \
    "$epoch" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
done

.venv/bin/python - "$HISTORY" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

history_path, report = map(Path, sys.argv[1:])
state = json.loads(history_path.read_text())
lines = [
    "# Huber200 hs2x8/os128 - eight epochs",
    "",
    f"Best Huber epoch: **{state['best_huber_epoch']}**.",
    f"Best CP-MAE epoch: **{state['best_cp_epoch']}**.",
    "",
    "| Epoch | Selection Huber | Selection CP MAE | Ranking Huber | Ranking CP MAE | WDL loss | Slope |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in state["epochs"]:
    lines.append(
        f"| {row['epoch']} | {row['selection_huber']:.8f} | "
        f"{row['selection_cp_mae']:.4f} | {row['ranking_huber']:.8f} | "
        f"{row['ranking_cp_mae']:.4f} | {row['ranking_wdl']:.6f} | "
        f"{row['ranking_slope']:.4f} |"
    )
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("\n".join(lines) + "\n")
PY

.venv/bin/python tools/train/export_phase_quantized_nnue.py \
  --checkpoint "$BEST_DIR/phase_component_best.pt" \
  --output "$BEST_DIR/phase_quantized_nnue.bin" \
  --parity-data "$EVAL_DATA" \
  --parity-output "$BEST_DIR/phase_quantized_nnue_parity.tsv" \
  --parity-samples 2048

if [[ -x build-release/phase_quantized_nnue_tests ]]; then
  build-release/phase_quantized_nnue_tests \
    "$BEST_DIR/phase_quantized_nnue.bin" \
    "$BEST_DIR/phase_quantized_nnue_parity.tsv"
fi

printf 'state=complete tag=%s best=%s report=%s updated_at=%s\n' \
  "$TAG" "$BEST_DIR/phase_component_best.pt" "$REPORT" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

if [[ -n "$CAFFEINATE_PID" ]]; then
  kill "$CAFFEINATE_PID" 2>/dev/null || true
  CAFFEINATE_PID=""
fi
trap - EXIT
