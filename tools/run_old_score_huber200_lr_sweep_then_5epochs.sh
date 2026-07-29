#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:-old_score_huber200_lr_sweep_then_5ep_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
SWEEP_ROOT="$MODEL_ROOT/lr_sweep"
CONTINUE_ROOT="$MODEL_ROOT/continuation"
BEST_DIR="$MODEL_ROOT/best"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
RUNNER_LOG="logs/$TAG.runner.log"
REPORT="reports/$TAG.md"
SWEEP_HISTORY="$MODEL_ROOT/lr_sweep.json"
HISTORY="$MODEL_ROOT/history.json"

TRAIN_DATA="data/robotmoon_old_score_train_200m_v1"
EVAL_DATA="data/robotmoon_old_score_val_1m_v1"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-models/quantized_scale_grid/old_score_huber200_resume_8epochs_200m_20260723_073124/best/phase_component_best.pt}"
SWEEP_SAMPLES="${SWEEP_SAMPLES:-50000000}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-200000000}"
SELECTION_SAMPLES="${SELECTION_SAMPLES:-500000}"
RANKING_SAMPLES="${RANKING_SAMPLES:-500000}"

# Constant-LR probes, all resumed from the same epoch-8 model and AdamW state.
LRS=(
  "0.000003"
  "0.000006"
  "0.000012"
  "0.000024"
)

mkdir -p "$MODEL_ROOT" "$SWEEP_ROOT" "$CONTINUE_ROOT" logs reports
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

.venv/bin/python - \
  "$TRAIN_DATA" "$EVAL_DATA" "$SOURCE_CHECKPOINT" \
  "$SWEEP_SAMPLES" "$EPOCH_SAMPLES" <<'PY'
import shutil
import sys
from pathlib import Path

import torch

from nn.compact_board_data import compact_paths

train, validation, checkpoint_path = map(Path, sys.argv[1:4])
sweep_samples, epoch_samples = map(int, sys.argv[4:6])
for path, expected_shards in ((train, 200), (validation, 1)):
    if not path.exists():
        raise SystemExit(f"missing data: {path}")
    actual = len(compact_paths(path))
    if actual != expected_shards:
        raise SystemExit(
            f"unexpected shard count: path={path} "
            f"expected={expected_shards} actual={actual}"
        )
if not (0 < sweep_samples <= 200_000_000):
    raise SystemExit(f"invalid sweep sample count: {sweep_samples}")
if not (0 < epoch_samples <= 200_000_000):
    raise SystemExit(f"invalid epoch sample count: {epoch_samples}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
for key in ("model_state", "optimizer_state"):
    if key not in checkpoint:
        raise SystemExit(f"source checkpoint lacks {key}")
expected = {
    "architecture": "F2",
    "phase_stacks": 8,
    "component_supervision": "total",
    "label_mode": "total",
    "loss_type": "huber",
    "huber_delta": 200.0,
    "hidden_scales": [2, 8],
    "output_scale": 128,
    "activation": "screlu_relu16_all",
}
for key, value in expected.items():
    if checkpoint.get(key) != value:
        raise SystemExit(
            f"source checkpoint metadata mismatch: {key} "
            f"expected={value!r} actual={checkpoint.get(key)!r}"
        )
free = shutil.disk_usage(Path.cwd()).free
if free < 1_500_000_000:
    raise SystemExit(f"insufficient free disk: {free} bytes")
print(
    f"lr_pipeline_preflight=pass free_bytes={free} "
    f"sweep_samples={sweep_samples} epoch_samples={epoch_samples}"
)
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'source_checkpoint=%s\n' "$SOURCE_CHECKPOINT"
  printf 'architecture=F2 shared transformer + 8 phase stacks\n'
  printf 'activation=screlu_relu16_all hidden_clip=181 divisor=128\n'
  printf 'scales=hidden_scale1=2 hidden_scale2=8 output_scale=128\n'
  printf 'objective=Huber200 total score, joint PSQT+positional, weight_decay=0\n'
  printf 'batch_size=8192 psqt_lr_multiplier=2\n'
  printf 'sweep_lrs=%s\n' "${LRS[*]}"
  printf 'sweep_samples_per_branch=%s\n' "$SWEEP_SAMPLES"
  printf 'sweep_seed=20260730 shared across all branches\n'
  printf 'sweep_schedule=constant LR\n'
  printf 'winner_metric=selection objective_loss (Huber200)\n'
  printf 'continuation_epochs=5\n'
  printf 'continuation_samples_per_epoch=%s\n' "$EPOCH_SAMPLES"
  printf 'continuation_lr_decay=0.8 per epoch; cosine to 0.8 of peak within epoch\n'
  printf 'best_checkpoint_metric=selection objective_loss (Huber200)\n'
} > "$MANIFEST"

run_training() {
  local output_dir="$1"
  local log="$2"
  local samples="$3"
  local lr="$4"
  local min_lr="$5"
  local seed="$6"
  local checkpoint="$7"
  local psqt_lr
  psqt_lr="$(awk -v value="$lr" 'BEGIN { printf "%.12f", 2 * value }')"

  if [[ ! -f "$output_dir/summary.json" ]]; then
    PYTHONUNBUFFERED=1 .venv/bin/python tools/train_phase_component_nnue.py \
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
      --train-samples "$samples" \
      --selection-samples "$SELECTION_SAMPLES" \
      --ranking-samples "$RANKING_SAMPLES" \
      --batch-size 8192 \
      --workers 4 \
      --torch-threads 8 \
      --shuffle-block-size 250000 \
      --lr "$lr" \
      --psqt-lr "$psqt_lr" \
      --min-lr "$min_lr" \
      --lr-warmup-steps 0 \
      --weight-decay 0 \
      --resume-checkpoint "$checkpoint" \
      --resume-optimizer \
      --saturation-batches 20 \
      --progress-batches 500 \
      --device cpu \
      --seed "$seed" > "$log" 2>&1
  fi
  test -f "$output_dir/summary.json"
  grep -q '"event":"training_complete"' "$log"
}

for lr in "${LRS[@]}"; do
  lr_name="${lr#0.}"
  output_dir="$SWEEP_ROOT/lr_$lr_name"
  log="logs/${TAG}_sweep_lr_${lr_name}.log"
  printf 'state=sweep_running lr=%s samples=%s updated_at=%s\n' \
    "$lr" "$SWEEP_SAMPLES" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  run_training \
    "$output_dir" "$log" "$SWEEP_SAMPLES" "$lr" "$lr" \
    20260730 "$SOURCE_CHECKPOINT"
  printf 'state=sweep_complete lr=%s updated_at=%s\n' \
    "$lr" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
done

best_line="$(
  .venv/bin/python - \
    "$SWEEP_ROOT" "$SWEEP_HISTORY" "${LRS[@]}" <<'PY'
import json
import sys
from pathlib import Path

sweep_root = Path(sys.argv[1])
history_path = Path(sys.argv[2])
lrs = [float(value) for value in sys.argv[3:]]
records = []
for lr in lrs:
    name = f"{lr:.6f}".split(".", 1)[1]
    output_dir = sweep_root / f"lr_{name}"
    summary = json.loads((output_dir / "summary.json").read_text())
    records.append(
        {
            "lr": lr,
            "checkpoint": str(output_dir / "phase_component_best.pt"),
            "selection_huber": summary["selection"]["objective_loss"],
            "selection_cp_mae": summary["selection"]["cp_mae"],
            "ranking_huber": summary["ranking"]["objective_loss"],
            "ranking_cp_mae": summary["ranking"]["cp_mae"],
            "ranking_wdl": summary["ranking"]["wdl_score_only_loss"],
            "ranking_slope": summary["ranking"]["slope"],
        }
    )
winner = min(records, key=lambda row: (row["selection_huber"], row["lr"]))
state = {
    "winner_metric": "selection objective_loss (Huber200)",
    "winner_lr": winner["lr"],
    "winner_checkpoint": winner["checkpoint"],
    "branches": records,
}
history_path.write_text(json.dumps(state, indent=2) + "\n")
print(f"{winner['lr']:.12f}|{winner['checkpoint']}")
PY
)"
IFS='|' read -r best_lr best_sweep_checkpoint <<< "$best_line"
printf 'state=sweep_winner lr=%s checkpoint=%s updated_at=%s\n' \
  "$best_lr" "$best_sweep_checkpoint" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

mkdir -p "$BEST_DIR"
cp "$best_sweep_checkpoint" "$BEST_DIR/phase_component_best.pt"
cp "$(dirname "$best_sweep_checkpoint")/ranking_errors.json" \
  "$BEST_DIR/ranking_errors.json"

.venv/bin/python - \
  "$SWEEP_HISTORY" "$HISTORY" "$BEST_DIR/phase_component_best.pt" <<'PY'
import json
import sys
from pathlib import Path

sweep_path, history_path, best_checkpoint = map(Path, sys.argv[1:])
sweep = json.loads(sweep_path.read_text())
winner = next(
    row for row in sweep["branches"] if row["lr"] == sweep["winner_lr"]
)
state = {
    "winner_lr": sweep["winner_lr"],
    "winner_metric": sweep["winner_metric"],
    "continuation_epochs": [],
    "best_source": "lr_sweep",
    "best_epoch": 0,
    "best_huber": winner["selection_huber"],
    "best_checkpoint": str(best_checkpoint),
}
history_path.write_text(json.dumps(state, indent=2) + "\n")
PY

resume_checkpoint="$best_sweep_checkpoint"
for epoch in 1 2 3 4 5; do
  epoch_padded="$(printf '%02d' "$epoch")"
  peak_lr="$(
    awk -v base="$best_lr" -v n="$epoch" \
      'BEGIN { printf "%.12f", base * (0.8 ^ (n - 1)) }'
  )"
  min_lr="$(awk -v value="$peak_lr" 'BEGIN { printf "%.12f", 0.8 * value }')"
  seed="$((20260730 + epoch))"
  output_dir="$CONTINUE_ROOT/epoch_$epoch_padded"
  log="logs/${TAG}_epoch_${epoch_padded}.log"

  printf 'state=continuation_running epoch=%s peak_lr=%s min_lr=%s samples=%s seed=%s updated_at=%s\n' \
    "$epoch" "$peak_lr" "$min_lr" "$EPOCH_SAMPLES" "$seed" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  run_training \
    "$output_dir" "$log" "$EPOCH_SAMPLES" "$peak_lr" "$min_lr" \
    "$seed" "$resume_checkpoint"

  .venv/bin/python - \
    "$epoch" "$peak_lr" "$min_lr" "$output_dir" "$HISTORY" "$BEST_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

epoch = int(sys.argv[1])
peak_lr = float(sys.argv[2])
min_lr = float(sys.argv[3])
output_dir = Path(sys.argv[4])
history_path = Path(sys.argv[5])
best_dir = Path(sys.argv[6])
summary = json.loads((output_dir / "summary.json").read_text())
state = json.loads(history_path.read_text())
record = {
    "epoch": epoch,
    "peak_lr": peak_lr,
    "min_lr": min_lr,
    "checkpoint": str(output_dir / "phase_component_best.pt"),
    "selection_huber": summary["selection"]["objective_loss"],
    "selection_cp_mae": summary["selection"]["cp_mae"],
    "ranking_huber": summary["ranking"]["objective_loss"],
    "ranking_cp_mae": summary["ranking"]["cp_mae"],
    "ranking_wdl": summary["ranking"]["wdl_score_only_loss"],
    "ranking_slope": summary["ranking"]["slope"],
}
state["continuation_epochs"].append(record)
if record["selection_huber"] < state["best_huber"]:
    state["best_huber"] = record["selection_huber"]
    state["best_source"] = "continuation"
    state["best_epoch"] = epoch
    state["best_checkpoint"] = str(best_dir / "phase_component_best.pt")
    shutil.copy2(
        output_dir / "phase_component_best.pt",
        best_dir / "phase_component_best.pt",
    )
    shutil.copy2(
        output_dir / "ranking_errors.json",
        best_dir / "ranking_errors.json",
    )
history_path.write_text(json.dumps(state, indent=2) + "\n")
print(json.dumps(record, separators=(",", ":")))
PY

  resume_checkpoint="$output_dir/phase_component_best.pt"
  printf 'state=continuation_complete epoch=%s updated_at=%s\n' \
    "$epoch" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
done

.venv/bin/python - \
  "$SWEEP_HISTORY" "$HISTORY" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

sweep_path, history_path, report_path = map(Path, sys.argv[1:])
sweep = json.loads(sweep_path.read_text())
history = json.loads(history_path.read_text())
lines = [
    "# Huber200 LR sweep and five-epoch continuation",
    "",
    "Winner criterion: **selection objective_loss (Huber200)**.",
    f"Sweep winner LR: **{sweep['winner_lr']:.2e}**.",
    (
        f"Best checkpoint source: **{history['best_source']}**, "
        f"epoch **{history['best_epoch']}**."
    ),
    f"Best selection Huber: **{history['best_huber']:.8f}**.",
    "",
    "## LR sweep",
    "",
    "| LR | Selection Huber | Selection CP MAE | Ranking Huber | Ranking CP MAE | WDL loss | Slope |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in sweep["branches"]:
    lines.append(
        f"| {row['lr']:.2e} | {row['selection_huber']:.8f} | "
        f"{row['selection_cp_mae']:.4f} | {row['ranking_huber']:.8f} | "
        f"{row['ranking_cp_mae']:.4f} | {row['ranking_wdl']:.6f} | "
        f"{row['ranking_slope']:.4f} |"
    )
lines.extend(
    [
        "",
        "## Five-epoch continuation",
        "",
        "| Epoch | Peak LR | Min LR | Selection Huber | Selection CP MAE | Ranking Huber | Ranking CP MAE | WDL loss | Slope |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
)
for row in history["continuation_epochs"]:
    lines.append(
        f"| {row['epoch']} | {row['peak_lr']:.2e} | {row['min_lr']:.2e} | "
        f"{row['selection_huber']:.8f} | {row['selection_cp_mae']:.4f} | "
        f"{row['ranking_huber']:.8f} | {row['ranking_cp_mae']:.4f} | "
        f"{row['ranking_wdl']:.6f} | {row['ranking_slope']:.4f} |"
    )
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines) + "\n")
PY

.venv/bin/python tools/export_phase_quantized_nnue.py \
  --checkpoint "$BEST_DIR/phase_component_best.pt" \
  --output "$BEST_DIR/phase_quantized_nnue.bin" \
  --parity-data "$EVAL_DATA" \
  --parity-output "$BEST_DIR/phase_quantized_nnue_parity.tsv" \
  --parity-samples 4096

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
