#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:-old_score_loss_ablation_50m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
REPORT="reports/$TAG.md"
TRAIN_DATA="data/robotmoon_old_score_train_50m_v1"
EVAL_DATA="data/robotmoon_old_score_val_1m_v1"

# label:loss_type:huber_delta. The WDL delta is ignored.
CONFIGS=(
  "huber200:huber:200"
  "huber600:huber:600"
  "wdl2p5:wdl:200"
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

.venv/bin/python - "$TRAIN_DATA" "$EVAL_DATA" <<'PY'
import sys
from pathlib import Path

from nn.compact_board_data import compact_paths

train, validation = map(Path, sys.argv[1:])
for path in (train, validation):
    if not path.exists():
        raise SystemExit(f"missing data: {path}")
train_shards = compact_paths(train)
validation_shards = compact_paths(validation)
if not train_shards or not validation_shards:
    raise SystemExit("old-score preflight found no CBin shards")
print(
    f"old_score_preflight=pass train_shards={len(train_shards)} "
    f"validation_shards={len(validation_shards)}"
)
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'labels=original RobotMoon Stockfish score, CBin raw score units\n'
  printf 'architecture=F2 shared transformer + 8 independent phase stacks\n'
  printf 'activation=screlu_relu16_all hidden_scales=8,16 output_scale=128\n'
  printf 'objective=total score, PSQT and positional optimized jointly\n'
  printf 'train_samples_per_config=50000000, same shuffled corpus and seed\n'
  printf 'selection_samples=500000 validation records 0..499999\n'
  printf 'ranking_samples=500000 validation records 500000..999999\n'
  printf 'losses=Huber200 Huber600 Stockfish-WDL-score-only-exp2.5\n'
  printf 'wdl_formula=in_offset270 out_offset270 in_scaling340 out_scaling380\n'
  printf 'cp_metric_target=clamp(raw_score*100/208,-2000,2000)\n'
} > "$MANIFEST"

printf 'state=running tag=%s configs=%s updated_at=%s\n' \
  "$TAG" "${#CONFIGS[@]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

for spec in "${CONFIGS[@]}"; do
  IFS=: read -r label loss_type huber_delta <<< "$spec"
  output_dir="$MODEL_ROOT/$label"
  log="logs/${TAG}_${label}.log"
  printf 'config=%s state=running updated_at=%s\n' \
    "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train_phase_component_nnue.py \
    --total-data "$TRAIN_DATA" \
    --eval-total-data "$EVAL_DATA" \
    --output-dir "$output_dir" \
    --arch F2 \
    --activation screlu_relu16_all \
    --phase-stacks 8 \
    --objective total \
    --loss-type "$loss_type" \
    --huber-delta "$huber_delta" \
    --wdl-exponent 2.5 \
    --wdl-input-offset 270 \
    --wdl-output-offset 270 \
    --wdl-input-scaling 340 \
    --wdl-output-scaling 380 \
    --hidden-scales 8 16 \
    --output-scale 128 \
    --train-samples 50000000 \
    --selection-samples 500000 \
    --ranking-samples 500000 \
    --batch-size 8192 \
    --workers 4 \
    --torch-threads 8 \
    --shuffle-block-size 250000 \
    --lr 0.0005 \
    --psqt-lr 0.001 \
    --min-lr 0.00005 \
    --lr-warmup-steps 610 \
    --weight-decay 0 \
    --saturation-batches 20 \
    --progress-batches 250 \
    --device cpu > "$log" 2>&1
  grep -q '"event":"training_complete"' "$log"
  printf 'config=%s state=complete updated_at=%s\n' \
    "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
done

.venv/bin/python - "$MODEL_ROOT" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
report = Path(sys.argv[2])
rows = []
for label in ("huber200", "huber600", "wdl2p5"):
    directory = root / label
    summary = json.loads((directory / "summary.json").read_text())
    errors = np.asarray(
        json.loads((directory / "ranking_errors.json").read_text()),
        dtype=np.float64,
    )
    rows.append((label, summary, errors))

rows.sort(key=lambda row: row[1]["ranking"]["cp_mae"])
best_label, _, best_errors = rows[0]
rng = np.random.default_rng(20260720)
lines = [
    "# Old-score loss ablation — 50M",
    "",
    "All configs use the same old RobotMoon labels, architecture, initialization seed,",
    "training order, and disjoint 500K selection / 500K paired-ranking samples.",
    "",
    "| Rank | Loss | Selection CP MAE | Ranking CP MAE | Ranking slope | Common WDL 2.5 loss | Paired CP delta vs best (95% block-bootstrap CI) |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for rank, (label, summary, errors) in enumerate(rows, 1):
    selection = summary["selection"]
    ranking = summary["ranking"]
    delta = errors - best_errors
    block_means = delta.reshape(1000, -1).mean(axis=1)
    indices = rng.integers(0, len(block_means), size=(5000, len(block_means)))
    boot = block_means[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    lines.append(
        f"| {rank} | `{label}` | {selection['cp_mae']:.4f} | "
        f"{ranking['cp_mae']:.4f} | {ranking['slope']:.4f} | "
        f"{ranking['wdl_score_only_loss']:.8f} | "
        f"{delta.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] |"
    )

lines.extend([
    "",
    "CP MAE is measured against clamp(raw_score * 100 / 208, -2000, 2000).",
    "WDL loss uses the unclamped original raw score and the official score-only transform.",
])
report.write_text("\n".join(lines) + "\n")
print(
    f"best_cp={best_label} "
    f"ranking_cp_mae={rows[0][1]['ranking']['cp_mae']:.6f}"
)
PY

printf 'state=complete tag=%s report=%s updated_at=%s\n' \
  "$TAG" "$REPORT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

if [[ -n "$CAFFEINATE_PID" ]]; then
  kill "$CAFFEINATE_PID" 2>/dev/null || true
  CAFFEINATE_PID=""
fi
trap - EXIT
