#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

TAG="${TAG:-old_score_huber200_low_scale_grid_5m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
REPORT="reports/$TAG.md"
TRAIN_DATA="data/robotmoon_old_score_train_50m_v1"
EVAL_DATA="data/robotmoon_old_score_val_1m_v1"

HS1_VALUES=(1 2 4 8)
HS2_VALUES=(1 2 4 8 16)
OUTPUT_SCALE_VALUES=(8 16 32 64 128)
CONFIG_COUNT=$((${#HS1_VALUES[@]} * ${#HS2_VALUES[@]} * ${#OUTPUT_SCALE_VALUES[@]}))

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

from chess_nnue.compact_board_data import compact_paths

train, validation = map(Path, sys.argv[1:])
for path, expected_shards in ((train, 50), (validation, 1)):
    if not path.exists():
        raise SystemExit(f"missing data: {path}")
    shards = compact_paths(path)
    if len(shards) != expected_shards:
        raise SystemExit(
            f"unexpected shard count: path={path} "
            f"expected={expected_shards} actual={len(shards)}"
        )
print("low_scale_grid_preflight=pass train_shards=50 validation_shards=1")
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'labels=original RobotMoon Stockfish score, CBin raw score units\n'
  printf 'architecture=F2 shared transformer + 8 independent phase stacks\n'
  printf 'activation=screlu_relu16_all hidden_clip=181 divisor=128\n'
  printf 'objective=total score, PSQT and positional optimized jointly\n'
  printf 'loss=Huber200\n'
  printf 'train_samples_per_config=5000000 unique random positions, same corpus order and seed\n'
  printf 'selection_samples=500000 validation records 0..499999\n'
  printf 'ranking_samples=500000 validation records 500000..999999\n'
  printf 'hs1_values=%s\n' "${HS1_VALUES[*]}"
  printf 'hs2_values=%s\n' "${HS2_VALUES[*]}"
  printf 'output_scale_values=%s\n' "${OUTPUT_SCALE_VALUES[*]}"
  printf 'config_count=%s\n' "$CONFIG_COUNT"
  printf 'baseline=hs8x16_os128\n'
  printf 'seed=20260720\n'
  printf 'cp_metric_target=clamp(raw_score*100/208,-2000,2000)\n'
} > "$MANIFEST"

# Exercise the lowest-scale corners and the existing baseline end to end before
# allowing the long grid to start.
SMOKE_CONFIGS=(
  "hs1x1_os8:1:1:8"
  "hs1x16_os128:1:16:128"
  "hs8x1_os8:8:1:8"
  "hs8x16_os128:8:16:128"
)
for spec in "${SMOKE_CONFIGS[@]}"; do
  IFS=: read -r label hs1 hs2 output_scale <<< "$spec"
  smoke_dir="$MODEL_ROOT/smoke_$label"
  smoke_log="logs/${TAG}_smoke_${label}.log"
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_phase_component_nnue.py \
    --total-data "$TRAIN_DATA" \
    --eval-total-data "$EVAL_DATA" \
    --output-dir "$smoke_dir" \
    --arch F2 \
    --activation screlu_relu16_all \
    --phase-stacks 8 \
    --objective total \
    --loss-type huber \
    --huber-delta 200 \
    --hidden-scales "$hs1" "$hs2" \
    --output-scale "$output_scale" \
    --train-samples 32768 \
    --selection-samples 8192 \
    --ranking-samples 8192 \
    --batch-size 2048 \
    --workers 0 \
    --torch-threads 8 \
    --shuffle-block-size 32768 \
    --lr 0.0005 \
    --psqt-lr 0.001 \
    --min-lr 0.00005 \
    --lr-warmup-steps 4 \
    --weight-decay 0 \
    --saturation-batches 2 \
    --progress-batches 0 \
    --device cpu > "$smoke_log" 2>&1
  grep -q '"event":"training_complete"' "$smoke_log"
done

.venv/bin/python - "$MODEL_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = sorted(root.glob("smoke_*/summary.json"))
if len(summaries) != 4:
    raise SystemExit(f"expected 4 smoke summaries, found {len(summaries)}")
for path in summaries:
    summary = json.loads(path.read_text())
    if summary["train"]["samples"] != 32768:
        raise SystemExit(f"wrong smoke sample count: {path}")
    for split in ("selection", "ranking"):
        for metric in ("cp_mae", "slope"):
            if not math.isfinite(summary[split][metric]):
                raise SystemExit(f"non-finite {split}.{metric}: {path}")
print("low_scale_grid_smoke=pass configs=4")
PY

printf 'state=running tag=%s configs=%s updated_at=%s\n' \
  "$TAG" "$CONFIG_COUNT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

for hs1 in "${HS1_VALUES[@]}"; do
  for hs2 in "${HS2_VALUES[@]}"; do
    for output_scale in "${OUTPUT_SCALE_VALUES[@]}"; do
      label="hs${hs1}x${hs2}_os${output_scale}"
      output_dir="$MODEL_ROOT/$label"
      log="logs/${TAG}_${label}.log"
      if [[ -f "$output_dir/summary.json" ]] \
        && grep -q '"event": "training_complete"' "$output_dir/summary.json"; then
        printf 'config=%s state=already_complete updated_at=%s\n' \
          "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
        continue
      fi
      if [[ -d "$output_dir" ]]; then
        mv "$output_dir" "${output_dir}.incomplete.$(date +%Y%m%d_%H%M%S)"
      fi
      printf 'config=%s state=running updated_at=%s\n' \
        "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
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
        --hidden-scales "$hs1" "$hs2" \
        --output-scale "$output_scale" \
        --train-samples 5000000 \
        --selection-samples 500000 \
        --ranking-samples 500000 \
        --batch-size 8192 \
        --workers 4 \
        --torch-threads 8 \
        --shuffle-block-size 250000 \
        --lr 0.0005 \
        --psqt-lr 0.001 \
        --min-lr 0.00005 \
        --lr-warmup-steps 61 \
        --weight-decay 0 \
        --saturation-batches 20 \
        --progress-batches 100 \
        --device cpu > "$log" 2>&1
      grep -q '"event":"training_complete"' "$log"
      printf 'config=%s state=complete updated_at=%s\n' \
        "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
    done
  done
done

.venv/bin/python - "$MODEL_ROOT" "$REPORT" "$CONFIG_COUNT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
report = Path(sys.argv[2])
expected = int(sys.argv[3])
rows = []
for summary_path in sorted(root.glob("hs*/summary.json")):
    label = summary_path.parent.name
    summary = json.loads(summary_path.read_text())
    rows.append((label, summary))
if len(rows) != expected:
    raise SystemExit(f"expected {expected} completed configs, got {len(rows)}")
rows.sort(key=lambda row: row[1]["ranking"]["cp_mae"])
best_label, best_summary = rows[0]
best_errors = np.asarray(
    json.loads((root / best_label / "ranking_errors.json").read_text()),
    dtype=np.float64,
)
rng = np.random.default_rng(20260720)

lines = [
    "# Old-score Huber200 low-scale grid - 5M",
    "",
    f"Best ranking config: `{best_label}`.",
    "",
    "All 100 configs use the same 5M training positions, order, seed, and disjoint validation splits.",
    "",
    "| Rank | Config | Selection CP MAE | Ranking CP MAE | Slope | Paired delta vs best (95% block-bootstrap CI) |",
    "|---:|---|---:|---:|---:|---:|",
]
for rank, (label, summary) in enumerate(rows, 1):
    errors = np.asarray(
        json.loads((root / label / "ranking_errors.json").read_text()),
        dtype=np.float64,
    )
    delta = errors - best_errors
    block_means = delta.reshape(1000, -1).mean(axis=1)
    indices = rng.integers(0, len(block_means), size=(2000, len(block_means)))
    boot = block_means[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    lines.append(
        f"| {rank} | `{label}` | {summary['selection']['cp_mae']:.4f} | "
        f"{summary['ranking']['cp_mae']:.4f} | {summary['ranking']['slope']:.4f} | "
        f"{delta.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] |"
    )

lines.extend([
    "",
    "## Best config by phase",
    "",
    "| Phase | Samples | CP MAE |",
    "|---:|---:|---:|",
])
for phase in best_summary["ranking"]["phase"]:
    mae = "n/a" if phase["cp_mae"] is None else f"{phase['cp_mae']:.4f}"
    lines.append(f"| {phase['phase']} | {phase['samples']} | {mae} |")

lines.extend([
    "",
    "## Best config saturation",
    "",
    "| Phase | Layer | Clip rate | Zero rate |",
    "|---:|---:|---:|---:|",
])
for phase in best_summary["saturation"]["phase"]:
    for layer in phase["layers"]:
        lines.append(
            f"| {phase['phase']} | {layer['layer']} | "
            f"{100 * layer['clip_rate']:.4f}% | {100 * layer['zero_rate']:.4f}% |"
        )
report.write_text("\n".join(lines) + "\n")
print(f"best={best_label} ranking_cp_mae={best_summary['ranking']['cp_mae']:.6f}")
PY

printf 'state=complete tag=%s report=%s updated_at=%s\n' \
  "$TAG" "$REPORT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

if [[ -n "$CAFFEINATE_PID" ]]; then
  kill "$CAFFEINATE_PID" 2>/dev/null || true
  CAFFEINATE_PID=""
fi
trap - EXIT
