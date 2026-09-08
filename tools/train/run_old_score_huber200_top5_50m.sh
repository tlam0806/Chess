#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

TAG="${TAG:-old_score_huber200_top5_50m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
REPORT="reports/$TAG.md"
TRAIN_DATA="data/robotmoon_old_score_train_50m_v1"
EVAL_DATA="data/robotmoon_old_score_val_1m_v1"

# Top five from old_score_huber200_low_scale_grid_5m_20260721_175501.
CONFIGS=(
  "hs2x8_os128:2:8:128"
  "hs1x16_os128:1:16:128"
  "hs4x8_os64:4:8:64"
  "hs2x16_os64:2:16:64"
  "hs4x4_os128:4:4:128"
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
import shutil
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
free = shutil.disk_usage(Path.cwd()).free
if free < 700_000_000:
    raise SystemExit(f"insufficient free disk for five checkpoints: {free} bytes")
print(
    f"top5_50m_preflight=pass train_shards=50 validation_shards=1 "
    f"free_bytes={free}"
)
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'source_grid=old_score_huber200_low_scale_grid_5m_20260721_175501\n'
  printf 'labels=original RobotMoon Stockfish score, CBin raw score units\n'
  printf 'architecture=F2 shared transformer + 8 independent phase stacks\n'
  printf 'activation=screlu_relu16_all hidden_clip=181 divisor=128\n'
  printf 'objective=total score, PSQT and positional optimized jointly\n'
  printf 'loss=Huber200\n'
  printf 'train_samples_per_config=50000000 unique random positions, same full corpus and seed\n'
  printf 'selection_samples=500000 validation records 0..499999\n'
  printf 'ranking_samples=500000 validation records 500000..999999\n'
  printf 'configs=%s\n' "${CONFIGS[*]}"
  printf 'seed=20260720\n'
  printf 'training=from_scratch\n'
  printf 'cp_metric_target=clamp(raw_score*100/208,-2000,2000)\n'
} > "$MANIFEST"

printf 'state=running tag=%s configs=%s updated_at=%s\n' \
  "$TAG" "${#CONFIGS[@]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

for spec in "${CONFIGS[@]}"; do
  IFS=: read -r label hs1 hs2 output_scale <<< "$spec"
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
for summary_path in sorted(root.glob("hs*/summary.json")):
    summary = json.loads(summary_path.read_text())
    rows.append((summary_path.parent.name, summary))
if len(rows) != 5:
    raise SystemExit(f"expected 5 completed configs, got {len(rows)}")
rows.sort(key=lambda row: row[1]["ranking"]["cp_mae"])
best_label, best_summary = rows[0]
best_errors = np.asarray(
    json.loads((root / best_label / "ranking_errors.json").read_text()),
    dtype=np.float64,
)
rng = np.random.default_rng(20260720)

lines = [
    "# Old-score Huber200 top-5 - 50M",
    "",
    f"Best ranking config: `{best_label}`.",
    "",
    "All configs were trained from scratch on the same 50M unique positions, order, and seed.",
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
    indices = rng.integers(0, len(block_means), size=(5000, len(block_means)))
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
