#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:-phase_total_loss_ablation_50m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS="logs/$TAG.status"
REPORT="reports/$TAG.md"
PSQT_DATA="data/stockfish_static_nnue_psqt_component_50m_v1"
POSITIONAL_DATA="data/stockfish_static_nnue_positional_component_50m_v1"
EVAL_PSQT_DATA="data/stockfish_static_nnue_psqt_component_val_1m_v1"
EVAL_POSITIONAL_DATA="data/stockfish_static_nnue_positional_component_val_1m_v1"
HUBER200_DIR="models/quantized_scale_grid/phase_component_causal_controls_50m_20260720_1330/phase8_total"

CONFIGS=(
  "huber600:huber:600"
  "huber1000:huber:1000"
  "mse:mse:200"
)

mkdir -p "$MODEL_ROOT" logs reports
printf 'state=running tag=%s updated_at=%s\n' \
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

for spec in "${CONFIGS[@]}"; do
  IFS=: read -r label loss_type delta <<< "$spec"
  output_dir="$MODEL_ROOT/$label"
  log="logs/${TAG}_${label}.log"
  printf 'config=%s state=running updated_at=%s\n' \
    "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train_phase_component_nnue.py \
    --psqt-data "$PSQT_DATA" \
    --positional-data "$POSITIONAL_DATA" \
    --eval-psqt-data "$EVAL_PSQT_DATA" \
    --eval-positional-data "$EVAL_POSITIONAL_DATA" \
    --output-dir "$output_dir" \
    --hidden-scales 8 16 \
    --output-scale 128 \
    --phase-stacks 8 \
    --objective total \
    --loss-type "$loss_type" \
    --huber-delta "$delta" \
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
    --progress-batches 250 > "$log" 2>&1
  grep -q '"event":"training_complete"' "$log"
  printf 'config=%s state=complete updated_at=%s\n' \
    "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
done

.venv/bin/python - "$MODEL_ROOT" "$HUBER200_DIR" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root, huber200, report = map(Path, sys.argv[1:])
entries = [("huber200", huber200)]
entries.extend((label, root / label) for label in ("huber600", "huber1000", "mse"))
rows = []
for label, directory in entries:
    summary = json.loads((directory / "summary.json").read_text())
    errors = np.asarray(
        json.loads((directory / "ranking_errors.json").read_text()),
        dtype=np.float64,
    )
    rows.append((label, summary["selection"], summary["ranking"], errors))
rows.sort(key=lambda row: row[2]["cp_mae"])
best_errors = rows[0][3]
rng = np.random.default_rng(20260720)

lines = [
    "# Eight-phase total-loss ablation — 50M",
    "",
    "Fixed architecture: F2, SCReLU-ReLU16-ReLU16, hs8x16/os128, 8 phase stacks.",
    "",
    "| Rank | Loss | Selection CP MAE | Ranking CP MAE | Slope | Intercept | Paired delta vs best (95% CI) |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for rank, (label, selection, ranking, errors) in enumerate(rows, 1):
    delta = errors - best_errors
    block_means = delta.reshape(1000, -1).mean(axis=1)
    bootstrap = block_means[
        rng.integers(0, 1000, size=(5000, 1000))
    ].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    lines.append(
        f"| {rank} | `{label}` | {selection['cp_mae']:.4f} | "
        f"{ranking['cp_mae']:.4f} | {ranking['slope']:.4f} | "
        f"{ranking['intercept_cp']:.3f} | {delta.mean():+.4f} "
        f"[{low:+.4f}, {high:+.4f}] |"
    )

lines.extend(["", "## Best loss by phase", "", "| Phase | Samples | CP MAE |", "|---:|---:|---:|"])
for phase in rows[0][2]["phase"]:
    lines.append(f"| {phase['phase']} | {phase['samples']} | {phase['cp_mae']:.4f} |")
report.write_text("\n".join(lines) + "\n")
print(json.dumps({
    "best": rows[0][0],
    "ranking_cp_mae": rows[0][2]["cp_mae"],
    "slope": rows[0][2]["slope"],
    "report": str(report),
}, separators=(",", ":")))
PY

printf 'state=complete report=%s updated_at=%s\n' \
  "$REPORT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
