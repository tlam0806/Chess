#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${TAG:-phase_component_grid_50m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
REPORT="reports/$TAG.md"
LABEL_STATUS="logs/stockfish_components_50m_v1.status"
PSQT_DATA="data/stockfish_static_nnue_psqt_component_50m_v1"
POSITIONAL_DATA="data/stockfish_static_nnue_positional_component_50m_v1"
EVAL_PSQT_DATA="data/stockfish_static_nnue_psqt_component_val_1m_v1"
EVAL_POSITIONAL_DATA="data/stockfish_static_nnue_positional_component_val_1m_v1"

# Twelve points around the previous winner hs8x16/os128. This keeps the run
# near six hours at the measured 50M/config throughput while independently
# varying both hidden scales and output scale.
CONFIGS=(
  "hs4x8_os64:4:8:64"
  "hs4x8_os128:4:8:128"
  "hs4x16_os64:4:16:64"
  "hs4x16_os128:4:16:128"
  "hs8x8_os64:8:8:64"
  "hs8x8_os128:8:8:128"
  "hs8x16_os64:8:16:64"
  "hs8x16_os128:8:16:128"
  "hs16x8_os64:16:8:64"
  "hs16x8_os128:16:8:128"
  "hs16x16_os64:16:16:64"
  "hs16x16_os128:16:16:128"
)

mkdir -p "$MODEL_ROOT" logs reports
printf 'state=waiting_for_component_labels tag=%s updated_at=%s\n' \
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
    printf 'state=failed tag=%s exit_code=%s updated_at=%s\n' \
      "$TAG" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  fi
}
trap on_exit EXIT

while ! grep -q '^state=complete' "$LABEL_STATUS" 2>/dev/null; do
  if grep -q '^state=failed' "$LABEL_STATUS" 2>/dev/null; then
    echo "component labeling failed" >&2
    exit 1
  fi
  sleep 30
done

.venv/bin/python - "$PSQT_DATA" "$POSITIONAL_DATA" "$EVAL_PSQT_DATA" \
  "$EVAL_POSITIONAL_DATA" <<'PY'
import sys
from pathlib import Path

train_psqt, train_positional, eval_psqt, eval_positional = map(Path, sys.argv[1:])
for left, right, expected in (
    (train_psqt, train_positional, 50),
    (eval_psqt, eval_positional, 1),
):
    left_names = sorted(path.name for path in left.glob("*.cbin.zst"))
    right_names = sorted(path.name for path in right.glob("*.cbin.zst"))
    if left_names != right_names or len(left_names) != expected:
        raise SystemExit(
            f"aligned component preflight failed: {left} {right} "
            f"counts={len(left_names)},{len(right_names)}"
        )
print("component_preflight=pass train_shards=50 eval_shards=1")
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'architecture=F2 shared_transformer + 8 independent 32x32x1 phase stacks\n'
  printf 'phase_formula=clamp((piece_count - 1) // 4, 0, 7)\n'
  printf 'component_objective=Huber200(PSQT) + Huber200(positional)\n'
  printf 'train_samples_per_config=50000000 unique random positions, same 50 shards for every config\n'
  printf 'selection_samples=500000 validation records 0..499999\n'
  printf 'ranking_samples=500000 validation records 500000..999999\n'
  printf 'activation=screlu_relu16_all hidden_clip=181 divisor=128\n'
  printf 'grid=%s\n' "${CONFIGS[*]}"
} > "$MANIFEST"

# A short end-to-end run catches label alignment, routing, gradients, checkpoint
# serialization, and validation before any 50M config is allowed to start.
SMOKE_DIR="$MODEL_ROOT/smoke"
PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_phase_component_nnue.py \
  --psqt-data "$PSQT_DATA" \
  --positional-data "$POSITIONAL_DATA" \
  --eval-psqt-data "$EVAL_PSQT_DATA" \
  --eval-positional-data "$EVAL_POSITIONAL_DATA" \
  --output-dir "$SMOKE_DIR" \
  --hidden-scales 8 16 \
  --output-scale 128 \
  --train-samples 32768 \
  --selection-samples 8192 \
  --ranking-samples 8192 \
  --batch-size 2048 \
  --workers 0 \
  --torch-threads 8 \
  --lr-warmup-steps 4 \
  --saturation-batches 2 \
  --progress-batches 0 > "logs/${TAG}_smoke.log" 2>&1
grep -q '"event":"training_complete"' "logs/${TAG}_smoke.log" \
  || { echo "phase/component smoke test failed" >&2; exit 1; }

printf 'state=running tag=%s configs=%s updated_at=%s\n' \
  "$TAG" "${#CONFIGS[@]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

for spec in "${CONFIGS[@]}"; do
  IFS=: read -r label hs1 hs2 output_scale <<< "$spec"
  output_dir="$MODEL_ROOT/$label"
  log="logs/${TAG}_${label}.log"
  printf 'config=%s state=running updated_at=%s\n' \
    "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_phase_component_nnue.py \
    --psqt-data "$PSQT_DATA" \
    --positional-data "$POSITIONAL_DATA" \
    --eval-psqt-data "$EVAL_PSQT_DATA" \
    --eval-positional-data "$EVAL_POSITIONAL_DATA" \
    --output-dir "$output_dir" \
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
    --huber-delta 200 \
    --saturation-batches 20 \
    --progress-batches 250 > "$log" 2>&1
  grep -q '"event":"training_complete"' "$log" \
    || { echo "training failed: $label" >&2; exit 1; }
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
for summary_path in sorted(root.glob("*/summary.json")):
    label = summary_path.parent.name
    if label == "smoke":
        continue
    summary = json.loads(summary_path.read_text())
    errors = np.asarray(
        json.loads((summary_path.parent / "ranking_errors.json").read_text()),
        dtype=np.float64,
    )
    ranking = summary["ranking"]
    rows.append((label, ranking, errors, summary["saturation"]))
if len(rows) != 12:
    raise SystemExit(f"expected 12 completed configs, got {len(rows)}")
rows.sort(key=lambda row: row[1]["cp_mae"])
best_label, _best_metrics, best_errors, _best_saturation = rows[0]
rng = np.random.default_rng(20260720)

lines = [
    "# Eight-phase component-supervised 50M grid",
    "",
    f"Best ranking config: `{best_label}`.",
    "",
    "| Rank | Config | Selection CP MAE | Ranking CP MAE | Slope | Paired delta vs best (95% block-bootstrap CI) |",
    "|---:|---|---:|---:|---:|---:|",
]
for rank, (label, metrics, errors, saturation) in enumerate(rows, 1):
    delta = errors - best_errors
    # Preserve locality with 1000 contiguous 500-position blocks, then
    # bootstrap block means. This is paired on exactly the same positions.
    block_means = delta.reshape(1000, -1).mean(axis=1)
    samples = block_means[rng.integers(0, len(block_means), size=(5000, len(block_means)))]
    boot = samples.mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    selection = json.loads((root / label / "summary.json").read_text())["selection"]
    lines.append(
        f"| {rank} | `{label}` | {selection['cp_mae']:.4f} | "
        f"{metrics['cp_mae']:.4f} | {metrics['slope']:.4f} | "
        f"{delta.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] |"
    )

lines.extend(["", "## Best config by phase", "", "| Phase | Samples | CP MAE |", "|---:|---:|---:|"])
for phase in rows[0][1]["phase"]:
    mae = "n/a" if phase["cp_mae"] is None else f"{phase['cp_mae']:.4f}"
    lines.append(f"| {phase['phase']} | {phase['samples']} | {mae} |")

lines.extend(["", "## Best config saturation", "", "| Phase | Layer | Clip rate | Zero rate |", "|---:|---:|---:|---:|"])
for phase in rows[0][3]["phase"]:
    for layer in phase["layers"]:
        lines.append(
            f"| {phase['phase']} | {layer['layer']} | "
            f"{100 * layer['clip_rate']:.4f}% | {100 * layer['zero_rate']:.4f}% |"
        )
report.write_text("\n".join(lines) + "\n")
print(f"best={best_label} ranking_cp_mae={rows[0][1]['cp_mae']:.6f}")
PY

printf 'state=complete tag=%s report=%s updated_at=%s\n' \
  "$TAG" "$REPORT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

if [[ -n "$CAFFEINATE_PID" ]]; then
  kill "$CAFFEINATE_PID" 2>/dev/null || true
  CAFFEINATE_PID=""
fi
trap - EXIT
pmset sleepnow
