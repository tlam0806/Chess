#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:-phase_component_causal_controls_50m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS="logs/$TAG.status"
REPORT="reports/$TAG.md"
PSQT_DATA="data/stockfish_static_nnue_psqt_component_50m_v1"
POSITIONAL_DATA="data/stockfish_static_nnue_positional_component_50m_v1"
EVAL_PSQT_DATA="data/stockfish_static_nnue_psqt_component_val_1m_v1"
EVAL_POSITIONAL_DATA="data/stockfish_static_nnue_positional_component_val_1m_v1"
OLD_LOG="logs/psqt_relu16_top7_pending_50m_1ep_20260720_034334_relu16_hs8x16_os128.log"
COMBINED_SUMMARY="models/quantized_scale_grid/phase_component_grid_50m_20260720_072627/hs8x16_os128/summary.json"

CONTROLS=(
  "phase8_total:8:total"
  "phase1_separate:1:separate"
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

for spec in "${CONTROLS[@]}"; do
  IFS=: read -r label phases objective <<< "$spec"
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
    --phase-stacks "$phases" \
    --objective "$objective" \
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
  grep -q '"event":"training_complete"' "$log"
  printf 'config=%s state=complete updated_at=%s\n' \
    "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
done

.venv/bin/python - "$OLD_LOG" "$MODEL_ROOT" "$COMBINED_SUMMARY" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

old_log, root, combined_path, report = map(Path, sys.argv[1:])
baseline = None
for line in old_log.read_text().splitlines():
    if '"event":"epoch"' in line:
        event = json.loads(line)
        baseline = float(event["val_cp"])
if baseline is None:
    raise SystemExit("old shared-total baseline metric not found")

phase_total = json.loads((root / "phase8_total" / "summary.json").read_text())
shared_separate = json.loads((root / "phase1_separate" / "summary.json").read_text())
combined = json.loads(combined_path.read_text())
b = float(phase_total["selection"]["cp_mae"])
c = float(shared_separate["selection"]["cp_mae"])
d = float(combined["selection"]["cp_mae"])

phase_effect_total = b - baseline
separate_effect_shared = c - baseline
phase_effect_separate = d - c
separate_effect_phase = d - b
interaction = d - b - c + baseline

lines = [
    "# Phase/component causal controls — 50M",
    "",
    "All four cells use F2, SCReLU-ReLU16-ReLU16, hs8x16/os128 and the same 500K selection records.",
    "",
    "| Dense routing | Training objective | Selection CP MAE |",
    "|---|---|---:|",
    f"| Shared stack | Total score | {baseline:.4f} |",
    f"| 8 phase stacks | Total score | {b:.4f} |",
    f"| Shared stack | Separate PSQT + positional | {c:.4f} |",
    f"| 8 phase stacks | Separate PSQT + positional | {d:.4f} |",
    "",
    "## Effects",
    "",
    f"- Phase split under total loss: {phase_effect_total:+.4f} CP.",
    f"- Separate supervision under shared stack: {separate_effect_shared:+.4f} CP.",
    f"- Phase split under separate supervision: {phase_effect_separate:+.4f} CP.",
    f"- Separate supervision under phase stacks: {separate_effect_phase:+.4f} CP.",
    f"- Interaction: {interaction:+.4f} CP.",
]
report.write_text("\n".join(lines) + "\n")
print(json.dumps({
    "baseline_shared_total": baseline,
    "phase8_total": b,
    "phase1_separate": c,
    "phase8_separate": d,
    "phase_effect_total": phase_effect_total,
    "separate_effect_shared": separate_effect_shared,
    "interaction": interaction,
    "report": str(report),
}, separators=(",", ":")))
PY

printf 'state=complete report=%s updated_at=%s\n' \
  "$REPORT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
