#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

TAG="${TAG:-old_score_huber200_batch_ablation_200m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
REPORT="reports/$TAG.md"
TRAIN_DATA="data/robotmoon_old_score_train_200m_v1"
EVAL_DATA="data/robotmoon_old_score_val_1m_v1"
SOURCE_MANIFEST="data/robotmoon_balanced_cp_200m_sf_static_eligible_v1.manifest.json"

# Same architecture/scales/objective/data; only batch-size-dependent optimizer
# settings differ. Learning rates use a conservative between-linear-and-sqrt
# scale-down from the established bs8192/lr5e-4 run.
CONFIGS=(
  "bs4096_lr3e4:4096:0.0003:0.0006:0.00003:4883"
  "bs2048_lr2e4:2048:0.0002:0.0004:0.00002:9766"
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

.venv/bin/python - "$TRAIN_DATA" "$EVAL_DATA" "$SOURCE_MANIFEST" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from chess_nnue.compact_board_data import compact_paths

train, validation, manifest_path = map(Path, sys.argv[1:])
for path, expected_shards in ((train, 200), (validation, 1)):
    if not path.exists():
        raise SystemExit(f"missing data: {path}")
    shards = compact_paths(path)
    if len(shards) != expected_shards:
        raise SystemExit(
            f"unexpected shard count: path={path} "
            f"expected={expected_shards} actual={len(shards)}"
        )
manifest = json.loads(manifest_path.read_text())
output = manifest["output"]
if output["records"] != 200_000_000 or output["shards"] != 200:
    raise SystemExit(f"bad 200M source manifest: {output}")
for bucket in manifest["selection"]["bins"]:
    if bucket["records"] != bucket["unique_source_records"]:
        raise SystemExit(f"non-unique source bucket: {bucket}")
    if bucket["repeated_records"] != 0:
        raise SystemExit(f"repeated source records: {bucket}")
free = shutil.disk_usage(Path.cwd()).free
if free < 500_000_000:
    raise SystemExit(f"insufficient free disk for two checkpoints: {free} bytes")
print(
    f"batch_ablation_preflight=pass train_shards=200 validation_shards=1 "
    f"unique_records={output['records']} free_bytes={free}"
)
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'baseline_checkpoint=from_scratch\n'
  printf 'data_manifest=%s\n' "$SOURCE_MANIFEST"
  printf 'labels=original RobotMoon Stockfish score, CBin raw score units\n'
  printf 'architecture=F2 shared transformer + 8 independent phase stacks\n'
  printf 'activation=screlu_relu16_all hidden_clip=181 divisor=128\n'
  printf 'scales=hidden_scale1=2 hidden_scale2=8 output_scale=128\n'
  printf 'objective=total score, PSQT and positional optimized jointly\n'
  printf 'loss=Huber200 weight_decay=0\n'
  printf 'train_samples_per_config=200000000 unique positions, one pass, same corpus order and seed\n'
  printf 'selection_samples=500000 validation records 0..499999\n'
  printf 'ranking_samples=500000 validation records 500000..999999\n'
  printf 'configs=%s\n' "${CONFIGS[*]}"
  printf 'seed=20260720\n'
  printf 'schedule=linear warmup then cosine decay\n'
  printf 'cp_metric_target=clamp(raw_score*100/208,-2000,2000)\n'
} > "$MANIFEST"

printf 'state=running tag=%s configs=%s updated_at=%s\n' \
  "$TAG" "${#CONFIGS[@]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

for spec in "${CONFIGS[@]}"; do
  IFS=: read -r label batch_size lr psqt_lr min_lr warmup_steps <<< "$spec"
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
  printf 'config=%s state=running batch_size=%s lr=%s updated_at=%s\n' \
    "$label" "$batch_size" "$lr" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
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
    --batch-size "$batch_size" \
    --workers 4 \
    --torch-threads 8 \
    --shuffle-block-size 250000 \
    --lr "$lr" \
    --psqt-lr "$psqt_lr" \
    --min-lr "$min_lr" \
    --lr-warmup-steps "$warmup_steps" \
    --weight-decay 0 \
    --saturation-batches 20 \
    --progress-batches 500 \
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

root, report = map(Path, sys.argv[1:])
rows = []
for summary_path in sorted(root.glob("bs*/summary.json")):
    summary = json.loads(summary_path.read_text())
    errors = np.asarray(
        json.loads((summary_path.parent / "ranking_errors.json").read_text()),
        dtype=np.float64,
    )
    rows.append((summary_path.parent.name, summary, errors))
if len(rows) != 2:
    raise SystemExit(f"expected 2 completed configs, got {len(rows)}")
rows.sort(key=lambda row: row[1]["ranking"]["cp_mae"])
best_label, best_summary, best_errors = rows[0]
rng = np.random.default_rng(20260720)
lines = [
    "# Old-score Huber200 batch-size ablation - 200M",
    "",
    f"Best ranking config: `{best_label}`.",
    "",
    "Both configs were trained from scratch for one pass over the same 200M unique positions.",
    "",
    "| Rank | Config | Selection CP MAE | Ranking CP MAE | WDL loss | Slope | Paired delta vs best (95% block-bootstrap CI) |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for rank, (label, summary, errors) in enumerate(rows, 1):
    delta = errors - best_errors
    block_means = delta.reshape(1000, -1).mean(axis=1)
    sampled = rng.integers(0, len(block_means), size=(5000, len(block_means)))
    boot = block_means[sampled].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    ranking = summary["ranking"]
    lines.append(
        f"| {rank} | `{label}` | {summary['selection']['cp_mae']:.4f} | "
        f"{ranking['cp_mae']:.4f} | {ranking['wdl_score_only_loss']:.6f} | "
        f"{ranking['slope']:.4f} | {delta.mean():+.4f} [{low:+.4f}, {high:+.4f}] |"
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
