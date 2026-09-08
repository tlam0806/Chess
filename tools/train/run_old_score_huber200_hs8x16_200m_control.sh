#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

TAG="${TAG:-old_score_huber200_hs8x16_200m_control_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="models/quantized_scale_grid/$TAG/hs8x16_os128"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
LOG="logs/${TAG}_hs8x16_os128.log"
TRAIN_DATA="data/robotmoon_old_score_train_200m_v1"
EVAL_DATA="data/robotmoon_old_score_val_1m_v1"
SOURCE_MANIFEST="data/robotmoon_balanced_cp_200m_sf_static_eligible_v1.manifest.json"

mkdir -p "$OUTPUT_DIR" logs

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
    shards = compact_paths(path)
    if len(shards) != expected_shards:
        raise SystemExit(
            f"unexpected shard count: path={path} "
            f"expected={expected_shards} actual={len(shards)}"
        )
manifest = json.loads(manifest_path.read_text())
output = manifest["output"]
if output["records"] != 200_000_000 or output["shards"] != 200:
    raise SystemExit(f"bad source manifest: {output}")
for bucket in manifest["selection"]["bins"]:
    if bucket["records"] != bucket["unique_source_records"]:
        raise SystemExit(f"non-unique source bucket: {bucket}")
    if bucket["repeated_records"] != 0:
        raise SystemExit(f"repeated source records: {bucket}")
free = shutil.disk_usage(Path.cwd()).free
if free < 250_000_000:
    raise SystemExit(f"insufficient free disk: {free}")
print(
    "hs8x16_200m_preflight=pass train_shards=200 validation_shards=1 "
    f"unique_records={output['records']} free_bytes={free}"
)
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'control_for=old_score_huber200_top2_200m_20260722_163841\n'
  printf 'data_manifest=%s\n' "$SOURCE_MANIFEST"
  printf 'labels=original RobotMoon Stockfish score, CBin raw score units\n'
  printf 'architecture=F2 shared transformer + 8 independent phase stacks\n'
  printf 'activation=screlu_relu16_all hidden_clip=181 divisor=128\n'
  printf 'objective=total score, PSQT and positional optimized jointly\n'
  printf 'loss=Huber200\n'
  printf 'config=hs8x16_os128:8:16:128\n'
  printf 'train_samples=200000000 unique positions, one pass\n'
  printf 'selection_samples=500000 validation records 0..499999\n'
  printf 'ranking_samples=500000 validation records 500000..999999\n'
  printf 'seed=20260720\n'
  printf 'training=from_scratch\n'
  printf 'batch_size=8192 workers=4 torch_threads=8\n'
  printf 'lr=0.0005 min_lr=0.00005 warmup_steps=2441 total_steps=24415\n'
  printf 'psqt_lr=0.001 weight_decay=0 shuffle_block_size=250000\n'
  printf 'cp_metric_target=clamp(raw_score*100/208,-2000,2000)\n'
} > "$MANIFEST"

printf 'state=running tag=%s config=hs8x16_os128 updated_at=%s\n' \
  "$TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

PYTHONUNBUFFERED=1 .venv/bin/python tools/train/train_phase_component_nnue.py \
  --total-data "$TRAIN_DATA" \
  --eval-total-data "$EVAL_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --arch F2 \
  --activation screlu_relu16_all \
  --phase-stacks 8 \
  --objective total \
  --loss-type huber \
  --huber-delta 200 \
  --hidden-scales 8 16 \
  --output-scale 128 \
  --train-samples 200000000 \
  --selection-samples 500000 \
  --ranking-samples 500000 \
  --batch-size 8192 \
  --workers 4 \
  --torch-threads 8 \
  --shuffle-block-size 250000 \
  --lr 0.0005 \
  --psqt-lr 0.001 \
  --min-lr 0.00005 \
  --lr-warmup-steps 2441 \
  --weight-decay 0 \
  --saturation-batches 20 \
  --progress-batches 500 \
  --device cpu > "$LOG" 2>&1

grep -q '"event":"training_complete"' "$LOG"
printf 'state=complete tag=%s config=hs8x16_os128 output=%s updated_at=%s\n' \
  "$TAG" "$OUTPUT_DIR" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
