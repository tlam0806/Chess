#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

TAG="${TAG:-old_score_huber200_resume_epoch2_200m_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="models/quantized_scale_grid/$TAG"
OUTPUT_DIR="$MODEL_ROOT/continued"
BEST_DIR="$MODEL_ROOT/best"
STATUS="logs/$TAG.status"
MANIFEST="logs/$TAG.manifest"
LOG="logs/$TAG.log"
REPORT="reports/$TAG.md"
TRAIN_DATA="data/robotmoon_old_score_train_200m_v1"
EVAL_DATA="data/robotmoon_old_score_val_1m_v1"
SOURCE_DIR="models/quantized_scale_grid/old_score_huber200_top2_200m_20260722_163841/hs2x8_os128"
SOURCE_CHECKPOINT="$SOURCE_DIR/phase_component_best.pt"

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
if checkpoint["metrics"]["ranking"]["cp_mae"] <= 0:
    raise SystemExit("source checkpoint has invalid ranking CP MAE")
free = shutil.disk_usage(Path.cwd()).free
if free < 500_000_000:
    raise SystemExit(f"insufficient free disk: {free} bytes")
print(
    f"resume_preflight=pass source_cp_mae="
    f"{checkpoint['metrics']['ranking']['cp_mae']:.6f} free_bytes={free}"
)
PY

{
  printf 'tag=%s\n' "$TAG"
  printf 'source_checkpoint=%s\n' "$SOURCE_CHECKPOINT"
  printf 'resume=model_state+AdamW_optimizer_state\n'
  printf 'architecture=F2 shared transformer + 8 independent phase stacks\n'
  printf 'activation=screlu_relu16_all hidden_clip=181 divisor=128\n'
  printf 'scales=hidden_scale1=2 hidden_scale2=8 output_scale=128\n'
  printf 'objective=total score, PSQT and positional optimized jointly\n'
  printf 'loss=Huber200 weight_decay=0\n'
  printf 'batch_size=8192\n'
  printf 'learning_rate=warmup 0.00002 to 0.0001 over 1000 steps, cosine to 0.00001\n'
  printf 'psqt_learning_rate=2x base LR\n'
  printf 'train_samples=200000000, second shuffled pass\n'
  printf 'seed=20260723\n'
  printf 'best_selection=lower ranking CP MAE between source and continued checkpoint\n'
} > "$MANIFEST"

printf 'state=running tag=%s updated_at=%s\n' \
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
  --hidden-scales 2 8 \
  --output-scale 128 \
  --train-samples 200000000 \
  --selection-samples 500000 \
  --ranking-samples 500000 \
  --batch-size 8192 \
  --workers 4 \
  --torch-threads 8 \
  --shuffle-block-size 250000 \
  --lr 0.0001 \
  --psqt-lr 0.0002 \
  --min-lr 0.00001 \
  --lr-warmup-steps 1000 \
  --warmup-start-lr 0.00002 \
  --weight-decay 0 \
  --resume-checkpoint "$SOURCE_CHECKPOINT" \
  --resume-optimizer \
  --saturation-batches 20 \
  --progress-batches 500 \
  --device cpu \
  --seed 20260723 > "$LOG" 2>&1

grep -q '"event":"training_complete"' "$LOG"
printf 'state=selecting_best tag=%s updated_at=%s\n' \
  "$TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

.venv/bin/python - \
  "$SOURCE_DIR" "$OUTPUT_DIR" "$BEST_DIR" "$REPORT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source_dir, continued_dir, best_dir, report = map(Path, sys.argv[1:])
source_checkpoint = source_dir / "phase_component_best.pt"
continued_checkpoint = continued_dir / "phase_component_best.pt"

import torch

source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
continued = torch.load(continued_checkpoint, map_location="cpu", weights_only=False)
source_cp = float(source["metrics"]["ranking"]["cp_mae"])
continued_cp = float(continued["metrics"]["ranking"]["cp_mae"])
if continued_cp < source_cp:
    winner = "continued_epoch2"
    winner_dir = continued_dir
    winner_checkpoint = continued_checkpoint
    winner_payload = continued
else:
    winner = "source_epoch1"
    winner_dir = source_dir
    winner_checkpoint = source_checkpoint
    winner_payload = source

best_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(winner_checkpoint, best_dir / "phase_component_best.pt")
shutil.copy2(winner_dir / "ranking_errors.json", best_dir / "ranking_errors.json")
selection = {
    "winner": winner,
    "source_checkpoint": str(source_checkpoint),
    "continued_checkpoint": str(continued_checkpoint),
    "source_ranking_cp_mae": source_cp,
    "continued_ranking_cp_mae": continued_cp,
    "best_checkpoint": str(best_dir / "phase_component_best.pt"),
    "best_ranking_cp_mae": float(winner_payload["metrics"]["ranking"]["cp_mae"]),
}
(best_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(
    "# Huber200 hs2x8/os128 resumed second pass\n\n"
    "| Checkpoint | Ranking CP MAE | WDL loss | Slope |\n"
    "|---|---:|---:|---:|\n"
    f"| Source epoch 1 | {source_cp:.4f} | "
    f"{source['metrics']['ranking']['wdl_score_only_loss']:.6f} | "
    f"{source['metrics']['ranking']['slope']:.4f} |\n"
    f"| Continued epoch 2 | {continued_cp:.4f} | "
    f"{continued['metrics']['ranking']['wdl_score_only_loss']:.6f} | "
    f"{continued['metrics']['ranking']['slope']:.4f} |\n\n"
    f"Selected best: `{winner}`.\n"
)
print(json.dumps(selection, separators=(",", ":")))
PY

printf 'state=complete tag=%s best=%s updated_at=%s\n' \
  "$TAG" "$BEST_DIR/phase_component_best.pt" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"

if [[ -n "$CAFFEINATE_PID" ]]; then
  kill "$CAFFEINATE_PID" 2>/dev/null || true
  CAFFEINATE_PID=""
fi
trap - EXIT
