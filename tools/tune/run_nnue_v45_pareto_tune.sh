#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:?usage: run_nnue_v45_pareto_tune.sh RUN_DIR}"
if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$REPO_ROOT/$RUN_DIR"
fi

SOURCE_RUN="${NNUE_V45_SOURCE_RUN:-logs/nnue_v43_all_prunes_balanced_baseline_12h_20260831_030546}"
PYTHON="${NNUE_V45_PYTHON:-.venv/bin/python}"
MUTATIONS="${NNUE_V45_MUTATIONS:-360}"
WORKERS="${NNUE_V45_WORKERS:-3}"
DURATION_SEC="${NNUE_V45_DURATION_SEC:-28800}"
SEED="${NNUE_V45_SEED:-20260902}"
HARD_NODE_RATIO="${NNUE_V45_HARD_NODE_RATIO:-2.0}"
BASELINE_PROFILE="${NNUE_V45_BASELINE_PROFILE:-v43-original}"
MINIMUM_DISK_FREE_BYTES="${NNUE_V45_MINIMUM_DISK_FREE_BYTES:-5368709120}"

if [[ ! -x "$PYTHON" ]]; then
  echo "python is not executable: $PYTHON" >&2
  exit 1
fi
for value in "$MUTATIONS" "$WORKERS" "$DURATION_SEC" "$SEED" "$MINIMUM_DISK_FREE_BYTES"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "mutations, workers, duration, and seed must be integers" >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR/bin"
RUN_TUNER="$RUN_DIR/bin/tune_nnue_v45_pareto.py"
if [[ ! -e "$RUN_TUNER" ]]; then
  cp tools/tune/tune_nnue_v45_pareto.py "$RUN_TUNER"
  cp tools/tune/tune_nnue_v43_all_prunes.py "$RUN_DIR/bin/tune_nnue_v43_all_prunes.py"
  cp tools/tune/tune_nnue_v43_aspiration.py "$RUN_DIR/bin/tune_nnue_v43_aspiration.py"
  chmod +x "$RUN_TUNER"
fi

exec "$PYTHON" "$RUN_TUNER" \
  --run-dir "$RUN_DIR" \
  --source-run "$SOURCE_RUN" \
  --mutation-count "$MUTATIONS" \
  --workers "$WORKERS" \
  --duration-sec "$DURATION_SEC" \
  --seed "$SEED" \
  --hard-node-ratio "$HARD_NODE_RATIO" \
  --baseline-profile "$BASELINE_PROFILE" \
  --minimum-disk-free-bytes "$MINIMUM_DISK_FREE_BYTES"
