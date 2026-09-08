#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:?usage: run_nnue_v46_timed_pipeline.sh RUN_DIR}"
if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$REPO_ROOT/$RUN_DIR"
fi

SOURCE_RUN="${NNUE_V46_SOURCE_RUN:-logs/nnue_v43_all_prunes_balanced_baseline_12h_20260831_030546}"
RACE_SOURCE="${NNUE_V46_RACE_SOURCE:-logs/nnue_v43_selfplay_race_20260901_030146}"
BOOK="${NNUE_V46_BOOK:-logs/nnue_v40_qsee_tune_20260817_005158/stockfish_balanced_openings_10ply_2400.txt}"
EXCLUDED="${NNUE_V46_EXCLUDED_OPENINGS:-logs/nnue_v45_protected_tournament_20260903_132948/openings}"
PYTHON="${NNUE_V46_PYTHON:-.venv/bin/python}"
MUTATIONS="${NNUE_V46_MUTATIONS:-480}"
WORKERS="${NNUE_V46_WORKERS:-4}"
TEACHER_SECONDS="${NNUE_V46_TEACHER_SECONDS:-43200}"
NODE_RATIO="${NNUE_V46_NODE_RATIO:-1.25}"

if [[ ! -x "$PYTHON" ]]; then
  echo "python is not executable: $PYTHON" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/bin"
RUNNER="$RUN_DIR/bin/run_nnue_v46_timed_pipeline.py"
if [[ ! -e "$RUNNER" ]]; then
  cp tools/run_nnue_v46_timed_pipeline.py "$RUNNER"
  cp tools/tune_nnue_v45_pareto.py "$RUN_DIR/bin/tune_nnue_v45_pareto.py"
  cp tools/tune_nnue_v43_all_prunes.py "$RUN_DIR/bin/tune_nnue_v43_all_prunes.py"
  cp tools/tune_nnue_v43_aspiration.py "$RUN_DIR/bin/tune_nnue_v43_aspiration.py"
  cp tools/run_nnue_v43_selfplay_race.py "$RUN_DIR/bin/run_nnue_v43_selfplay_race.py"
  chmod +x "$RUNNER"
fi

exec "$PYTHON" "$RUNNER" \
  --run-dir "$RUN_DIR" \
  --source-run "$SOURCE_RUN" \
  --evaluation-cache-run logs/nnue_v45_pareto_8h_20260902_024232 \
  --evaluation-cache-run logs/nnue_v45_pareto_broad_12h_20260902_231700 \
  --binary "$RACE_SOURCE/artifacts/nnue_v43_time_gauntlet" \
  --model "$RACE_SOURCE/artifacts/phase_quantized_nnue.bin" \
  --book "$BOOK" \
  --exclude-openings-dir "$EXCLUDED" \
  --mutation-count "$MUTATIONS" \
  --workers "$WORKERS" \
  --teacher-duration-sec "$TEACHER_SECONDS" \
  --hard-node-ratio "$NODE_RATIO"
