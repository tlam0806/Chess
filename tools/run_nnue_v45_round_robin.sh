#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:?usage: run_nnue_v45_round_robin.sh RUN_DIR}"
if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$REPO_ROOT/$RUN_DIR"
fi

V45_RUN="${NNUE_V45_TOURNAMENT_SOURCE:-logs/nnue_v45_pareto_8h_20260902_024232}"
RACE_SOURCE="${NNUE_V45_RACE_SOURCE:-logs/nnue_v43_selfplay_race_20260901_030146}"
PYTHON="${NNUE_V45_TOURNAMENT_PYTHON:-.venv/bin/python}"
BINARY="$RACE_SOURCE/artifacts/nnue_v43_time_gauntlet"
MODEL="$RACE_SOURCE/artifacts/phase_quantized_nnue.bin"
BOOK="${NNUE_V45_TOURNAMENT_BOOK:-logs/nnue_v40_qsee_tune_20260817_005158/stockfish_balanced_openings_10ply_2400.txt}"

if [[ ! -x "$PYTHON" ]]; then
  echo "python is not executable: $PYTHON" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/bin"
RUNNER="$RUN_DIR/bin/run_nnue_v45_round_robin.py"
if [[ ! -e "$RUNNER" ]]; then
  cp tools/run_nnue_v45_round_robin.py "$RUNNER"
  cp tools/run_nnue_v43_selfplay_race.py "$RUN_DIR/bin/run_nnue_v43_selfplay_race.py"
  chmod +x "$RUNNER"
fi

exec "$PYTHON" "$RUNNER" \
  --run-dir "$RUN_DIR" \
  --v45-run "$V45_RUN" \
  --binary "$BINARY" \
  --model "$MODEL" \
  --book "$BOOK"
