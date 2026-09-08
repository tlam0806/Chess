#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:?usage: run_nnue_v43_aspiration_tune.sh RUN_DIR}"
if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$REPO_ROOT/$RUN_DIR"
fi

# V43 intentionally consumes the same sealed, game-disjoint corpus and model
# as V42.  Everything executable is snapshotted under the run directory so a
# resumed experiment cannot silently pick up a newer scalar searcher/tuner.
DATASET_DIR="${NNUE_V43_ASPIRATION_DATASET_DIR:-data/nnue_v42_aspiration_lichess_20260830}"
MODEL="${NNUE_V43_ASPIRATION_MODEL:-models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin}"
BUILD_DIR="${NNUE_V43_ASPIRATION_BUILD_DIR:-build-v43-aspiration-release}"
BUILD_LOCK_FILE="${NNUE_V43_ASPIRATION_BUILD_LOCK_FILE:-${BUILD_DIR}.nnue_v43_aspiration.lock}"
RUN_BINARY_DIR="$RUN_DIR/bin"
RUN_INPUT_DIR="$RUN_DIR/inputs"
EVALUATOR_BINARY="$RUN_BINARY_DIR/evaluate_nnue_v43_selective"
GAUNTLET_BINARY="$RUN_BINARY_DIR/nnue_v43_time_gauntlet"
TUNER_SOURCE="${NNUE_V43_ASPIRATION_TUNER_SOURCE:-tools/tune/tune_nnue_v43_aspiration.py}"
RUN_TUNER="$RUN_BINARY_DIR/tune_nnue_v43_aspiration.py"
RUN_MODEL="$RUN_INPUT_DIR/phase_quantized_nnue.bin"
PYTHON="${NNUE_V43_ASPIRATION_PYTHON:-.venv/bin/python}"
# This is the cumulative elapsed ceiling for the run, not a fresh allowance
# per invocation. Resume a PAUSED run only with a strictly larger ceiling.
DURATION_SEC="${NNUE_V43_ASPIRATION_CUMULATIVE_DURATION_SEC:-${NNUE_V43_ASPIRATION_DURATION_SEC:-28800}}"
WORKERS="${NNUE_V43_ASPIRATION_WORKERS:-4}"
BUILD_JOBS="${NNUE_V43_ASPIRATION_BUILD_JOBS:-8}"
# Keep the V42 seed so the scalar/range experiments see the same deterministic
# samples and mutation stream wherever their common configuration permits it.
SEED="${NNUE_V43_ASPIRATION_SEED:-20260829}"
CAFFEINATE_PID=""
SHLOCK_HELD=0
TUNER_PID=""
TUNER_HAS_PROCESS_GROUP=0

mkdir -p "$RUN_DIR"

if [[ -s "$RUN_DIR/runner.pid" ]]; then
  prior_pid="$(<"$RUN_DIR/runner.pid")"
  if [[ "$prior_pid" =~ ^[0-9]+$ ]] && kill -0 "$prior_pid" 2>/dev/null; then
    echo "another V43 aspiration tune runner is active: pid=$prior_pid" >&2
    exit 1
  fi
fi

printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
rm -f \
  "$RUN_DIR/JOB_COMPLETE" \
  "$RUN_DIR/JOB_FAILED" \
  "$RUN_DIR/JOB_PAUSED"
touch "$RUN_DIR/RUNNING"

write_status() {
  local state="$1"
  local detail="$2"
  printf 'state=%s detail=%s pid=%s updated_at=%s\n' \
    "$state" "$detail" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$RUN_DIR/status.txt"
}

finish() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$TUNER_PID" ]] && kill -0 "$TUNER_PID" 2>/dev/null; then
    if [[ $TUNER_HAS_PROCESS_GROUP -eq 1 ]]; then
      kill -TERM -- "-$TUNER_PID" 2>/dev/null || true
    else
      kill -TERM "$TUNER_PID" 2>/dev/null || true
    fi
    wait "$TUNER_PID" 2>/dev/null || true
  fi
  if [[ -n "$CAFFEINATE_PID" ]]; then
    kill "$CAFFEINATE_PID" 2>/dev/null || true
    wait "$CAFFEINATE_PID" 2>/dev/null || true
  fi
  if [[ $SHLOCK_HELD -eq 1 ]]; then
    rm -f "$BUILD_LOCK_FILE"
    SHLOCK_HELD=0
  fi
  rm -f "$RUN_DIR/RUNNING" "$RUN_DIR/runner.pid"
  if [[ $status -eq 0 && -f "$RUN_DIR/DONE" ]]; then
    touch "$RUN_DIR/JOB_COMPLETE"
    write_status complete tuner_finished
  elif [[ $status -eq 0 && -f "$RUN_DIR/PAUSED.json" ]]; then
    touch "$RUN_DIR/JOB_PAUSED"
    write_status paused duration_budget_exhausted
  else
    if [[ $status -eq 0 ]]; then
      status=1
    fi
    printf '%s\n' "$status" > "$RUN_DIR/JOB_FAILED"
    write_status failed "exit_$status"
  fi
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
  CAFFEINATE_PID=$!
fi

for split in tune selection holdout; do
  if [[ ! -s "$DATASET_DIR/$split.tsv" ]]; then
    echo "missing or empty dataset split: $DATASET_DIR/$split.tsv" >&2
    exit 1
  fi
done
if [[ ! -s "$DATASET_DIR/manifest.json" ]]; then
  echo "missing dataset manifest: $DATASET_DIR/manifest.json" >&2
  exit 1
fi
if [[ ! -s "$MODEL" ]]; then
  echo "missing or empty model: $MODEL" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "python is not executable: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c '
import json
import sys
manifest = json.load(open(sys.argv[1]))
valid = (
    manifest.get("format") == "nnue-v42-lichess-game-disjoint-v2"
    and manifest.get("schema_version") == 2
    and all(
        int(manifest.get("splits", {}).get(split, {}).get("count", 0)) > 0
        for split in ("tune", "selection", "holdout")
    )
)
raise SystemExit(0 if valid else 1)
' "$DATASET_DIR/manifest.json"; then
  echo "dataset manifest is not the sealed Lichess game-disjoint schema v2" >&2
  exit 1
fi
for numeric in "$DURATION_SEC" "$WORKERS" "$BUILD_JOBS"; do
  if [[ ! "$numeric" =~ ^[1-9][0-9]*$ ]]; then
    echo "duration, workers and build jobs must be positive integers" >&2
    exit 1
  fi
done

snapshot_file() {
  local source="$1"
  local destination="$2"
  local mode="$3"
  local temporary="${destination}.tmp.$$"
  mkdir -p "$(dirname -- "$destination")"
  cp "$source" "$temporary"
  chmod "$mode" "$temporary"
  if [[ -e "$destination" ]]; then
    if ! cmp -s "$destination" "$temporary"; then
      rm -f "$temporary"
      echo "immutable run snapshot differs: $destination" >&2
      return 1
    fi
    rm -f "$temporary"
  else
    mv "$temporary" "$destination"
  fi
}

build_targets() {
  if [[ ! -s "$TUNER_SOURCE" ]]; then
    echo "missing or empty tuner source for new snapshot: $TUNER_SOURCE" >&2
    return 1
  fi
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCHESS_BUILD_EXPERIMENTS=ON
  cmake --build "$BUILD_DIR" \
    --clean-first \
    --target evaluate_nnue_v43_selective nnue_v43_time_gauntlet \
    -j "$BUILD_JOBS"
  snapshot_file \
    "$BUILD_DIR/evaluate_nnue_v43_selective" "$EVALUATOR_BINARY" 755
  snapshot_file \
    "$BUILD_DIR/nnue_v43_time_gauntlet" "$GAUNTLET_BINARY" 755
  snapshot_file "$TUNER_SOURCE" "$RUN_TUNER" 755
  snapshot_file "$MODEL" "$RUN_MODEL" 644
}

if [[ -x "$EVALUATOR_BINARY" && -x "$GAUNTLET_BINARY" \
      && -x "$RUN_TUNER" && -s "$RUN_MODEL" ]]; then
  write_status validating immutable_artifacts
  snapshot_file "$MODEL" "$RUN_MODEL" 644
else
  mkdir -p "$(dirname -- "$BUILD_LOCK_FILE")"
  write_status waiting build_lock
  if command -v lockf >/dev/null 2>&1; then
    (
      exec 9>"$BUILD_LOCK_FILE"
      lockf 9
      write_status building immutable_run_artifacts
      build_targets
    )
  elif command -v flock >/dev/null 2>&1; then
    (
      exec 9>"$BUILD_LOCK_FILE"
      flock 9
      write_status building immutable_run_artifacts
      build_targets
    )
  elif command -v shlock >/dev/null 2>&1; then
    if ! shlock -f "$BUILD_LOCK_FILE" -p $$; then
      echo "shared build is locked: $BUILD_LOCK_FILE" >&2
      exit 1
    fi
    SHLOCK_HELD=1
    write_status building immutable_run_artifacts
    build_targets
    rm -f "$BUILD_LOCK_FILE"
    SHLOCK_HELD=0
  else
    echo "no supported global build lock tool (lockf, flock or shlock)" >&2
    exit 1
  fi
fi

write_status running aspiration_tuner
TUNER_COMMAND=(
  "$PYTHON" "$RUN_TUNER"
  --binary "$EVALUATOR_BINARY" \
  --dataset-dir "$DATASET_DIR" \
  --model "$RUN_MODEL" \
  --run-dir "$RUN_DIR" \
  --duration-sec "$DURATION_SEC" \
  --preflight-size 128 \
  --exploration-size 750 \
  --selection-size 3000 \
  --holdout-size 4000 \
  --preflight-depth 6 \
  --exploration-depth 6 \
  --selection-depth 7 \
  --holdout-depth 8 \
  --global-candidates 64 \
  --refinement-candidates 64 \
  --max-researches 6 \
  --mean-score-clamp-cp 1500 \
  --workers "$WORKERS" \
  --seed "$SEED"
)

if command -v setsid >/dev/null 2>&1; then
  setsid "${TUNER_COMMAND[@]}" &
  TUNER_PID=$!
  TUNER_HAS_PROCESS_GROUP=1
elif command -v perl >/dev/null 2>&1; then
  perl -MPOSIX -e 'POSIX::setsid() >= 0 or die "setsid failed: $!\n"; exec @ARGV' \
    "${TUNER_COMMAND[@]}" &
  TUNER_PID=$!
  TUNER_HAS_PROCESS_GROUP=1
else
  "${TUNER_COMMAND[@]}" &
  TUNER_PID=$!
fi
wait "$TUNER_PID"
TUNER_PID=""
