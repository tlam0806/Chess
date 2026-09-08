#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:?usage: run_nnue_v43_all_prunes_tune.sh RUN_DIR}"
if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$REPO_ROOT/$RUN_DIR"
fi

DATASET_DIR="${NNUE_V43_ALL_PRUNES_DATASET_DIR:-data/nnue_v43_all_prunes_lichess_20260831}"
SOURCE_DATABASE="${NNUE_V43_ALL_PRUNES_SOURCE_DATABASE:-data/lichess_bot_2300_2600_population_v1/collector.sqlite3}"
EXCLUDE_DATASET_DIR="${NNUE_V43_ALL_PRUNES_EXCLUDE_DATASET_DIR:-data/nnue_v42_aspiration_lichess_20260830}"
MODEL="${NNUE_V43_ALL_PRUNES_MODEL:-models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin}"
TUNER_SOURCE="${NNUE_V43_ALL_PRUNES_TUNER_SOURCE:-tools/tune/tune_nnue_v43_all_prunes.py}"
TUNER_INFRA_SOURCE="${NNUE_V43_ALL_PRUNES_TUNER_INFRA_SOURCE:-tools/tune/tune_nnue_v43_aspiration.py}"
DATASET_BUILDER_SOURCE="${NNUE_V43_ALL_PRUNES_DATASET_BUILDER_SOURCE:-tools/data/build_nnue_v43_all_prunes_dataset.py}"
BUILD_DIR="${NNUE_V43_ALL_PRUNES_BUILD_DIR:-build-v43-all-prunes-release}"
# One repository-wide lock is intentional.  A --clean-first CMake build in a
# second tuning runner must not race this one even if it chose another build
# directory.
BUILD_LOCK_FILE="${NNUE_GLOBAL_BUILD_LOCK_FILE:-$REPO_ROOT/build/.nnue-global-build.lock}"
PYTHON="${NNUE_V43_ALL_PRUNES_PYTHON:-.venv/bin/python}"
WORKERS="${NNUE_V43_ALL_PRUNES_WORKERS:-4}"
BUILD_JOBS="${NNUE_V43_ALL_PRUNES_BUILD_JOBS:-8}"
SEED="${NNUE_V43_ALL_PRUNES_SEED:-20260831}"
# Cumulative wall-clock ceiling, not a fresh allowance per resume.  Twelve
# hours is the final-tune default; a paused run may only be resumed with an
# equal or larger ceiling.
DURATION_SEC="${NNUE_V43_ALL_PRUNES_CUMULATIVE_DURATION_SEC:-${NNUE_V43_ALL_PRUNES_DURATION_SEC:-43200}}"

RUN_BINARY_DIR="$RUN_DIR/bin"
RUN_INPUT_DIR="$RUN_DIR/inputs"
RUN_DATASET_DIR="$RUN_INPUT_DIR/dataset"
EVALUATOR_BINARY="$RUN_BINARY_DIR/evaluate_nnue_v43_selective"
GAUNTLET_BINARY="$RUN_BINARY_DIR/nnue_v43_time_gauntlet"
RUN_TUNER="$RUN_BINARY_DIR/tune_nnue_v43_all_prunes.py"
RUN_TUNER_INFRA="$RUN_BINARY_DIR/tune_nnue_v43_aspiration.py"
RUN_DATASET_BUILDER="$RUN_BINARY_DIR/build_nnue_v43_all_prunes_dataset.py"
RUN_MODEL="$RUN_INPUT_DIR/phase_quantized_nnue.bin"
CONTRACT="$RUN_INPUT_DIR/run_contract.json"
DURATION_REQUEST="$RUN_INPUT_DIR/cumulative_duration_sec.txt"
RUN_LOCK_FILE="$RUN_DIR/.runner.lock"

CAFFEINATE_PID=""
TUNER_PID=""
TUNER_HAS_PROCESS_GROUP=0

mkdir -p "$RUN_DIR"
# Hold an advisory lock on a dedicated descriptor for the runner's entire
# lifetime.  Unlike checking then writing runner.pid, acquisition is atomic;
# a competing invocation cannot clear status markers or replace the PID.  The
# lock file itself is deliberately persistent: unlinking a locked inode would
# let a new process lock a different inode at the same pathname.
exec 8>"$RUN_LOCK_FILE"
if command -v lockf >/dev/null 2>&1; then
  if ! lockf -t 0 8; then
    exec 8>&-
    echo "another V43 all-prunes runner already holds $RUN_LOCK_FILE" >&2
    exit 1
  fi
elif command -v flock >/dev/null 2>&1; then
  if ! flock -n 8; then
    exec 8>&-
    echo "another V43 all-prunes runner already holds $RUN_LOCK_FILE" >&2
    exit 1
  fi
else
  exec 8>&-
  echo "no supported per-run lock tool (lockf or flock)" >&2
  exit 1
fi

for numeric in "$DURATION_SEC" "$WORKERS" "$BUILD_JOBS"; do
  if [[ ! "$numeric" =~ ^[1-9][0-9]*$ ]]; then
    echo "duration, workers and build jobs must be positive integers" >&2
    exit 1
  fi
done
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "python is not executable: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c '
import numpy
if not numpy.__version__:
    raise SystemExit("numpy has no version")
'; then
  echo "the exact tuner interpreter must provide numpy: $PYTHON" >&2
  exit 1
fi

printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
rm -f "$RUN_DIR/JOB_COMPLETE" "$RUN_DIR/JOB_FAILED" "$RUN_DIR/JOB_PAUSED"
touch "$RUN_DIR/RUNNING"

write_status() {
  local state="$1"
  local detail="$2"
  printf 'state=%s detail=%s pid=%s updated_at=%s\n' \
    "$state" "$detail" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$RUN_DIR/runner_status.txt"
}

read_tuner_state() {
  "$PYTHON" -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
try:
    value = json.loads(path.read_text()).get("state", "")
except (OSError, json.JSONDecodeError):
    value = ""
print(value)
' "$RUN_DIR/status.json" 2>/dev/null || true
}

finish() {
  local status=$?
  local tuner_state=""
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
  rm -f "$RUN_DIR/RUNNING" "$RUN_DIR/runner.pid"

  tuner_state="$(read_tuner_state)"
  if [[ $status -eq 0 && "$tuner_state" == "DONE" \
        && -s "$RUN_DIR/summary.json" ]]; then
    touch "$RUN_DIR/JOB_COMPLETE"
    write_status complete tuner_finished
  elif [[ $status -eq 0 && "$tuner_state" == "PAUSED" ]]; then
    touch "$RUN_DIR/JOB_PAUSED"
    write_status paused cumulative_duration_budget_exhausted
  else
    if [[ $status -eq 0 ]]; then
      status=1
    fi
    printf '%s\n' "$status" > "$RUN_DIR/JOB_FAILED"
    write_status failed "exit_${status}_tuner_${tuner_state:-missing}"
  fi
  exec 8>&-
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w "$$" &
  CAFFEINATE_PID=$!
fi

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
      echo "partial immutable snapshot differs: $destination" >&2
      return 1
    fi
    rm -f "$temporary"
  else
    mv "$temporary" "$destination"
  fi
}

validate_source_dataset() {
  "$PYTHON" -c '
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest_path = root / "manifest.json"
try:
    manifest = json.loads(manifest_path.read_text())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid dataset manifest: {error}")
splits = ("tune", "selection", "aspiration_refresh", "holdout")
if manifest.get("format") != "nnue-v43-all-prunes-game-disjoint-v3":
    raise SystemExit("wrong final-prune dataset format")
if manifest.get("schema_version") != 3:
    raise SystemExit("wrong final-prune dataset schema")
for split in splits:
    metadata = manifest.get("splits", {}).get(split, {})
    file_metadata = manifest.get("split_files", {}).get(split, {})
    path = root / f"{split}.tsv"
    if int(metadata.get("count", 0)) <= 0 or not path.is_file():
        raise SystemExit(f"missing or empty split metadata: {split}")
    payload = path.read_bytes()
    if not payload:
        raise SystemExit(f"empty split file: {split}")
    if file_metadata.get("file") != path.name:
        raise SystemExit(f"wrong split filename: {split}")
    if int(file_metadata.get("bytes", -1)) != len(payload):
        raise SystemExit(f"split byte count mismatch: {split}")
    if file_metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise SystemExit(f"split digest mismatch: {split}")
exclusions = manifest.get("exclusions", {})
overlap = manifest.get("overlap", {})
if int(exclusions.get("unique_game_ids", 0)) <= 0:
    raise SystemExit("prior game exclusions are absent")
required_overlap_keys = {
    "games_between_new_splits",
    "position_or_mirror_keys_between_new_splits",
    "games_with_prior_v42_v43",
    "position_or_mirror_keys_with_prior_v42_v43",
}
if (
    not isinstance(overlap, dict)
    or set(overlap) != required_overlap_keys
    or any(int(overlap[key]) != 0 for key in required_overlap_keys)
):
    raise SystemExit("dataset manifest overlap proof is incomplete or nonzero")
' "$1"
}

build_dataset_if_needed() {
  if [[ -s "$DATASET_DIR/manifest.json" ]]; then
    validate_source_dataset "$DATASET_DIR"
    return
  fi
  for source in "$SOURCE_DATABASE" "$EXCLUDE_DATASET_DIR/manifest.json" \
      "$DATASET_BUILDER_SOURCE"; do
    if [[ ! -s "$source" ]]; then
      echo "missing input required to build final dataset: $source" >&2
      return 1
    fi
  done
  mkdir -p "$DATASET_DIR"
  "$PYTHON" "$DATASET_BUILDER_SOURCE" \
    --database "$SOURCE_DATABASE" \
    --exclude-dataset-dir "$EXCLUDE_DATASET_DIR" \
    --output-dir "$DATASET_DIR" \
    --seed "$SEED"
  validate_source_dataset "$DATASET_DIR"
}

build_targets() {
  for source in "$MODEL" "$TUNER_SOURCE" "$TUNER_INFRA_SOURCE" \
      "$DATASET_BUILDER_SOURCE"; do
    if [[ ! -s "$source" ]]; then
      echo "missing or empty immutable input: $source" >&2
      return 1
    fi
  done
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCHESS_BUILD_EXPERIMENTS=ON
  cmake --build "$BUILD_DIR" --clean-first \
    --target evaluate_nnue_v43_selective nnue_v43_time_gauntlet \
    -j "$BUILD_JOBS"
  snapshot_file \
    "$BUILD_DIR/evaluate_nnue_v43_selective" "$EVALUATOR_BINARY" 755
  snapshot_file "$BUILD_DIR/nnue_v43_time_gauntlet" "$GAUNTLET_BINARY" 755
  snapshot_file "$TUNER_SOURCE" "$RUN_TUNER" 755
  snapshot_file "$TUNER_INFRA_SOURCE" "$RUN_TUNER_INFRA" 644
  snapshot_file "$DATASET_BUILDER_SOURCE" "$RUN_DATASET_BUILDER" 755
  snapshot_file "$MODEL" "$RUN_MODEL" 644
  mkdir -p "$RUN_DATASET_DIR"
  for name in tune.tsv selection.tsv aspiration_refresh.tsv holdout.tsv \
      manifest.json; do
    snapshot_file "$DATASET_DIR/$name" "$RUN_DATASET_DIR/$name" 644
  done
}

write_contract() {
  local temporary="${CONTRACT}.tmp.$$"
  "$PYTHON" -c '
import hashlib, json, pathlib, sys
output = pathlib.Path(sys.argv[1])
run_input = pathlib.Path(sys.argv[2])
seed = int(sys.argv[3])
workers = int(sys.argv[4])
relative_files = sys.argv[5:]
files = {}
for relative in relative_files:
    path = run_input.parent / relative
    payload = path.read_bytes()
    files[relative] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
contract = {
    "schema_version": 2,
    "experiment": "nnue-v43-final-all-prunes-balanced-baseline-v2",
    "tuner_schema_version": 2,
    "tuner_experiment": "v43-final-joint-all-prunes-balanced-baseline-v2",
    "prune_stage_aspiration": {
        "name": "v43_balanced_aspiration",
        "config_sha256": (
            "906570ae8ede9e87540847228cd75eef707a64aa5873163be665e220c8b2515e"
        ),
        "config": {
            "enabled": True,
            "min_depth": 2,
            "delta_base_cp": 68,
            "delta_divisor": 33700,
            "expansion_factor_per_mille": 2290,
            "max_fail_high_reductions": 1,
            "mean_score_new_weight_per_mille": 370,
            "max_researches": 6,
            "mean_score_clamp_cp": 1500,
        },
        "fixed_identically_through_selection": True,
    },
    "production_baseline_config_hash": (
        "e770be6261511ed0bc391852c27d69cbe203204c3b5db96e8cc329daf7772d48"
    ),
    "seed": seed,
    "workers": workers,
    "files": files,
}
output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
' "$temporary" "$RUN_INPUT_DIR" "$SEED" "$WORKERS" \
    bin/evaluate_nnue_v43_selective \
    bin/nnue_v43_time_gauntlet \
    bin/tune_nnue_v43_all_prunes.py \
    bin/tune_nnue_v43_aspiration.py \
    bin/build_nnue_v43_all_prunes_dataset.py \
    inputs/phase_quantized_nnue.bin \
    inputs/dataset/tune.tsv \
    inputs/dataset/selection.tsv \
    inputs/dataset/aspiration_refresh.tsv \
    inputs/dataset/holdout.tsv \
    inputs/dataset/manifest.json
  mv "$temporary" "$CONTRACT"
}

validate_contract() {
  "$PYTHON" -c '
import hashlib, importlib.util, json, pathlib, sys
contract_path = pathlib.Path(sys.argv[1])
root = contract_path.parent.parent
seed = int(sys.argv[2])
workers = int(sys.argv[3])
try:
    contract = json.loads(contract_path.read_text())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid immutable run contract: {error}")
if contract.get("schema_version") != 2:
    raise SystemExit("unsupported immutable run contract")
if contract.get("experiment") != (
    "nnue-v43-final-all-prunes-balanced-baseline-v2"
):
    raise SystemExit("wrong experiment in immutable run contract")
if contract.get("tuner_schema_version") != 2 or contract.get(
    "tuner_experiment"
) != "v43-final-joint-all-prunes-balanced-baseline-v2":
    raise SystemExit("wrong tuner semantics in immutable run contract")
expected_aspiration = {
    "name": "v43_balanced_aspiration",
    "config_sha256": (
        "906570ae8ede9e87540847228cd75eef707a64aa5873163be665e220c8b2515e"
    ),
    "config": {
        "enabled": True,
        "min_depth": 2,
        "delta_base_cp": 68,
        "delta_divisor": 33700,
        "expansion_factor_per_mille": 2290,
        "max_fail_high_reductions": 1,
        "mean_score_new_weight_per_mille": 370,
        "max_researches": 6,
        "mean_score_clamp_cp": 1500,
    },
    "fixed_identically_through_selection": True,
}
if contract.get("prune_stage_aspiration") != expected_aspiration:
    raise SystemExit("wrong Balanced aspiration in immutable run contract")
if contract.get("production_baseline_config_hash") != (
    "e770be6261511ed0bc391852c27d69cbe203204c3b5db96e8cc329daf7772d48"
):
    raise SystemExit("wrong production baseline in immutable run contract")
if int(contract.get("seed", -1)) != seed:
    raise SystemExit("resume seed differs from immutable run contract")
if int(contract.get("workers", -1)) != workers:
    raise SystemExit("resume worker count differs from immutable run contract")
files = contract.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit("immutable run contract has no artifacts")
for relative, expected in files.items():
    path = root / relative
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SystemExit(f"missing immutable artifact {relative}: {error}")
    if len(payload) != int(expected.get("bytes", -1)):
        raise SystemExit(f"immutable artifact size mismatch: {relative}")
    if hashlib.sha256(payload).hexdigest() != expected.get("sha256"):
        raise SystemExit(f"immutable artifact digest mismatch: {relative}")
tuner_path = root / "bin/tune_nnue_v43_all_prunes.py"
sys.path.insert(0, str(tuner_path.parent))
spec = importlib.util.spec_from_file_location(
    "balanced_baseline_contract_tuner", tuner_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load snapshotted all-prunes tuner")
tuner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tuner
spec.loader.exec_module(tuner)
if (
    getattr(tuner, "SCHEMA_VERSION", None) != 2
    or getattr(tuner, "EXPERIMENT", None)
        != "v43-final-joint-all-prunes-balanced-baseline-v2"
    or tuner.PRODUCTION_BASELINE_ASPIRATION.canonical()
        != expected_aspiration["config"]
    or tuner.PRODUCTION_BASELINE_ASPIRATION.hash
        != expected_aspiration["config_sha256"]
    or tuner.Candidate().hash
        != contract["production_baseline_config_hash"]
):
    raise SystemExit(
        "snapshotted tuner is not the pinned Balanced-baseline experiment")
' "$CONTRACT" "$SEED" "$WORKERS"
  validate_source_dataset "$RUN_DATASET_DIR"
}

prepare_new_run() {
  write_status preparing sealed_dataset_and_binaries
  build_dataset_if_needed
  validate_source_dataset "$DATASET_DIR"
  build_targets
  write_contract
}

if [[ -s "$CONTRACT" ]]; then
  write_status validating immutable_resume
  validate_contract
else
  mkdir -p "$(dirname -- "$BUILD_LOCK_FILE")"
  write_status waiting global_build_lock
  if command -v lockf >/dev/null 2>&1; then
    (
      exec 9>"$BUILD_LOCK_FILE"
      lockf 9
      prepare_new_run
    )
  elif command -v flock >/dev/null 2>&1; then
    (
      exec 9>"$BUILD_LOCK_FILE"
      flock 9
      prepare_new_run
    )
  else
    echo "no supported global build lock tool (lockf or flock)" >&2
    exit 1
  fi
  validate_contract
fi

if [[ -s "$DURATION_REQUEST" ]]; then
  previous_duration="$(<"$DURATION_REQUEST")"
  if [[ ! "$previous_duration" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid previous cumulative duration request" >&2
    exit 1
  fi
  if (( DURATION_SEC < previous_duration )); then
    echo "resume cumulative duration may not decrease: $DURATION_SEC < $previous_duration" >&2
    exit 1
  fi
fi
duration_temporary="${DURATION_REQUEST}.tmp.$$"
printf '%s\n' "$DURATION_SEC" > "$duration_temporary"
mv "$duration_temporary" "$DURATION_REQUEST"

write_status running final_all_prunes_balanced_baseline_tuner
TUNER_COMMAND=(
  "$PYTHON" "$RUN_TUNER"
  --binary "$EVALUATOR_BINARY"
  --dataset-dir "$RUN_DATASET_DIR"
  --model "$RUN_MODEL"
  --run-dir "$RUN_DIR"
  --duration-sec "$DURATION_SEC"
  --workers "$WORKERS"
  --seed "$SEED"
)

if command -v setsid >/dev/null 2>&1; then
  setsid "${TUNER_COMMAND[@]}" &
  TUNER_PID=$!
  TUNER_HAS_PROCESS_GROUP=1
elif command -v perl >/dev/null 2>&1; then
  perl -MPOSIX -e \
    'POSIX::setsid() >= 0 or die "setsid failed: $!\n"; exec @ARGV' \
    "${TUNER_COMMAND[@]}" &
  TUNER_PID=$!
  TUNER_HAS_PROCESS_GROUP=1
else
  "${TUNER_COMMAND[@]}" &
  TUNER_PID=$!
fi
wait "$TUNER_PID"
TUNER_PID=""
