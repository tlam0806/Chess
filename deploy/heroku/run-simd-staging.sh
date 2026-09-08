#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  CHESS_HEROKU_STAGING_APP=APP deploy/heroku/run-simd-staging.sh \
    --app APP [--size basic] [--profile smoke|full] [--runs N] [--deploy-only]

The app must already exist and use the container stack.  This script never
creates an app, reads/sets a Lichess token, or scales a worker.  The explicit
CHESS_HEROKU_STAGING_APP latch prevents an accidental production deployment.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_REL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"
PARITY_REL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue_parity.tsv"
PRODUCTION_APP="${CHESS_HEROKU_PRODUCTION_APP:-stormy-garden-92984}"
APP_NAME=""
DYNO_SIZE="basic"
PROFILE="smoke"
RUNS=""
DEPLOY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      APP_NAME="${2:-}"
      shift 2
      ;;
    --size)
      DYNO_SIZE="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --runs)
      RUNS="${2:-}"
      shift 2
      ;;
    --deploy-only)
      DEPLOY_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$APP_NAME" ]]; then
  echo "--app is required; there is deliberately no default app" >&2
  exit 2
fi
if [[ "$APP_NAME" == "$PRODUCTION_APP" ]]; then
  echo "Refusing to use production app: $APP_NAME" >&2
  exit 2
fi
if [[ "${CHESS_HEROKU_STAGING_APP:-}" != "$APP_NAME" ]]; then
  echo "Safety latch mismatch." >&2
  echo "Set CHESS_HEROKU_STAGING_APP=$APP_NAME to confirm this dedicated staging app." >&2
  exit 2
fi
if [[ "$PROFILE" != "smoke" && "$PROFILE" != "full" ]]; then
  echo "--profile must be smoke or full" >&2
  exit 2
fi
if [[ -n "$RUNS" && ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--runs must be a positive integer" >&2
  exit 2
fi
for command in git heroku python3 rsync; do
  if ! command -v "$command" >/dev/null; then
    echo "Missing required command: $command" >&2
    exit 2
  fi
done
if ! heroku auth:whoami >/dev/null 2>&1; then
  echo "Heroku CLI is not logged in. Run: heroku login" >&2
  exit 2
fi
if ! heroku apps:info --app "$APP_NAME" >/dev/null 2>&1; then
  echo "Staging app does not exist or is not accessible: $APP_NAME" >&2
  exit 2
fi
if ! heroku stack --app "$APP_NAME" | grep -Eq '^\* +container$'; then
  echo "Staging app must already use the container stack: $APP_NAME" >&2
  echo "Set it explicitly with: heroku stack:set container --app $APP_NAME" >&2
  exit 2
fi
ACTIVE_DYNOS="$(
  heroku ps --app "$APP_NAME" --json \
    | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))'
)"
if [[ "$ACTIVE_DYNOS" != "0" ]]; then
  echo "Refusing to replace an app that currently has $ACTIVE_DYNOS active dyno(s)." >&2
  echo "Use a dedicated, idle benchmark staging app." >&2
  exit 2
fi
if [[ ! -f "$REPO_ROOT/$MODEL_REL" || ! -f "$REPO_ROOT/$PARITY_REL" ]]; then
  echo "Production model or parity fixture is missing" >&2
  exit 2
fi

STAGE_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "$STAGE_DIR"
}
trap cleanup EXIT

mkdir -p \
  "$STAGE_DIR/benchmarks" \
  "$STAGE_DIR/deploy/heroku" \
  "$STAGE_DIR/$(dirname "$MODEL_REL")"
cp "$REPO_ROOT/CMakeLists.txt" "$STAGE_DIR/"
rsync -a "$REPO_ROOT/cmake/" "$STAGE_DIR/cmake/"
rsync -a "$REPO_ROOT/include/" "$STAGE_DIR/include/"
rsync -a "$REPO_ROOT/src/" "$STAGE_DIR/src/"
rsync -a "$REPO_ROOT/tools/" "$STAGE_DIR/tools/"
rsync -a "$REPO_ROOT/tests/" "$STAGE_DIR/tests/"
cp "$REPO_ROOT/benchmarks/uci_platform_v1.json" "$STAGE_DIR/benchmarks/"
cp "$REPO_ROOT/deploy/heroku/benchmark-config-v41.yml" \
  "$STAGE_DIR/deploy/heroku/"
cp "$REPO_ROOT/deploy/heroku/Dockerfile.simd-benchmark" \
  "$STAGE_DIR/Dockerfile"
cp "$REPO_ROOT/deploy/heroku/heroku-simd-benchmark.yml" \
  "$STAGE_DIR/heroku.yml"
cp "$REPO_ROOT/$MODEL_REL" "$STAGE_DIR/$MODEL_REL"
cp "$REPO_ROOT/$PARITY_REL" "$STAGE_DIR/$PARITY_REL"

# Only explicitly selected source/artifact paths enter this context.  Refuse
# common secret filenames anyway so a future copy-list expansion fails closed.
if find "$STAGE_DIR" -type f \
    \( -name '.env' -o -name '.env.*' -o -name '.token.env' \) \
    -print -quit | grep -q .; then
  echo "Refusing staging snapshot containing an environment/token file" >&2
  exit 2
fi

git -C "$STAGE_DIR" init -q
git -C "$STAGE_DIR" config user.name "Chess SIMD Staging"
git -C "$STAGE_DIR" config user.email "staging@localhost"
git -C "$STAGE_DIR" add .
SNAPSHOT_TREE="$(git -C "$STAGE_DIR" write-tree)"
git -C "$STAGE_DIR" commit -qm "Benchmark NNUE SIMD snapshot $SNAPSHOT_TREE"

echo "Staging app: $APP_NAME"
echo "Snapshot tree: $SNAPSHOT_TREE"
echo "Deploying benchmark-only image; no token/config mutation and no worker scaling."
git -C "$STAGE_DIR" remote add heroku "https://git.heroku.com/$APP_NAME.git"
git -C "$STAGE_DIR" push heroku HEAD:main --force

# A non-web process is normally created at formation zero.  State it
# explicitly after every release so the image can only run as a one-off dyno.
heroku ps:scale benchmark=0 --app "$APP_NAME" >/dev/null
if [[ "$DEPLOY_ONLY" -eq 1 ]]; then
  echo "Staging image deployed at formation benchmark=0."
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$REPO_ROOT/logs/nnue_simd_matrix_${PROFILE}_${STAMP}"
MATRIX_COMMAND=(
  python3 "$REPO_ROOT/tools/benchmark/run_heroku_nnue_backend_matrix.py"
  --app "$APP_NAME"
  --production-app "$PRODUCTION_APP"
  --process-type benchmark
  --size "$DYNO_SIZE"
  --profile "$PROFILE"
  --source-label "staging-tree:$SNAPSHOT_TREE"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$RUNS" ]]; then
  MATRIX_COMMAND+=(--runs "$RUNS")
fi
"${MATRIX_COMMAND[@]}"

echo "All forced backend parity checks and benchmarks passed."
echo "Artifacts: $OUTPUT_DIR"
