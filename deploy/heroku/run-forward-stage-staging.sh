#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  CHESS_HEROKU_STAGING_APP=APP deploy/heroku/run-forward-stage-staging.sh \
    --app APP --source-root CLEAN_GIT_WORKTREE \
    [--profile smoke|full] [--runs N] [--deploy-only]

The source root must be a clean Git worktree containing the exact benchmark
candidate and production model. The app must already exist, use the container
stack, and be idle. This script refuses the production app, deploys only a
benchmark image, keeps benchmark formation at zero, and runs only explicitly
requested one-off Basic dynos.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_REL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"
PARITY_REL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue_parity.tsv"
CANONICAL_PRODUCTION_APP="stormy-garden-92984"
PRODUCTION_APP="${CHESS_HEROKU_PRODUCTION_APP:-$CANONICAL_PRODUCTION_APP}"
APP_NAME=""
SOURCE_ROOT=""
PROFILE="smoke"
RUNS=""
DEPLOY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      APP_NAME="${2:-}"
      shift 2
      ;;
    --source-root)
      SOURCE_ROOT="${2:-}"
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
if [[ "$APP_NAME" == "$CANONICAL_PRODUCTION_APP" || "$APP_NAME" == "$PRODUCTION_APP" ]]; then
  echo "Refusing to use production app: $APP_NAME" >&2
  exit 2
fi
for command in git heroku python3 rsync shasum tar tee; do
  if ! command -v "$command" >/dev/null; then
    echo "Missing required command: $command" >&2
    exit 2
  fi
done
if [[ -z "$SOURCE_ROOT" ]]; then
  echo "--source-root is required; dirty workspace snapshots are forbidden" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source root does not exist: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd -P)"
if ! git -C "$SOURCE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Source root must be a Git worktree: $SOURCE_ROOT" >&2
  exit 2
fi
GIT_TOPLEVEL="$(git -C "$SOURCE_ROOT" rev-parse --show-toplevel)"
GIT_TOPLEVEL="$(cd "$GIT_TOPLEVEL" && pwd -P)"
if [[ "$SOURCE_ROOT" != "$GIT_TOPLEVEL" ]]; then
  echo "--source-root must name the Git worktree root: $GIT_TOPLEVEL" >&2
  exit 2
fi
SOURCE_DIRTY="$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)"
if [[ -n "$SOURCE_DIRTY" ]]; then
  echo "Refusing dirty source root; commit/stash/remove all tracked and untracked changes." >&2
  exit 2
fi
SOURCE_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
SOURCE_TREE="$(git -C "$SOURCE_ROOT" rev-parse 'HEAD^{tree}')"
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
  exit 2
fi
ACTIVE_DYNOS="$(
  heroku ps --app "$APP_NAME" --json \
    | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))'
)"
if [[ "$ACTIVE_DYNOS" != "0" ]]; then
  echo "Refusing an app with $ACTIVE_DYNOS active dyno(s); use an idle staging app." >&2
  exit 2
fi
if [[ ! -f "$SOURCE_ROOT/$MODEL_REL" || ! -f "$SOURCE_ROOT/$PARITY_REL" ]]; then
  echo "Production model or parity fixture is missing" >&2
  exit 2
fi
for required_path in \
  CMakeLists.txt cmake include src experiments tools tests \
  benchmarks/uci_platform_v1.json \
  deploy/heroku/benchmark-config-v41.yml; do
  if [[ ! -e "$SOURCE_ROOT/$required_path" ]]; then
    echo "Clean source root is missing: $required_path" >&2
    exit 2
  fi
done
for staging_path in \
  deploy/heroku/Dockerfile.simd-benchmark \
  deploy/heroku/heroku-simd-benchmark.yml \
  tools/benchmark/run_heroku_nnue_forward_stages.py \
  tools/benchmark/run_local_nnue_forward_stages.py \
  tools/benchmark/run_nnue_forward_stages_remote.py; do
  if [[ ! -f "$REPO_ROOT/$staging_path" ]]; then
    echo "Workspace staging infrastructure is missing: $staging_path" >&2
    exit 2
  fi
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$REPO_ROOT/logs/nnue_forward_stages_heroku_${PROFILE}_${STAMP}"
PROVENANCE_DIR="$OUTPUT_DIR/provenance"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing artifact directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$PROVENANCE_DIR"

STAGE_DIR="$(mktemp -d)"
RELEASE_TOUCHED=0
cleanup() {
  if [[ "$RELEASE_TOUCHED" -eq 1 ]]; then
    heroku ps:scale benchmark=0 --app "$APP_NAME" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$STAGE_DIR"
}
trap cleanup EXIT

mkdir -p \
  "$STAGE_DIR/benchmarks" \
  "$STAGE_DIR/deploy/heroku" \
  "$STAGE_DIR/$(dirname "$MODEL_REL")"
cp "$SOURCE_ROOT/CMakeLists.txt" "$STAGE_DIR/"
rsync -a "$SOURCE_ROOT/cmake/" "$STAGE_DIR/cmake/"
rsync -a "$SOURCE_ROOT/include/" "$STAGE_DIR/include/"
rsync -a "$SOURCE_ROOT/src/" "$STAGE_DIR/src/"
rsync -a "$SOURCE_ROOT/experiments/" "$STAGE_DIR/experiments/"
rsync -a "$SOURCE_ROOT/tools/" "$STAGE_DIR/tools/"
rsync -a "$SOURCE_ROOT/tests/" "$STAGE_DIR/tests/"
cp "$SOURCE_ROOT/benchmarks/uci_platform_v1.json" "$STAGE_DIR/benchmarks/"
cp "$SOURCE_ROOT/deploy/heroku/benchmark-config-v41.yml" "$STAGE_DIR/deploy/heroku/"
cp "$REPO_ROOT/deploy/heroku/Dockerfile.simd-benchmark" "$STAGE_DIR/Dockerfile"
cp "$REPO_ROOT/deploy/heroku/heroku-simd-benchmark.yml" "$STAGE_DIR/heroku.yml"
cp "$REPO_ROOT/tools/benchmark/run_nnue_forward_stages_remote.py" \
  "$STAGE_DIR/tools/benchmark/run_nnue_forward_stages_remote.py"
cp "$SOURCE_ROOT/$MODEL_REL" "$STAGE_DIR/$MODEL_REL"
cp "$SOURCE_ROOT/$PARITY_REL" "$STAGE_DIR/$PARITY_REL"

if find "$STAGE_DIR" -type f \
    \( -name '.env' -o -name '.env.*' -o -name '.token.env' \
       -o -name '.netrc' -o -name 'credentials.json' -o -name 'token.txt' \
       -o -name '*.pem' -o -name '*.key' \
       -o -name 'id_rsa' -o -name 'id_ed25519' \) \
    -print -quit | grep -q .; then
  echo "Refusing staging snapshot containing an environment/token file" >&2
  exit 2
fi

git -C "$STAGE_DIR" init -q
git -C "$STAGE_DIR" config user.name "Chess Forward Stage Staging"
git -C "$STAGE_DIR" config user.email "staging@localhost"
git -C "$STAGE_DIR" add .
SNAPSHOT_TREE="$(git -C "$STAGE_DIR" write-tree)"
git -C "$STAGE_DIR" commit -qm "Benchmark NNUE forward stages $SNAPSHOT_TREE"
SNAPSHOT_COMMIT="$(git -C "$STAGE_DIR" rev-parse HEAD)"

git -C "$STAGE_DIR" archive --format=tar.gz \
  --output "$PROVENANCE_DIR/staged-source.tar.gz" HEAD
tar -tzf "$PROVENANCE_DIR/staged-source.tar.gz" >/dev/null
cp "$STAGE_DIR/Dockerfile" "$PROVENANCE_DIR/Dockerfile"
cp "$STAGE_DIR/heroku.yml" "$PROVENANCE_DIR/heroku.yml"
cp "$STAGE_DIR/tools/benchmark/run_nnue_forward_stages_remote.py" \
  "$PROVENANCE_DIR/remote-runner.py"
cp "$REPO_ROOT/tools/benchmark/run_heroku_nnue_forward_stages.py" \
  "$PROVENANCE_DIR/local-heroku-runner.py"
cp "$REPO_ROOT/tools/benchmark/run_local_nnue_forward_stages.py" \
  "$PROVENANCE_DIR/local-arm-runner.py"
cp "$REPO_ROOT/deploy/heroku/run-forward-stage-staging.sh" \
  "$PROVENANCE_DIR/staging-script.sh"
printf '%s\n' \
  "source_commit=$SOURCE_COMMIT" \
  "source_tree=$SOURCE_TREE" \
  "staged_commit=$SNAPSHOT_COMMIT" \
  "staged_tree=$SNAPSHOT_TREE" \
  > "$PROVENANCE_DIR/source-identity.txt"

write_provenance_hashes() {
  (
    cd "$OUTPUT_DIR"
    ARTIFACT_FILES=(
      provenance/Dockerfile
      provenance/heroku.yml
      provenance/local-arm-runner.py
      provenance/local-heroku-runner.py
      provenance/remote-runner.py
      provenance/source-identity.txt
      provenance/staged-source.tar.gz
      provenance/staging-script.sh
    )
    if [[ -f heroku-build.log ]]; then
      ARTIFACT_FILES+=(heroku-build.log)
    fi
    shasum -a 256 "${ARTIFACT_FILES[@]}" > provenance/SHA256SUMS
  )
}
write_provenance_hashes

echo "Staging app: $APP_NAME"
echo "Clean source commit: $SOURCE_COMMIT"
echo "Clean source tree: $SOURCE_TREE"
echo "Snapshot commit: $SNAPSHOT_COMMIT"
echo "Snapshot tree: $SNAPSHOT_TREE"
echo "Deploying benchmark-only image; formation remains benchmark=0."
git -C "$STAGE_DIR" remote add heroku "https://git.heroku.com/$APP_NAME.git"
git -C "$STAGE_DIR" push heroku HEAD:main --force 2>&1 \
  | tee "$OUTPUT_DIR/heroku-build.log"
RELEASE_TOUCHED=1
heroku ps:scale benchmark=0 --app "$APP_NAME" >/dev/null
write_provenance_hashes
if [[ "$DEPLOY_ONLY" -eq 1 ]]; then
  echo "Staging image deployed at formation benchmark=0."
  echo "Artifacts: $OUTPUT_DIR"
  exit 0
fi

RUNNER_COMMAND=(
  python3 "$REPO_ROOT/tools/benchmark/run_heroku_nnue_forward_stages.py"
  --app "$APP_NAME"
  --production-app "$PRODUCTION_APP"
  --process-type benchmark
  --profile "$PROFILE"
  --source-label "source-commit:$SOURCE_COMMIT:source-tree:$SOURCE_TREE:staging-tree:$SNAPSHOT_TREE"
  --output-dir "$OUTPUT_DIR"
  --local-model "$SOURCE_ROOT/$MODEL_REL"
)
if [[ -n "$RUNS" ]]; then
  RUNNER_COMMAND+=(--runs "$RUNS")
fi
"${RUNNER_COMMAND[@]}"

heroku ps:scale benchmark=0 --app "$APP_NAME" >/dev/null
echo "All VNNI parity, identity, checksum, and stage timing gates passed."
echo "Artifacts: $OUTPUT_DIR"
