#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOT_SOURCE="${LICHESS_BOT_SOURCE:-/Users/tunglamnguyen/lichess-bot}"
TOKEN_FILE="${LICHESS_TOKEN_FILE:-$BOT_SOURCE/.token.env}"
MODEL_REL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"
ENGINE_TARGET="uci_nnue_v43"
PRODUCTION_CONFIG_HASH="28c848b51bd93c402e873a0154e1fc61c953efa683898dc4a67d1fecd5a76aa8"
APP_NAME="${1:-}"
SOURCE_REF="${CHESS_DEPLOY_SOURCE_REF:-HEAD}"
SET_LICHESS_TOKEN="${CHESS_SET_LICHESS_TOKEN:-0}"
DEPLOY_SCALE="${CHESS_DEPLOY_SCALE:-1}"
DYNO_SIZE="${CHESS_HEROKU_DYNO_SIZE:-basic}"

if [[ "$SET_LICHESS_TOKEN" != "0" && "$SET_LICHESS_TOKEN" != "1" ]]; then
  echo "CHESS_SET_LICHESS_TOKEN must be 0 or 1"
  exit 2
fi
if [[ "$DEPLOY_SCALE" != "0" && "$DEPLOY_SCALE" != "1" ]]; then
  echo "CHESS_DEPLOY_SCALE must be 0 or 1"
  exit 2
fi

if ! command -v heroku >/dev/null; then
  echo "Heroku CLI is missing. Install once with:"
  echo "  brew tap heroku/brew && brew install heroku"
  exit 2
fi
if ! heroku auth:whoami >/dev/null 2>&1; then
  echo "Heroku CLI is not logged in. Run: heroku login"
  exit 2
fi
if [[ ! -d "$BOT_SOURCE/.git" || ! -f "$BOT_SOURCE/requirements.txt" ]]; then
  echo "Missing lichess-bot source: $BOT_SOURCE"
  exit 2
fi
if [[ ! -f "$REPO_ROOT/$MODEL_REL" ]]; then
  echo "Missing NNUE model: $REPO_ROOT/$MODEL_REL"
  exit 2
fi

SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse --verify "$SOURCE_REF^{commit}")"
SOURCE_TREE="$(git -C "$REPO_ROOT" rev-parse "$SOURCE_COMMIT^{tree}")"
BOT_COMMIT="$(git -C "$BOT_SOURCE" rev-parse --verify HEAD^{commit})"

load_lichess_token() {
  if [[ -f "$TOKEN_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$TOKEN_FILE"
  fi
  if [[ -z "${LICHESS_BOT_TOKEN:-}" ]]; then
    read -r -s -p "Lichess bot token: " LICHESS_BOT_TOKEN
    echo
  fi
  LICHESS_BOT_TOKEN="$(printf '%s' "$LICHESS_BOT_TOKEN" | tr -d '[:space:]')"
  if [[ -z "$LICHESS_BOT_TOKEN" ]]; then
    echo "Lichess token is empty"
    exit 2
  fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

mkdir -p "$STAGE_DIR/lichess-bot" "$STAGE_DIR/$(dirname "$MODEL_REL")"

# Package the exact committed engine tree, never the surrounding dirty
# working tree. The ignored production model is copied separately below.
git -C "$REPO_ROOT" archive --format=tar "$SOURCE_COMMIT" -- \
  CMakeLists.txt include src tools tests benchmarks \
  deploy/heroku/Dockerfile deploy/heroku/heroku.yml \
  deploy/lichess/config-nnue-v43.yml \
  | tar -xf - -C "$STAGE_DIR"
cp "$STAGE_DIR/deploy/heroku/Dockerfile" "$STAGE_DIR/Dockerfile"
cp "$STAGE_DIR/deploy/heroku/heroku.yml" "$STAGE_DIR/heroku.yml"
cp "$REPO_ROOT/$MODEL_REL" "$STAGE_DIR/$MODEL_REL"

# Package only files committed by upstream lichess-bot. Local tokens, helper
# scripts and editor files cannot enter the release image.
git -C "$BOT_SOURCE" archive --format=tar "$BOT_COMMIT" \
  | tar -xf - -C "$STAGE_DIR/lichess-bot"

MODEL_SHA256="$(sha256_file "$STAGE_DIR/$MODEL_REL")"
CONFIG_SHA256="$(sha256_file "$STAGE_DIR/deploy/lichess/config-nnue-v43.yml")"
SUITE_SHA256="$(sha256_file "$STAGE_DIR/benchmarks/uci_platform_v1.json")"
python3 "$STAGE_DIR/tools/nnue_v43_production_profile.py" \
  "$STAGE_DIR/deploy/lichess/config-nnue-v43.yml" \
  --expect "$PRODUCTION_CONFIG_HASH" >/dev/null
python3 - "$STAGE_DIR/release-manifest.json" \
  "$SOURCE_COMMIT" "$SOURCE_TREE" "$BOT_COMMIT" \
  "$MODEL_REL" "$MODEL_SHA256" "$CONFIG_SHA256" "$SUITE_SHA256" \
  "$ENGINE_TARGET" "$PRODUCTION_CONFIG_HASH" <<'PY'
import json
import sys

(
    output,
    source_commit,
    source_tree,
    lichess_bot_commit,
    model_path,
    model_sha256,
    config_sha256,
    suite_sha256,
    engine_target,
    production_config_hash,
) = sys.argv[1:]
payload = {
    "schema_version": 2,
    "source_commit": source_commit,
    "source_tree": source_tree,
    "lichess_bot_commit": lichess_bot_commit,
    "engine_target": engine_target,
    "production_config_hash": production_config_hash,
    "model_path": model_path,
    "model_sha256": model_sha256,
    "config_sha256": config_sha256,
    "suite_sha256": suite_sha256,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

git -C "$STAGE_DIR" init -q
git -C "$STAGE_DIR" config user.name "Chess Bot Deploy"
git -C "$STAGE_DIR" config user.email "deploy@localhost"
git -C "$STAGE_DIR" add -f .
if git -C "$STAGE_DIR" ls-files \
    | grep -Eq '(^|/)(\.token\.env|\.env($|\.))'; then
  echo "Refusing deploy: a secret environment file was staged"
  exit 2
fi

git -C "$STAGE_DIR" commit -qm "Deploy provisional NNUE V45 pruning profile"

APP_CREATED=0
if [[ -z "$APP_NAME" ]]; then
  APP_NAME="$(heroku create --stack container --json | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["name"])')"
  APP_CREATED=1
else
  if ! heroku apps:info --app "$APP_NAME" >/dev/null 2>&1; then
    heroku create "$APP_NAME" --stack container
    APP_CREATED=1
  else
    if ! heroku stack --app "$APP_NAME" | grep -Eq '^\* +container$'; then
      echo "Existing app must already use the container stack: $APP_NAME"
      exit 2
    fi
  fi
fi

if [[ "$APP_CREATED" == "1" || "$SET_LICHESS_TOKEN" == "1" ]]; then
  load_lichess_token
  heroku config:set \
    "LICHESS_BOT_TOKEN=$LICHESS_BOT_TOKEN" \
    --app "$APP_NAME" >/dev/null
elif ! heroku config --json --app "$APP_NAME" \
    | python3 -c 'import json,sys; raise SystemExit(0 if "LICHESS_BOT_TOKEN" in json.load(sys.stdin) else 1)'; then
  echo "Existing app has no LICHESS_BOT_TOKEN config key"
  exit 2
fi
git -C "$STAGE_DIR" remote add heroku \
  "https://git.heroku.com/$APP_NAME.git"
git -C "$STAGE_DIR" push heroku HEAD:main --force
if [[ "$DEPLOY_SCALE" == "1" ]]; then
  heroku ps:scale "worker=1:$DYNO_SIZE" --app "$APP_NAME"
else
  echo "Release deployed with formation unchanged (CHESS_DEPLOY_SCALE=0)."
fi

echo
echo "Heroku app: $APP_NAME"
echo "Source commit: $SOURCE_COMMIT"
echo "Source tree: $SOURCE_TREE"
echo "lichess-bot commit: $BOT_COMMIT"
echo "Model SHA-256: $MODEL_SHA256"
echo "Live logs: heroku logs --tail --app $APP_NAME"
heroku ps --app "$APP_NAME"
