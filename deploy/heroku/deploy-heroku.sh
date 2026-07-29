#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOT_SOURCE="${LICHESS_BOT_SOURCE:-/Users/tunglamnguyen/lichess-bot}"
TOKEN_FILE="${LICHESS_TOKEN_FILE:-$BOT_SOURCE/.token.env}"
MODEL_REL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"
APP_NAME="${1:-}"

if ! command -v heroku >/dev/null; then
  echo "Heroku CLI is missing. Install once with:"
  echo "  brew tap heroku/brew && brew install heroku"
  exit 2
fi
if ! heroku auth:whoami >/dev/null 2>&1; then
  echo "Opening Heroku login..."
  heroku login
fi
if [[ ! -d "$BOT_SOURCE" || ! -f "$BOT_SOURCE/requirements.txt" ]]; then
  echo "Missing lichess-bot source: $BOT_SOURCE"
  exit 2
fi
if [[ ! -f "$REPO_ROOT/$MODEL_REL" ]]; then
  echo "Missing NNUE model: $REPO_ROOT/$MODEL_REL"
  exit 2
fi

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

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

mkdir -p \
  "$STAGE_DIR/deploy/heroku" \
  "$STAGE_DIR/deploy/lichess" \
  "$STAGE_DIR/$(dirname "$MODEL_REL")"
cp "$REPO_ROOT/CMakeLists.txt" "$STAGE_DIR/"
rsync -a "$REPO_ROOT/include/" "$STAGE_DIR/include/"
rsync -a "$REPO_ROOT/src/" "$STAGE_DIR/src/"
rsync -a "$REPO_ROOT/tools/" "$STAGE_DIR/tools/"
cp "$REPO_ROOT/deploy/heroku/Dockerfile" "$STAGE_DIR/deploy/heroku/"
cp "$REPO_ROOT/deploy/heroku/heroku.yml" "$STAGE_DIR/heroku.yml"
cp "$REPO_ROOT/deploy/lichess/config-nnue-v38.yml" \
  "$STAGE_DIR/deploy/lichess/"
cp "$REPO_ROOT/$MODEL_REL" "$STAGE_DIR/$MODEL_REL"
rsync -a \
  --exclude '.git/' \
  --exclude 'logs/' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  "$BOT_SOURCE/" "$STAGE_DIR/lichess-bot/"

git -C "$STAGE_DIR" init -q
git -C "$STAGE_DIR" config user.name "Chess Bot Deploy"
git -C "$STAGE_DIR" config user.email "deploy@localhost"
git -C "$STAGE_DIR" add .
git -C "$STAGE_DIR" commit -qm "Deploy NNUE V38 Lichess bot"

if [[ -z "$APP_NAME" ]]; then
  APP_NAME="$(heroku create --stack container --json | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["name"])')"
else
  if ! heroku apps:info --app "$APP_NAME" >/dev/null 2>&1; then
    heroku create "$APP_NAME" --stack container
  else
    heroku stack:set container --app "$APP_NAME"
  fi
fi

heroku config:set \
  "LICHESS_BOT_TOKEN=$LICHESS_BOT_TOKEN" \
  --app "$APP_NAME"
git -C "$STAGE_DIR" remote add heroku \
  "https://git.heroku.com/$APP_NAME.git"
git -C "$STAGE_DIR" push heroku HEAD:main --force
heroku ps:scale worker=1:basic --app "$APP_NAME"

echo
echo "Heroku app: $APP_NAME"
echo "Live logs: heroku logs --tail --app $APP_NAME"
heroku ps --app "$APP_NAME"
