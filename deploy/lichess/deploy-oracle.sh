#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 ubuntu@ORACLE_PUBLIC_IP [SSH_PRIVATE_KEY]"
  exit 2
fi

REMOTE="$1"
SSH_KEY="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOKEN_FILE="${LICHESS_TOKEN_FILE:-/Users/tunglamnguyen/lichess-bot/.token.env}"
MODEL_REL="models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin"

SSH_ARGS=(-o ServerAliveInterval=30 -o ServerAliveCountMax=4)
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY")
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
  echo "Token is empty"
  exit 2
fi

echo "Checking SSH connection to $REMOTE ..."
ssh "${SSH_ARGS[@]}" "$REMOTE" 'uname -m; printf "remote_home=%s\n" "$HOME"'

REMOTE_HOME="$(ssh "${SSH_ARGS[@]}" "$REMOTE" 'printf "%s" "$HOME"')"
REMOTE_USER="$(ssh "${SSH_ARGS[@]}" "$REMOTE" 'id -un')"
ENGINE_ROOT="$REMOTE_HOME/chess-engine"
BOT_ROOT="$REMOTE_HOME/lichess-bot"

echo "Installing server dependencies ..."
ssh "${SSH_ARGS[@]}" "$REMOTE" \
  'sudo apt-get update &&
   sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
     build-essential cmake git python3 python3-venv rsync'

echo "Uploading engine source ..."
rsync -az --delete \
  -e "ssh ${SSH_ARGS[*]}" \
  --exclude '.git/' \
  --exclude '.codex_checkpoints/' \
  --exclude 'build/' \
  --exclude 'build-*' \
  --exclude 'data/' \
  --exclude 'logs/' \
  --exclude 'reports/' \
  --exclude 'models/' \
  "$REPO_ROOT/" "$REMOTE:$ENGINE_ROOT/"

ssh "${SSH_ARGS[@]}" "$REMOTE" "mkdir -p '$ENGINE_ROOT/$(dirname "$MODEL_REL")'"
rsync -az -e "ssh ${SSH_ARGS[*]}" \
  "$REPO_ROOT/$MODEL_REL" \
  "$REMOTE:$ENGINE_ROOT/$MODEL_REL"

echo "Building ARM Release binary ..."
ssh "${SSH_ARGS[@]}" "$REMOTE" \
  "cmake -S '$ENGINE_ROOT' -B '$ENGINE_ROOT/build-release' -DCMAKE_BUILD_TYPE=Release -DCHESS_BUILD_EXPERIMENTS=ON &&
   cmake --build '$ENGINE_ROOT/build-release' --target uci_nnue_v38 -j 2"

echo "Installing lichess-bot ..."
ssh "${SSH_ARGS[@]}" "$REMOTE" \
  "if [[ ! -d '$BOT_ROOT/.git' ]]; then
     git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git '$BOT_ROOT'
   else
     git -C '$BOT_ROOT' pull --ff-only
   fi
   python3 -m venv '$BOT_ROOT/venv'
   '$BOT_ROOT/venv/bin/pip' install --upgrade pip
   '$BOT_ROOT/venv/bin/pip' install -r '$BOT_ROOT/requirements.txt'"

echo "Installing config and systemd service ..."
ssh "${SSH_ARGS[@]}" "$REMOTE" \
  "sed 's#__ENGINE_ROOT__#$ENGINE_ROOT#g' \
     '$ENGINE_ROOT/deploy/lichess/config-nnue-v38.yml' \
     > '$BOT_ROOT/config-nnue-v38.yml'"

TOKEN_TMP="$(mktemp)"
trap 'rm -f "$TOKEN_TMP"' EXIT
chmod 600 "$TOKEN_TMP"
printf 'LICHESS_BOT_TOKEN=%s\n' "$LICHESS_BOT_TOKEN" > "$TOKEN_TMP"
scp "${SSH_ARGS[@]}" "$TOKEN_TMP" "$REMOTE:/tmp/lichess-nnue-v38.env"

ssh "${SSH_ARGS[@]}" "$REMOTE" \
  "sudo install -m 600 /tmp/lichess-nnue-v38.env /etc/lichess-nnue-v38.env
   rm -f /tmp/lichess-nnue-v38.env
   sudo tee /etc/systemd/system/lichess-nnue-v38.service >/dev/null <<'EOF'
[Unit]
Description=Lichess NNUE V38 bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$REMOTE_USER
WorkingDirectory=$BOT_ROOT
EnvironmentFile=/etc/lichess-nnue-v38.env
ExecStart=$BOT_ROOT/venv/bin/python $BOT_ROOT/lichess-bot.py --config $BOT_ROOT/config-nnue-v38.yml
Restart=always
RestartSec=5
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF
   sudo systemctl daemon-reload
   sudo systemctl enable --now lichess-nnue-v38
   sleep 2
   sudo systemctl --no-pager --full status lichess-nnue-v38"

echo
echo "Deployment complete."
echo "Logs: ssh ${SSH_ARGS[*]} $REMOTE 'sudo journalctl -u lichess-nnue-v38 -f'"
