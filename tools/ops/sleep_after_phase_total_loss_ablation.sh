#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${1:?usage: sleep_after_phase_total_loss_ablation.sh TAG}"
STATUS="logs/$TAG.status"
WATCH_STATUS="logs/${TAG}_sleep.status"

printf 'state=waiting training_status=%s updated_at=%s\n' \
  "$STATUS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$WATCH_STATUS"

while true; do
  if grep -q '^state=complete ' "$STATUS" 2>/dev/null; then
    printf 'state=sleeping updated_at=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_STATUS"
    /usr/bin/pmset sleepnow
    exit 0
  fi
  if grep -q '^state=failed ' "$STATUS" 2>/dev/null; then
    printf 'state=cancelled_reason_training_failed updated_at=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_STATUS"
    exit 1
  fi
  sleep 30
done
