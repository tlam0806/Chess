#!/usr/bin/env bash
set -euo pipefail

tag="${TAG:?TAG must name the long-run grid}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

log="logs/${tag}_sleep_watcher.log"
report="reports/${tag}.md"
printf '%s watcher_started tag=%s\n' "$(date '+%F %T')" "$tag" >>"$log"

caffeinate -dimsu &
caffeinate_pid=$!
cleanup() {
  kill "$caffeinate_pid" 2>/dev/null || true
  wait "$caffeinate_pid" 2>/dev/null || true
}
trap cleanup EXIT

while true; do
  screens="$(screen -list 2>/dev/null || true)"
  if ! rg -q \
    '[.]nnue_ablation_5m|[.]nnue_ablation_metrics|[.]nnue_formula_longrun' \
    <<<"$screens"; then
    break
  fi
  sleep 30
done

if [[ ! -s "$report" ]]; then
  printf '%s not_sleeping reason=missing_report report=%s\n' \
    "$(date '+%F %T')" "$report" >>"$log"
  exit 1
fi

cleanup
trap - EXIT
printf '%s training_complete sleeping report=%s\n' \
  "$(date '+%F %T')" "$report" >>"$log"
osascript -e 'tell application "System Events" to sleep'
