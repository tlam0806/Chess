#!/bin/zsh
set -u

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h}"
cd "$repo_root"

main_session="stockfish_static_nnue_200m"
status_file="logs/stockfish_static_nnue_psqtpos_200m.status"
watcher_log="logs/stockfish_static_nnue_psqtpos_200m_sleep_watcher.log"

print "$(date '+%F %T') watcher_started session=$main_session" >> "$watcher_log"

# Prevent idle sleep while collecting, labelling, and validating. Closing the
# lid can still suspend a Mac; the processes resume after it wakes.
caffeinate -dimsu &
caffeinate_pid=$!
cleanup() {
  kill "$caffeinate_pid" 2>/dev/null || true
  wait "$caffeinate_pid" 2>/dev/null || true
}
trap cleanup EXIT

while screen -ls 2>/dev/null \
    | grep -Eq "[.]${main_session}[[:space:]]"; do
  sleep 30
done

state="$(sed -n 's/^state=\([^ ]*\).*/\1/p' "$status_file" 2>/dev/null)"
if [[ "$state" != "complete" ]]; then
  print "$(date '+%F %T') not_sleeping state=${state:-missing}" >> "$watcher_log"
  exit 1
fi

cleanup
trap - EXIT
print "$(date '+%F %T') pipeline_complete sleeping" >> "$watcher_log"
if ! osascript -e 'tell application "System Events" to sleep' >> "$watcher_log" 2>&1; then
  print "$(date '+%F %T') apple_events_sleep_failed fallback=pmset" >> "$watcher_log"
  /usr/bin/pmset sleepnow >> "$watcher_log" 2>&1
fi
