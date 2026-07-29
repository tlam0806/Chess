#!/bin/zsh
set -u

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h}"
cd "$repo_root"

status_file="logs/stockfish_static_nnue_psqtpos_200m.status"
log="logs/stockfish_static_nnue_psqtpos_200m.log"

write_status() {
  print "state=$1 updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
}

write_status queued
while screen -ls 2>/dev/null \
    | grep -Eq '[.]nnue_relu16_low_hs_10ep|[.]nnue_relu16_holdout_queue'; do
  sleep 30
done

write_status running
STOCKFISH_LABEL_WORKERS="${STOCKFISH_LABEL_WORKERS:-5}" \
  tools/collect_stockfish_static_nnue_200m.sh >> "$log" 2>&1
code=$?
if [[ $code -eq 0 ]]; then
  write_status complete
else
  print "state=failed exit_code=$code updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
fi
exit $code
