#!/bin/bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
run_dir="${1:?usage: run_nnue_v43_selfplay_race.sh RUN_DIR [runner args...]}"
shift

command=("$repo_dir/.venv/bin/python"
  "$repo_dir/tools/match/run_nnue_v43_selfplay_race.py"
  --run-dir "$run_dir" "$@")

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu "${command[@]}"
fi
exec "${command[@]}"
