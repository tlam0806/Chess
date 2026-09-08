#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/workspace/Chess}"
choice_env="$repo_root/logs/f2m_worker_choice.env"

cd "$repo_root"
bash tools/benchmark/benchmark_f2m_runpod_workers.sh

# The file is generated locally by the benchmark above and contains only two
# integer assignments: WORKERS and EVAL_WORKERS.
set -a
source "$choice_env"
set +a

exec bash tools/train/run_old_score_huber200_f2m_500m_runpod.sh
