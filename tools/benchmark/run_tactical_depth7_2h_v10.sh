#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CMAKE_BIN="/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake"
OUTPUT="data/tactical_disagreement_depth7_2h_v10.jsonl"
LOG="data/tactical_disagreement_depth7_2h_v10.log"

"${CMAKE_BIN}" --build build --target tactical_disagreement_export

rm -f "${OUTPUT}"
find data -maxdepth 1 -name 'tactical_disagreement_depth7_2h_v10.jsonl.part*' -delete

./build/tactical_disagreement_export \
  --shallow-depth 3 \
  --deep-depth 7 \
  --min-diff 300 \
  --random-plies 16 \
  --max-plies 100 \
  --time-limit-seconds 7200 \
  --limit 150000 \
  --threads 6 \
  --progress-interval 5000 \
  --random-play-percent 25 \
  --seed 20260618 \
  --output "${OUTPUT}" \
  2>&1 | tee "${LOG}"
