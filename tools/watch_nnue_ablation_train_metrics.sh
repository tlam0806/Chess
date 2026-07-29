#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tag="${TAG:?TAG must name the active ablation run}"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
metrics_dir="reports/${tag}_train_metrics"
mkdir -p "$metrics_dir"
branches=(float_cp_mse float_cp_huber quantized_cp_mse quantized_cp_huber)

while true; do
  all_complete=1
  for branch in "${branches[@]}"; do
    log="logs/${tag}_${branch}.log"
    checkpoint="models/${tag}/${branch}/quant_nnue_arch_F2_current.pt"
    if [[ ! -f "$log" ]]; then
      all_complete=0
      continue
    fi
    epoch="$($python_bin - "$log" <<'PY'
import json, sys
latest = 0
for line in open(sys.argv[1]):
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if event.get("event") == "epoch":
        latest = max(latest, int(event["epoch"]))
print(latest)
PY
)"
    if [[ "$epoch" -gt 0 && -f "$checkpoint" ]]; then
      output="$metrics_dir/${branch}_epoch${epoch}.json"
      if [[ ! -f "$output" ]]; then
        if ! "$python_bin" tools/evaluate_nnue_regression_metrics.py \
          --checkpoint "$checkpoint" \
          --expected-epoch "$epoch" \
          --data data/robotmoon_ablation_fixed5m_unique_v1 \
          --data-format cbin \
          --split all \
          --max-samples 500000 \
          --batch-size 8192 \
          --workers 0 \
          --torch-threads 1 \
          --output "$output" \
          >"$output.tmp" 2>&1; then
          rm -f "$output" "$output.tmp"
        else
          rm -f "$output.tmp"
        fi
      fi
    fi
    if ! rg -q '"event":"training_complete"' "$log"; then
      all_complete=0
    fi
  done
  if [[ "$all_complete" -eq 1 ]]; then
    break
  fi
  sleep 5
done
