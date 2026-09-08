#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 TUNER_PID SOURCE_RUN_DIR OUTPUT_RUN_DIR" >&2
    exit 2
fi

tuner_pid="$1"
source_run_dir="$2"
output_run_dir="$3"
results_log="${source_run_dir}/results.jsonl"
watcher_log="${output_run_dir}/watcher.log"

mkdir -p "${output_run_dir}"
exec >>"${watcher_log}" 2>&1

echo "watcher_started tuner_pid=${tuner_pid} source=${source_run_dir}"

while kill -0 "${tuner_pid}" 2>/dev/null; do
    if [[ -f "${results_log}" ]] &&
       rg -q '"kind": "selection"' "${results_log}"; then
        echo "selection_detected stopping_capped_pipeline"
        pkill -TERM -P "${tuner_pid}" 2>/dev/null || true
        kill -TERM "${tuner_pid}" 2>/dev/null || true
        while kill -0 "${tuner_pid}" 2>/dev/null; do
            sleep 1
        done
        break
    fi
    sleep 5
done

if ! [[ -f "${results_log}" ]] ||
   ! rg -q '"kind": "selection"' "${results_log}"; then
    echo "watcher_abort tuner_ended_before_selection" >&2
    exit 1
fi

echo "selection_all_started"
python3 tools/tune/tune_nnue_v38_selective.py \
    --binary build-release/evaluate_nnue_v38_selective \
    --dataset-dir "${source_run_dir}/dataset" \
    --model models/quantized_scale_grid/old_score_huber200_lr_sweep_then_5ep_20260724_142758/best/phase_quantized_nnue.bin \
    --run-dir "${output_run_dir}" \
    --duration-sec 28800 \
    --selection-only-from-log "${results_log}" \
    --fixed-tune-size 1000 \
    --wide-mutations \
    --frontier-cap 0 \
    --ranking-target-abs-cp 1500 \
    --tune-depth 6 \
    --selection-depth 6 \
    --holdout-depth 7 \
    --seed 20260728
echo "selection_all_complete"
