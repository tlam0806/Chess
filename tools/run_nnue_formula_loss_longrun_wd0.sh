#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
tag="${TAG:-nnue_formula_loss_longrun_wd0_$(date +%Y%m%d_%H%M%S)}"
jobs="${JOBS:-4}"
mkdir -p logs "models/$tag" reports

specs=(
  "c181d128_hs32x16__mse 181 128 32 16 cp_mse 200 0.75"
  "c181d128_hs32x16__mix75 181 128 32 16 cp_mse_huber 200 0.75"
  "c181d128_hs32x16__huber600 181 128 32 16 cp_huber 600 0.75"
  "c181d128_hs32x16__huber1000 181 128 32 16 cp_huber 1000 0.75"
  "c181d128_hs91x91__mse 181 128 91 91 cp_mse 200 0.75"
  "c181d128_hs91x91__mix75 181 128 91 91 cp_mse_huber 200 0.75"
  "c181d128_hs91x91__huber600 181 128 91 91 cp_huber 600 0.75"
  "c181d128_hs91x91__huber1000 181 128 91 91 cp_huber 1000 0.75"
  "c255d256_hs64x64__mse 255 256 64 64 cp_mse 200 0.75"
  "c255d256_hs64x64__mix75 255 256 64 64 cp_mse_huber 200 0.75"
  "c255d256_hs64x64__huber600 255 256 64 64 cp_huber 600 0.75"
  "c255d256_hs64x64__huber1000 255 256 64 64 cp_huber 1000 0.75"
)

run_config() {
  local stage="$1" name="$2" clip="$3" divisor="$4" hs1="$5" hs2="$6"
  local loss="$7" delta="$8" mix="$9"
  local output_dir="models/$tag/$name"
  local log="logs/${tag}_${name}.log"
  local epochs=4 learning_rate=0.0005
  if [[ "$stage" == "stage2" ]]; then
    epochs=8
    learning_rate=0.0001
  fi
  local command=("$python_bin" tools/train_quantized_nnue_architecture.py \
    --arch F2 \
    --data data/robotmoon_ablation_fixed5m_unique_v1 \
    --data-format cbin \
    --train-data-all-records \
    --eval-data data/robotmoon_val500k_balanced_target75u25c_v3 \
    --eval-data-format cbin \
    --eval-data-all-records-for-val \
    --forward-mode quantized \
    --loss-type "$loss" \
    --mse-mix-weight "$mix" \
    --cp-huber-delta "$delta" \
    --activation screlu_all \
    --hidden-clip "$clip" \
    --screlu-divisor "$divisor" \
    --quantization-convention scale_clean \
    --feature-weight-scale "$clip" \
    --linear-weight-scale 64 \
    --output-weight-scale 16 \
    --fixed-hidden-scales "$hs1" "$hs2" \
    --fixed-output-scale 16 \
    --screlu-init-fraction 0.25 \
    --screlu-first-bias-fraction 0.1 \
    --epochs "$epochs" \
    --patience 9 \
    --batch-size 4096 \
    --lr "$learning_rate" \
    --lr-schedule constant \
    --weight-decay 0 \
    --device cpu \
    --workers 0 \
    --eval-workers 0 \
    --torch-threads 2 \
    --train-max-samples 5000000 \
    --val-max-samples 500000 \
    --calibration-max-batches 25 \
    --shuffle-block-size 250000 \
    --progress-batches 250 \
    --eval-progress-batches 0 \
    --log-initial-saturation \
    --skip-final-test \
    --seed 20260718 \
    --output-dir "$output_dir")
  if [[ "$stage" == "stage2" ]]; then
    command+=(
      --resume-checkpoint "$output_dir/quant_nnue_arch_F2_current.pt"
    )
  fi
  "${command[@]}" >>"$log" 2>&1
}

run_grouped() {
  local stage="$1"; shift
  local pids=() names=() spec active=0
  for spec in "$@"; do
    read -r name clip divisor hs1 hs2 loss delta mix <<<"$spec"
    run_config "$stage" "$name" "$clip" "$divisor" "$hs1" "$hs2" "$loss" "$delta" "$mix" &
    pids+=("$!")
    names+=("$name")
    active=$((active + 1))
    if [[ "$active" -ge "$jobs" ]]; then
      for index in "${!pids[@]}"; do
        wait "${pids[$index]}" || { echo "failed ${names[$index]}"; return 1; }
      done
      pids=()
      names=()
      active=0
    fi
  done
  if [[ "$active" -gt 0 ]]; then
    for index in "${!pids[@]}"; do
      wait "${pids[$index]}" || { echo "failed ${names[$index]}"; return 1; }
    done
  fi
}

selection_file="reports/${tag}_selection.json"
if [[ "${RESUME_SELECTION_ONLY:-0}" != "1" ]]; then
  run_grouped stage1 "${specs[@]}"
  mapfile_output="$($python_bin tools/select_nnue_longrun_candidates.py \
    --tag "$tag" --log-dir logs --count 4 --output "$selection_file")"
else
  mapfile_output="$($python_bin - "$selection_file" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["selected"]:
    print(row["branch"])
PY
)"
fi
selected_specs=()
while IFS= read -r selected; do
  [[ -n "$selected" ]] || continue
  for spec in "${specs[@]}"; do
    read -r name _ <<<"$spec"
    if [[ "$name" == "$selected" ]]; then
      selected_specs+=("$spec")
      break
    fi
  done
done <<<"$mapfile_output"

run_grouped stage2 "${selected_specs[@]}"

"$python_bin" tools/summarize_nnue_loss_quantization_ablation.py \
  --tag "$tag" --log-dir logs --output "reports/${tag}.md"
