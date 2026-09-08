#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="${REPO_ROOT:-/workspace/Chess}"
cd "$repo_root"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

base_tag="${BASE_TAG:-old_score_huber200_f2m_independent_500m_20260824}"
tag="${TAG:-${base_tag}_low_lr_e13e14}"
data_dir="${DATA_DIR:-$repo_root/data/robotmoon_test80_2024_500m_unique_v2_shards_1m}"
data_manifest="${DATA_MANIFEST:-$repo_root/data/robotmoon_test80_2024_500m_unique_v2.manifest.json}"
base_dir="$repo_root/models/quantized_scale_grid/$base_tag"
base_checkpoint="$base_dir/phase_component_best.pt"
output_dir="$repo_root/models/quantized_scale_grid/$tag"
log="$repo_root/logs/$tag.runpod.log"
status="$repo_root/logs/$tag.runpod.status"
terminate_log="$output_dir/pod_terminate.log"

workers="${WORKERS:-14}"
eval_workers="${EVAL_WORKERS:-10}"
torch_threads="${TORCH_THREADS:-2}"
auto_terminate_pod="${AUTO_TERMINATE_POD:-0}"
pod_id="${POD_ID:-${RUNPOD_POD_ID:-}}"
dry_run="${DRY_RUN:-0}"

# Epochs 1-12 reproduce the schedule stored in the selected checkpoint. Only
# epochs 13 and 14 are executed after resuming epoch 12. They deliberately use
# constant, smaller rates so this is a clean low-LR branch, not a continuation
# from the worse epoch-14 optimizer state.
epoch_peak_lrs=(
  0.000500 0.000100 0.000060 0.000040
  0.000025 0.0000175 0.00001225 0.000008575
  0.000006 0.000005 0.000005 0.000005
  0.0000025 0.000001
)
epoch_min_lrs=(
  0.000050 0.000060 0.000040 0.000025
  0.0000175 0.00001225 0.000008575 0.000006
  0.000005 0.000005 0.000005 0.000005
  0.0000025 0.000001
)

if [[ "$auto_terminate_pod" != "0" && "$auto_terminate_pod" != "1" ]]; then
  printf 'AUTO_TERMINATE_POD must be 0 or 1, got %q\n' "$auto_terminate_pod" >&2
  exit 2
fi

mkdir -p "$output_dir" "$repo_root/logs"
progress_checkpoint="$output_dir/phase_epoch_in_progress.pt"
current_checkpoint="$output_dir/phase_component_current.pt"
selected_checkpoint="$output_dir/phase_component_best.pt"

if [[ -f "$progress_checkpoint" ]]; then
  resume_checkpoint="$progress_checkpoint"
  resume_source="in_epoch"
elif [[ -f "$current_checkpoint" ]]; then
  resume_checkpoint="$current_checkpoint"
  resume_source="current"
else
  resume_checkpoint="$selected_checkpoint"
  resume_source="epoch12_best"
fi

train_command=(
  python3 tools/train/train_phase_nnue_until_overfit.py
  --data "$data_dir"
  --output-dir "$output_dir"
  --arch F2M
  --phase-layout independent
  --epochs 14
  --patience 2
  --min-delta-loss 0.00005
  --batch-size 8192
  --workers "$workers"
  --eval-workers "$eval_workers"
  --torch-threads "$torch_threads"
  --device cuda
  --train-max-samples 490000000
  --val-max-samples 5000000
  --test-max-samples 5000000
  --train-probe-max-samples 1000000
  --split-mod 100 --val-mod 98 --test-mod 99
  --shuffle-block-size 250000
  --lr-steps-per-epoch 59815
  --lr-warmup-steps 2000
  --epoch-peak-lrs "${epoch_peak_lrs[@]}"
  --epoch-min-lrs "${epoch_min_lrs[@]}"
  --checkpoint-samples 50000000
  --progress-batches 500
  --saturation-batches 20
  --resume-checkpoint "$resume_checkpoint"
  --skip-final-test
  --seed 20260824
)

if [[ "$dry_run" == "1" ]]; then
  printf 'dry_run=pass resume_source=%s resume_checkpoint=%s peak_lrs=%s min_lrs=%s auto_terminate=%s\n' \
    "$resume_source" "$resume_checkpoint" \
    "${#epoch_peak_lrs[@]}" "${#epoch_min_lrs[@]}" "$auto_terminate_pod"
  printf 'command='
  printf '%q ' "${train_command[@]}"
  printf '\n'
  exit 0
fi

load_runpod_runtime() {
  if [[ -r /etc/rp_environment ]]; then
    set +u
    set -a
    # shellcheck disable=SC1091
    source /etc/rp_environment
    set +a
    set -u
  fi
  resolved_pod_id="$pod_id"
  if [[ -z "$resolved_pod_id" ]]; then
    resolved_pod_id="${RUNPOD_POD_ID:-${RUNPOD_PODID:-}}"
  fi
}

on_exit() {
  local code=$?
  local resolved_pod_id=""
  local terminate_code=0
  trap - EXIT

  if (( code != 0 )); then
    printf 'state=failed exit_code=%s updated_at=%s\n' \
      "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
  fi
  if [[ "$auto_terminate_pod" == "1" ]]; then
    load_runpod_runtime
    if [[ -z "$resolved_pod_id" ]]; then
      printf 'state=pod_terminate_failed reason=missing_pod_id updated_at=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
      exit 3
    fi
    printf 'state=pod_terminate_requested pod_id=%s training_exit_code=%s updated_at=%s\n' \
      "$resolved_pod_id" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
    sync
    set +e
    runpodctl remove pod "$resolved_pod_id" >> "$terminate_log" 2>&1
    terminate_code=$?
    set -e
    if (( terminate_code != 0 && code == 0 )); then
      code=$terminate_code
    fi
  fi
  exit "$code"
}
trap on_exit EXIT

printf 'state=preflight tag=%s resume_source=%s updated_at=%s\n' \
  "$tag" "$resume_source" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"

if [[ "$auto_terminate_pod" == "1" ]] && ! command -v runpodctl >/dev/null 2>&1; then
  printf 'runpodctl is required when AUTO_TERMINATE_POD=1\n' >&2
  exit 2
fi
if [[ ! -f "$base_checkpoint" ]]; then
  printf 'missing selected epoch-12 checkpoint: %s\n' "$base_checkpoint" >&2
  exit 2
fi

python3 - "$data_dir" "$data_manifest" "$base_checkpoint" "$workers" <<'PY'
import json
import sys
from pathlib import Path

import torch

from chess_nnue.compact_board_data import validate_compact_dataset

data, manifest_path, checkpoint_path = map(Path, sys.argv[1:4])
workers = int(sys.argv[4])
paths = validate_compact_dataset(data)
manifest = json.loads(manifest_path.read_text())
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if len(paths) != 500 or manifest["output"]["records"] != 500_000_000:
    raise SystemExit("500M corpus manifest/shard count mismatch")
if manifest["validation"]["independent_128bit_fingerprint_duplicates"] != 0:
    raise SystemExit("500M corpus is not unique")
if int(checkpoint.get("epoch", -1)) != 12:
    raise SystemExit(f"base checkpoint must be epoch 12, got {checkpoint.get('epoch')!r}")
if abs(float(checkpoint.get("best_val_loss", 0.0)) - 0.06424574248955349) > 1e-12:
    raise SystemExit("base checkpoint validation loss mismatch")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
cpu_count = __import__("os").cpu_count() or 1
if workers >= cpu_count:
    raise SystemExit(f"workers={workers} leaves no CPU for trainer (cpus={cpu_count})")
print(
    "low_lr_preflight=pass "
    f"gpu={torch.cuda.get_device_name(0)!r} checkpoint_epoch=12 "
    f"best_val_loss={checkpoint['best_val_loss']} workers={workers}"
)
PY

if [[ "$resume_source" == "epoch12_best" ]]; then
  cp -p "$base_checkpoint" "$selected_checkpoint"
fi

printf 'state=training tag=%s resume_source=%s lr13=2.5e-6 lr14=1e-6 updated_at=%s\n' \
  "$tag" "$resume_source" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
PYTHONUNBUFFERED=1 "${train_command[@]}" >> "$log" 2>&1

printf 'state=verifying tag=%s updated_at=%s\n' \
  "$tag" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
python3 tools/train/verify_phase_training_checkpoint.py \
  --checkpoint "$selected_checkpoint" \
  --arch F2M \
  --phase-layout independent > "$output_dir/checkpoint_verification.json"
python3 tools/train/export_phase_quantized_nnue.py \
  --checkpoint "$selected_checkpoint" \
  --output "$output_dir/phase_quantized_nnue.bin" \
  --parity-data "$data_dir" \
  --parity-output "$output_dir/phase_quantized_nnue_parity.tsv" \
  --parity-samples 16384 > "$output_dir/export.log"

printf 'state=complete tag=%s updated_at=%s\n' \
  "$tag" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
